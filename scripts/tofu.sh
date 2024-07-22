master_port=18765
model=llama2-7b
lr=1e-05
gradient_accumulation_steps=1
num_epochs=10
seed=42


usage() {
    echo "Usage: $0 [-f forget_loss] [-s save_dir_root] [-g gpu_ids] [-n num_dist]" 1>&2
    exit 1
}

while getopts ":f:s:g:n:" opt; do
    case ${opt} in
    f )
        forget_loss=$OPTARG
        ;;
    s )
        save_dir_root=$OPTARG
        ;;
    g )
        gpu_ids=$OPTARG
        ;;
    n )
        num_dist=$OPTARG
        ;;
    * )
        usage
        ;;
    esac
done


for split in forget10 # forget05 forget01
do
    if [ $forget_loss == "npo" ] || [ $forget_loss == "grad_diff" ]
    then
        retain_strength=1
    else
        retain_strength=0
    fi

    if [ $forget_loss == "intervention" ]
    then
        save_dir=${save_dir_root}/locuslab/tofu_ft_llama2-7b/${forget_loss}/${num_dist}_${seed}_${split}
    else
        save_dir=${save_dir_root}/locuslab/tofu_ft_llama2-7b/${forget_loss}/${seed}_${split}
        num_dist=1
    fi

    # Construct teacher distribution
    CUDA_VISIBLE_DEVICES=${gpu_ids} python teacher.py --config-name=forget_tofu.yaml split=${split} save_dir_root=${save_dir_root} forget_loss=${forget_loss} teacher.N=${num_dist}

    # Train student model
    CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget_tofu.yaml split=${split} model_family=${model} lr=${lr} gradient_accumulation_steps=${gradient_accumulation_steps} forget_loss=${forget_loss} num_epochs=${num_epochs} retain_strength=${retain_strength} seed=${seed} save_dir=$save_dir teacher.N=${num_dist} save_dir_root=${save_dir_root}

    # TOFU evaluation
    bash scripts/eval_all.sh -d $save_dir -m $model -s $split -g ${gpu_ids}
done
