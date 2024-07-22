master_port=18765
model=llama2-7b
num_epochs=10
gradient_accumulation_steps=10
retain_strength=0
lr=1e-05
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


for split in forget_2_1 # Possible splits: forget_2_2 forget_2_3 forget_2_4 forget_2_5 forget_20_1 forget_20_2 forget_20_3 forget_100
do
    if [ $split == "forget_100" ]
    then
        num_epochs=2
    fi

    if [[ "${split:0:9}" == "forget_2_" ]]
    then
        gradient_accumulation_steps=1
    fi

    if [ $forget_loss == "npo" ] || [ $forget_loss == "grad_diff" ]
    then
        retain_strength=1
        lr=4e-05
    fi

    if [ $forget_loss == "intervention" ]
    then
        save_dir=${save_dir_root}/meta-llama/Llama-2-7b-chat-hf/${forget_loss}/${num_dist}_${seed}_${split}
    else
        save_dir=${save_dir_root}/meta-llama/Llama-2-7b-chat-hf/${forget_loss}/${seed}_${split}
        num_dist=1
    fi

    # Construct teacher distribution
    CUDA_VISIBLE_DEVICES=${gpu_ids} python teacher.py --config-name=forget_wpu.yaml split=${split} save_dir_root=${save_dir_root} forget_loss=${forget_loss} teacher.N=${num_dist}

    # Train student model
    CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget_wpu.yaml split=${split} model_family=${model} lr=${lr} gradient_accumulation_steps=${gradient_accumulation_steps} forget_loss=${forget_loss} num_epochs=${num_epochs} retain_strength=${retain_strength} seed=${seed} save_dir=$save_dir teacher.N=${num_dist} save_dir_root=${save_dir_root}

    # MSJ attack
    CUDA_VISIBLE_DEVICES=${gpu_ids} python msj_attack.py split=${split} forget_loss=${forget_loss} save_dir=$save_dir
    # Soft embedding attack
    CUDA_VISIBLE_DEVICES=${gpu_ids} python soft_embed_attack.py split=${split} forget_loss=${forget_loss} save_dir=$save_dir
    # GPT evaluation
    CUDA_VISIBLE_DEVICES=${gpu_ids} python eval_gpt.py split=${split} forget_loss=${forget_loss} save_dir=$save_dir
done
