usage() {
    echo "Usage: $0 [-m model] [-s split] [-c checkpoint] [-g gpuid]" 1>&2
    exit 1
}

while getopts ":m:s:c:g:" opt; do
    case ${opt} in
        m )
            model=$OPTARG
            ;;
        s )
            split=$OPTARG
            ;;
        c )
            checkpoint=$OPTARG
            ;;
        g )
            gpuid=$OPTARG
            ;;
        * )
            usage
            ;;
    esac
done

echo "Model: $model"
echo "Split: $split"
echo "Checkpoint: $checkpoint"
echo "GPU ID: $gpuid"

CUDA_VISIBLE_DEVICES=$gpuid python evaluate_util.py model_family=$model \
    split=$split prompt_unlearn=false batch_size=64 model_path=$checkpoint save_dir=${checkpoint}/eval

# aggregate results
python aggregate_eval_tofu.py ckpt_result=${checkpoint}/eval/eval_log_aggregated.json save_file=${checkpoint}/eval/aggregate_stat.csv split=$split
