usage() {
    echo "Usage: $0 [-d model_dir] [-m model] [-s split] [-g gpu_ids]" 1>&2
    exit 1
}

while getopts ":d:m:s:g:" opt; do
    case ${opt} in
        d )
        model_dir=$OPTARG
        ;;
        m )
        model=$OPTARG
        ;;
        s )
        split=$OPTARG
        ;;
        g )
        gpu_ids=$OPTARG
        ;;
        * )
        usage
        ;;
    esac
done

# Split gpu_ids into an array
IFS=',' read -r -a gpu_ids <<< "$gpu_ids"

echo "Model dir: $model_dir"
echo "Model: $model"
echo "Split: $split"

num_devices=2

rawcheckpoints=($(find $model_dir -maxdepth 1 -type d | grep "checkpoint-[0-9]*"))
checkpoints=($(printf "%s\n" "${rawcheckpoints[@]}" | awk -F'-' '{print $NF, $0}' | sort -n | cut -d' ' -f2-))

# Evaluate two checkpoints at a time
for ((i=0; i<${#checkpoints[@]}; i+=1)); do
    checkpoint="${checkpoints[$i]}"
    gpuid=${gpu_ids[$((i % num_devices))]}

    bash scripts/eval_and_aggregate.sh -m $model -s $split -c $checkpoint -g $gpuid &

    if (((i+1) % $num_devices == 0)); then
        wait
    fi
done
wait

# remove files starting with model-*
for ((i=0; i<${#checkpoints[@]}; i+=1)); do
    checkpoint="${checkpoints[$i]}"
    rm ${checkpoint}/model-*
done
