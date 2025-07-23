#!/bin/bash
set -euo pipefail

GPUS_PER_NODE=4    # GPUs per node
optimizers=(APTS)
batch_sizes=(200) #(200)
epochs=25 #25
nodes_SGD_Adam=(2)  # total GPUs desired
nodes_APTS=(8)
trial_numbers=(1)
datasets=(cifar10)

SGD_cifar10_lr=0.01
Adam_cifar10_lr=0.001

partition="debug"
time="00:30:00"

# choose smallest #nodes so world_size % nodes == 0 and (world_size/nodes) ≤ GPUS_PER_NODE
calc_nodes() {
    local world_size=$1
    for n in $(seq 1 "$world_size"); do
        local tpn=$(( world_size / n ))
        if (( world_size % n == 0 && tpn <= GPUS_PER_NODE )); then
            echo "$n"
            return
        fi
    done
    echo "$world_size"
}

submit_job() {
    local opt=$1 bs=$2 ds=$3 trial=$4 lr=$5 world_size=$6
    local nodes=$(calc_nodes "$world_size")
    local tpn=$(( world_size / nodes ))
    echo "Tasks per node: ${tpn}"
    local name="${opt}_n${nodes}x${tpn}_${ds}_${bs}_lr${lr}_${epochs}_t${trial}"
    sbatch \
      --partition=${partition} \
      --nodes="$nodes" \
      --ntasks-per-node="${tpn}" \
      --gres=gpu:"${tpn}" \
      --job-name="${name}" \
      --output="log_files/${name}.out" \
      --error="log_files/${name}.err" \
      --time="${time}" \
      parallel_test.job \
         "$opt" "$bs" "$lr" "$trial" "$epochs" "$ds" "$world_size"
}

for opt in "${optimizers[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for ds in "${datasets[@]}"; do
      for trial in "${trial_numbers[@]}"; do
        if [[ "$opt" == "SGD" || "$opt" == "Adam" ]]; then
          for world in "${nodes_SGD_Adam[@]}"; do
            [[ "$opt" == "SGD" && "$ds" == "cifar10" ]] && lr=$SGD_cifar10_lr || lr=$Adam_cifar10_lr
            submit_job "$opt" "$bs" "$ds" "$trial" "$lr" "$world"
          done
        else
          for world in "${nodes_APTS[@]}"; do
            submit_job "$opt" "$bs" "$ds" "$trial" 0.001 "$world"
          done
        fi
      done
    done
  done
done
