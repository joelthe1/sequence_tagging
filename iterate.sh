#! /bin/bash

# set gpus to use
# export CUDA_VISIBLE_DEVICES=-1
export CUDA_VISIBLE_DEVICES=7

# set current state of iteration
state="/lfs1/joel/experiments/sequence_tagging3/state.txt"
models="/lfs1/joel/experiments/sequence_tagging3/model/*"

# clean previous models
rm -r $models

declare -a allsplits=("85")
# declare -a allsplits=("85" "99" "98" "97" "96" "95" "94" "93" "10" "20" "30" "40" "60" "70" "80" "90")
iterations=10

for s in "${allsplits[@]}"; do
    increments=("0")
    increments+=("$s")
    # echo "${increments[@]}"
    for inc in "${increments[@]}"; do
	for iter in $(seq 1 $iterations); do
	    # write current state
	    printf "%s\n%s\n%s" "$inc" "$iter" "$s" > $state

	    # train model
	    if [ "$inc" == "0" ]; then
		python train_base.py
		python evaluate.py
		break
	    else
		python train_iter.py
		python evaluate.py
	    fi
	done
    done
    mv /lfs1/joel/experiments/sequence_tagging3/model/0 /lfs1/joel/experiments/sequence_tagging3/model/0-"$s"
done
