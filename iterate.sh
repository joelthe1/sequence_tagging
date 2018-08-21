#! /bin/bash

# set gpus to use
# export CUDA_VISIBLE_DEVICES=-1
export CUDA_VISIBLE_DEVICES=6

# set current state of iteration
state="/lfs1/joel/experiments/sequence_tagging/state.txt"
models="/lfs1/joel/experiments/sequence_tagging/model/*"

# clean previous models
rm -r $models

# declare -a increments=( "a" "b" "c" "d" )
declare -a increments=("0" "93")
iterations=10

for inc in "${increments[@]}"; do
    for iter in $(seq 1 $iterations); do
	# write current state
	printf "%s\n%s" "$inc" "$iter" > $state

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
