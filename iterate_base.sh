#! /bin/bash

# set gpus to use
# export CUDA_VISIBLE_DEVICES=-1
export CUDA_VISIBLE_DEVICES=7

# set current state of iteration
state="/lfs1/joel/experiments/sequence_tagging2/state.txt"
models="/lfs1/joel/experiments/sequence_tagging2/model/*"

# clean previous models
rm -r $models

declare -a increments=("1" "3" "5" "7" "10" "20" "30" "40" "50" "60" "70" "80" "90")

for inc in "${increments[@]}"; do
    # write current state
    printf "0\n%s"  "$inc" > $state
    
    # train model
    python train_base.py
    python evaluate.py
done
