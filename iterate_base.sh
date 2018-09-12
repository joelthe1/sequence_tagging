#! /bin/bash

# set gpus to use
# export CUDA_VISIBLE_DEVICES=-1
export CUDA_VISIBLE_DEVICES=7

# set current state of iteration
state="/lfs1/joel/experiments/sequence_tagging/state.txt"
models="/lfs1/joel/experiments/sequence_tagging/model/*"

# clean previous models
rm -r $models

declare -a increments=("15" "1" "2" "3" "4" "5" "6" "7" "10" "20" "30" "40" "50" "60" "70" "80" "90")
# declare -a increments=("1")

for inc in "${increments[@]}"; do
    # write current state
    printf "0\n%s"  "$inc" > $state
    
    # train model
    python train_base.py
    python evaluate.py
done
