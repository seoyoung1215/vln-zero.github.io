#!/bin/bash
# import run
CHUNKS=1

# # ANSR
# CONFIG_PATH="VLN_CE/vlnce_baselines/config/ANSR_baselines/ansr.yaml"
# SAVE_PATH="runs/gpt-4.1-ansr"

# ANSR Gibson
CONFIG_PATH="VLN_CE/vlnce_baselines/config/ANSR_baselines/ansr_gibson.yaml"
SAVE_PATH="runs/gpt-4.1-ansr-gibson"


# R2R
# CONFIG_PATH="VLN_CE/vlnce_baselines/config/r2r_baselines/zero_r2r.yaml"
# SAVE_PATH="runs/gpt-4.1-r2r"


# #RxR
# CONFIG_PATH="vlnce_baselines/config/rxr_baselines/zero_rxr.yaml"
# SAVE_PATH="runs/gpt-4.1-rxr" 


for IDX in $(seq 0 $((CHUNKS-1))); do
    echo $(( IDX % 8 ))
    CUDA_VISIBLE_DEVICES=$(( IDX % 8 )) python run.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --split-id $IDX \
    --result-path $SAVE_PATH &
    
done

wait

