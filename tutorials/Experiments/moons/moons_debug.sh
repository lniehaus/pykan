#!/bin/bash

export EXPERIMENT_NAME="wdn-debug"
export JOB_NUM=10
export SEED=0
export BASE_FUN="silu"
#export BASE_FUN="zero"
export DATA_NOISE_LEVEL=0.5
export STEPS=3 #100

export width=1
export depth=1
export noise=1


echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"

python moons.py --native_noise_scale True --hidden_width $width --hidden_depth $depth --spline_noise_scale $noise --steps $STEPS --experiment_name $EXPERIMENT_NAME --data_noise_level $DATA_NOISE_LEVEL --base_fun $BASE_FUN --seed $SEED

