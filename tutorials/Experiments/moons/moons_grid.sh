#!/bin/bash


export EXPERIMENT_NAME="width-depth-noise-03"
export SEED=0
export BASE_FUN="silu"
#export BASE_FUN="zero"
export TOTAL_EXPERIMENTS=7
export DATA_NOISE_LEVEL=0.5
export STEPS=100

hidden_width=(1 5 10 15 20)
hidden_depth=(1 5 10 15 20)
spline_noise_array=(0.001 0.01 0.1 1.0 10 100 1000)

#hidden_width=(3 4)
#hidden_depth=(1 2)
#spline_noise_array=(5 6)

echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
for width in "${hidden_width[@]}"; do
    for depth in "${hidden_depth[@]}"; do
        for noise in "${spline_noise_array[@]}"; do
            echo "width: $width | depth: $depth | noise: $noise"
            python moons.py --hidden_width $width --hidden_depth $depth --spline_noise_scale $noise   --steps $STEPS --experiment_name $EXPERIMENT_NAME --data_noise_level $DATA_NOISE_LEVEL --base_fun $BASE_FUN --seed $SEED &
            sleep 5
        done
    done
done

