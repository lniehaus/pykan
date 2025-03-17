#!/bin/bash

export EXPERIMENT_NAME="wdn-coef-07"
export JOB_NUM=10
export SEED=1
export BASE_FUN="silu"
#export BASE_FUN="zero"
export DATA_NOISE_LEVEL=0.5
export STEPS=100

hidden_width=(3 2 1)
hidden_depth=(3 2 1)

#hidden_width=(2)
#hidden_depth=(3)
#spline_noise_array=(0.001 0.01 0.1 1.0 10 100 1000)
#spline_noise_array=(0.2 0.4 0.6 0.8 1 2 4 6 8 10)
spline_noise_array=(1.0 1.2 1.4 1.6 1.8 2.0)

echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"

first_execution=true  # Flag to track the first execution

for width in "${hidden_width[@]}"; do
    for depth in "${hidden_depth[@]}"; do
        for noise in "${spline_noise_array[@]}"; do
            echo "width: $width | depth: $depth | noise: $noise"
            python moons.py --device_index 0 --native_noise_scale True --hidden_width $width --hidden_depth $depth --spline_noise_scale $noise --steps $STEPS --experiment_name $EXPERIMENT_NAME --data_noise_level $DATA_NOISE_LEVEL --base_fun $BASE_FUN --seed $SEED &
            #python moons.py --device_index 1 --native_noise_scale True --sp_trainable True --sb_trainable True --affine_trainable True --update_grid True --hidden_width $width --hidden_depth $depth --spline_noise_scale $noise --steps $STEPS --experiment_name $EXPERIMENT_NAME --data_noise_level $DATA_NOISE_LEVEL --base_fun $BASE_FUN --seed $SEED &
            #exit

            sleep 1

            # Wait for 10 seconds after the first execution
            if $first_execution; then
                sleep 10
                first_execution=false  # Set the flag to false after the first execution
            fi

            # Limit the number of concurrent jobs to 10
            while [ $(jobs -r | wc -l) -ge $JOB_NUM ]; do
                sleep 1
            done
        done
    done
done

# Wait for all background jobs to finish
wait