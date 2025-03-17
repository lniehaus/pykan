#!/bin/bash

export EXPERIMENT_NAME="wdn-native-full-02"
export JOB_NUM=20
export SEED=1
export BASE_FUN="silu"
#export BASE_FUN="zero"
export DATA_NOISE_LEVEL=0.5
export STEPS=100

hidden_width=(20 15 10 5 1)
hidden_depth=(20 15 10 5 1)
spline_noise_array=(0.001 0.01 0.1 1.0 10 100 1000)

#hidden_width=(3 4)
#hidden_depth=(1 2)
#spline_noise_array=(5 6)

echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"

first_execution=true  # Flag to track the first execution

for width in "${hidden_width[@]}"; do
    for depth in "${hidden_depth[@]}"; do
        for noise in "${spline_noise_array[@]}"; do
            echo "width: $width | depth: $depth | noise: $noise"
            #python moons.py --native_noise_scale True --hidden_width $width --hidden_depth $depth --spline_noise_scale $noise --steps $STEPS --experiment_name $EXPERIMENT_NAME --data_noise_level $DATA_NOISE_LEVEL --base_fun $BASE_FUN --seed $SEED &
            python moons.py --native_noise_scale True --sp_trainable True --sb_trainable True --affine_trainable True --update_grid True --hidden_width $width --hidden_depth $depth --spline_noise_scale $noise --steps $STEPS --experiment_name $EXPERIMENT_NAME --data_noise_level $DATA_NOISE_LEVEL --base_fun $BASE_FUN --seed $SEED &
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