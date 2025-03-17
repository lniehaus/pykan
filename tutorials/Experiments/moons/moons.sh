#!/bin/bash


export EXPERIMENT_NAME="noise48-width15"
export BASE_FUN="silu"
#export BASE_FUN="zero"
export TOTAL_EXPERIMENTS=7
export DATA_NOISE_LEVEL=0.5
export SPLINE_NOISE_SCALE=0.1
export STEPS=1000

spline_noise_array=(0.001 0.01 0.1 1.0 10 100 1000)

# Loop from 0 to 9 using C-style syntax
for ((i=0; i<$TOTAL_EXPERIMENTS; i++)); do
    echo "SEEED: $i"

    #DATA_NOISE_LEVEL=$(echo "scale=4; $i / $TOTAL_EXPERIMENTS" | bc)

    #SPLINE_NOISE_SCALE=$(echo "scale=4; $i / $TOTAL_EXPERIMENTS" | bc)
    #SPLINE_NOISE_SCALE=$(echo "scale=4; $i / $TOTAL_EXPERIMENTS" | bc)
    #SPLINE_NOISE_SCALE=$(echo "($SPLINE_NOISE_SCALE+1) * 100" | bc)
    #SPLINE_NOISE_SCALE=$(echo "scale=4; l($SPLINE_NOISE_SCALE) / l(10)" | bc -l)
    #SPLINE_NOISE_SCALE=$(echo "$SPLINE_NOISE_SCALE * 1" | bc)
    #SPLINE_NOISE_SCALE=$(echo "scale=4; $i / $TOTAL_EXPERIMENTS * 100" | bc)

    SPLINE_NOISE_SCALE=${spline_noise_array[i]}
    
    # Execute the Python scripts in the background
    python moons.py --base_width 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 --steps $STEPS --experiment_name $EXPERIMENT_NAME --data_noise_level $DATA_NOISE_LEVEL --spline_noise_scale $SPLINE_NOISE_SCALE --base_fun $BASE_FUN --seed $i &
    sleep 5
    # pid1=$!
    # sleep 10
    
    # python moons.py --base_width 5 --experiment_name $EXPERIMENT_NAME --data_noise_level $DATA_NOISE_LEVEL --spline_noise_scale $SPLINE_NOISE_SCALE  --base_fun $BASE_FUN --seed $i --mode abs &
    # pid2=$!
    # sleep 10
    
    # python moons.py --base_width 5 --experiment_name $EXPERIMENT_NAME --data_noise_level $DATA_NOISE_LEVEL --spline_noise_scale $SPLINE_NOISE_SCALE  --base_fun $BASE_FUN --seed $i --mode sigmoid &
    # pid3=$!
    # sleep 10
    
    # python moons.py --base_width 5 --experiment_name $EXPERIMENT_NAME --data_noise_level $DATA_NOISE_LEVEL --spline_noise_scale $SPLINE_NOISE_SCALE --base_fun $BASE_FUN --seed $i --mode relu &
    # pid4=$!
    # sleep 10
    
    # # Wait for all background processes to finish
    # wait $pid1
    # wait $pid2
    # wait $pid3
    # wait $pid4

done