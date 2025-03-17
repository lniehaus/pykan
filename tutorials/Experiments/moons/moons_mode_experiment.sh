#!/bin/bash


export EXPERIMENT_NAME="base-06"
#export BASE_FUN="silu"
export BASE_FUN="zero"
export DATA_NOISE_LEVEL=0.5
export TOTAL_EXPERIMENTS=10
export WIDTH=5
export DEPTH=1

# Loop from 0 to 9 using C-style syntax
for ((i=0; i<$TOTAL_EXPERIMENTS; i++)); do
    echo "SEEED: $i"

    NOISE=$(echo "scale=4; $i / $TOTAL_EXPERIMENTS" | bc)
    
    # Execute the Python scripts in the background
    python moons.py  --data_noise_level $NOISE --hidden_width $WIDTH --hidden_depth $DEPTH --experiment_name $EXPERIMENT_NAME --base_fun $BASE_FUN --seed $i &
    pid1=$!
    sleep 10
    
    python moons.py  --data_noise_level $NOISE --hidden_width $WIDTH --hidden_depth $DEPTH --experiment_name $EXPERIMENT_NAME --base_fun $BASE_FUN --seed $i --mode abs &
    pid2=$!
    sleep 10
    
    python moons.py  --data_noise_level $NOISE --hidden_width $WIDTH --hidden_depth $DEPTH --experiment_name $EXPERIMENT_NAME --base_fun $BASE_FUN --seed $i --mode sigmoid &
    pid3=$!
    sleep 10
    
    python moons.py  --data_noise_level $NOISE --hidden_width $WIDTH --hidden_depth $DEPTH --experiment_name $EXPERIMENT_NAME --base_fun $BASE_FUN --seed $i --mode relu &
    pid4=$!
    sleep 10
    
    # Wait for all background processes to finish
    wait $pid1
    wait $pid2
    wait $pid3
    wait $pid4
done