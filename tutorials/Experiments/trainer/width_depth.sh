#!/bin/bash

# Execution
export JOB_NUM=10
# Utility
export experiment_name="width_depth_03"
export device_index=1
export seed=0
# Model
export hidden_width=3
export hidden_depth=5
export steps=100
export grid=5
export k=3
export mode='default'
export base_fun='zero'
export spline_noise_scale=0.3
export init_mode='default'
# Trainable
export sp_trainable=false
export sb_trainable=false
export affine_trainable=false
export update_grid=false
# Dataset
export dataset='random'
export moon_noise_level=0.5
export random_distribution='uniform'
export random_input_dim=2
export random_output_dim=1
export random_uniform_range_min=-1
export random_uniform_range_max=1
export random_normal_mean=0
export random_normal_std=1
# Eval & Plots
export symbolic_regression=false
export plot_initialized_model=true
export plot_trained_model=true
export save_video=false


echo "EXPERIMENT_NAME: $experiment_name"

widths=(2 5 10 15 20)
depths=(2 5 10 15 20)
input_dims=(2 5 10 15 20)

for hidden_depth in "${depths[@]}"; do
    for hidden_width in "${widths[@]}"; do
        for random_input_dim in "${input_dims[@]}"; do
            python src/trainer.py \
                --experiment_name $experiment_name \
                --device_index $device_index \
                --seed $seed \
                --hidden_width $hidden_width \
                --hidden_depth $hidden_depth \
                --steps $steps \
                --grid $grid \
                --k $k \
                --mode $mode \
                --base_fun $base_fun \
                --spline_noise_scale $spline_noise_scale \
                --init_mode $init_mode \
                --sp_trainable $sp_trainable \
                --sb_trainable $sb_trainable \
                --affine_trainable $affine_trainable \
                --update_grid $update_grid \
                --dataset $dataset \
                --moon_noise_level $moon_noise_level \
                --random_distribution $random_distribution \
                --random_input_dim $random_input_dim \
                --random_output_dim $random_output_dim \
                --random_uniform_range_min $random_uniform_range_min \
                --random_uniform_range_max $random_uniform_range_max \
                --random_normal_mean $random_normal_mean \
                --random_normal_std $random_normal_std \
                --symbolic_regression $symbolic_regression \
                --plot_initialized_model $plot_initialized_model \
                --plot_trained_model $plot_trained_model \
                --save_video $save_video \
                &

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