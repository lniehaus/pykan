#!/bin/bash

# Execution
export JOB_NUM=20
# Utility
export experiment_name="width_noise_dataset_02"
export device_index=0
export seed=0
# Model
export hidden_width=5
export hidden_depth=5
export steps=100
export grid=5
export k=3
export mode='default'
#export base_fun='zero'
export spline_noise_scale=0.3
export init_mode='native_noise'
# Trainable On
# export base_fun='silu'
# export sp_trainable=true
# export sb_trainable=true
# export affine_trainable=true
# export update_grid=true
# Trainable Off
export base_fun='zero'
export sp_trainable=false
export sb_trainable=false
export affine_trainable=false
export update_grid=false
# Dataset
export dataset='random'
export moon_noise_level=0.5
export random_distribution='uniform'
export random_input_dim=50
export random_output_dim=50
export random_uniform_range_min=-1
export random_uniform_range_max=1
export random_normal_mean=0
export random_normal_std=1
# Eval & Plots
export symbolic_regression=false
export plot_initialized_model=true
export plot_trained_model=true
export save_video=false
export save_model=false # deep models scale horribly and are super big when saved (Why)

echo "EXPERIMENT_NAME: $experiment_name"


#widths=(20 15 10 5 4 3 2)
#noise_scales=(2 1 0.5 0.3 0.1 0.05 0.01)

widths=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50)
noise_scales=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4)
datasets=('moon' 'random')

index=0
for hidden_width in "${widths[@]}"; do
    for dataset in "${datasets[@]}"; do
        for spline_noise_scale in "${noise_scales[@]}"; do
            toggle_devive_index=$((index % 2))
            python src/trainer.py \
                --experiment_name $experiment_name \
                --device_index $toggle_devive_index \
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
                --random_input_dim $hidden_width \
                --random_output_dim $hidden_width \
                --random_uniform_range_min $random_uniform_range_min \
                --random_uniform_range_max $random_uniform_range_max \
                --random_normal_mean $random_normal_mean \
                --random_normal_std $random_normal_std \
                --symbolic_regression $symbolic_regression \
                --plot_initialized_model $plot_initialized_model \
                --plot_trained_model $plot_trained_model \
                --save_video $save_video \
                --save_model $save_model \
                &
            
            index=$((index + 1))  # Increment the index counter

            sleep 10

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