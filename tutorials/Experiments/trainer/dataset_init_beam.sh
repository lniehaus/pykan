#!/bin/bash

# Execution
export JOB_NUM=2
# Utility
export experiment_name="dataset_init_10"
export device_index=1
export seed=0
# Model
export hidden_width=6
export hidden_depth=5
export steps=100
export grid=1
export k=3
export mode='default'
export base_fun='zero'
export spline_noise_scale=0.3
export init_mode='default'
# Trainable On
export sp_trainable=true
export sb_trainable=true
export affine_trainable=true
export update_grid=true
# Trainable Off
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


#widths=(10 20 30 40 50 60 70 80 90 100)
#depths=(100 90 80 70 60 50 40 30 20 10)
#depths=(10 9 8 7 6 5 4 3 2 1)
#depths=(100 50 10 5 1)
#depths=(5 4 3 2 1)
#init_modes=('default' 'native_noise' 'width_in' 'width_out' 'xavier_in' 'xavier_out' 'xavier_torch')
#init_modes=('default' 'native_noise' 'width_in' 'xavier_in' 'xavier_torch' 'width_in_num' 'xavier_in_num' 'width_in_out' 'xavier_in_out' 'width_in_out_num' 'xavier_in_out_num')
init_modes=('default-0_1' 'default-0_3' 'width_in' 'xavier_in' 'xavier_torch' 'width_in_out' 'xavier_in_out')
#datasets=('random' 'moon' 'mnist' 'cifar10')
datasets=('cifar10' 'mnist' 'moon' 'random')

index=0
for dataset in "${datasets[@]}"; do
    for init_mode in "${init_modes[@]}"; do
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
            --random_output_dim $random_output_dim \
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