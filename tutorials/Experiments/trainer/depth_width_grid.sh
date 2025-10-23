#!/bin/bash

# Execution
export JOB_NUM=1
# Utility
export experiment_name="depth_width_grid_38"
export device_index=1
export seed=0
# Model
export hidden_form='square'
#export hidden_form='linear'
#export hidden_form='kat'
#export hidden_width=2
export hidden_width=5
export hidden_depth=1
export steps=10_000
export grid=5
#export grid=40
#export k=4
export k=3
#export k=2
#export k=1
export mode='default'
export spline_noise_scale=0.3
export init_mode='native_noise'
export grid_mode='default'
export grid_bound=1.0
export learning_rate=1e-3
export lamb=0.0
#export lamb=1.0
#export lamb=0.5
#export lamb=0.1
#export lamb=0.01
#export lamb=0.00005
export lamb_l1=1.0
export lamb_entropy=2.0
#export lamb_entropy=10.0
#export lamb_entropy=0.0
export lamb_coef=0.0
export lamb_coefdiff=0.0
export reg_metric='edge_forward_spline_n'
#export reg_metric='edge_forward_sum'
#export reg_metric='edge_forward_spline_u'
#export reg_metric='edge_backward'
#export reg_metric='node_backward'
#export optimizer='LBFGS' # Adam LBFGS
#export optimizer='Adam'
export optimizer='Muon'

# Trainable On
export base_fun='silu'
export sp_trainable=true
export sb_trainable=true
#export affine_trainable=true
#export update_grid=true
# Trainable Off
# export base_fun='zero'
# export sp_trainable=false
# export sb_trainable=false
export affine_trainable=false
export update_grid=false
# Dataset
#export dataset='boxes_2d'
export dataset='spiral'
#export dataset='mnist1d'
#export dataset='iris'
export moon_noise_level=0.5
export random_distribution='uniform'
export random_input_dim=5
export random_output_dim=2
export random_uniform_range_min=-1
export random_uniform_range_max=1
export random_normal_mean=0
export random_normal_std=1
export boxes_n_classes=4
export boxes_datapoints_per_class=10
export boxes_normal_std=0.0
#export boxes_normal_std=-1.0
export spiral_n_classes=20 #3
export spiral_n_samples=1000
export spiral_n_samples=$((spiral_n_classes * 500))
export spiral_noise=0.4
export task='classification'
#export task='regression'
export output_layer_mode='default'
#export output_layer_mode='linear'
export classification_loss='cross_entropy'
#export classification_loss='mse'

# Eval & Plots
export symbolic_regression=false
export plot_initialized_model=true
export plot_trained_model=true
export save_video=false
#export save_video=true
export save_model=false # deep models scale horribly and are super big when saved (Why)

echo "EXPERIMENT_NAME: $experiment_name"

# depths=(10 5 2 1 0)
# widths=(20 10 5 2)
# grids=(40 20 10 5)

# depths=(20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0)
# widths=(100 95 90 85 80 75 70 65 60 55 50 45 40 35 30 25 20 15 10 5 4 3 2 1)
# grids=(200 190 180 170 160 150 140 130 120 110 100 90 80 70 60 50 40 30 20 10 5 4 3 2 1)

# depths=(40 38 36 34 32 30 28 26 24 22 20 18 16 14 12 10 8 6 4 3 2 1 0)
# depths=(100 95 90 85 80 75 70 65 60 50 45 40 35 30 25 20 15 10 8 6 4 3 2 1 0)
# depths=(100 90 80 70 60 50 45 40 35 30 25 20 15 10 8 6 4 3 2 1 0)
# widths=(400 380 360 340 320 300 280 260 240 220 200 190 180 170 160 150 140 130 120 110 100 90 80 70 60 50 40 30 20 10 5 4 3 2 1)
# grids=(400 380 360 340 320 300 280 260 240 220 200 180 160 140 120 100 80 60 40 20 10 5 4 3 2 1)


# depths=(1000 695 483 335 233 162 112 78 54 37 26 18 12 8 6 4 2 1 0)
# widths=(1000 695 483 335 233 162 112 78 54 37 26 18 12 8 6 4 2 1)
# grids=(1000 695 483 335 233 162 112 78 54 37 26 18 12 8 6 4 2 1)

depths=(695 483 335 233 162 112 78 54 37 26 18 12 8 6 4 2 1 0)
widths=(695 483 335 233 162 112 78 54 37 26 18 12 8 6 4 2 1)
grids=(695 483 335 233 162 112 78 54 37 26 18 12 8 6 4 2 1)

depths=(112 78 54 37 26 18 12 8 6 4 2 1 0)
widths=(112 78 54 37 26 18 12 8 6 4 2 1)
grids=(112 78 54 37 26 18 12 8 6 4 2 1)

first_execution=true
index=0

echo "DEPTH RUNS"
experiment_name_mod="${experiment_name}_depth"
for depth_value in "${depths[@]}"; do
    toggle_device_index=$((index % 2))
    echo "Running with dataset: $dataset, hidden_depth: $depth_value, hidden_width: $hidden_width, grid: $grid on device_index: $toggle_device_index"
    python src/trainer.py \
        --experiment_name $experiment_name_mod \
        --device_index $device_index \
        --seed $seed \
        --hidden_form $hidden_form \
        --hidden_width $hidden_width \
        --hidden_depth $depth_value \
        --steps $steps \
        --grid $grid \
        --k $k \
        --mode $mode \
        --base_fun $base_fun \
        --spline_noise_scale $spline_noise_scale \
        --init_mode $init_mode \
        --grid_mode $grid_mode \
        --grid_bound $grid_bound \
        --learning_rate $learning_rate \
        --lamb $lamb \
        --lamb_l1 $lamb_l1 \
        --lamb_entropy $lamb_entropy \
        --lamb_coef $lamb_coef \
        --lamb_coefdiff $lamb_coefdiff \
        --reg_metric $reg_metric \
        --optimizer $optimizer \
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
        --boxes_n_classes $boxes_n_classes \
        --boxes_datapoints_per_class $boxes_datapoints_per_class \
        --boxes_normal_std $boxes_normal_std \
        --spiral_n_classes $spiral_n_classes \
        --spiral_n_samples $spiral_n_samples \
        --spiral_noise $spiral_noise \
        --task $task \
        --classification_loss $classification_loss \
        --output_layer_mode $output_layer_mode \
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
        sleep 60
        first_execution=false  # Set the flag to false after the first execution
    fi

    # Limit the number of concurrent jobs to 10
    while [ $(jobs -r | wc -l) -ge $JOB_NUM ]; do
        sleep 1
    done
done

first_execution=true
echo "WIDTH RUNS"
experiment_name_mod="${experiment_name}_width"
for width_value in "${widths[@]}"; do
    toggle_device_index=$((index % 2))
    echo "Running with dataset: $dataset, hidden_depth: $hidden_depth, hidden_width: $width_value, grid: $grid on device_index: $toggle_device_index"
    python src/trainer.py \
        --experiment_name $experiment_name_mod \
        --device_index $device_index \
        --seed $seed \
        --hidden_form $hidden_form \
        --hidden_width $width_value \
        --hidden_depth $hidden_depth \
        --steps $steps \
        --grid $grid \
        --k $k \
        --mode $mode \
        --base_fun $base_fun \
        --spline_noise_scale $spline_noise_scale \
        --init_mode $init_mode \
        --grid_mode $grid_mode \
        --grid_bound $grid_bound \
        --learning_rate $learning_rate \
        --lamb $lamb \
        --lamb_l1 $lamb_l1 \
        --lamb_entropy $lamb_entropy \
        --lamb_coef $lamb_coef \
        --lamb_coefdiff $lamb_coefdiff \
        --reg_metric $reg_metric \
        --optimizer $optimizer \
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
        --boxes_n_classes $boxes_n_classes \
        --boxes_datapoints_per_class $boxes_datapoints_per_class \
        --boxes_normal_std $boxes_normal_std \
        --spiral_n_classes $spiral_n_classes \
        --spiral_n_samples $spiral_n_samples \
        --spiral_noise $spiral_noise \
        --task $task \
        --classification_loss $classification_loss \
        --output_layer_mode $output_layer_mode \
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
        sleep 60
        first_execution=false  # Set the flag to false after the first execution
    fi

    # Limit the number of concurrent jobs to 10
    while [ $(jobs -r | wc -l) -ge $JOB_NUM ]; do
        sleep 1
    done
done
    
first_execution=true
echo "GRID RUNS"
experiment_name_mod="${experiment_name}_grid"
for grid_value in "${grids[@]}"; do
    toggle_device_index=$((index % 2))
    echo "Running with dataset: $dataset, hidden_depth: $hidden_depth, hidden_width: $hidden_width, grid: $grid_value on device_index: $toggle_device_index"
    python src/trainer.py \
        --experiment_name $experiment_name_mod \
        --device_index $device_index \
        --seed $seed \
        --hidden_form $hidden_form \
        --hidden_width $hidden_width \
        --hidden_depth $hidden_depth \
        --steps $steps \
        --grid $grid_value \
        --k $k \
        --mode $mode \
        --base_fun $base_fun \
        --spline_noise_scale $spline_noise_scale \
        --init_mode $init_mode \
        --grid_mode $grid_mode \
        --grid_bound $grid_bound \
        --learning_rate $learning_rate \
        --lamb $lamb \
        --lamb_l1 $lamb_l1 \
        --lamb_entropy $lamb_entropy \
        --lamb_coef $lamb_coef \
        --lamb_coefdiff $lamb_coefdiff \
        --reg_metric $reg_metric \
        --optimizer $optimizer \
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
        --boxes_n_classes $boxes_n_classes \
        --boxes_datapoints_per_class $boxes_datapoints_per_class \
        --boxes_normal_std $boxes_normal_std \
        --spiral_n_classes $spiral_n_classes \
        --spiral_n_samples $spiral_n_samples \
        --spiral_noise $spiral_noise \
        --task $task \
        --classification_loss $classification_loss \
        --output_layer_mode $output_layer_mode \
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
        sleep 60
        first_execution=false  # Set the flag to false after the first execution
    fi

    # Limit the number of concurrent jobs to 10
    while [ $(jobs -r | wc -l) -ge $JOB_NUM ]; do
        sleep 1
    done
done
