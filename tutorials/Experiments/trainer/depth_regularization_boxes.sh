#!/bin/bash

# Execution
export JOB_NUM=20
# Utility
export experiment_name="depth_regularization_boxes_01"
export device_index=1
export seed=0
# Model
export hidden_form='square'
#export hidden_form='linear'
#export hidden_form='kat'
export hidden_width=2
export hidden_depth=1
export steps=10_000
export grid=5
#export grid=40
export k=3
export mode='default'
export spline_noise_scale=0.3
export init_mode='native_noise'
export grid_mode='default'
export grid_bound=1.0
export learning_rate=1e-2
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
export optimizer='Adam'

# Trainable On
export base_fun='silu'
export sp_trainable=true
export sb_trainable=true
export affine_trainable=true
export update_grid=true
# Trainable Off
# export base_fun='zero'
# export sp_trainable=false
# export sb_trainable=false
# export affine_trainable=false
# export update_grid=false
# Dataset
export dataset='boxes_2d'
#export dataset='spiral'
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
#export boxes_normal_std=0.0
export boxes_normal_std=-1.0
export spiral_n_classes=3 #3
export spiral_n_samples=1000
export spiral_n_samples=$((spiral_n_classes * 500))
export spiral_noise=0.0
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
export save_model=false # deep models scale horribly and are super big when saved (Why)

echo "EXPERIMENT_NAME: $experiment_name"

reg_metrics=('edge_forward_spline_n' 'edge_forward_sum' 'edge_forward_spline_u' 'edge_backward' 'node_backward')
lambs=(1e-4 1e-5 1e-6 0.0)
datasets=('boxes_2d')
depths=(10 5 2 1 0)

first_execution=true
index=0

for reg_metric in "${reg_metrics[@]}"; do
    for lamb in "${lambs[@]}"; do
        for dataset in "${datasets[@]}"; do
            for hidden_depth in "${depths[@]}"; do
                toggle_device_index=$((index % 2))
                echo "Running with dataset_complexity: $dataset_complexity, dataset: $dataset, hidden_depth: $hidden_depth on device_index: $toggle_device_index"
                python src/trainer.py \
                    --experiment_name $experiment_name \
                    --device_index $toggle_device_index \
                    --seed $seed \
                    --hidden_form $hidden_form \
                    --hidden_width $hidden_width \
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
done