#!/bin/bash

# Utility
export experiment_name="train-debug-4"
export device_index=1
export seed=0
# Model
export hidden_form='square'
export hidden_form='linear'
export hidden_form='kat'
export hidden_width=3
export hidden_depth=1
#export hidden_depth=10
#export hidden_depth=5
export steps=5_000
export grid=5
#export grid=3
export k=3
#export k=5
export mode='default'
export spline_noise_scale=0.3
export init_mode='default'
#export init_mode='default-0_1'
#export init_mode='default-0_5'
#export init_mode='xavier_in_out'
#export init_mode='xavier_in' 
#export init_mode='kaiming_in'
#export init_mode='xavier_torch'
export grid_mode='default'
#export grid_mode='xavier'
#export grid_mode='xavier_10'
#export grid_mode='xavier_x'
export grid_bound=2.0
#export grid_mode='xavier'
#export grid_mode='xavier_10'
#export learning_rate=1.0
#export learning_rate=0.001
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
export affine_trainable=false
export update_grid=false
# Dataset
#export dataset='random'
#export dataset='moon'
#export dataset='make_classification'
#export dataset='mnist'
#export dataset='cifar10'
#export dataset='mnist1d'
export dataset='boxes_2d'
#export dataset='and'
#export dataset='or'
#export dataset='xor'
export moon_noise_level=0.5
export random_distribution='uniform'
export random_input_dim=5
export random_output_dim=2
export random_uniform_range_min=-1
export random_uniform_range_max=1
export random_normal_mean=0
export random_normal_std=1
export boxes_n_classes=$((3**2))
export boxes_datapoints_per_class=10
export task='classification'
#export task='regression'
# Eval & Plots
export symbolic_regression=false
export plot_initialized_model=true
export plot_trained_model=true
export save_video=false


echo "EXPERIMENT_NAME: $experiment_name"

python src/trainer.py \
    --experiment_name $experiment_name \
    --device_index $device_index \
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
    --task $task \
    --symbolic_regression $symbolic_regression \
    --plot_initialized_model $plot_initialized_model \
    --plot_trained_model $plot_trained_model \
    --save_video $save_video \
