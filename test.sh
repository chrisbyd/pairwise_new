#!/usr/bin/env bash
#python test.py --dataset 'sysu' --resume 'sysu_sigmoidcrossentropy_bn_relu_drop_0.0_lr_1.0e-02_dim_512_lamda1_1.0_lamda2_0.1_resnet50__mode__indoor_best.t' --mode 'indoor'  --method 'sc'
#python test.py --dataset regdb --resume 'regdb_sigmoidcrossentropy_bn_relu_drop_0.0_lr_1.0e-02_dim_2048_lamda1_1.0_lamda2_0.1_lamda3_0.05_resnet50_trial_1_best.t' --trial 1
python test.py --dataset 'sysu' --resume 'sysu_sigmoidcrossentropy_bn_relu_drop_0.0_lr_1.0e-02_dim_2048_lamda1_1.0_lamda2_0.1_lamda3_0.05_resnet50__mode__indoor_best.t' --mode 'indoor'  --method 'sc'