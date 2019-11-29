#!/usr/bin/env bash
echo "First test the sole identity loss"
python train.py --dataset sysu --lr 0.01  --drop 0.0 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.1 --lamda3 0.08 --method 'sc' --mode 'indoor'
python train.py --dataset sysu --lr 0.01  --drop 0.5 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.1 --lamda3 0.2 --method 'sc' --mode 'indoor'
#python train.py --dataset sysu --lr 0.01  --drop 0.0 --trial 1 --gpu 0  --lamda1 0 --lamda2 1 --method 'sc' --mode 'all'
#python train.py --dataset sysu --lr 0.01  --drop 0.5 --trial 1 --gpu 0  --lamda1 0 --lamda2 1 --method 'sc' --mode 'all'