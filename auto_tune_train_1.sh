#!/usr/bin/env bash
echo "start tuning the parameter"
echo "method bd dropout 0.0 lambda1 0.1 lamda2 1"

echo "method bd dropout 0.0 lambda1 0.1 lamda2 0.1"
python train.py --dataset sysu --lr 0.01  --drop 0.0 --trial 1 --gpu 0  --lamda1 0.1 --lamda2 0.1 --method 'bd' --mode 'indoor'

echo "method bd dopout 0.5 lambda1 0.1 lamda2 0.1"
python train.py --dataset sysu --lr 0.01  --drop 0.5 --trial 1 --gpu 0  --lamda1 0.1 --lamda2 0.1 --method 'bd' 

echo "method bd dopout 0.1 lambda1 0.1 lamda2 0.1"
python train.py --dataset sysu --lr 0.01  --drop 0.1 --trial 1 --gpu 0  --lamda1 0.1 --lamda2 0.1 --method 'bd' 
