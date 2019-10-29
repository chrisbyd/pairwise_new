#!/usr/bin/env bash
echo "start tuning the parameter"
python train.py --dataset sysu --lr 0.01  --drop 0.0 --trial 1 --gpu 0  --lamda1 1 --lamda2 1 --method 'sc' --mode 'indoor'
python train.py --dataset sysu --lr 0.01  --drop 0.0 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.1 --method 'sc' --mode 'indoor'
python train.py --dataset sysu --lr 0.01  --drop 0.2 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.1 --method 'sc' --mode 'indoor'
python train.py --dataset sysu --lr 0.01  --drop 0.5 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.1 --method 'sc' --mode 'indoor'

