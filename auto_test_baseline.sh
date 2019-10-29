#!/usr/bin/env bash
echo "First test the sole identity loss"
python train.py --dataset sysu --lr 0.01  --drop 0.0 --trial 1 --gpu 0  --lamda1 1 --lamda2 1 --method 'id' --mode 'indoor'
python train.py --dataset sysu --lr 0.01  --drop 0.5 --trial 1 --gpu 0  --lamda1 1 --lamda2 1 --method 'id' --mode 'indoor'
python train.py --dataset sysu --lr 0.01  --drop 0.0 --trial 1 --gpu 0  --lamda1 1 --lamda2 1 --method 'id' --mode 'all'
python train.py --dataset sysu --lr 0.01  --drop 0.0 --trial 1 --gpu 0  --lamda1 1 --lamda2 1 --method 'id' --mode 'all'
