#!/usr/bin/env bash
echo 'start training on regdb'
python train.py --dataset regdb --lr 0.01  --drop 0.4 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.1 --lamda3 0.05  --method 'sc'
python train.py --dataset regdb --lr 0.01  --drop 0.2 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.1 --lamda3 0.2 --method 'sc'
python train.py --dataset regdb --lr 0.01  --drop 0.0 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.1 --lamda3 0.1  --method 'sc'
python train.py --dataset regdb --lr 0.01  --drop 0.0 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.01 --lamda3 0.05  --method 'sc'

#python train.py --dataset regdb --lr 0.01  --drop 0.0 --trial 1 --gpu 0  --lamda1 1 --lamda2 1   --method 'sc'
#python train.py --dataset regdb --lr 0.01  --drop 0.1 --trial 1 --gpu 0  --lamda1 0.1 --lamda2 0.01 --method 'sc'
#python train.py --dataset regdb --lr 0.01  --drop 0.1 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.1 --method 'sc'
#python train.py --dataset regdb --lr 0.01  --drop 0.5 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.01 --method 'sc'
#python train.py --dataset regdb --lr 0.01  --drop 0.5 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.1 --method 'sc'auto