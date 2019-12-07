#!/usr/bin/env bash
echo 'start training regdb'

python train.py --dataset regdb --lr 0.01 --drop 0.5 --trial 1 --gpu 0 --lamda1 1 --lamda2 0.01 --method 'sc'
python train.py --dataset regdb --lr 0.01 --drop 0.0 --trial 1 --gpu 0 --lamda1 1 --lamda2 0.1 --method 'sc'
#python train.py --dataset regdb --lr 0.01 --drop 0.0 --trial 1 --gpu 0 --lamda1 0. 1 --lamda2 1 --method 'bd'
#python train.py --dataset regdb --lr 0.01 --drop 0.0 --trial 1 --gpu 0 --lamda1 0.1 --lamda2 0.1 --method 'bd'
