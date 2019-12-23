echo "First test the autoencoder with sc loss"

#python train.py --dataset sysu --lr 0.01  --drop 0.0 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.1 --lamda3 0.04 --method 'sc' --mode 'all'
python train.py --dataset regdb --lr 0.01  --drop 0.0 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.1 --lamda3 0.04 --method 'sc' --mode 'all'

#python train.py --dataset sysu --lr 0.01  --drop 0.1 --trial 1 --gpu 0  --lamda1 1 --lamda2 0.1 --lamda3 0.1 --method 'sc' --mode 'all'