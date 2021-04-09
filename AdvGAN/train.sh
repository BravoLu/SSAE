MODEL=resnet
DATA=imagenette
DATA_DIR='/user/lintao/shaohao/data/'
GPU='4'
python train.py --model $MODEL --data $DATA --dir $DATA_DIR --gpu $GPU
