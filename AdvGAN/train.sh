# MODEL=resnet
# MODEL=efficientnet
# MODEL=mobilenet
# MODEL=googlenet
MODEL=densenet
# DATA=imagenette
DATA=caltech101
DATA_DIR='/user/lintao/shaohao/data/'
GPU='2,3'
python train.py --model $MODEL --data $DATA --dir $DATA_DIR --gpu $GPU
