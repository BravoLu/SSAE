MODEL=resnet
DATA=imagenette
DATA_DIR=/raid/home/bravolu/data
python train.py --model $MODEL --data $DATA --dir $DATA_DIR
