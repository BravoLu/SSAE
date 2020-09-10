MODEL=resnet
DATA=imagenette
DATA_DIR=''
python train.py --model $MODEL --data $DATA --dir $DATA_DIR
