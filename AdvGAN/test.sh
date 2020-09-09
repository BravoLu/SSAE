MODEL=resnet
DATA=imagenette
DATA_DIR=/raid/home/bravolu/data
python test.py --model $MODEL --data $DATA --dir $DATA_DIR
