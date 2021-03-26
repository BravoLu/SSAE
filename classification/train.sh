dataset=imagenette
target=resnet
gpu=6,7
dir='~/shaohao/data'
echo python train.py --dataset ${dataset} --target ${target} --ckpt target/${dataset}_${target}.pth --gpu ${gpu} --dir ${dir} 
python train.py --dataset ${dataset} --target ${target} --ckpt target/${dataset}_${target}.pth --gpu ${gpu} --dir ${dir} 
