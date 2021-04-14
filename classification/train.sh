# dataset=imagenette
dataset=caltech101
# target=resnet
# target=efficientnet
# target=googlenet
target=densenet
gpu=4,5
dir='~/shaohao/data'
# first step
echo python train.py --dataset ${dataset} --target ${target} --ckpt target/${dataset}_${target}.pth --gpu ${gpu} --dir ${dir} 
python train.py --dataset ${dataset} --target ${target} --ckpt target/${dataset}_${target}.pth --gpu ${gpu} --dir ${dir} 
# second step
echo python train.py --dataset ${dataset} --target ${target} --ckpt target/${dataset}_${target}.pth --gpu ${gpu} --dir ${dir} --saliency 
python train.py --dataset ${dataset} --target ${target} --ckpt target/${dataset}_${target}.pth --gpu ${gpu} --dir ${dir} --saliency