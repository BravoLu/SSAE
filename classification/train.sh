dataset=imagenette
target=resnet
gpu=2,3
dir='~/shaohao/data'
echo python train.py --dataset ${dataset} --target ${target} --ckpt target/${dataset}_${target}.pth --gpu ${gpu} --dir ${dir} --saliency 
python train.py --dataset ${dataset} --target ${target} --ckpt target/${dataset}_${target}.pth --gpu ${gpu} --dir ${dir} --saliency
