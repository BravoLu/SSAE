dataset=imagenette
target=googlenet
gpu=0,1,2,3
dir='../data'
echo python attack_baseline.py --dataset ${dataset} --target ${target} --ckpt target/${dataset}_${target}.pth --gpu ${gpu} --dir ${dir} --saliency
python train.py --dataset ${dataset} --target ${target} --ckpt target/${dataset}_${target}.pth --gpu ${gpu} --dir ${dir}
