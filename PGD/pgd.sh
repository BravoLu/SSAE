# dataset=imagenette
dataset=caltech101
#dataset=cifar10
# target=efficientnet
# target=resnet
target=googlenet
# target=mobilenet
# target=densenet
dir='~/shaohao/data'
gpu=2,4
#gpu=5
echo python pgd.py --dataset ${dataset} --target ${target} --gpu ${gpu} --dir ${dir} 
python pgd.py --dataset ${dataset} --target ${target} --gpu ${gpu} --dir ${dir} 
