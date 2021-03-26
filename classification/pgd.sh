dataset=imagenette
#dataset=cifar10
#target=efficientnet
#target=resnet
target=googlenet
#target=mobilenet
#target=densenet
dir='../data'
gpu=0,1,2,3
#gpu=5
echo python pgd.py --dataset ${dataset} --target ${target} --gpu ${gpu} --dir ${dir} --saliency
python pgd.py --dataset ${dataset} --target ${target} --gpu ${gpu} --dir ${dir} --saliency
