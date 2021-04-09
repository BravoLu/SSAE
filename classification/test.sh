dataset=imagenette
#target=resnet
#target=efficientnet
#target=googlenet
#target=mobilenet
target=resnet
dir='~/shaohao/data'
gpu=0
echo python test.py --dataset ${dataset} --target ${target} --gpu ${gpu} --dir ${dir} --saliency 
python test.py --dataset ${dataset} --target ${target} --gpu ${gpu} --dir ${dir} --saliency
  
