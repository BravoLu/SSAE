dataset=imagenette
#target=resnet
#target=efficientnet
#target=googlenet
#target=mobilenet
target=resnet
dir='~/shaohao/data'
gpu=2
echo python test.py --dataset ${dataset} --target ${target} --gpu ${gpu} --dir ${dir} 
python test.py --dataset ${dataset} --target ${target} --gpu ${gpu} --dir ${dir} 
  
