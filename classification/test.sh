dataset=imagenette
target=resnet
dir=''
gpu=0,1,2,3
echo python test.py --dataset ${dataset} --target ${target} --gpu ${gpu} --dir ${dir} --saliency
python test.py --dataset ${dataset} --target ${target} --gpu ${gpu} --dir ${dir} --saliency
  
