# @Author: Lu Shaohao(Bravo)
# @Date:   2020-06-20 22:18:53
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2020-06-22 23:41:18
#python test.py --target pcb --ckpt ../targets/reid/pcb_market1501.pth --dataset Market1501 -g ../logs/reid/pcb_eps0.100000_alpha0.000000/Best_G.pth --gpu 5
#python test.py --target aligned --ckpt ../targets/reid/aligned_market1501.pth --dataset Market1501 --g ../logs/reid/aligned_baseline_0.05/Best_G.pth --gpu 3
#python test_baseline.py --target ide --ckpt ../targets/reid/ide_cuhk03.pth --dataset CUHK03 -g ../logs/results/CUHK03_ide_eps0.1_alpha0_lr1e-4/epoch53.pth --log ../vis/ide_baseline_cuhk03 --gpu 7
target=ide
dataset=CUHK03
model=saliency
#dataset=Market1501
dir=/raid/home/bravolu/data
gpu=0,1
echo python test.py --target ${target} --ckpt ../targets/reid/${target}_${dataset}.pth --dataset ${dataset} -g ../logs/reid/${dataset}_${target}_${model}//Best_G.pth --log ../vis/${dataset}_${target}_${model} --gpu ${gpu} --saliency --dir ${dir} 
python test.py --target ${target} --ckpt ../targets/reid/${target}_${dataset}.pth --dataset ${dataset} -g ../logs/reid/${dataset}_${target}_${model}//Best_G.pth --log ../vis/${dataset}_${target}_${model} --gpu ${gpu} --saliency --dir ${dir}
