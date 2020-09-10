# @Author: Lu Shaohao(Bravo)
# @Date:   2020-06-20 23:50:31
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2020-06-23 11:01:20
#python attack_baseline.py --target mudeep --ckpt ../targets/reid/mudeep_market1501.pth --lr 1e-4 --gpu 0,2
#python attack_baseline.py --target ide --net FCN16s --ckpt ../targets/reid/ide_market1501.pth --lr 1e-4 --gpu 1,6
#python attack_baseline.py --target spgan --ckpt ../targets/reid/spgan_market1501.pth --lr 1e-4 --gpu 3
#python attack_baseline.py --target hhl --ckpt ../targets/reid/hhl_market1501.pth --lr 1e-4 --gpu 4
#python attack_baseline.py --target hacnn --ckpt ../targets/reid/hacnn_market1501.pth --lr 1e-4 --gpu 5
#python attack_baseline.py --target hhl --dataset CUHK03 --ckpt ../targets/reid/hhl_cuhk03.pth --lr 1e-4 --gpu 6
#target='sbl'
#target='mgn'
#dataset='Market1501'
target='ide'
dataset=CUHK03
gpu=2,3
dir='/raid/home/bravolu/data'
echo python train.py --dir ${dir}  --target ${target} --dataset ${dataset} --ckpt ../targets/reid/${target}_${dataset}.pth --lr 1e-4 --gpu ${gpu} --saliency 
python train.py --dir ${dir} --target ${target} --dataset ${dataset} --ckpt ../targets/reid/${target}_${dataset}.pth --lr 1e-4 --gpu ${gpu} --saliency 
