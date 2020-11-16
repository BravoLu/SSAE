# Adversarial Attack for Image Retrieval

## Requirements

### dataset

*  Market1501
* cuhk03

### target model 
1. mkdir the directory
```shell
    mkdir target
```

2. download the [target model files](https://pan.baidu.com/s/18lVuCO2knotUzD8oNUI3fg)(password:3914) into ./target directory.

## Train 

* Train the baseline for 40 epochs first.

```shell
python train.py --dataset DATASET_NAME --target TARGET_MODEL_NAME \\
				--ckpt TARGET_MODEL_WEIGHT --dir DATA_DIR --gpu YOUR_GPU_ID \\
				--lr LR_RATE 
```

* Train the SSAE for 150 epochs.

```shell
python train.py --dataset DATASET_NAME --target TARGET_MODEL_NAME \\
				--ckpt TARGET_MODEL_WEIGHT --dir DATA_DIR --saliency --gpu YOUR_GPU_ID \\
				--lr LR_RATE 
```

## Test

```shell
python test.py --dataset DATASET_NAME --target TARGET_MODEL_NAME --dir DATA_DIR --saliency --gpu YOUR_GPU_ID -g YOUR_AUTOENCODER_WEIGHT --log OUTPUT_LOG
```
