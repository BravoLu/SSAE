# Adversarial Attack for Image Classification

## Requirements

### dataset

*  cifar10
* [imagenette]( https://github.com/fastai/fastai )

## Train 

* Train the baseline for 40 epochs first.

```shell
python train.py --dataset DATASET_NAME --target TARGET_MODEL_NAME \\
				--ckpt TARGET_MODEL_WEIGHT --dir DATA_DIR --gpu YOUR_GPU_ID
```

* Train the SSAE for 150 epochs.

```shell
python train.py --dataset DATASET_NAME --target TARGET_MODEL_NAME \\
				--ckpt TARGET_MODEL_WEIGHT --dir DATA_DIR --saliency --gpu YOUR_GPU_ID
```

## Test

```shell
python test.py --dataset DATASET_NAME --target TARGET_MODEL_NAME --dir DATA_DIR --saliency --gpu YOUR_GPU_ID
```

