# this repository is the implement of pretrain code for ResNet18 and ResNet34 which are missing from tf.keras.applications.

## prepare dataset

download ImageNet and unzip it.

## train

train with the following command

```shell
python3 train.py --batch_size=<batch size> --imagenet_path=<path/to/imagenet> [--use_tfrecord] --model=(resnet18|resnet34)
```

## save model

save model from checkpoint with the following command, the saved model will be located under directory models.

```shell
python3 train.py --model=(resnet18|resnet34) --save_model
```
