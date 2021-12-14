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

## pretrained model

the pretrained model of ResNet18 and ResNet34 are included in repository. the performance of the pretrained model is listed in the following table.

| model | train accuracy | eval accuracy |
| ----- | -------------- | ------------- |
| [resnet18](models/resnet18.h5) | 0.6717 |  0.6447 |
| [resnet34](models/resnet34.h5) | 0.7236 |  0.6712 |

