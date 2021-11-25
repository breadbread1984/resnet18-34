#!/usr/bin/python3

import tensorflow as tf;

def ResnetBlock(in_channels, out_channels, down_sample = False):
  inputs = tf.keras.Input((None, None, in_channels)); # inputs.shape = (batch, height, width, in_channels)
  if down_sample:
    shortcut = tf.keras.layers.Conv2D(out_channels, kernel_size = (1,1), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.HeNormal())(inputs);
    shortcut = tf.keras.layers.BatchNormalization()(shortcut);
  else:
    shortcut = inputs;
  results = tf.keras.layers.Conv2D(out_channels, kernel_size = (3,3), strides = (2,2) if down_sample else (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.HeNormal())(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(out_channels, kernel_size = (3,3), strides = (1,1) if down_sample else (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.HeNormal())(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.Add()([results, shortcut]);
  results = tf.keras.layers.ReLU()(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def ResNet18(**kwargs):
  inputs = tf.keras.Input((None, None, 3));
  results = tf.keras.layers.Conv2D(64, kernel_size = (7,7), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.HeNormal())(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'same')(results);
  results = ResnetBlock(64, 64)(results);
  results = ResnetBlock(64, 64)(results);
  results = ResnetBlock(64, 128, down_sample = True)(results);
  results = ResnetBlock(128, 128)(results);
  results = ResnetBlock(128, 256, down_sample = True)(results);
  results = ResnetBlock(256, 256)(results);
  results = ResnetBlock(256, 512, down_sample = True)(results);
  results = ResnetBlock(512, 512)(results);
  results = tf.keras.layers.GlobalAveragePooling2D()(results); # results.shape = (batch, 512)
  return tf.keras.Model(inputs = inputs, outputs = results, **kwargs);

def ImageNetRN18():
  inputs = tf.keras.Input((None, None, 3));
  results = ResNet18(name = 'resnet18')(inputs);
  results = tf.keras.layers.Dense(1000, activation = tf.keras.activations.softmax)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def ResNet34(**kwargs):
  inputs = tf.keras.Input((None, None, 3));
  results = tf.keras.layers.Conv2D(64, kernel_size = (7,7), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.HeNormal())(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'same')(results);
  results = ResnetBlock(64, 64)(results);
  results = ResnetBlock(64, 64)(results);
  results = ResnetBlock(64, 64)(results);
  results = ResnetBlock(64, 128, down_sample = True)(results);
  results = ResnetBlock(128, 128)(results);
  results = ResnetBlock(128, 128)(results);
  results = ResnetBlock(128, 128)(results);
  results = ResnetBlock(128, 256, down_sample = True)(results);
  results = ResnetBlock(256, 256)(results);
  results = ResnetBlock(256, 256)(results);
  results = ResnetBlock(256, 256)(results);
  results = ResnetBlock(256, 256)(results);
  results = ResnetBlock(256, 256)(results);
  results = ResnetBlock(256, 512, down_sample = True)(results);
  results = ResnetBlock(512, 512)(results);
  results = ResnetBlock(512, 512)(results);
  results = tf.keras.layers.GlobalAveragePooling2D()(results); # results.shape = (batch, 512)
  return tf.keras.Model(inputs = inputs, outputs = results, **kwargs);

def ImageNetRN34():
  inputs = tf.keras.Input((None, None, 3));
  results = ResNet34(name = 'resnet34')(inputs);
  results = tf.keras.layers.Dense(1000, activation = tf.keras.activations.softmax)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":
  rn18 = ImageNetRN18();
  rn18.save('rn18.h5');
  rn34 = ImageNetRN34();
  rn34.save('rn34.h5');

