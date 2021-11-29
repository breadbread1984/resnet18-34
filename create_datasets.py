#!/usr/bin/python3

from os import listdir;
from os.path import join, exists, splitext;
from random import shuffle;
import numpy as np;
import cv2;
import tensorflow as tf;

class ImageNet(object):
  def __init__(self, root_dir):
    self.root_dir = root_dir;
    self.label2cls = dict();
    self.train_list = list();
    self.eval_list = list();
    for cls in listdir(join(root_dir, 'train')):
      if cls not in self.label2cls:
        self.label2cls[cls] = len(self.label2cls);
      for img in listdir(join(root_dir, 'train', cls)):
        if splitext(img)[1].lower() == '.jpeg':
          self.train_list.append((join(root_dir, 'train', cls, img), self.label2cls[cls]));
    for cls in listdir(join(root_dir, 'val')):
      assert cls in self.label2cls;
      for img in listdir(join(root_dir, 'val', cls)):
        if splitext(img)[1].lower() == '.jpeg':
          self.eval_list.append((join(root_dir, 'val', cls, img), self.label2cls[cls]));
    shuffle(self.train_list);
    shuffle(self.eval_list);
  def data_generator(self, trainset = True):
    dataset = self.train_list if trainset == True else self.eval_list;
    def gen():
      for sample in dataset:
        img_path, label = sample;
        img = cv2.imread(img_path)[...,::-1]; # convert BRG to RGB
        img = img.astype(np.float32);
        label = label.astype(np.int32);
        yield img, label;
    return gen;
  def train_parse_function(self, img, label):
    height = tf.shape(img)[0];
    width = tf.shape(img)[1];
    # sample crop size
    area = tf.cast(height * width, dtype = tf.float32);
    target_area = tf.random.uniform(minval = 0.08, maxval = 1., shape = (10,)) * area; # targt_area.shape = (10,)
    aspect_ratio = tf.math.exp(tf.random.uniform(minval = tf.math.log(0.75), maxval = tf.math.log(1.33), shape = (10,))); # aspect_ratio.shape = (10,)
    sample_w = tf.cast(tf.math.sqrt(target_area * aspect_ratio), dtype = tf.int32); # w.shape = (10,)
    sample_h = tf.cast(tf.math.sqrt(target_area / aspect_ratio), dtype = tf.int32); # h.shape = (10,)
    sample_x = tf.random.uniform(minval = tf.zeros_like(sample_w), maxval = tf.math.maximum(0, width - sample_w + 1), shape = (10,), dtype = tf.int32); # sample_x.shape = (10,)
    sample_y = tf.random.uniform(minval = tf.zeros_like(sample_h), maxval = tf.math.maximum(0, height - sample_h + 1), shape = (10,), dtype = tf.int32); # sample_y.shape = (10,)
    # fallback (center) crop size
    fallback_w = tf.where(
      tf.math.less(width / height, 0.75),
      width, 
      tf.where(
        tf.math.greater(width / height, 1.33),
        tf.cast(height * 1.33, dtype = tf.int32),
        width));
    fallback_h = tf.where(
      tf.math.less(width / height, 0.75),
      tf.cast(width / 0.75, dtype = tf.int32),
      tf.where(
        tf.math.greater(width / height, 1.33),
        height,
        height));
    fallback_x = (height - fallback_h) // 2;
    fallback_y = (width - fallback_w) // 2;
    # concat crop size
    w = tf.concat([sample_w, fallback_w], axis = 0); # w.shape = (11,)
    h = tf.concat([sample_h, fallback_h], axis = 0); # h.shape = (11,)
    x = tf.concat([sample_x, fallback_x], axis = 0); # x.shape = (11,)
    y = tf.concat([sample_y, fallback_y], axis = 0); # y.shape = (11,)
    ok = tf.math.logical_and(
        tf.math.logical_and(tf.math.greater(w,0), tf.math.less_equal(w,width)),
        tf.math.logical_and(tf.math.greater(h,0), tf.math.less_equal(h,height))
      ); # ok.shape = (10,)
    box = tf.stack([y / height, x / width, (y + h) / height, (x + w) / width], axis = -1); # box.shape = (11, 4)
    filtered_box = tf.boolean_mask(box, ok); # filtered_box.shape = (box_num, 4)
    bbox = filtered_box[:1,...]; # bbox.shap = (1, 4)
    cropped = tf.image.crop_and_resize(tf.expand_dims(img, axis = 0), bbox, tf.zeros((1,), dtype = tf.int32), [224,224], 'bilinear');
    flipped = tf.image.random_flip_left_right(cropped);
    sample = tf.squeeze(flipped, axis = 0) / 255.; # sample.shape = (1, 224, 224, 3)
    sample = (sample - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225]); # sample.shap = (224, 224, 3)
    return sample, label;
  def test_parse_function(self, img, label):
    img = tf.expand_dims(img, axis = 0); # sample.shape = (1, height, width, 3)
    img = tf.image.resize(img, (256, 256)); # sample.shape = (1, 256, 256, 3)
    img = tf.image.central_crop(img, 0.875); # sample.shape = (1, 224, 224, 3)
    sample = tf.squeeze(img, axis = 0) / 255.; # sample.shape = (224, 224, 3)
    sample = (sample - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225]); # sample.shap = (224, 224, 3)
    return sample, label;
  def load_datasets(self,):
    trainset = tf.data.Dataset.from_generator(self.data_generator(True), (tf.float32, tf.int32), (tf.TensorShape([None, None, 3]), tf.TensorShape([]))).map(self.train_parse_function);
    testset = tf.data.Dataset.from_generator(self.data_generator(False), (tf.float32, tf.int32), (tf.TensorShape([None, None, 3]), tf.TensorShape([]))).map(self.test_parse_function);
    return trainset, testset;

if __name__ == "__main__":
  imagenet = ImageNet('/home/devdata/dataset/imagenet_raw');
  trainset, testset = imagenet.load_datasets();
  for image, label in trainset:
    image = 255. * (image * tf.constant([0.229, 0.224, 0.225]) + tf.constant([0.485, 0.456, 0.406]));
    image = image.numpy().astype(np.uint8);
    cv2.imshow('sample', image);
    cv2.waitKey();
