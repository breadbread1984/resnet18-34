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
    height, width, _ = img.shape;
    # sample crop size
    area = height * width;
    target_area = tf.random.uniform(minval = 0.08, maxval = 1., shape = (10,)) * area; # targt_area.shape = (10,)
    aspect_ratio = tf.math.exp(tf.random.uniform(minval = tf.math.log(0.75), maxval = tf.math.log(1.33), shape = (10,))); # aspect_ratio.shape = (10,)
    sample_w = tf.cast(tf.math.sqrt(target_area * aspect_ratio), dtype = tf.int32); # w.shape = (10,)
    sample_h = tf.cast(tf.math.sqrt(target_area / aspect_ratio), dtype = tf.int32); # h.shape = (10,)
    sample_x = tf.random.uniform(minval = 0, maxval = tf.math.maximum(0, height - sample_h + 1), shape = (), dtype = tf.int32); # sample_x.shape = (10,)
    sample_y = tf.random.uniform(minval = 0, maxval = tf.math.maximum(0, width - sample_w + 1), shape = (), dtype = tf.int32); # sample_y.shape = (10,)
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
    ok = tf.math.logical_and(
        tf.math.logical_and(tf.math.greater(w,0), tf.math.less_equal(w,width)),
        tf.math.logical_and(tf.math.greater(h,0), tf.math.less_equal(h,height))
      ); # ok.shape = (10,)
    
  def test_parse_function(self, img, label):
    
  def load_datasets(self,):
    trainset = tf.data.Dataset.from_generator(self.data_generator(True), (tf.float32, tf.int32), (tf.TensorShape([None, None, 3]), tf.TensorShape([]))).map(self.train_parse_function);
    testset = tf.data.Dataset.from_generator(self.data_generator(False), (tf.float32, tf.int32), (tf.TensorShape([None, None, 3]), tf.TensorShape([]))).map(self.test_parse_function);
    return trainset, testset;

if __name__ == "__main__":
  imagenet = ImageNet('/home/devdata/dataset/imagenet_raw');
