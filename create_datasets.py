#!/usr/bin/python3

from re import search;
from os import listdir;
from os.path import join, exists, splitext;
from random import shuffle;
import numpy as np;
import cv2;
import tensorflow as tf;

class ImageNet(object):
  def __init__(self, root_dir, use_tfrecord = False):
    self.root_dir = root_dir;
    self.use_tfrecord = use_tfrecord;
    if self.use_tfrecord == False:
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
  def load_tfrecord(self, trainset = True):
    subdir = 'train' if trainset == True else 'validation';
    filenames = [join(self.root_dir, subdir, filename) for filename in listdir(join(self.root_dir, subdir)) if search('-of-', filename)];
    filenames = sorted(filenames);
    dataset = tf.data.Dataset.from_tensor_slices(filenames);
    dataset = dataset.shuffle(buffer_size = len(filenames));
    dataset = dataset.flat_map(tf.data.TFRecordDataset);
    return dataset;
  def tfrecord_parse_function(self, serialized_example):
    feature = tf.io.parse_single_example(
      serialized_example,
      features = {
        'image/encoded': tf.FixedLenFeature((), dtype=tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature((), dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature((), dtype=tf.string, default_value=''),
      }
    );
    image = tf.io.decode_jpeg(feature['image/encoded']);
    image = tf.cast(image, dtype = tf.float32);
    label = tf.cast(feature['image/class/label'], dtype=tf.int32) - 1;
    tf.debugging.Assert(tf.math.logical_and(tf.math.greater_equal(label, 0), tf.math.less(label, 1000)), [label]);
    return image, label;
  def data_generator(self, trainset = True):
    dataset = self.train_list if trainset == True else self.eval_list;
    def gen():
      for sample in dataset:
        img_path, label = sample;
        img = cv2.imread(img_path)[...,::-1]; # convert BRG to RGB
        img = img.astype(np.float32);
        label = label;
        yield img, label;
    return gen;
  def train_parse_function(self, img, label):
    height = tf.cast(tf.shape(img)[0], dtype = tf.float32);
    width = tf.cast(tf.shape(img)[1], dtype = tf.float32);
    # sample crop size
    area = tf.cast(height * width, dtype = tf.float32);
    target_area = tf.random.uniform(minval = 0.08, maxval = 1., shape = (10,)) * area; # targt_area.shape = (10,)
    aspect_ratio = tf.math.exp(tf.random.uniform(minval = tf.math.log(0.75), maxval = tf.math.log(1.33), shape = (10,))); # aspect_ratio.shape = (10,)
    sample_w = tf.math.floor(tf.math.sqrt(target_area * aspect_ratio)); # w.shape = (10,)
    sample_h = tf.math.floor(tf.math.sqrt(target_area / aspect_ratio)); # h.shape = (10,)
    sample_x = tf.math.floor(tf.random.uniform(minval = tf.zeros_like(sample_w, dtype = tf.float32), maxval = tf.math.maximum(0., width - sample_w + 1.), shape = (10,), dtype = tf.float32)); # sample_x.shape = (10,)
    sample_y = tf.math.floor(tf.random.uniform(minval = tf.zeros_like(sample_h, dtype = tf.float32), maxval = tf.math.maximum(0., height - sample_h + 1.), shape = (10,), dtype = tf.float32)); # sample_y.shape = (10,)
    # fallback (center) crop size
    fallback_w = tf.where(
      tf.math.less(width / height, 0.75),
      width, # if width < 0.75 * height
      tf.where(
        tf.math.greater(width / height, 1.33),
        tf.math.round(height * 1.33), # if width > 1.33 * height
        width)); # if 0.75 * height < width < 1.33 * height
    fallback_h = tf.where(
      tf.math.less(width / height, 0.75),
      tf.math.round(width / 0.75), # if width < 0.75 * height
      tf.where(
        tf.math.greater(width / height, 1.33),
        height, # if width > 1.33 * height
        height)); # if 0.75 * height < width < 1.33 * height
    fallback_x = (height - fallback_h) // 2;
    fallback_y = (width - fallback_w) // 2;
    # concat crop size
    w = tf.concat([sample_w, tf.expand_dims(fallback_w, axis = -1)], axis = 0); # w.shape = (11,)
    h = tf.concat([sample_h, tf.expand_dims(fallback_h, axis = -1)], axis = 0); # h.shape = (11,)
    x = tf.concat([sample_x, tf.expand_dims(fallback_x, axis = -1)], axis = 0); # x.shape = (11,)
    y = tf.concat([sample_y, tf.expand_dims(fallback_y, axis = -1)], axis = 0); # y.shape = (11,)
    ok = tf.math.logical_and(
        tf.math.logical_and(tf.math.greater(w,0), tf.math.less_equal(w, width)),
        tf.math.logical_and(tf.math.greater(h,0), tf.math.less_equal(h, height))
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
    if self.use_tfrecord:
      trainset = self.load_tfrecord(True).map(self.tfrecord_parse_function, num_parallel_calls = tf.data.AUTOTUNE).map(self.train_parse_function, num_parallel_calls = tf.data.AUTOTUNE);
      testset = self.load_tfrecord(False).map(self.tfrecord_parse_function, num_parallel_calls = tf.data.AUTOTUNE).map(self.test_parse_function, num_parallel_calls = tf.data.AUTOTUNE);
    else:
      trainset = tf.data.Dataset.from_generator(self.data_generator(True), (tf.float32, tf.int32), (tf.TensorShape([None, None, 3]), tf.TensorShape([]))).map(self.train_parse_function, num_parallel_calls = tf.data.AUTOTUNE);
      testset = tf.data.Dataset.from_generator(self.data_generator(False), (tf.float32, tf.int32), (tf.TensorShape([None, None, 3]), tf.TensorShape([]))).map(self.test_parse_function, num_parallel_calls = tf.data.AUTOTUNE);
    return trainset, testset;

if __name__ == "__main__":
  imagenet = ImageNet('/home/devdata/dataset/imagenet_raw', use_tfrecord = False);
  trainset, testset = imagenet.load_datasets();
  count = 10;
  for image, label in trainset:
    image = 255. * (image * tf.constant([0.229, 0.224, 0.225]) + tf.constant([0.485, 0.456, 0.406]));
    image = image.numpy().astype(np.uint8);
    cv2.imshow('sample', image);
    cv2.waitKey();
    count -= 1;
    if count <= 0: break;
  imagenet = ImageNet('/home/devdata/dataset/imagenet', use_tfrecord = True);
  trainset, testset = imagenet.load_datasets();
  count = 10;
  for image, label in trainset:
    image = 255. * (image * tf.constant([0.229, 0.224, 0.225]) + tf.constant([0.485, 0.456, 0.406]));
    image = image.numpy().astype(np.uint8);
    cv2.imshow('sample', image);
    cv2.waitKey();
    count -= 1;
    if count <= 0: break;
  
