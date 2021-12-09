#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
from absl import flags, app;
import tensorflow as tf;
from models import ImageNetRN18, ImageNetRN34;
from create_datasets import ImageNet;

FLAGS = flags.FLAGS;
flags.DEFINE_integer('batch_size', default = 256, help = 'batch size');
flags.DEFINE_integer('decay_epoch', default = 30, help = 'how many epochs for each decay');
flags.DEFINE_float('decay_rate', default = 0.1, help = 'decay rate');
flags.DEFINE_float('lr', default = 0.1, help = 'learning rate');
flags.DEFINE_integer('epochs', default = 90, help = 'epochs');
flags.DEFINE_float('momentum', default = 0.9, help = 'momentum');
flags.DEFINE_enum('model', default = 'resnet18', enum_values = ['resnet18', 'resnet34'], help = 'which model to train');
flags.DEFINE_string('imagenet_path', default = None, help = 'path to raw imagenet directory');
flags.DEFINE_string('checkpoint', default = 'checkpoints', help = 'path to checkpoint');
flags.DEFINE_boolean('save_model', default = False, help = 'whether to save model from checkpoint');

def main(unused_argv):

  if exists(join(FLAGS.checkpoint, 'ckpt')):
    model = tf.keras.models.load_model(join(FLAGS.checkpoint, 'ckpt'), custom_objects = {'tf': tf}, compile = True);
    optimizer = model.optimizer;
    if FLAGS.save_model:
      if not exists('models'): mkdir('models');
      model.get_layer(FLAGS.model).save(join('models', '%s.h5' % FLAGS.model));
      exit();
  else:
    model = ImageNetRN18() if FLAGS.model == 'resnet18' else ImageNetRN34();
    optimizer = tf.keras.optimizers.SGD(
      tf.keras.optimizers.schedules.ExponentialDecay(FLAGS.lr, decay_steps = FLAGS.decay_epochs * 1281167 / FLAGS.batch_size, decay_rate = FLAGS.decay_rate), 
      momemtum = FLAGS.momentum);
    model.compile(optimizer = optimizer,
                  loss = [tf.keras.losses.SparseCategoricalCrossentropy(name = 'ce_loss')],
                  metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name = 'acc')]);
  # create dataset
  imagenet = ImageNet(FLAGS.imagenet_path);
  options = tf.data.Options();
  options.autotune.enabled = True;
  trainset, testset = imagenet.load_datasets();
  trainset = trainset.with_options(options).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size);
  testset = testset.with_options(options).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size);
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.checkpoint),
    tf.keras.callbacks.ModelCheckpoint(filepath = join(FLAGS.checkpoint, 'ckpt'), save_freq = 1000),
  ];
  model.fit(trainst, epochs = FLAGS.epochs, validation_data = testset, callbacks = callbacks);

if __name__ == "__main__":
  app.run(main);
