#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
from absl import flags, app;
import tensorflow as tf;
from models import ImageNetRN18, ImageNetRN34;
from create_datasets import ImageNet;

FLAGS = flags.FLAGS;
flags.DEFINE_integer('batch_size', default = 256, help = 'batch size');
flags.DEFINE_integer('decay_epochs', default = 30, help = 'how many epochs for each decay');
flags.DEFINE_float('decay_rate', default = 0.1, help = 'decay rate');
flags.DEFINE_float('lr', default = 0.1, help = 'learning rate');
flags.DEFINE_integer('epochs', default = 90, help = 'epochs');
flags.DEFINE_float('momentum', default = 0.9, help = 'momentum');
flags.DEFINE_enum('model', default = 'resnet18', enum_values = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help = 'which model to train');
flags.DEFINE_string('imagenet_path', default = None, help = 'path to imagenet directory');
flags.DEFINE_boolean('use_tfrecord', default = False, help = 'whether to use dataset in tfrecord format');
flags.DEFINE_string('checkpoint', default = 'checkpoints', help = 'path to checkpoint');
flags.DEFINE_boolean('save_model', default = False, help = 'whether to save model from checkpoint');

def main(unused_argv):

  # 1) create model
  if FLAGS.model == 'resnet18':
    model = ImageNetRN18();
  elif FLAGS.model == 'resnet34':
    model = ImageNetRN34();
  elif FLAGS.model == 'resnet50':
    model = tf.keras.applications.resnet50.ResNet50(weights = None, include_top = True);
  elif FLAGS.model == 'resnet101':
    model = tf.keras.applications.resnet.ResNet101(weights = None, include_top = True);
  elif FLAGS.model == 'resnet152':
    model = tf.keras.applications.resnet.ResNet152(weights = None, include_top = True);
  else:
    raise Exception('invalid model!');
  # 2) create optimizer
  optimizer = tf.keras.optimizers.SGD(
    tf.keras.optimizers.schedules.ExponentialDecay(FLAGS.lr, decay_steps = FLAGS.decay_epochs * 1281167 / FLAGS.batch_size, decay_rate = FLAGS.decay_rate), 
    momentum = FLAGS.momentum);
  # 3) create dataset
  imagenet = ImageNet(FLAGS.imagenet_path, FLAGS.use_tfrecord);
  trainset, testset = imagenet.load_datasets();
  options = tf.data.Options();
  options.autotune.enabled = True;
  trainset = trainset.with_options(options).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size);
  testset = testset.with_options(options).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size);
  # 4) restore from existing checkpoint
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # 5) create log
  log = tf.summary.create_file_writr('checkpoints');
  # 6) metrics
  train_ce = tf.keras.metrics.Mean();
  eval_acc = tf.keras.metrics.SparseCategoricalAccuracy();
  for epoch in range(FLAGS.epochs):
    for inputs, labels in trainset:
      with tf.GradientTape() as tape:
        preds = model(inputs);
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(labels, inputs);
      grads = tape.gradient(loss, model.trainable_variables);
      optimizer.apply_gradients(zip(grads, model.trainable_variables));
      train_ce.update_state(loss);
    for inputs, labels in testset:
      preds = model(inputs);
      eval_acc.update_state(labels, preds);
    with log.as_default():
      tf.summary.scalar('loss', train_ce.result(), step = optimizer.iterations);
      tf.summary.scalar('accuracy', eval_acc.result(), step = optimizer.iterations);
    train_ce.reset_states();
    eval_acc.reset_states();
    # save checkpoint
    checkpoint.save(join('checkpoint', 'ckpt'));

if __name__ == "__main__":
  app.run(main);
