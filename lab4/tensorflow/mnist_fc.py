#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import sys
import argparse
import numpy as np

# Global flags for convenience.
FLAGS = None

# Parameters.
NUM_PIXELS = 784
NUM_CLASSES = 10
BATCH_SIZE = 50
TRAIN_STEPS = 36000
HID_1 = 100
HID_2 = 50


def train_and_test(_):

    # Check if log_dir exists, if so delete contents.
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Import data.
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Define placeholders for batch of training images and labels.
    x = tf.placeholder(tf.float32, shape=(None, NUM_PIXELS), name='input_data')
    y = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name='hot_vector')

    # Define variables for weights and biases
    # of the three fully connected layers.
    W1 = tf.Variable(tf.truncated_normal(shape=(NUM_PIXELS, HID_1),
                                         stddev=0.1), name='weights1')
    W2 = tf.Variable(tf.truncated_normal(shape=(HID_1, HID_2),
                                         stddev=0.1), name='weights2')
    W3 = tf.Variable(tf.truncated_normal(shape=(HID_2, NUM_CLASSES),
                                         stddev=0.1), name='weights3')
    b1 = tf.Variable(tf.constant(0.1, shape=(HID_1,)), name='bias1')
    b2 = tf.Variable(tf.constant(0.1, shape=(HID_2,)), name='bias2')
    b3 = tf.Variable(tf.constant(0.1, shape=(NUM_CLASSES,)), name='bias3')

    # computation graph
    """ inp1: [BATCH_SIZE x  784]
        W:    [704        x Hid1]
        out:  [BATCH_SIZE x Hid1]

        inp2: [BATCH_SIZE x Hid1]
        W:    [Hid1       x Hid2]
        out:  [BATCH_SIZE x Hid2]

        inp3: [BATCH_SIZE x Hid2]
        W:    [Hid2       x   10]
        out:  [BATCH_SIZE x   10]
    """

    h1 = tf.matmul(x, W1) + b1
    out1 = tf.nn.relu(h1)
    h2 = tf.matmul(out1, W2) + b2
    out2 = tf.nn.relu(h2)
    h3 = tf.matmul(out2, W3) + b3
    out = h3

    # Define loss function.
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out)
    loss = tf.reduce_mean(loss, axis=0)

    # Make the loss a "summary" to visualise it in tensorboard.
    tf.summary.scalar('loss', loss)

    # tf.summary.histogram('weights1', W1)
    # tf.summary.histogram('weights2', W2)
    # tf.summary.histogram('weights3', W3)

    # Define the optimizer and what is optimizing.
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train_step = optimizer.minimize(loss)

    # Measure accuracy on the batch and make it a summary for tensorboard.
    a = tf.argmax(y, axis=1)
    b = tf.argmax(out, axis=1)
    acc = tf.equal(a, b)
    acc = tf.cast(acc, tf.float32)
    accuracy = tf.reduce_mean(acc)
    tf.summary.scalar('accuracy', accuracy)

    # Create session.
    sess = tf.InteractiveSession()

    # Merge summaries for tensorboard.
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)

    # Initialize variables.
    tf.global_variables_initializer().run()

    for i in range(1, TRAIN_STEPS):
        batch_xs , batch_ys = mnist.train.next_batch(BATCH_SIZE)
        summary_train, _ = sess.run([merged, train_step],
                                    feed_dict={x:batch_xs, y:batch_ys})
        train_writer.add_summary(summary_train, i)

    batch_xs , batch_ys = mnist.test.next_batch(10000)
    test_accuracy = sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys})
    print('Test accuracy: %.4f' % test_accuracy)


if __name__ == '__main__':
    # Use nice argparse module to aprte cli arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data_dir/',
                        help='Directory for training data')
    parser.add_argument('--log_dir', type=str, default='./log_dir/',
                        help='Directory for Tensorboard event files')
    FLAGS, unparsed = parser.parse_known_args()

    # app.run is a simple wrapper that parses
    # flags and sends them to main function.
    tf.app.run(main=train_and_test, argv=[sys.argv[0]] + unparsed)
