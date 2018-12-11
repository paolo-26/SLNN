#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import sys
import argparse

# global flags for convenience
FLAGS = None

# Parameters
NUM_PIXELS = 784
NUM_CLASSES = 10
BATCH_SIZE = 100
TRAIN_STEPS = 10
HID_1 = 15
HID_2 = 20


def train_and_test(_):

    # Check if log_dir exists, if so delete contents
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

#################################################################################
############################	YOUR CODE HERE   ################################

    # define placeholders for batch of training images and labels
    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_PIXELS), name='input_data')
    y = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_CLASSES), name='hot_vector')

    # define variables for weights and biases of the three fully connected layers
    W1 = tf.Variable(tf.truncated_normal(shape=(NUM_PIXELS, HID_1),
                                         stddev=1/NUM_PIXELS),
                                         name='weights1')

    W2 = tf.Variable(tf.truncated_normal(shape=(HID_1, HID_2),
                                         stddev=1/NUM_PIXELS),
                                         name='weights2')

    W3 = tf.Variable(tf.truncated_normal(shape=(HID_2, NUM_CLASSES),
                                         stddev=1/NUM_PIXELS),
                                         name='weights3')

    b1 = tf.Variable(tf.truncated_normal(shape=(1, HID_1),
                                         stddev=1/NUM_PIXELS),
                                         name='bias1')

    b2 = tf.Variable(tf.truncated_normal(shape=(1, HID_2),
                                         stddev=1/NUM_PIXELS),
                                         name='bias2')

    b3 = tf.Variable(tf.truncated_normal(shape=(1, NUM_CLASSES),
                                         stddev=1/NUM_PIXELS),
                                         name='bias3')

    # computation graph
    """ inp1: [BATCH_SIZE x 784]
        W:    [704        x 15]
        out:  [BATCH_SIZE x 15]

        inp2: [BATCH_SIZE x 15]
        W:    [15         x 20]
        out:  [BATCH_SIZE x 20]

        inp3: [BATCH_SIZE x 20]
        W:    [20         x 10]
        out:  [BATCH_SIZE x 10]
    """

    h1 = tf.matmul(x, W1) + b1
    out = tf.nn.relu(h1)

    h2 = tf.matmul(out, W2) + b2
    out = tf.nn.relu(h2)

    h3 = tf.matmul(out, W3) + b3
    out = tf.nn.relu(h3)


    # define loss function
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out)

    # make the loss a "summary" to visualise it in tensorboard
    tf.summary.scalar('loss', loss)

    tf.summary.histogram('weights1', W1)
    tf.summary.histogram('weights2', W2)
    tf.summary.histogram('weights3', W3)

    # define the optimizer and what is optimizing
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train_step = optimizer.minimize(loss)

    # measure accuracy on the batch and make it a summary for tensorboard
    correct_prediction = tf.equal(tf.argmax(loss,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # create session
    sess = tf.InteractiveSession()

    # merge summaries for tensorboard
    merged = tf.summary.merge_all()

    # initialize variables
    tf.global_variables_initializer().run()

    # training iterations: fetch training batch and run
    batch_xs , batch_ys = mnist.train.next_batch(BATCH_SIZE)
#    sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})

    # after training fetch test set and measure accuracy
    accuracy_value = sess.run(accuracy, feed_dict={x: batch_xs, y:batch_ys})
###################################################################################

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    summary_train = sess.run([merged, train_step], feed_dict={x:batch_xs, y:batch_ys})
    train_writer.add_summary(summary_train, i)




if __name__ == '__main__':

    # use nice argparse module to aprte cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data_dir/', help='Directory for training data')
    parser.add_argument('--log_dir', type=str, default='./log_dir/', help='Directory for Tensorboard event files')
    FLAGS, unparsed = parser.parse_known_args()
    # app.run is a simple wrapper that parses flags and sends them to main function
    tf.app.run(main=train_and_test, argv=[sys.argv[0]] + unparsed)
