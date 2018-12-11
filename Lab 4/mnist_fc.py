#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import sys
import argparse

# global flags for convenience
FLAGS=None

# Parameters
NUM_PIXELS = 784
NUM_CLASSES = 10
BATCH_SIZE =
TRAIN_STEPS =


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


	# define variables for weights and biases of the three fully connected layers


	# computation graph


	# define loss function


	# make the loss a "summary" to visualise it in tensorboard


	# define the optimizer and what is optimizing


	# measure accuracy on the batch and make it a summary for tensorboard


	# create session


	# merge summaries for tensorboard


	# initialize variables


	# training iterations: fetch training batch and run


	# after training fetch test set and measure accuracy


###################################################################################







if __name__ == '__main__':

	# use nice argparse module to aprte cli arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='./data_dir/', help='Directory for training data')
	parser.add_argument('--log_dir', type=str, default='./log_dir/', help='Directory for Tensorboard event files')
	FLAGS, unparsed = parser.parse_known_args()
	# app.run is a simple wrapper that parses flags and sends them to main function
	tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)
