#!/usr/bin/env python

import tensorflow as tf

# GTSRB(German Traffic Sign Recognition Benchmark)
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 3
NUM_CLASSES = 43
NUM_EPOCH = 1


def deepnn(x):

    x_image = tf.reshape(x, [-1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS])
    img_summary = tf.summary.image('Input_images', x_image)

    with tf.variable_scope('Conv_1'):
        W_conv1 = weight_variable([5, 5, IMG_CHANNELS, 32])
        tf.add_to_collection('decay_weights', W_conv1)
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)
        h_pool1 = avg_pool_3x3(h_conv1)

    with tf.variable_scope('Conv_2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        tf.add_to_collection('decay_weights', W_conv2)
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
        h_pool2 = avg_pool_3x3(h_conv2)

    with tf.variable_scope('Conv_3'):
        W_conv3 = weight_variable([5, 5, 64, 128])
        tf.add_to_collection('decay_weights', W_conv3)
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 2) + b_conv3)
        h_pool3 = max_pool_3x3(h_conv3)

    with tf.variable_scope('FC_1'):
        h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 128])
        W_fc1 = weight_variable([4 * 4 * 128, 2048])
        tf.add_to_collection('decay_weights', W_fc1)
        b_fc1 = bias_variable([2048])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    with tf.variable_scope('FC_2'):
        W_fc2 = weight_variable([2048, NUM_CLASSES])
        tf.add_to_collection('decay_weights', W_fc2)
        b_fc2 = bias_variable([NUM_CLASSES])
        y_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y_fc2, img_summary


def conv2d(x, W, p):
    output = tf.nn.conv2d(
        x, W, strides=[1, 1, 1, 1], padding='VALID', name='convolution')
    return tf.pad(output,
                  tf.constant([[0, 0], [
                      p,
                      p,
                  ], [p, p], [0, 0]]), "CONSTANT")


def conv2d_same(x, W, p):
    return tf.nn.conv2d(
        x, W, strides=[1, 1, 1, 1], padding='SAME', name='convolution')


def avg_pool_3x3(x):
    output = tf.nn.avg_pool(
        x,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='VALID',
        name='pooling')
    return tf.pad(output,
                  tf.constant([[0, 0], [
                      0,
                      1,
                  ], [0, 1], [0, 0]]), "CONSTANT")


def max_pool_3x3(x):
    output = tf.nn.max_pool(
        x,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='VALID',
        name='pooling2')
    return tf.pad(output,
                  tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]]), "CONSTANT")


def max_pool_2x2(x):
    output = tf.nn.max_pool(
        x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='VALID',
        name='pooling3')
    return tf.pad(output,
                  tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]]), "CONSTANT")


def max_pool_2x2_same(x):
    return tf.nn.max_pool(
        x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pooling3')


def weight_variable(shape):
    weight_init = tf.random_uniform(shape, -0.05, 0.05)
    return tf.Variable(weight_init, name='weights')


def weight_variable_xavier(name, shape):
    return tf.get_variable(
        name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    bias_init = tf.random_uniform(shape, -0.05, 0.05)
    return tf.Variable(bias_init, name='biases')


def bias_variable_const(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
