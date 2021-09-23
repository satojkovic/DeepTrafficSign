#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# The MIT License (MIT)
# Copyright (c) 2016 satojkovic

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import tensorflow as tf
import os

PATCH_SIZE = 5
NUM_CLASSES = 43
NUM_EPOCH = 10
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
BATCH_SIZE = 128


def params():
    # weights and biases
    params = {}

    params['w_conv1'] = tf.get_variable(
        'w_conv1',
        shape=[PATCH_SIZE, PATCH_SIZE, 3, 32],
        initializer=tf.contrib.layers.xavier_initializer())
    params['b_conv1'] = tf.Variable(tf.constant(0.1, shape=[32]))

    params['w_conv2'] = tf.get_variable(
        'w_conv2',
        shape=[PATCH_SIZE, PATCH_SIZE, 32, 64],
        initializer=tf.contrib.layers.xavier_initializer())
    params['b_conv2'] = tf.Variable(tf.constant(0.1, shape=[64]))

    params['w_conv3'] = tf.get_variable(
        'w_conv3',
        shape=[PATCH_SIZE, PATCH_SIZE, 64, 128],
        initializer=tf.contrib.layers.xavier_initializer())
    params['b_conv3'] = tf.Variable(tf.constant(0.1, shape=[128]))

    params['w_fc1'] = tf.get_variable(
        'w_fc1',
        shape=[4 * 4 * 128, 2048],
        initializer=tf.contrib.layers.xavier_initializer())
    params['b_fc1'] = tf.Variable(tf.constant(0.1, shape=[2048]))

    params['w_fc2'] = tf.get_variable(
        'w_fc2',
        shape=[2048, NUM_CLASSES],
        initializer=tf.contrib.layers.xavier_initializer())
    params['b_fc2'] = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))

    return params


def cnn(data, model_params, keep_prob):
    # First layer
    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(
            data, model_params['w_conv1'], [1, 1, 1, 1], padding='SAME') +
        model_params['b_conv1'])
    h_pool1 = tf.nn.max_pool(
        h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second layer
    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(
            h_pool1, model_params['w_conv2'], [1, 1, 1, 1], padding='SAME') +
        model_params['b_conv2'])
    h_pool2 = tf.nn.max_pool(
        h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Third layer
    h_conv3 = tf.nn.relu(
        tf.nn.conv2d(
            h_pool2, model_params['w_conv3'], [1, 1, 1, 1], padding='SAME') +
        model_params['b_conv3'])
    h_pool3 = tf.nn.max_pool(
        h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer
    conv_layer_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 128])
    h_fc1 = tf.nn.relu(
        tf.matmul(conv_layer_flat, model_params['w_fc1']) +
        model_params['b_fc1'])
    h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

    # Output layer
    out = tf.matmul(h_fc1, model_params['w_fc2']) + model_params['b_fc2']

    return out


class TrafficSignRecognizer:
    def __init__(self, mode, model_dir):
        assert mode in {'train', 'inference'}
        self.mode = mode
        self.model_dir = model_dir
        self.recognizer_model = self.build(mode)

    def build(self, mode):
        assert mode in {'train', 'inference'}
        input_image = tf.keras.layers.Input(
            shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
        x = tf.keras.layers.Conv2D(
            32, PATCH_SIZE, padding='same', activation='relu')(input_image)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(
            64, PATCH_SIZE, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(
            128, PATCH_SIZE, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(4*4*128, activation='relu')(x)
        if mode == 'train':
            x = tf.keras.layers.Dropout(0.5)(x, training=True)
        outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        return tf.keras.Model(inputs=input_image, outputs=outputs)

    def compile(self, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.recognizer_model.compile(
            optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, train_dataset, train_labels, learning_rate):
        assert self.mode == 'train', 'Create model in train mode'

        # Callbacks
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(
                self.model_dir, 'log_dir'), histogram_freq=0, write_graph=True),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_dir, 'ckpt_dir'), verbose=0, save_weights_only=True),
        ]

        # Compile
        self.compile(learning_rate)

        # Do training
        n_epochs = (len(train_dataset) // BATCH_SIZE) + 1
        self.recognizer_model.fit(
            train_dataset, train_labels, callbacks=callbacks, epochs=n_epochs, validation_split=0.2)


if __name__ == '__main__':
    print('For training')
    tsr = TrafficSignRecognizer(mode='train', model_dir='train_logs')
    tsr.recognizer_model.summary()

    print('For inference')
    tsr_inference = TrafficSignRecognizer(
        mode='inference', model_dir='inference_logs')
    tsr_inference.recognizer_model.summary()
