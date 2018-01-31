# The MIT License (MIT)
# Copyright (c) 2018 satojkovic

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

#!/usr/bin/env python

import tensorflow as tf
import os
import common
import joblib
import numpy as np
import model as model

BATCH_SIZE = 128


def accuracy(predictions, labels):
    return (100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])


def load_dataset_and_labels(dataset_fname, train_or_test):
    data = joblib.load(dataset_fname)
    if train_or_test == 'train':
        dataset, labels = data['train_bboxes'], data['train_classIds']
    else:
        dataset, labels = data['test_bboxes'], data['test_classIds']
    return dataset, labels


def main():
    # Load dataset and label
    train_dataset, train_labels = load_dataset_and_labels(
        common.TRAIN_PKL_FILENAME, 'train')
    test_dataset, test_labels = load_dataset_and_labels(
        common.TEST_PKL_FILENAME, 'test')
    n_train_dataset = train_dataset.shape[0]
    n_test_dataset = test_dataset.shape[0]

    with tf.Graph().as_default(), tf.Session() as sess:
        # Inputs
        x = tf.placeholder(
            tf.float32,
            shape=[
                BATCH_SIZE, model.IMG_HEIGHT, model.IMG_WIDTH,
                model.IMG_CHANNELS
            ])
        y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, model.NUM_CLASSES])

        # Inputs for test
        tf_test_dataset = tf.constant(test_dataset, dtype=tf.float32)

        # Instantiate convolutional neural network
        model_params = model.params()
        logits = model.cnn(x, model_params, keep_prob=0.5)

        # Training computation
        with tf.name_scope('loss'):
            loss = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=y))
            tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(
            model.cnn(tf_test_dataset, model_params, keep_prob=1.0))

        # Merge all summaries
        merged = tf.summary.merge_all()
        train_fwriter = tf.summary.FileWriter(
            os.path.join(os.getcwd(), 'train'))

        # Add ops to save and restore all the variables
        saver = tf.train.Saver()

        #
        # Training
        #
        sess.run(tf.global_variables_initializer())
        for epoch in range(model.NUM_EPOCH):
            for idx in range(0, n_train_dataset, BATCH_SIZE):
                offset = (idx * BATCH_SIZE) % (n_train_dataset - BATCH_SIZE)
                x_batch = train_dataset[offset:offset + BATCH_SIZE, :, :, :]
                y_batch = train_labels[offset:offset + BATCH_SIZE, :]
                feed_dict = {x: x_batch, y: y_batch}
                _, l, predictions = sess.run(
                    [optimizer, loss, train_prediction], feed_dict=feed_dict)
                # Print batch results
                print('[epoch %d] Mini-batch loss at %d: %f' % (epoch, idx, l))
                print('[epoch %d] Minibatch accuracy: %.1f%%' %
                      (epoch, accuracy(predictions, y_batch)))

        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(),
                                                 test_labels))

        # Save the trained model to disk.
        save_dir = "models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "deep_traffic_sign_model")
        saved = saver.save(sess, save_path)
        print("Model saved in file: %s" % saved)


if __name__ == '__main__':
    main()
