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

#!/usr/bin/env python

import model_sof as model
import tensorflow as tf
import os

BATCH_SIZE = 32


def main():
    with tf.Graph().as_default(), tf.Session() as sess:
        # Inputs
        x = tf.placeholder(
            tf.float32,
            shape=[
                BATCH_SIZE, model.IMG_HEIGHT, model.IMG_WIDTH,
                model.IMG_CHANNELS
            ])
        y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, model.NUM_CLASSES])

        # Instantiate convolutional neural network
        logits, img_summary = model.deepnn(x)

        # Training computation
        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y)
            tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

        # Merge all summaries
        merged = tf.summary.merge_all()
        train_fwriter = tf.summary.FileWriter(
            os.path.join(os.getcwd(), 'train'))

        #
        # Training
        #

        # Load batches


if __name__ == '__main__':
    main()
