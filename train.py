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

import numpy as np
import joblib
from config import get_default_cfg
from model import TrafficSignRecognizer
import argparse


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='train_model_dir',
                        help='Path to model directory')
    args = parser.parse_args()

    config = get_default_cfg()

    # Load dataset and label
    train_dataset, train_labels = load_dataset_and_labels(
        config.TRAIN_PKL_FILENAME, 'train')
    test_dataset, test_labels = load_dataset_and_labels(
        config.TEST_PKL_FILENAME, 'test')

    # Create model and train
    model = TrafficSignRecognizer(mode='train', model_dir=args.model_dir)
    model.train(train_dataset, train_labels, learning_rate=1e-4)


if __name__ == '__main__':
    main()
