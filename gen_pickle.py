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
# -*- coding: utf-8 -*-

import cv2
import os
import pandas as pd
import re
import joblib
import numpy as np
from model import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
from config import get_default_cfg
from util import *


def save_as_pickle(train_or_test, bboxes, classIds, pkl_fname, shuffle=False):
    if shuffle:
        shuffled_idx = np.random.permutation(len(bboxes))
        save_bboxes = np.array(bboxes)[shuffled_idx]
        save_classIds = np.array(classIds)[shuffled_idx]
    else:
        save_bboxes = np.array(bboxes)
        save_classIds = np.array(classIds)

    if train_or_test == 'train':
        save = {'train_bboxes': save_bboxes, 'train_classIds': save_classIds}
    else:
        save = {'test_bboxes': save_bboxes, 'test_classIds': save_classIds}
    joblib.dump(save, pkl_fname, compress=5)


def preproc(bboxes, classIds):
    preproced_bboxes = np.zeros(bboxes.shape)

    # Histogram equalization on color image
    for i, bbox in enumerate(bboxes):
        img = cv2.cvtColor(bbox, cv2.COLOR_BGR2YCrCb)
        split_img = cv2.split(img)
        split_img = list(split_img)
        split_img[0] = cv2.equalizeHist(split_img[0])
        eq_img = cv2.merge(split_img)
        eq_img = cv2.cvtColor(eq_img, cv2.COLOR_YCrCb2BGR)

        # Scaling in [0, 1]
        eq_img = (eq_img / 255.).astype(np.float32)

        # Append bboxes
        preproced_bboxes[i] = eq_img
    return preproced_bboxes, classIds


def aug_by_flip(bboxes, classIds):
    aug_bboxes = np.zeros(
        (0, bboxes.shape[1], bboxes.shape[2], bboxes.shape[3]), dtype=np.uint8)
    aug_classIds = np.zeros((0, classIds.shape[1]), dtype=np.int32)
    n_classes = NUM_CLASSES

    # This classification is referenced to below.
    # https://navoshta.com/traffic-signs-classification/
    #
    # horizontal flip class
    hflip_cls = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    # vertical flip class
    vflip_cls = np.array([1, 5, 12, 15, 17])
    # hozirontal and then vertical flip
    hvflip_cls = np.array([32, 40])
    # horizontal flip but the class change
    hflip_cls_changed = np.array([
        [19, 20],
        [33, 34],
        [36, 37],
        [38, 39],
        [20, 19],
        [34, 33],
        [37, 36],
        [39, 38],
    ])

    for c in range(n_classes):
        idxes = np.where(classIds == c)[0]
        src = bboxes[idxes]
        srcIds = classIds[idxes]

        if c in hflip_cls:
            # list of images(Ids) that flipped horizontally
            dst = src[:, ::-1, :, :]
            # append to bbox and classIds
            aug_bboxes = np.append(aug_bboxes, dst, axis=0)
            aug_classIds = np.append(aug_classIds, srcIds, axis=0)
        if c in vflip_cls:
            # list of images(Ids) that flipped vertically
            dst = src[:, :, ::-1, :]
            # append to bbox and classIds
            aug_bboxes = np.append(aug_bboxes, dst, axis=0)
            aug_classIds = np.append(aug_classIds, srcIds, axis=0)
        if c in hvflip_cls:
            # list of images(Ids) that flipped horizontally and vertiaclly
            dst = src[:, ::-1, :, :]
            dst = dst[:, :, ::-1, :]
            # append to bbox and classIds
            aug_bboxes = np.append(aug_bboxes, dst, axis=0)
            aug_classIds = np.append(aug_classIds, srcIds, axis=0)
        if c in hflip_cls_changed[:, 0]:
            dst = src[:, ::-1, :, :]
            dstIds = np.asarray([
                hflip_cls_changed[hflip_cls_changed[:, 0] == c][0][1]
                for i in range(len(srcIds))
            ])
            # append to bbox and classIds
            aug_bboxes = np.append(aug_bboxes, dst, axis=0)
            aug_classIds = np.append(
                aug_classIds, np.expand_dims(dstIds, axis=1), axis=0)
    return np.append(bboxes, aug_bboxes, axis=0), \
        np.append(classIds, aug_classIds, axis=0)


def main():
    config = get_default_cfg()
    train_gt_csvs = get_gt_csvs(config.TRAIN_ROOT_DIR)
    test_gt_csvs = get_gt_csvs(config.TEST_ROOT_DIR)

    train_bboxes, train_classIds = parse_gt_csv(train_gt_csvs,
                                                config.TRAIN_SIZE)
    test_bboxes, test_classIds = parse_gt_csv(test_gt_csvs, config.TEST_SIZE)
    print('train dataset {}, labels {}'.format(train_bboxes.shape,
                                               train_classIds.shape))
    print('test dataset {}, labels {}'.format(test_bboxes.shape,
                                              test_classIds.shape))

    # Preprocessing and apply data augmentation method
    train_bboxes, train_classIds = preproc(train_bboxes, train_classIds)
    print('train dataset(after preprocessing) {}, labels {}'.format(
        train_bboxes.shape, train_classIds.shape))

    # flip
    train_bboxes, train_classIds = aug_by_flip(train_bboxes, train_classIds)
    print(
        'train dataset(after data augmentation) {}'.format(len(train_bboxes)))

    # Convert classIds to one hot vector
    train_one_hot_classIds = np.eye(
        NUM_CLASSES)[train_classIds.reshape(len(train_classIds))]
    test_one_hot_classIds = np.eye(
        NUM_CLASSES)[test_classIds.reshape(len(test_classIds))]

    # Save bboxes and classIds as pickle
    save_as_pickle(
        'train',
        train_bboxes,
        train_one_hot_classIds,
        config.TRAIN_PKL_FILENAME,
        shuffle=True)
    save_as_pickle('test', test_bboxes, test_one_hot_classIds,
                   config.TEST_PKL_FILENAME)


if __name__ == '__main__':
    main()
