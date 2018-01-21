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
# -*- coding: utf-8 -*-

import cv2
import os
import pandas as pd
import re
import joblib
import numpy as np
import model_sof as model
import copy

TRAIN_ROOT_DIR = os.path.join('GTSRB', 'Final_training')
TRAIN_PKL_FILENAME = 'traffic_sign_train_dataset.pickle'

TEST_ROOT_DIR = os.path.join('GTSRB', 'Final_test')
TEST_PKL_FILENAME = 'traffic_sign_test_dataset.pickle'


def gt_csv_getline(gt_csvs):
    for gt_csv in gt_csvs:
        df = pd.io.parsers.read_csv(gt_csv, delimiter=';', skiprows=0)
        n_lines = df.shape[0]
        for i in range(n_lines):
            img_file_path = os.path.join(
                os.path.dirname(gt_csv), df.loc[i, 'Filename'])
            # bbox include (Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2)
            bbox = {
                'Width': df.loc[i, 'Width'],
                'Height': df.loc[i, 'Height'],
                'Roi.X1': df.loc[i, 'Roi.X1'],
                'Roi.Y1': df.loc[i, 'Roi.Y1'],
                'Roi.X2': df.loc[i, 'Roi.X2'],
                'Roi.Y2': df.loc[i, 'Roi.Y2']
            }
            classId = df.loc[i, 'ClassId']
            yield (img_file_path, bbox, classId)


def get_gt_csvs(root_dir):
    gt_csvs = [
        os.path.join(root, f)
        for root, dirs, files in os.walk(root_dir) for f in files
        if re.search(r'.csv', f)
    ]
    return gt_csvs


def parse_gt_csv(gt_csvs):
    bboxes = []
    classIds = []
    for (img_file_path, bbox, classId) in gt_csv_getline(gt_csvs):
        # Crop ground truth bounding box
        img = cv2.imread(img_file_path)
        gt_bbox = img[bbox['Roi.Y1']:bbox['Roi.Y2'], bbox['Roi.X1']:bbox[
            'Roi.X2']]

        # Append bbox and classId
        bboxes.append(gt_bbox)
        classIds.append(classId)
    return bboxes, classIds


def save_as_pickle(train_or_test, bboxes, classIds, pkl_fname):
    if train_or_test == 'train':
        save = {'train_bboxes': bboxes, 'train_classIds': classIds}
    else:
        save = {'test_bboxes': bboxes, 'test_classIds': classIds}
    joblib.dump(save, pkl_fname, compress=True)


def preproc(bboxes, classIds):
    preproced_bboxes = []

    # Histogram equalization on color image
    for bbox in bboxes:
        img = cv2.cvtColor(bbox, cv2.COLOR_BGR2YCrCb)
        split_img = cv2.split(img)
        split_img[0] = cv2.equalizeHist(split_img[0])
        eq_img = cv2.merge(split_img)
        eq_img = cv2.cvtColor(eq_img, cv2.COLOR_YCrCb2BGR)

        # Scaling in [0, 1]
        eq_img = (eq_img / 255.).astype(np.float32)
        preproced_bboxes.append(eq_img)

    return preproced_bboxes, classIds


def aug_by_flip(bboxes, classIds):
    aug_bboxes = copy.deepcopy(bboxes)
    aug_classIds = copy.deepcopy(classIds)
    n_classes = model.NUM_CLASSES

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
        idxes = np.where(np.array(classIds) == c)
        src = np.array(bboxes)[idxes]
        srcIds = np.array(classIds)[idxes]

        if c in hflip_cls:
            # list of images(Ids) that flipped horizontally
            dst = [s[::-1, :, :] for s in src]
            dstIds = srcIds
            # append to bbox and classIds
            aug_bboxes += dst
            aug_classIds += list(dstIds)
        if c in vflip_cls:
            # list of images(Ids) that flipped vertically
            dst = [s[:, ::-1, :] for s in src]
            dstIds = srcIds
            # append to bbox and classIds
            aug_bboxes += dst
            aug_classIds += list(dstIds)
        if c in hvflip_cls:
            # list of images(Ids) that flipped horizontally and vertiaclly
            dst = [s[::-1, :, :] for s in src]
            dst = [d[:, ::-1, :] for d in dst]
            dstIds = srcIds
            # append to bbox and classIds
            aug_bboxes += dst
            aug_classIds += list(dstIds)
        if c in hflip_cls_changed[:, 0]:
            dst = [s[::-1, :, :] for s in src]
            dstIds = [
                hflip_cls_changed[hflip_cls_changed[:, 0] == c]
                for si in srcIds
            ]
            # append to bbox and classIds
            aug_bboxes += dst
            aug_classIds += dstIds

    return aug_bboxes, aug_classIds


def main():
    train_gt_csvs = get_gt_csvs(TRAIN_ROOT_DIR)
    test_gt_csvs = get_gt_csvs(TEST_ROOT_DIR)

    train_bboxes, train_classIds = parse_gt_csv(train_gt_csvs)
    test_bboxes, test_classIds = parse_gt_csv(test_gt_csvs)
    print('train dataset {}'.format(len(train_bboxes)))
    print('test dataset {}'.format(len(test_bboxes)))

    # Preprocessing and apply data augmentation method
    train_bboxes, train_classIds = preproc(train_bboxes, train_classIds)

    # flip
    train_bboxes, train_classIds = aug_by_flip(train_bboxes, train_classIds)
    print(
        'train dataset(after data augmentation) {}'.format(len(train_bboxes)))

    # Save bboxes and classIds as pickle
    save_as_pickle('train', train_bboxes, train_classIds, TRAIN_PKL_FILENAME)
    save_as_pickle('test', test_bboxes, test_classIds, TEST_PKL_FILENAME)


if __name__ == '__main__':
    main()
