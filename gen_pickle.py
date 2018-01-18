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

TRAIN_ROOT_DIR = os.path.join('GTSRB', 'Final_training')
TRAIN_PKL_FILENAME = 'traffic_sign_train_dataset.pickle'


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


def main():
    gt_csvs = [
        os.path.join(root, f)
        for root, dirs, files in os.walk(TRAIN_ROOT_DIR) for f in files
        if re.search(r'.csv', f)
    ]

    train_bboxes = []
    train_classIds = []
    for (img_file_path, bbox, classId) in gt_csv_getline(gt_csvs):
        # Crop ground truth bounding box
        img = cv2.imread(img_file_path)
        gt_bbox = img[bbox['Roi.Y1']:bbox['Roi.Y2'], bbox['Roi.X1']:bbox[
            'Roi.X2']]

        # Append bbox and classId
        train_bboxes.append(gt_bbox)
        train_classIds.append(classId)

    # Save bboxes and classIds as pickle
    save = {'train_bboxes': train_bboxes, 'train_classIds': train_classIds}
    joblib.dump(save, TRAIN_PKL_FILENAME, compress=True)


if __name__ == '__main__':
    main()
