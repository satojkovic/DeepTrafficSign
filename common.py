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
import os
import re

GTSRB_ROOT_DIR = 'GTSRB'

TRAIN_ROOT_DIR = os.path.join(GTSRB_ROOT_DIR, 'Final_Training')
TRAIN_PKL_FILENAME = 'traffic_sign_train_dataset.pickle'
TRAIN_SIZE = len([
    f
    for root, dirs, files in os.walk(os.path.join(TRAIN_ROOT_DIR, 'Images'))
    for f in files if re.search(r'.ppm', f)
])

TEST_ROOT_DIR = os.path.join(GTSRB_ROOT_DIR, 'Final_Test')
TEST_PKL_FILENAME = 'traffic_sign_test_dataset.pickle'
TEST_SIZE = len([
    f
    for root, dirs, files in os.walk(os.path.join(TEST_ROOT_DIR, 'Images'))
    for f in files if re.search(r'.ppm', f)
])
