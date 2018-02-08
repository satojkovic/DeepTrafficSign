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
import sys
import tensorflow as tf
import numpy as np
import model
import argparse
import os
import util
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import common


def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_fn', help='image filename')
    return parser.parse_args()


def traffic_sign_recognition(sess, img, obj_proposal, graph_params):
    # recognition results
    recog_results = {}
    recog_results['obj_proposal'] = obj_proposal

    # Resize image
    if img.shape != model.IMG_SHAPE:
        img = cv2.resize(img, (model.IMG_SHAPE[0], model.IMG_SHAPE[1]))

    # Pre-processing(Hist equalization)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    split_img = cv2.split(img)
    split_img[0] = cv2.equalizeHist(split_img[0])
    eq_img = cv2.merge(split_img)
    eq_img = cv2.cvtColor(eq_img, cv2.COLOR_YCrCb2BGR)
    # Scaling in [0, 1]
    eq_img = (eq_img / 255.).astype(np.float32)
    eq_img = np.expand_dims(eq_img, axis=0)

    # Traffic sign recognition
    pred = sess.run(
        [graph_params['pred']],
        feed_dict={graph_params['target_image']: eq_img})
    recog_results['pred_class'] = np.argmax(pred)
    recog_results['pred_prob'] = np.max(pred)

    return recog_results


def setup_graph():
    graph_params = {}
    graph_params['graph'] = tf.Graph()
    with graph_params['graph'].as_default():
        model_params = model.params()
        graph_params['target_image'] = tf.placeholder(
            tf.float32,
            shape=(1, model.IMG_HEIGHT, model.IMG_WIDTH, model.IMG_CHANNELS))
        logits = model.cnn(
            graph_params['target_image'], model_params, keep_prob=1.0)
        graph_params['pred'] = tf.nn.softmax(logits)
        graph_params['saver'] = tf.train.Saver()
    return graph_params


def cls2name(cls):
    SIGNNAMES_FILE = 'signnames.csv'
    signnames_ = np.loadtxt(
        os.path.join(common.GTSRB_ROOT_DIR, SIGNNAMES_FILE),
        delimiter=',',
        dtype=np.str)
    # skip first row
    signnames = signnames_[1:]
    # dictionary that convert class number to sign name
    to_name = {s[0]: s[1] for s in signnames}

    # convert class name to signname
    name = to_name[str(cls)]
    return name


def main():
    args = parse_cmdline()
    img_fn = os.path.abspath(args.img_fn)
    if not os.path.exists(img_fn):
        print('Not found: {}'.format(img_fn))
        sys.exit(-1)
    else:
        print('Target image: {}'.format(img_fn))

    # Loaa target image
    target_image = cv2.imread(img_fn)

    # Get object proposals
    object_proposals = util.get_object_proposals(target_image)

    # Setup computation graph
    graph_params = setup_graph()

    # Model initialize
    sess = tf.Session(graph=graph_params['graph'])
    tf.global_variables_initializer()
    if os.path.exists('models'):
        save_path = os.path.join('models', 'deep_traffic_sign_model')
        graph_params['saver'].restore(sess, save_path)
        print('Model restored')
    else:
        print('Initialized')

    # traffic sign recognition
    results = []
    for obj_proposal in object_proposals:
        x, y, w, h = obj_proposal
        crop_image = target_image[y:y + h, x:x + w]
        results.append(
            traffic_sign_recognition(sess, crop_image, obj_proposal,
                                     graph_params))
    """
    del_idx = []
    for i, result in enumerate(results):
        if result['pred_class'] == common.CLASS_NAME[-1]:
            del_idx.append(i)
    results = np.delete(results, del_idx)
    """
    # Non-max suppression
    nms_results = util.nms(results, pred_prob_th=0.999999, iou_th=0.4)

    # Draw rectangles on the target image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))

    for result in nms_results:
        print(result)
        (x, y, w, h) = result['obj_proposal']
        ax.text(
            x,
            y,
            cls2name(result['pred_class']),
            fontsize=13,
            bbox=dict(facecolor='red', alpha=0.7))
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()


if __name__ == '__main__':
    main()
