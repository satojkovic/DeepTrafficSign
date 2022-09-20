import argparse
import json
from collections import OrderedDict, defaultdict
import cv2
import os

from config import get_default_cfg


def info():
    tmp = OrderedDict()
    tmp['description'] = 'German Traffic Sign Recognition Benchmark (GTSRB)'
    tmp['url'] = 'https://benchmark.ini.rub.de/'
    tmp['version'] = '1.0'
    tmp['year'] = '2022'
    tmp['contributor'] = 'satojkovic'
    tmp['data_created'] = '2022/09/20'
    return tmp


def licences():
    tmp = OrderedDict()
    tmp['id'] = 1
    tmp['url'] = 'Unknown'
    tmp['name'] = 'Unknown'
    return tmp


def images(image_idxes, image_sizes):
    pass


def annotations(annots, image_idxes):
    pass


def categories(cfg):
    pass


def get_annots(cfg, mode):
    pass


def get_image_idxes(annots):
    pass


def get_image_sizes(annots, cfg):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', dest='cfg_file',
                        default=None, type=str, help='Path to config file.')
    parser.add_argument('--mode', dest='mode',
                        default='train', type=str, help='train or test.')
    parser.add_argument('--output_dir', dest='output_dir',
                        default='.', type=str, help='Path to output directory.')
    args = parser.parse_args()

    cfg = get_default_cfg()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    annots = get_annots(cfg, args.mode)
    image_idxes = get_image_idxes(annots)
    image_sizes = get_image_sizes(annots, cfg)

    query_list = ['info', 'licenses', 'images',
                  'annotations', 'categories']
    js = OrderedDict()
    for i, query in enumerate(query_list):
        tmp = ''
        if query == 'info':
            tmp = info()
        if query == 'licenses':
            tmp = licences()
        if query == 'images':
            tmp = images(image_idxes, image_sizes)
        if query == 'annotations':
            tmp = annotations(annots, image_idxes)
        if query == 'categories':
            tmp = categories(cfg)

        js[query] = tmp

    output_fn = 'gtsrb_train.json' if args.mode == 'train' else 'gtsrb_test.json'
    if args.output_dir != '.' and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, output_fn), 'w') as f:
        json.dump(js, f, indent=2)
