import pathlib
import os
import re
import tensorflow as tf
import pandas as pd

class GTSRB:
    def __init__(self, data_root):
        self.data_root = pathlib.Path(data_root)
        self.train_dir = self.data_root.joinpath('Final_training')
        self.test_dir = self.data_root.joinpath('Final_test')
        self._train_gt_csvs = self._get_gt_csvs(self.train_dir)
        self._test_gt_csvs = self._get_gt_csvs(self.test_dir)

    def _get_gt_csvs(self, root_dir):
        gt_csvs = [
            str(self.data_root.joinpath(f))
            for root, dirs, files in os.walk(root_dir) for f in files
            if re.search(r'.csv', f)
        ]
        return gt_csvs

    def _gt_csv_getline(self, gt_csvs):
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

    def create_tf_examples(self, output_path):
        pass

if __name__ == "__main__":
    gtsrb = GTSRB('GTSRB')
