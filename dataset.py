import pathlib
import os
import re

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

if __name__ == "__main__":
    gtsrb = GTSRB('GTSRB')
