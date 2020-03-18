import pathlib

class GTSRB:
    def __init__(self, data_root):
        self.data_root = pathlib.Path(data_root)
        self.train_dir = self.data_root.joinpath('Final_training')
        self.test_dir = self.data_root.joinpath('Final_test')
