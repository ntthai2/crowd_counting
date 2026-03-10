import os
import logging
from utils.logger import setlogger


class Trainer(object):
    def __init__(self, args):
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        setlogger(os.path.join(self.save_dir, 'train.log'))  # set logger
        for k, v in args.__dict__.items():  # save args
            logging.info("{}: {}".format(k, v))
        self.args = args

    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        pass

    def train(self):
        """training one epoch"""
        pass
