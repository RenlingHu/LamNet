import os
import logging
import sys
if sys.path[-1] != os.getcwd():
    sys.path.append(os.getcwd())

import time
import json


class BasicLogger(object):
    def __init__(self, path):
        #
        self.logger = logging.getLogger(path)
        #
        self.logger.setLevel(logging.DEBUG)
        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                        "%Y-%m-%d %H:%M:%S")

        if not self.logger.handlers:
            # Create a file handler
            file_handler = logging.FileHandler(path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)

            # use StreamHandler for print
            print_handler = logging.StreamHandler()
            print_handler.setLevel(logging.DEBUG)
            print_handler.setFormatter(formatter)

            # Add the handlers to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(print_handler)

    def noteset(self, message):
        self.logger.noteset(message)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


def create_dir(dir_list):
    """Create directories if they don't exist"""
    assert isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

class TrainLogger(BasicLogger):
    """Logger class specifically for training with additional functionality"""
    def __init__(self, args, create=True, repeat=None):
        self.args = args
        self.repeat = repeat

        # Create unique save tag with timestamp and parameters
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        savetag = timestamp + '_' + args.system + '_' + 'repeat' + str(repeat) + '_' + 'aue_weight' + str(args.use_aue_weight)

        # Set up directory structure
        save_dir = args.save_dir
        if save_dir == None:
            raise Exception('save_dir can not be None!')
        train_save_dir = os.path.join(save_dir, savetag)
        self.log_dir = os.path.join(train_save_dir, 'log', 'train')
        self.model_dir = os.path.join(train_save_dir, 'model')
        self.result_dir = os.path.join(train_save_dir, 'result')

        if create:
            # Create directories and initialize logger
            create_dir([self.log_dir, self.model_dir, self.result_dir])
            print(self.log_dir)
            log_path = os.path.join(self.log_dir, 'Train.log')
            super().__init__(log_path)
            self.record_config(args)

    def record_config(self, config):
        """Save configuration arguments to JSON file"""
        with open(os.path.join(self.log_dir, 'args.json'), 'w') as f:
            f.write(json.dumps(vars(self.args)))

    def get_log_dir(self):
        """Return log directory path if it exists"""
        if hasattr(self, 'log_dir'):
            return self.log_dir
        else:
            return None

    def get_model_dir(self):
        """Return model directory path if it exists"""
        if hasattr(self, 'model_dir'):
            return self.model_dir
        else:
            return None

    def get_result_dir(self):
        """Return results directory path if it exists"""
        if hasattr(self, 'result_dir'):
            return self.result_dir
        else:
            return None