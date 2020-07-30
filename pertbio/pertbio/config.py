"""
This module defines the reader for the configuration file
"""

import json


class Config(object):
    """read in config file"""
    def __init__(self, config_file):

        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        self.experiment_type = "random partition" if "experiment_type" not in config_dict else None
        self.experiment_id = "Debugging" if "experiment_id" not in config_dict else None
        self.experiment_id = "model11.ckpt" if "ckpt_name" not in config_dict else None

        self.ckpt_name = config_dict['ckpt_name']
        self.model_prefix = config_dict['model_prefix']
        self.stages = config_dict['stages']
        self.model = config_dict['model']
        self.max_iter = config_dict['max_iter'] if "max_iter" in config_dict else None
        self.weight_loss = config_dict['weight_loss'] if 'weight_loss' in config_dict else 'None'
        self.__dict__.update(config_dict)
