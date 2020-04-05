import json


class Config(object):
    def __init__(self, config_file):

        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        self.experiment_type = "random partition" if "experiment_type" not in config_dict else None
        self.experiment_id = "Debugging" if "experiment_id" not in config_dict else None
        self.experiment_id = "model11.ckpt" if "ckpt_name" not in config_dict else None

        self.__dict__.update(config_dict)
