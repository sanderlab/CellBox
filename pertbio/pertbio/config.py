import json

class Config(object):
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
            self.__dict__.update(config_dict)
