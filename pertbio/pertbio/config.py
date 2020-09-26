"""
This module defines the reader for the configuration file
"""

import json


class Config(object):
    """read in config file"""
    def __init__(self, config_file):

        with open(config_file, "r") as f:
            config_dict = json.load(f)

        self.experiment_id = config_dict["experiment_id"] if "experiment_id" in config_dict else "Debugging"
        self.model_prefix = config_dict["model_prefix"] if "model_prefix" in config_dict else "seed"
        self.ckpt_name = config_dict["ckpt_name"] if "ckpt_name" in config_dict else "model.ckpt"
        self.export_verbose = config_dict["export_verbose"] if "export_verbose" in config_dict else 2

        self.experiment_type = config_dict["experiment_type"] if "experiment_type" in config_dict else "random partition"
        assert self.experiment_type in ["random partition", "leave one out (w/o single)",
                                        "leave one out (w/ single)", "single to combo"], NotImplementedError
        self.sparse_data = config_dict["sparse_data"] if "sparse_data" in config_dict else False
        self.batchsize = config_dict["batchsize"] if "batchsize" in config_dict else 8
        self.trainset_ratio = config_dict["trainset_ratio"] if "trainset_ratio" in config_dict else 0.7
        self.validset_ratio = config_dict["validset_ratio"] if "validset_ratio" in config_dict else 0.8
        self.n_batches_eval = config_dict["n_batches_eval"] if "n_batches_eval" in config_dict else None
        self.add_noise_level = config_dict["add_noise_level"] if "add_noise_level" in config_dict else 0

        self.dT = config_dict["dT"] if "dT" in config_dict else 0.1
        self.ode_solver = config_dict["ode_solver"] if "ode_solver" in config_dict else "heun"
        assert self.ode_solver in ["heun", "euler", "rk4", "midpoint"], NotImplementedError
        self.envelope_form = config_dict["envelope_form"] if "envelope_form" in config_dict else "tanh"
        assert self.envelope_form in ["tanh", "polynomial", "hill"], NotImplementedError
        self.envelope = config_dict["envelope"] if "envelope" in config_dict else 0
        assert self.envelope in [0, 1, 2], NotImplementedError
        self.pert_form = config_dict["pert_form"] if "pert_form" in config_dict else "by u"
        assert self.pert_form in ["by u", "fix x"], NotImplementedError
        self.ode_degree = config_dict["ode_degree"] if "ode_degree" in config_dict else 1
        assert self.ode_degree in [1, 2], NotImplementedError
        self.ode_last_steps = config_dict["ode_last_steps"] if "ode_last_steps" in config_dict else 2

        self.n_iter_buffer = config_dict["n_iter_buffer"] if "n_iter_buffer" in config_dict else 5
        self.n_iter_patience = config_dict["n_iter_patience"] if "n_iter_patience" in config_dict else 100
        self.weight_loss = config_dict["weight_loss"] if "weight_loss" in config_dict else "None"
        assert self.weight_loss in ["expr", "None"], NotImplementedError
        self.l1lambda = config_dict["l1lambda"] if "l1lambda" in config_dict else 0
        self.l2lambda = config_dict["l2lambda"] if "l2lambda" in config_dict else 0

        self.__dict__.update(config_dict)
