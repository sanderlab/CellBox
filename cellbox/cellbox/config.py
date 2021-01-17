"""
This module defines the reader for the configuration file
"""

import json


class Config(object):
    """
    An object reading in a json file of model configuration

    ### Output files
    experiment_id (str): A string that used for result folder export (default: "Debugging"). The full
                         output path would be "./results/$experiment_id$_$an_md5_identifier$/$model_prefic$_$seed$/"
    model_prefix (str): A name for each sub-experiment with a specific random seed, default: 'seed', The full
                         output path would be "./results/$experiment_id$_$an_md5_identifier$/$model_prefic$_$seed$/"
    ckpt_name (str): A name for tensorflow checkpoint files (default: "model.ckpt").
    model (str): the type of model. Currently supported input: ["CellBox", "NN", "LinReg"].
    export_verbose (int): export verbose, supported [0: no output, 1: params only, 2: params + prediction (default),
                                                     3: output for each iteration]

    ### Input files
    pert_file (str): file path to the perturbation matrix.
    expr_file (str): file path to the cell response matrix.
    sparse_data (bool): whether to use sparse matrix (scipy.sparse) as input data, default: False.
    n_x (int) : Number of nodes in the interaction network.
                The nodes should be ordered in protein nodes -> activity nodes -> phenotypic nodes
    n_protein_nodes (int): idx of the last protein nodes in the interaction network
    n_activity_nodes (int): idx of the last activity nodes in the interaction network
    experiment_type (str): A string that describes the experiment type and data partition.
                           Supported input: ["random partition"(default), "leave one out (w/o single)",
                                             "leave one out (w/ single)", "single to combo"].
    trainset_ratio (float): The data partition ratio between (train + valid) : test. (Default: 0.7)
    validset_ratio (float): The data partition ratio between train : valid. (Default: 0.8)
    batchsize (int): The batch size for each iteration. (Default: 8)
    n_batches_eval (int): Maximum number of batches for model evaluation (use when the validation/test dataset is large)
    add_noise_level (float): the sigma of Gaussian noise added onto expr data file during training, default: 0.

    ### Details for the ODE kernel (See kernel.py for more details)
    envelope_form (str): the nonlinear envelope function in a string format, supported: ["tanh" (default), "polynomial"]
    envelope (int): idx of envelope function form of dx/dt, {0: dx/dt = epsilon*phi(Sigma+u)-alpha*x (default),
                    1: dx/dt = epsilon*[phi(Sigma)+u]-alpha*x, 2: dx/dt = epsilon*phi(Sigma)+psi*u-alpha*x}.
    polynomial_k (int): the parameter k for polynomial interactions, only used when envelope_form='polynomial'.
    pert_form (str): the perturbation form, supported: ['by u' (default), 'fix x']
    dT (float): dt unit used in numerical ODE solving, default: 0.1.
    ode_degree (int): See kernel.py for more details, default: 1.
    ode_solver (str): The ODE solver methods used for numerical simulation,
                      supported input: ["euler", "midpoint", "heun"(default), "rk4"]
    ode_last_steps (int): The number of last iterations used to determine oscillation, default: 2.

    ### Traning procedure
    seed (int): The random seed used for numpy (e.g. data partition) and tensorflow (model training),
                default: working idx + 1000 (see main.py for more details)
    stages (dict): A dictionary with all the training procedures. Check out the ./configs/ for example stages.
                   Note that the configs fed to the `substage` dict would override the configs fed into the `Config`
                   instance.
    n_epoch (int): Maximum number of epoches per substage.
    n_iter (int): Maximum number of iterations per epoch.
    n_iter_buffer (int): The moving window for eval losses during training. (Default: 5)
    n_iter_patience (int): How many iterations without buffered loss on validation dataset decreases would result in
                           an earlystop in training (for each substage). (Default: 100)

    weight_loss (str): Whether to add a weight to loss metrics, currently supported input: ["expr", "None"(default)]
    lr_val (float): The learning rate for Adam Optimizer. Note that we used the default beta1 and beta2 for Adam.
    l1lambda (float): l1 regularization strength, default: 0.
    l2lambda (float): l2 regularization strength, default: 0.
    """

    def __init__(self, config_file):

        with open(config_file, "r") as f:
            config_dict = json.load(f)

        self.experiment_id = config_dict["experiment_id"] if "experiment_id" in config_dict else "Debugging"
        self.model_prefix = config_dict["model_prefix"] if "model_prefix" in config_dict else "seed"
        self.ckpt_name = config_dict["ckpt_name"] if "ckpt_name" in config_dict else "model.ckpt"
        self.export_verbose = config_dict["export_verbose"] if "export_verbose" in config_dict else 2

        self.experiment_type = config_dict["experiment_type"] if "experiment_type" in config_dict else "random partition"
        assert self.experiment_type in ["random partition", "random partition with replicates", "leave one out (w/o single)",
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
