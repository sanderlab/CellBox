import json

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.experiment_id = config['experiment_id']
        self.experiment_type = config['experiment_type']
        self.model_prefix = config['model_prefix']

        self.seed = config['seed']
        self.iterations = config['iterations']
        self.n_iter_buffer = config['n_iter_buffer']
        self.save_iterations = config['save_iterations']
        self.dT = config['dT']

        self.pert_file = config['pert_file']
        self.expr_file = config['expr_file']
        self.trainset_ratio = config['trainset_ratio']
        self.validset_ratio = config['validset_ratio']

        self.node_index_file = config['node_index_file']
        self.pert_v_file = config['pert_v_file']
        self.prediction_output_file = config['prediction_output_file']

        self.n_x = config['n_x']
        self.n_protein_nodes = config['n_protein_nodes']
        self.n_activity_nodes = config['n_activity_nodes']

        self.tail_iters = config['tail_iters']
        self.dropout_percent = config['dropout_percent']
        self.loss_min = config['loss_min']

        self.ckpt_name = config['ckpt_name']
        self.stages = config['stages']
