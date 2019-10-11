import datetime
import pertbio
import sys
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import shutil
import argparse

parser = argparse.ArgumentParser(description='CellBox main script')
parser.add_argument('-config', '--experiment_config_path', required=True, type=str, help="Path of experiment config")
parser.add_argument('-i', '--working_index', default=0)
parser.add_argument('-drug', '--drug_index', required=False)
master_args = parser.parse_args()

def set_seed(seed):
    seed = int(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)

def prepare_workdir(cfg):
    # Read Data
    cfg.pert = pd.read_csv(cfg.pert_file, header=None)
    cfg.expr = pd.read_csv(cfg.expr_file, header=None)
    cfg.node_index = pd.read_csv(cfg.node_index_file, header=None, names=None)
    cfg.loo = pd.read_csv("data/loo_label.csv", header=None)

    # Create Output Folder
    try:
        experiment_path = 'results/{}_{}'.format(cfg.experiment_id, md5)
        os.makedirs(experiment_path)
    except:
        pass
    os.chdir(experiment_path)

    if "leave one out" in cfg.experiment_type:
        try:
            cfg.model_prefix = '{}_{}'.format(cfg.model_prefix, cfg.drug_index)
        except:
            raise Exception('Drug index not specified')
    cfg.working_index = cfg.model_prefix + "_" + str(working_index).zfill(3)

    try:
        shutil.rmtree(cfg.working_index)
    except:
        pass
    os.makedirs(cfg.working_index)
    os.chdir(cfg.working_index)
    with open("record_eval.csv", 'w') as f:
        f.write("iter,train_loss,valid_loss,train_mse,valid_mse,test_mse,time_elapsed\n")

    # Load dataset
    dataset = pertbio.dataset.factory(cfg)
    return dataset


if __name__ == '__main__':
    ### Launching expr
    working_index = master_args.working_index
    cfg = pertbio.config.Config(master_args.experiment_config_path)
    cfg.ckpt_path_full = os.path.join('./', cfg.ckpt_name)   
    md5 = pertbio.utils.md5(str(vars(cfg)))
    try:
        cfg.drug_index = master_args.drug_index
    except:
        pass

    try:
        set_seed(random_seed)
    except:
        seed = int(working_index) + 1000
        set_seed(seed)
    print(vars(cfg))

    cfg.dataset = prepare_workdir(cfg)

    logger = pertbio.utils.time_logger(time_logger_step = 1, hierachy = 3)
    args = cfg
    for i, stage in enumerate(cfg.stages):
        logger.log("Training on stage {}/{} ...".format(i+1,len(cfg.stages)))
        args.sub_stages = stage['sub_stages']
        args.n_T = stage['nT']
        pertbio.train.train_model(args)
