import sys
import pertbio
import os
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


def set_seed(in_seed):
    int_seed = int(in_seed)
    tf.compat.v1.set_random_seed(int_seed)
    np.random.seed(int_seed)


def prepare_workdir(in_cfg):
    # Read Data
    in_cfg.pert = pd.read_csv(in_cfg.pert_file, header=None, dtype=np.float32)
    in_cfg.expr = pd.read_csv(in_cfg.expr_file, header=None, dtype=np.float32)
    in_cfg.node_index = pd.read_csv(in_cfg.node_index_file, header=None, names=None)
    in_cfg.loo = pd.read_csv("data/loo_label.csv", header=None)

    # Create Output Folder
    experiment_path = 'results/{}_{}'.format(in_cfg.experiment_id, md5)
    try:
        os.makedirs(experiment_path)
    except Exception:
        pass

    os.chdir(experiment_path)

    if "leave one out" in in_cfg.experiment_type:
        try:
            in_cfg.model_prefix = '{}_{}'.format(in_cfg.model_prefix, in_cfg.drug_index)
        except Exception('Drug index not specified') as e:
            raise e

    in_cfg.working_index = in_cfg.model_prefix + "_" + str(working_index).zfill(3)

    try:
        shutil.rmtree(in_cfg.working_index)
    except Exception:
        pass
    os.makedirs(in_cfg.working_index)
    os.chdir(in_cfg.working_index)
    with open("record_eval.csv", 'w') as f:
        f.write("epoch,iter,train_loss,valid_loss,train_mse,valid_mse,test_mse,time_elapsed\n")

    # Load dataset
    dataset = pertbio.dataset.factory(cfg)

    print('Working directory is ready at {}.'.format(experiment_path))
    return dataset


if __name__ == '__main__':
    # Launching expr
    working_index = master_args.working_index
    cfg = pertbio.config.Config(master_args.experiment_config_path)
    cfg.ckpt_path_full = os.path.join('./', cfg.ckpt_name)
    md5 = pertbio.utils.md5(str(vars(cfg)))
    cfg.drug_index = master_args.drug_index if hasattr(master_args, "drug_index") else None
    cfg.seed = working_index + cfg.seed if hasattr(cfg, "seed") else working_index + 1000
    set_seed(cfg.seed)
    print(vars(cfg))

    cfg.dataset = prepare_workdir(cfg)

    logger = pertbio.utils.TimeLogger(time_logger_step=1, hierachy=3)
    args = cfg
    for i, stage in enumerate(cfg.stages):
        logger.log("Training on stage {}/{} ...".format(i + 1, len(cfg.stages)))
        args.sub_stages = stage['sub_stages']
        args.n_T = stage['nT']
        pertbio.train.train_model(args)
