import cellbox
import os
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import shutil
import argparse
import json
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.compat.v1.errors import OutOfRangeError

tf.disable_v2_behavior()

def set_seed(in_seed):
    int_seed = int(in_seed)
    tf.compat.v1.set_random_seed(int_seed)
    np.random.seed(int_seed)


def prepare_workdir(in_cfg, working_index, md5):
    # Read Data
    in_cfg.root_dir = os.getcwd()
    in_cfg.node_index = pd.read_csv(in_cfg.node_index_file, header=None, names=None) \
        if hasattr(in_cfg, 'node_index_file') else pd.DataFrame(np.arange(in_cfg.n_x))

    # Create Output Folder
    experiment_path = 'results/{}_{}'.format(in_cfg.experiment_id, md5)
    try:
        os.makedirs(experiment_path)
    except Exception:
        pass
    out_cfg = vars(in_cfg)
    out_cfg = {key: out_cfg[key] for key in out_cfg if type(out_cfg[key]) is not pd.DataFrame}
    os.chdir(experiment_path)
    json.dump(out_cfg, open('config.json', 'w'), indent=4)

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

    print('Working directory is ready at {}.'.format(experiment_path))
    return 0


def get_dataloader(experiment_config_path, tensorflow_code=True, drug_index=None):
    """
    Get either the Tensorflow or Pytorch dataloader
    """
    working_index = 0
    cfg = cellbox.config.Config(experiment_config_path)
    cfg.drug_index = drug_index
    cfg.ckpt_path_full = os.path.join('./', cfg.ckpt_name)
    md5 = cellbox.utils.md5(cfg)
    cfg.seed = working_index + cfg.seed if hasattr(cfg, "seed") else working_index + 1000
    set_seed(cfg.seed)
    #print(vars(cfg))

    prepare_workdir(cfg, working_index, md5)
    logger = cellbox.utils.TimeLogger(time_logger_step=1, hierachy=3)
    args = cfg
    dataloader_list = []
    for i, stage in enumerate(cfg.stages):
        set_seed(cfg.seed)
        if tensorflow_code:
            cfg = cellbox.dataset.factory(cfg)
            iter_train = cfg.iter_train
            iter_monitor = cfg.iter_monitor
            feed_dict = cfg.feed_dicts
            data_dict = {
                "iter_train": cfg.iter_train,
                "iter_monitor": cfg.iter_monitor,
                "feed_dict": cfg.feed_dicts
            }
        else:
            cfg = cellbox.dataset_torch.factory(cfg)
            iter_train = cfg.iter_train
            iter_monitor = cfg.iter_monitor
            data_dict = {
                "iter_train": cfg.iter_train,
                "iter_monitor": cfg.iter_monitor,
            }
        #args.sub_stages = stage['sub_stages']
        #args.n_T = stage['nT']
        #model = cellbox.model.factory(args)
        dataloader_list.append(data_dict)

    return dataloader_list, cfg


def yield_data_from_tensorflow_dataloader(dataloader, feed_dict):
    """
    Yield the data from Tensorflow dataloader until it is exhausted
    """
    config = tf.compat.v1.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = off

    # Launching session
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    items_pert, items_expr = [], []
    next_element = dataloader.get_next()

    sess.run(dataloader.initializer, feed_dict=feed_dict['train_set'])

    try:
        while True:
            result1, result2 = sess.run(next_element, feed_dict=feed_dict['train_set'])
            items_pert.append(result1)
            items_expr.append(result2)
    except OutOfRangeError:
        return items_pert, items_expr
    

def yield_data_from_pytorch_dataloader(dataloader):
    """
    Yield the data from Pytorch dataloader until it is exhausted
    """
    items_pert, items_expr = [], []
    for pert, expr in dataloader:
        items_pert.append(pert)
        items_expr.append(expr)
    return items_pert, items_expr


def s2c_row_inds(loo_label_dir):
    """
    Identify the rows of the dataset that only has one drug.
    The information is stored in the loo_label file
    """
    loo_label = pd.read_csv(loo_label_dir, header=None)
    rows_with_single_drugs = loo_label.index[(loo_label[[0, 1]] == 0).any(axis=1)].tolist()
    rows_with_multiple_drugs = list(set(list(range(loo_label.shape[0]))) - set(rows_with_single_drugs))
    return rows_with_single_drugs, rows_with_multiple_drugs


def loo_row_inds(loo_label_dir, cfg):
    """
    Identify the rows of the dataset that leaves out one specific drug
    There is some complication in this function, check https://github.com/sanderlab/CellBox/issues/48
    """
    drug_indices_map = []
    for drug_index in range(14):
        double_idx = cfg.loo.all(axis=1)
        testidx = (cfg.loo == drug_index).any(axis=1)

        if cfg.experiment_type == 'leave one out (w/o single)':
            singles = False
        elif cfg.experiment_type == 'leave one out (w/ single)':
            singles = True

        if singles:
            testidx = pd.concat([testidx, double_idx], axis=1)
            testidx = testidx.all(axis=1)

        loo_label = pd.read_csv(loo_label_dir, header=None)[testidx]
        for i in range(14):
            if (loo_label == i).any(axis=1).all():
                drug_indices_map.append(i)
                break
        
    print(f"Drug indices map: {drug_indices_map}")
    true_drug_index = drug_indices_map[cfg.drug_index]
    loo_label = pd.read_csv(loo_label_dir, header=None)
    ind_rows = loo_label.index[(loo_label[[0, 1]] == true_drug_index).any(axis=1)].tolist()
    return np.array(ind_rows)