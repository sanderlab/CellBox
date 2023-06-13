"""
This module defines the data partitioning for different training schemes,
including single-to-combo (s2c), leave-one-drug-out cross-validations (loo),
and random partition of the dataset.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from scipy import sparse



def factory(cfg):
    """formulate training dataset"""
    # Prepare data
    # To replace cfg.pert_in and cfg.expr_out, should it take the full pert and expr datasets?
    if cfg.sparse_data:
        #cfg.pert_in = tf.compat.v1.sparse.placeholder(tf.float32, [None, cfg.n_x], name='pert_in')
        #cfg.expr_out = tf.compat.v1.sparse.placeholder(tf.float32, [None, cfg.n_x], name='expr_out')
        cfg.pert = sparse.load_npz(os.path.join(cfg.root_dir, cfg.pert_file))
        cfg.expr = sparse.load_npz(os.path.join(cfg.root_dir, cfg.expr_file))
    else:
        #cfg.pert_in = tf.compat.v1.placeholder(tf.float32, [None, cfg.n_x], name='pert_in')
        #cfg.expr_out = tf.compat.v1.placeholder(tf.float32, [None, cfg.n_x], name='expr_out')
        cfg.pert = pd.read_csv(os.path.join(cfg.root_dir, cfg.pert_file), header=None, dtype=np.float32)
        cfg.expr = pd.read_csv(os.path.join(cfg.root_dir, cfg.expr_file), header=None, dtype=np.float32)
    
    group_df = pd.DataFrame(np.where(cfg.pert != 0), index=['row_id', 'pert_idx']).T.groupby('row_id')
    max_combo_degree = group_df.pert_idx.count().max()
    cfg.loo = pd.DataFrame(group_df.pert_idx.apply(
        lambda x: pad_and_realign(x, max_combo_degree, cfg.n_activity_nodes - 1)
    ).tolist())

    # add noise
    if cfg.add_noise_level > 0:
        np.random.seed(cfg.seed)
        assert not cfg.sparse_data, "Adding noise to sparse data format is yet to be supported"
        cfg.expr.iloc[:] = cfg.expr.values + np.random.normal(loc=0, scale=cfg.add_noise_level, size=cfg.expr.shape)

    # Data partition
    if cfg.experiment_type == 'random partition' or cfg.experiment_type == 'full data':
        cfg.dataset = random_partition(cfg)

    elif cfg.experiment_type == 'leave one out (w/o single)':
        cfg.dataset = loo(cfg, singles=False)

    elif cfg.experiment_type == 'leave one out (w/ single)':
        cfg.dataset = loo(cfg, singles=True)

    elif cfg.experiment_type == 'single to combo':
        cfg.dataset = s2c(cfg)

    elif cfg.experiment_type == 'random partition with replicates':
        cfg.dataset = random_partition_with_replicates(cfg)

    # Prepare feed_dicts
    #cfg.feed_dicts = {
    #    'train_set': {
    #        cfg.pert_in: cfg.dataset['pert_train'],
    #        cfg.expr_out: cfg.dataset['expr_train'],
    #    },
    #    'valid_set': {
    #        cfg.pert_in: cfg.dataset['pert_valid'],
    #        cfg.expr_out: cfg.dataset['expr_valid'],
    #    },
    #    'test_set': {
    #        cfg.pert_in: cfg.dataset['pert_test'],
    #        cfg.expr_out: cfg.dataset['expr_test']
    #    }
    #}
    cfg = get_tensors(cfg)

    return cfg


def pad_and_realign(x, length, idx_shift=0):
    x -= idx_shift
    padded = np.pad(x, (0, length - len(x)), 'constant')
    return padded


def get_tensors(cfg):
    # prepare training placeholders
    #cfg.l1_lambda_placeholder = tf.compat.v1.placeholder(tf.float32, name='l1_lambda')
    #cfg.l2_lambda_placeholder = tf.compat.v1.placeholder(tf.float32, name='l2_lambda')
    #cfg.lr = tf.compat.v1.placeholder(tf.float32, name='lr')
    cfg.l1_lambda_placeholder = 0.0
    cfg.l2_lambda_placeholder = 0.0
    cfg.lr = 0.0

    # Prepare dataset iterators (these can be replaced with DataLoader)
    #dataset = tf.data.Dataset.from_tensor_slices((cfg.pert_in, cfg.expr_out))
    #cfg.iter_train = tf.compat.v1.data.make_initializable_iterator(
    #    dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True).batch(cfg.batchsize))
    #cfg.iter_monitor = tf.compat.v1.data.make_initializable_iterator(
    #    dataset.repeat().shuffle(buffer_size=1024, reshuffle_each_iteration=True).batch(cfg.batchsize))
    #cfg.iter_eval = tf.compat.v1.data.make_initializable_iterator(dataset.batch(cfg.batchsize))
    train_dataset = TensorDataset(
        torch.from_numpy(cfg.dataset["pert_train"]), torch.from_numpy(cfg.dataset["expr_train"])
    )
    val_dataset = TensorDataset(
        torch.from_numpy(cfg.dataset["pert_valid"]), torch.from_numpy(cfg.dataset["expr_valid"])
    )
    test_dataset = TensorDataset(
        torch.from_numpy(cfg.dataset["pert_test"]), torch.from_numpy(cfg.dataset["expr_test"])
    )

    cfg.iter_train = DataLoader(
        train_dataset, batch_size=cfg.batchsize, shuffle=True
    )
    cfg.iter_monitor = DataLoader(
        val_dataset, batch_size=cfg.batchsize, shuffle=True
    )
    cfg.iter_eval = DataLoader(
        test_dataset, batch_size=cfg.batchsize, shuffle=True
    )

    return cfg


def s2c(cfg):
    """data parition for single-to-combo experiments"""
    double_idx = cfg.loo.all(axis=1)
    testidx = double_idx

    nexp, _ = cfg.pert.shape
    nvalid = nexp - sum(testidx)
    ntrain = int(nvalid * cfg.validset_ratio)

    valid_pos = np.random.choice(range(nvalid), nvalid, replace=False)
    dataset = {
        "node_index": cfg.node_index,
        "pert_full": cfg.pert,
        "train_pos": valid_pos[:ntrain],
        "valid_pos": valid_pos[ntrain:],
        "test_pos": testidx
    }

    if cfg.sparse_data:
        dataset.update({
            "pert_train": sparse_to_feedable_arrays(cfg.pert[~testidx][valid_pos[:ntrain]]),
            "pert_valid": sparse_to_feedable_arrays(cfg.pert[~testidx][valid_pos[ntrain:]]),
            "pert_test": sparse_to_feedable_arrays(cfg.pert[testidx]),
            "expr_train": sparse_to_feedable_arrays(cfg.expr[~testidx][valid_pos[:ntrain]]),
            "expr_valid": sparse_to_feedable_arrays(cfg.expr[~testidx][valid_pos[ntrain:]]),
            "expr_test": sparse_to_feedable_arrays(cfg.expr[testidx])
        })
    else:
        dataset.update({
            "pert_train": cfg.pert[~testidx].iloc[valid_pos[:ntrain], :].values,
            "pert_valid": cfg.pert[~testidx].iloc[valid_pos[ntrain:], :].values,
            "pert_test": cfg.pert[testidx].values,
            "expr_train": cfg.expr[~testidx].iloc[valid_pos[:ntrain], :].values,
            "expr_valid": cfg.expr[~testidx].iloc[valid_pos[ntrain:], :].values,
            "expr_test": cfg.expr[testidx].values
        })

    return dataset


def loo(cfg, singles):
    """data parition for leave-one-drug-out experiments"""
    drug_index = int(cfg.drug_index)
    double_idx = cfg.loo.all(axis=1)

    testidx = (cfg.loo == drug_index).any(axis=1)

    if singles:
        testidx = pd.concat([testidx, double_idx], axis=1)
        testidx = testidx.all(axis=1)

    nexp, _ = cfg.pert.shape
    nvalid = nexp - sum(testidx)
    ntrain = int(nvalid * cfg.validset_ratio)

    valid_pos = np.random.choice(range(nvalid), nvalid, replace=False)
    dataset = {
        "node_index": cfg.node_index,
        "pert_full": cfg.pert,
        "train_pos": valid_pos[:ntrain],
        "valid_pos": valid_pos[ntrain:],
        "test_pos": testidx
    }

    if cfg.sparse_data:
        dataset.update({
            "pert_train": sparse_to_feedable_arrays(cfg.pert[~testidx][valid_pos[:ntrain]]),
            "pert_valid": sparse_to_feedable_arrays(cfg.pert[~testidx][valid_pos[ntrain:]]),
            "pert_test": sparse_to_feedable_arrays(cfg.pert[testidx]),
            "expr_train": sparse_to_feedable_arrays(cfg.expr[~testidx][valid_pos[:ntrain]]),
            "expr_valid": sparse_to_feedable_arrays(cfg.expr[~testidx][valid_pos[ntrain:]]),
            "expr_test": sparse_to_feedable_arrays(cfg.expr[testidx])
        })
    else:
        dataset.update({
            "pert_train": cfg.pert[~testidx].iloc[valid_pos[:ntrain], :].values,
            "pert_valid": cfg.pert[~testidx].iloc[valid_pos[ntrain:], :].values,
            "pert_test": cfg.pert[testidx].values,
            "expr_train": cfg.expr[~testidx].iloc[valid_pos[:ntrain], :].values,
            "expr_valid": cfg.expr[~testidx].iloc[valid_pos[ntrain:], :].values,
            "expr_test": cfg.expr[testidx].values
        })

    return dataset


def random_partition(cfg):
    """random dataset partition"""
    nexp, _ = cfg.pert.shape
    nvalid = int(nexp * cfg.trainset_ratio)
    ntrain = int(nvalid * cfg.validset_ratio)
    try:
        random_pos = np.genfromtxt('random_pos.csv', defaultfmt='%d')
    except Exception:
        random_pos = np.random.choice(range(nexp), nexp, replace=False)
        np.savetxt('random_pos.csv', random_pos, fmt='%d')

    dataset = {
        "node_index": cfg.node_index,
        "pert_full": cfg.pert,
        "train_pos": random_pos[:ntrain],
        "valid_pos": random_pos[ntrain:nvalid],
        "test_pos": random_pos[nvalid:]
    }

    if cfg.sparse_data:
        dataset.update({
            "pert_train": sparse_to_feedable_arrays(cfg.pert[random_pos[:ntrain], :]),
            "pert_valid": sparse_to_feedable_arrays(cfg.pert[random_pos[ntrain:nvalid], :]),
            "pert_test": sparse_to_feedable_arrays(cfg.pert[random_pos[nvalid:], :]),
            "expr_train": sparse_to_feedable_arrays(cfg.expr[random_pos[:ntrain], :]),
            "expr_valid": sparse_to_feedable_arrays(cfg.expr[random_pos[ntrain:nvalid], :]),
            "expr_test": sparse_to_feedable_arrays(cfg.expr[random_pos[nvalid:], :])
        })
    else:
        dataset.update({
            "pert_train": cfg.pert.iloc[random_pos[:ntrain], :].values,
            "pert_valid": cfg.pert.iloc[random_pos[ntrain:nvalid], :].values,
            "pert_test": cfg.pert.iloc[random_pos[nvalid:], :].values,
            "expr_train": cfg.expr.iloc[random_pos[:ntrain], :].values,
            "expr_valid": cfg.expr.iloc[random_pos[ntrain:nvalid], :].values,
            "expr_test": cfg.expr.iloc[random_pos[nvalid:], :].values
        })

    return dataset


def random_partition_with_replicates(cfg):
    """random dataset partition"""
    nexp = len(np.unique(cfg.loo, axis=0))
    nvalid = int(nexp * cfg.trainset_ratio)
    ntrain = int(nvalid * cfg.validset_ratio)
    conds_train_idx = np.random.choice(range(nexp), nexp, replace=False)
    pos_train = [idx for idx in range(nexp) if idx in conds_train_idx[:ntrain]]
    pos_valid = [idx for idx in range(nexp) if idx in conds_train_idx[ntrain:nvalid]]
    pos_test = [idx for idx in range(nexp) if idx in conds_train_idx[nvalid:]]

    try:
        random_pos = np.genfromtxt('random_pos.csv', defaultfmt='%d')
    except Exception:
        random_pos = np.concatenate([pos_train, pos_valid, pos_test])
        np.savetxt('random_pos.csv', random_pos, fmt='%d')

    dataset = {
        "node_index": cfg.node_index,
        "pert_full": cfg.pert,
        "train_pos": random_pos[:ntrain],
        "valid_pos": random_pos[ntrain:nvalid],
        "test_pos": random_pos[nvalid:]
    }

    if cfg.sparse_data:
        dataset.update({
            "pert_train": sparse_to_feedable_arrays(cfg.pert[random_pos[:ntrain], :]),
            "pert_valid": sparse_to_feedable_arrays(cfg.pert[random_pos[ntrain:nvalid], :]),
            "pert_test": sparse_to_feedable_arrays(cfg.pert[random_pos[nvalid:], :]),
            "expr_train": sparse_to_feedable_arrays(cfg.expr[random_pos[:ntrain], :]),
            "expr_valid": sparse_to_feedable_arrays(cfg.expr[random_pos[ntrain:nvalid], :]),
            "expr_test": sparse_to_feedable_arrays(cfg.expr[random_pos[nvalid:], :])
        })
    else:
        dataset.update({
            "pert_train": cfg.pert.iloc[random_pos[:ntrain], :].values,
            "pert_valid": cfg.pert.iloc[random_pos[ntrain:nvalid], :].values,
            "pert_test": cfg.pert.iloc[random_pos[nvalid:], :].values,
            "expr_train": cfg.expr.iloc[random_pos[:ntrain], :].values,
            "expr_valid": cfg.expr.iloc[random_pos[ntrain:nvalid], :].values,
            "expr_test": cfg.expr.iloc[random_pos[nvalid:], :].values
        })

    return dataset



def sparse_to_feedable_arrays(npz):
    """convert sparse matrix to arrays"""
    coo = npz.tocoo()
    indices = [[i, j] for i, j in zip(coo.row, coo.col)]
    values = coo.data
    dense_shape = coo.shape
    return indices, values, dense_shape
