"""
This module defines the data partitioning for different training schemes,
including single-to-combo (s2c), leave-one-drug-out cross-validations (loo),
and random partition of the dataset.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse


def factory(cfg):
    """formulate training dataset"""
    # Prepare data
    if cfg.sparse_data:
        cfg.pert_in = tf.compat.v1.sparse.placeholder(tf.float32, [None, cfg.n_x], name='pert_in')
        cfg.expr_out = tf.compat.v1.sparse.placeholder(tf.float32, [None, cfg.n_x], name='expr_out')
        cfg.pert = sparse.load_npz(os.path.join(cfg.root_dir, cfg.pert_file))
        cfg.expr = sparse.load_npz(os.path.join(cfg.root_dir, cfg.expr_file))
    else:
        cfg.pert_in = tf.compat.v1.placeholder(tf.float32, [None, cfg.n_x], name='pert_in')
        cfg.expr_out = tf.compat.v1.placeholder(tf.float32, [None, cfg.n_x], name='expr_out')
        cfg.pert = pd.read_csv(os.path.join(cfg.root_dir, cfg.pert_file), header=None, dtype=np.float32)
        cfg.expr = pd.read_csv(os.path.join(cfg.root_dir, cfg.expr_file), header=None, dtype=np.float32)

    # add noise
    if hasattr(cfg, "add_noise_level") and cfg.add_noise_level > 0:
        np.random.seed(cfg.seed)
        try:
            assert not cfg.sparse_data
        except Exception:
            raise Exception("Adding noise to sparse data format is yet to be supported")
        cfg.expr.iloc[:] = cfg.expr.values + np.random.normal(loc=0, scale=cfg.add_noise_level, size=cfg.expr.shape)

    # prepare training placeholders
    cfg.l1_lambda_placeholder = tf.compat.v1.placeholder(tf.float32, name='l1_lambda')
    cfg.l2_lambda_placeholder = tf.compat.v1.placeholder(tf.float32, name='l2_lambda')
    cfg.lr = tf.compat.v1.placeholder(tf.float32, name='lr')

    # Prepare dataset iterators
    dataset = tf.data.Dataset.from_tensor_slices((cfg.pert_in, cfg.expr_out))
    cfg.iter_train = tf.compat.v1.data.make_initializable_iterator(
        dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True).batch(cfg.batchsize))
    cfg.iter_monitor = tf.compat.v1.data.make_initializable_iterator(
        dataset.repeat().shuffle(buffer_size=1024, reshuffle_each_iteration=True).batch(cfg.batchsize))
    cfg.iter_eval = tf.compat.v1.data.make_initializable_iterator(dataset.batch(cfg.batchsize))

    # Data partition
    if cfg.experiment_type == 'random partition' or cfg.experiment_type == 'full data':
        cfg.dataset = random_partition(cfg)

    elif cfg.experiment_type == 'leave one out (w/o single)':
        cfg.dataset = loo(cfg, singles=False)

    elif cfg.experiment_type == 'leave one out (w/ single)':
        cfg.dataset = loo(cfg, singles=True)

    elif cfg.experiment_type == 'single to combo':
        cfg.dataset = s2c(cfg)
    else:
        raise Exception('Invalid experiment type. \nValid options: [random partition, leave one out (w/o single), '
                        'leave one out (w/ single), full data, single to combo]')

    # Prepare feed_dicts
    cfg.feed_dicts = {
        'train_set': {
            cfg.pert_in: cfg.dataset['pert_train'],
            cfg.expr_out: cfg.dataset['expr_train'],
        },
        'valid_set': {
            cfg.pert_in: cfg.dataset['pert_valid'],
            cfg.expr_out: cfg.dataset['expr_valid'],
        },
        'test_set': {
            cfg.pert_in: cfg.dataset['pert_test'],
            cfg.expr_out: cfg.dataset['expr_test']
        }
    }
    return cfg


def s2c(cfg):
    """single-to-combo trials"""
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
            "pert_train": npz_to_feedable_arrays(cfg.pert[~testidx][valid_pos[:ntrain]]),
            "pert_valid": npz_to_feedable_arrays(cfg.pert[~testidx][valid_pos[ntrain:]]),
            "pert_test": npz_to_feedable_arrays(cfg.pert[testidx]),
            "expr_train": npz_to_feedable_arrays(cfg.expr[~testidx][valid_pos[:ntrain]]),
            "expr_valid": npz_to_feedable_arrays(cfg.expr[~testidx][valid_pos[ntrain:]]),
            "expr_test": npz_to_feedable_arrays(cfg.expr[testidx])
        })
    else:
        dataset.update({
            "pert_train": cfg.pert[~testidx].iloc[valid_pos[:ntrain], :].values,
            "pert_valid": cfg.pert[~testidx].iloc[valid_pos[ntrain:], :].values,
            "pert_test": cfg.pert[testidx],
            "expr_train": cfg.expr[~testidx].iloc[valid_pos[:ntrain], :].values,
            "expr_valid": cfg.expr[~testidx].iloc[valid_pos[ntrain:], :].values,
            "expr_test": cfg.expr[testidx]
        })

    # TODO: class Dataset of Sample instances

    return dataset


def loo(cfg, singles):
    """leave-one-drug-out trials"""
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
            "pert_train": npz_to_feedable_arrays(cfg.pert[~testidx][valid_pos[:ntrain]]),
            "pert_valid": npz_to_feedable_arrays(cfg.pert[~testidx][valid_pos[ntrain:]]),
            "pert_test": npz_to_feedable_arrays(cfg.pert[testidx]),
            "expr_train": npz_to_feedable_arrays(cfg.expr[~testidx][valid_pos[:ntrain]]),
            "expr_valid": npz_to_feedable_arrays(cfg.expr[~testidx][valid_pos[ntrain:]]),
            "expr_test": npz_to_feedable_arrays(cfg.expr[testidx])
        })
    else:
        dataset.update({
            "pert_train": cfg.pert[~testidx].iloc[valid_pos[:ntrain], :].values,
            "pert_valid": cfg.pert[~testidx].iloc[valid_pos[ntrain:], :].values,
            "pert_test": cfg.pert[testidx],
            "expr_train": cfg.expr[~testidx].iloc[valid_pos[:ntrain], :].values,
            "expr_valid": cfg.expr[~testidx].iloc[valid_pos[ntrain:], :].values,
            "expr_test": cfg.expr[testidx]
        })

    return dataset


def random_partition(cfg):
    """random partition of data"""
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
            "pert_train": npz_to_feedable_arrays(cfg.pert[random_pos[:ntrain], :]),
            "pert_valid": npz_to_feedable_arrays(cfg.pert[random_pos[ntrain:nvalid], :]),
            "pert_test": npz_to_feedable_arrays(cfg.pert[random_pos[nvalid:], :]),
            "expr_train": npz_to_feedable_arrays(cfg.expr[random_pos[:ntrain], :]),
            "expr_valid": npz_to_feedable_arrays(cfg.expr[random_pos[ntrain:nvalid], :]),
            "expr_test": npz_to_feedable_arrays(cfg.expr[random_pos[nvalid:], :])
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


def npz_to_feedable_arrays(npz):
    """convert sparse matrix to arrays"""
    coo = npz.tocoo()
    indices = [[i, j] for i, j in zip(coo.row, coo.col)]
    values = coo.data
    dense_shape = coo.shape
    return indices, values, dense_shape
