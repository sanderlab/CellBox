import numpy as np
import pandas as pd
import tensorflow as tf
import os


def factory(cfg):

        cfg.pert_in = tf.compat.v1.placeholder(tf.float32, [None, cfg.n_x], name='pert_in')
        cfg.expr_out = tf.compat.v1.placeholder(tf.float32, [None, cfg.n_x], name='expr_out')
        cfg.pert = pd.read_csv(os.path.join(cfg.root_dir, cfg.pert_file), header=None, dtype=np.float32)
        cfg.expr = pd.read_csv(os.path.join(cfg.root_dir, cfg.expr_file), header=None, dtype=np.float32)
    cfg.l1_lambda = tf.compat.v1.placeholder(tf.float32, name='lambda')
    cfg.lr = tf.compat.v1.placeholder(tf.float32, name='lr')

    # Prepare dataset iterators
    dataset = tf.data.Dataset.from_tensor_slices((cfg.pert_in, cfg.expr_out))
    cfg.iter_train = tf.compat.v1.data.make_initializable_iterator(
        dataset.batch(cfg.batchsize).shuffle(buffer_size=1024, reshuffle_each_iteration=True))
    cfg.iter_monitor = tf.compat.v1.data.make_initializable_iterator(
        dataset.repeat().batch(cfg.batchsize).shuffle(buffer_size=1024, reshuffle_each_iteration=True))
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
        'train_set' : {
            cfg.pert_in: cfg.dataset['pert_train'],
            cfg.expr_out: cfg.dataset['expr_train'],
        },
        'valid_set': {
            cfg.pert_in: cfg.dataset['pert_valid'],
            cfg.expr_out: cfg.dataset['expr_valid'],
        },
        'test_set':{
            cfg.pert_in: cfg.dataset['pert_test'],
            cfg.expr_out: cfg.dataset['expr_test']
        }
    }
    return cfg


def s2c(cfg):
    double_idx = cfg.loo.all(axis=1)
    testidx = double_idx

    nexp, n_x = cfg.pert.shape
    nvalid = nexp - sum(testidx)
    ntrain = int(nvalid * cfg.validset_ratio)

    valid_pos = np.random.choice(range(nvalid), nvalid, replace=False)
    pert_train = cfg.pert[~testidx]
    train_data = cfg.expr[~testidx]
    dataset = {
        "node_index": cfg.node_index,
        "pert_train": pert_train.iloc[valid_pos[:ntrain], :].values,
        "pert_valid": pert_train.iloc[valid_pos[ntrain:], :].values,
        "pert_test": cfg.pert[testidx],
        "pert_full": cfg.pert,
        "train_data": train_data.iloc[valid_pos[:ntrain], :].values,
        "valid_data": train_data.iloc[valid_pos[ntrain:], :].values,
        "test_data": cfg.expr[testidx],
        "train_pos": valid_pos[:ntrain].values,
        "valid_pos": valid_pos[ntrain:].values,
        "test_pos": testidx
    }

    # TODO: class Dataset of Sample instances

    return dataset


def loo(cfg, singles):
    drug_index = int(cfg.drug_index)
    double_idx = cfg.loo.all(axis=1)

    testidx = (cfg.loo == drug_index).any(axis=1)

    if singles:
        testidx = pd.concat([testidx, double_idx], axis=1)
        testidx = testidx.all(axis=1)

    nexp, n_x = cfg.pert.shape
    nvalid = nexp - sum(testidx)
    ntrain = int(nvalid * cfg.validset_ratio)

    valid_pos = np.random.choice(range(nvalid), nvalid, replace=False)
    pert_train = cfg.pert[~testidx]
    train_data = cfg.expr[~testidx]

    dataset = {
        "node_index": cfg.node_index,
        "pert_train": pert_train.iloc[valid_pos[:ntrain], :],
        "pert_valid": pert_train.iloc[valid_pos[ntrain:], :],
        "pert_test": cfg.pert[testidx],
        "pert_full": cfg.pert,
        "train_data": train_data.iloc[valid_pos[:ntrain], :],
        "valid_data": train_data.iloc[valid_pos[ntrain:], :],
        "test_data": cfg.expr[testidx]
    }

    return dataset


def random_partition(cfg):

    nexp, n_x = cfg.pert.shape
    nvalid = int(nexp * cfg.trainset_ratio)
    ntrain = int(nvalid * cfg.validset_ratio)
    try:
        random_pos = np.genfromtxt('random_pos.csv')
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

        dataset.update({
            "pert_train": cfg.pert.iloc[random_pos[:ntrain], :].values,
            "pert_valid": cfg.pert.iloc[random_pos[ntrain:nvalid], :].values,
            "pert_test": cfg.pert.iloc[random_pos[nvalid:], :].values,
            "expr_train": cfg.expr.iloc[random_pos[:ntrain], :].values,
            "expr_valid": cfg.expr.iloc[random_pos[ntrain:nvalid], :].values,
            "expr_test": cfg.expr.iloc[random_pos[nvalid:], :].values
        })

    return dataset
