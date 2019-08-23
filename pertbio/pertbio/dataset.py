import tensorflow as tf
import numpy as np
from pertbio.utils import loss, optimize
import pandas as pd

def factory(cfg):

    if cfg.experiment_type == 'random partition' or cfg.experiment_type == 'full data':
        return random_partition(cfg)

    if cfg.experiment_type == 'leave one out (w/o single)':
        return loo(cfg, singles = False)

    if cfg.experiment_type == 'leave one out (w/ single)':
        return loo(cfg, singles = True)

    if cfg.experiment_type == 'single to combo':
        return s2c(cfg)

    raise Exception('Invalid experiment type. \nValid options: [random partition, leave one out (w/o single), leave one out (w/ single), full data, single to combo]')

def s2c(cfg):
    double_idx = cfg.loo.all(axis=1)
    single_idx = ~double_idx
    testidx = double_idx

    nexp, n_x = cfg.pert.shape
    nvalid = nexp - sum(testidx)
    ntrain = int(nvalid * cfg.validset_ratio)

    valid_pos = np.random.choice(range(nvalid), nvalid, replace = False)
    pert_train = cfg.pert[~testidx]
    train_data = cfg.expr[~testidx]

    dataset = {
        "node_index": cfg.node_index,
        "pert_train": pert_train.iloc[valid_pos[:ntrain],:],
        "pert_valid": pert_train.iloc[valid_pos[ntrain:],:],
        "pert_test": cfg.pert[testidx],
        "train_data": train_data.iloc[valid_pos[:ntrain],:],
        "valid_data": train_data.iloc[valid_pos[ntrain:],:],
        "test_data": cfg.expr[testidx]
    }

    return dataset



def loo(cfg, singles):
    drug_index = int(cfg.drug_index)
    double_idx = cfg.loo.all(axis=1)
    single_idx = ~double_idx

    testidx = (cfg.loo==drug_index).any(axis=1)

    if singles:
        testidx = pd.concat([testidx,double_idx],axis=1)
        testidx = testidx.all(axis=1)

    nexp, n_x = cfg.pert.shape
    nvalid = nexp - sum(testidx)
    ntrain = int(nvalid * cfg.validset_ratio)

    valid_pos = np.random.choice(range(nvalid), nvalid, replace = False)
    pert_train = cfg.pert[~testidx]
    train_data = cfg.expr[~testidx]

    dataset = {
        "node_index": cfg.node_index,
        "pert_train": pert_train.iloc[valid_pos[:ntrain],:],
        "pert_valid": pert_train.iloc[valid_pos[ntrain:],:],
        "pert_test": cfg.pert[testidx],
        "train_data": train_data.iloc[valid_pos[:ntrain],:],
        "valid_data": train_data.iloc[valid_pos[ntrain:],:],
        "test_data": cfg.expr[testidx]
    }

    return dataset


def random_partition(cfg):
    nexp, n_x = cfg.pert.shape
    nvalid = int(nexp * cfg.trainset_ratio)
    ntrain = int(nvalid * cfg.validset_ratio)
    try:
        random_pos = np.genfromtxt('random_pos.csv')
    except:
        random_pos = np.random.choice(range(nexp), nexp, replace = False)
        np.savetxt('random_pos.csv', random_pos, fmt = '%d')

    dataset = {
        "node_index": cfg.node_index,
        "pert_train": cfg.pert.iloc[random_pos[:ntrain], :],
        "pert_valid": cfg.pert.iloc[random_pos[ntrain:nvalid], :],
        "pert_test": cfg.pert.iloc[random_pos[nvalid:], :],
        "train_data": cfg.expr.iloc[random_pos[:ntrain], :],
        "valid_data": cfg.expr.iloc[random_pos[ntrain:nvalid], :],
        "test_data": cfg.expr.iloc[random_pos[nvalid:], :]
    }

    return dataset
