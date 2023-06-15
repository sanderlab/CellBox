"""
This module defines utility functions during training, including the loss function,
the optimizer, the timer, and the md5 key for each configuration
"""

import time
import hashlib
import torch
import torch.nn as nn
import json

def loss(x_gold, x_hat, W, l1=0, l2=0, weight=1.):
    """
    Evaluate loss

    Args:
        - x_gold, x_hat, W (torch.tensor)
        - l1, l2, weight (float)
    Returns:
        - A single-value loss tensor, e.g. loss_mse = tensor(5)
    """
    #if isinstance(x_gold, tf.SparseTensor):
    #    x_gold = tf.sparse.to_dense(x_gold)
    loss_mse = torch.mean(torch.square(x_gold - x_hat)*torch.abs(weight))
    l1_loss = l1 * torch.sum(torch.abs(W))
    l2_loss = l2 * torch.sum(torch.square(torch.abs(W)))
    loss_full = loss_mse + l1_loss + l2_loss
    return loss_full, loss_mse


def optimize(loss_in, lr, optimizer=torch.optim.Adam, var_list=None):
    """
    Optimize the training loss using Adam
    Do we need this at all for Pytorch?
    Args:
        loss_in (float): training loss, mean squared error + L1 regularization term
        lr (float): placeholder for learning rate
        optimizer: default tf.train.AdamOptimizer
        var_list: list of vars to be optimized
    Returns:
        opt_op (optimizer): op to optimize the training loss
        loss (loss): training loss, including regularization if applicable
    """
    #if var_list is None:
    #    var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
    #with tf.compat.v1.variable_scope("optimization", reuse=tf.compat.v1.AUTO_REUSE):
    #    opt = optimizer(lr)
    #    opt_op = opt.minimize(loss_in, var_list=var_list)
    #return opt_op
    pass


class TimeLogger:
    """calculate training time"""
    def __init__(self, time_logger_step=1, hierachy=1):
        self.time_logger_step = time_logger_step
        self.step_count = 0
        self.hierachy = hierachy
        self.time = time.time()

    def log(self, s):
        """time log"""
        if self.step_count % self.time_logger_step == 0:
            print("#" * 4 * self.hierachy, " ", s, "  --time elapsed: %.2f" % (time.time() - self.time))
            self.time = time.time()
            self.step_count += 1


def md5(obj):
    """
    returns a hashed with md5 string of the key
    """
    key = json.dumps(vars(obj), sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()
