"""
This module defines utility functions during training, including the loss function,
the optimizer, the timer, and the md5 key for each configuration
"""

import time
import hashlib
import tensorflow as tf


def loss(x_gold, x_hat, W, l1=0, l2=0, weight=1.):
    """evaluate loss"""
    if isinstance(x_gold, tf.SparseTensor):
        x_gold = tf.sparse.to_dense(x_gold)

    with tf.compat.v1.variable_scope("loss", reuse=True):
        loss_mse = tf.reduce_mean(tf.square(x_gold - x_hat) * tf.abs(weight))
        l1_loss = l1 * tf.reduce_sum(tf.abs(W))
        l2_loss = l2 * tf.reduce_sum(tf.square(tf.abs(W)))
        loss_full = loss_mse + l1_loss + l2_loss
    return loss_full, loss_mse


def optimize(loss_in, lr, optimizer=tf.compat.v1.train.AdamOptimizer, var_list=None):
    """
    Optimize the training loss using Adam

    Args:
        loss_in (float): training loss, mean squared error + L1 regularization term
        lr (float): placeholder for learning rate
        optimizer: default tf.train.AdamOptimizer
        var_list: list of vars to be optimized
    Returns:
        opt_op (optimizer): op to optimize the training loss
        loss (loss): training loss, including regularization if applicable
    """
    if var_list is None:
        var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
    with tf.compat.v1.variable_scope("optimization", reuse=tf.compat.v1.AUTO_REUSE):
        opt = optimizer(lr)
        opt_op = opt.minimize(loss_in, var_list=var_list)
    return opt_op


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


def md5(key):
    """
    returns a hashed with md5 string of the key
    """
    return hashlib.md5(key.encode()).hexdigest()
