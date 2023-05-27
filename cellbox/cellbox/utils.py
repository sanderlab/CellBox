"""
This module defines utility functions during training, including the loss function,
the optimizer, the timer, and the md5 key for each configuration
"""

import time
import hashlib
import tensorflow.compat.v1 as tf
import json
from typing import Union, Tuple, Any
tf.disable_v2_behavior()

def loss(
    x_gold: Union[tf.SparseTensor, tf.Tensor],
    x_hat: Union[tf.SparseTensor, tf.Tensor],
    W: tf.Variable,
    l1: float = 0.0,
    l2: float = 0.0,
    weight: float = 1.0) -> Tuple[tf.Tensor, tf.Tensor]:
    """Evaluates the losses.
    
    Args:
        x_gold: The ground truth tensor for values. Expected shape: [B, N].
        x_hat: The predicted tensor for values. Expected shape: [B, N].
        W: The variable for the the weight matrix, used for regularization.
        l1: The lambda 1 for L1 loss.
        l2: The lambda 2 for L2 loss.
        weight: The balance weight for the MSE loss when calculating the total loss. 

    Returns:
        A tuple of the total loss (including regularization) and the raw MSE loss.
    """
    if isinstance(x_gold, tf.SparseTensor):
        x_gold = tf.sparse.to_dense(x_gold)

    with tf.compat.v1.variable_scope("loss", reuse=True):
        loss_mse = tf.reduce_mean(tf.square(x_gold - x_hat) * tf.abs(weight))
        l1_loss = l1 * tf.reduce_sum(tf.abs(W))
        l2_loss = l2 * tf.reduce_sum(tf.square(tf.abs(W)))
        loss_full = loss_mse + l1_loss + l2_loss
    return loss_full, loss_mse


def optimize(
    loss_in: tf.Tensor,
    lr: tf.Variable,
    optimizer: tf.compat.v1.train.Optimizer = tf.compat.v1.train.AdamOptimizer,
    var_list: Any = None):
    """
    Optimizes the training loss using by default the Adam optimizer.

    Args:
        loss_in (float): training loss, mean squared error + L1 regularization term
        lr (float): placeholder for learning rate
        optimizer: default tf.train.AdamOptimizer
        var_list (list): a list of vars to be optimized

    Returns:
        A tuple of two items:
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
    """The class used to measure and log the training time."""
    def __init__(self, time_logger_step: int = 1, hierachy: int = 1) -> None:
        """Creates the logger instance and initialize the variables.
        
        Args:
            time_logger_step: The listening frequency, default=1. 
            hierachy: The header #s of the printed log string during logging. Used
                when running `grep` from the log files.
        """
        self.time_logger_step = time_logger_step
        self.step_count = 0
        self.hierachy = hierachy
        self.time = time.time()

    def log(self, s: str) -> None:
        """Track and log the time difference."""
        if self.step_count % self.time_logger_step == 0:
            print("#" * 4 * self.hierachy, " ", s, "  --time elapsed: %.2f" % (time.time() - self.time))
            self.time = time.time()
            self.step_count += 1


def md5(obj: Any) -> str:
    """Returns a hashed with md5 string of the key. Used as identifiers for file I/O."""
    key = json.dumps(vars(obj), sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()
