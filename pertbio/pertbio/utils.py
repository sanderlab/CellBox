import tensorflow as tf
import numpy as np
import time

def loss(x_gold, x_hat,l1, W):
    with tf.variable_scope("loss", reuse=True):
        loss_mse = tf.reduce_mean(tf.square((x_gold - x_hat)))
        loss = loss_mse + l1 * tf.reduce_sum(tf.abs(W))
    return loss, loss_mse

def optimize(loss, lr, optimizer=tf.train.AdamOptimizer):
    """
    Optimize the training loss using Adam

    Args:
        loss (float): training loss, mean squared error + L1 regularization term
        lr (float): placeholder for learning rate
        optimizer: default tf.train.AdamOptimizer
    Returns:
        opt_op (optimizer): op to optimize the training loss
        loss (loss): training loss, including regularization if applicable
    """
    with tf.variable_scope("optimization", reuse=tf.AUTO_REUSE):
        opt = optimizer(lr)
        opt_op = opt.minimize(loss)
    return opt_op, loss

class time_logger():
    def __init__(self, time_logger_step = 1, hierachy = 1):
        self.time_logger_step = time_logger_step
        self.step_count = 0
        self.hierachy = hierachy
        self.time = time.time()

    def log(self, s):
        if self.step_count%self.time_logger_step==0:
            print("#" * 4 * self.hierachy, " ", s, "  --time elapsed: %.2f"%(time.time() - self.time))
            self.time = time.time()
            self.step_count += 1
