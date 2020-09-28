"""
This module defines the training of the model
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.compat.v1.errors import OutOfRangeError
import cellbox
from cellbox.utils import TimeLogger


def train_substage(model, sess, lr_val, l1_lambda, l2_lambda, n_epoch, n_iter, n_iter_buffer, n_iter_patience, args):
    """
    Training function that does one stage of training. The stage training can be repeated and modified to give better
    training result.

    Args:
        model (CellBox): an CellBox instance
        sess (tf.Session): current session, need reinitialization for every nT
        lr_val (float): learning rate (read in from config file)
        l1_lambda (float): l1 regularization weight
        l2_lambda (float): l2 regularization weight
        n_epoch (int): maximum number of epochs
        n_iter (int): maximum number of iterations
        n_iter_buffer (int): training loss moving average window
        n_iter_patience (int): training loss tolerance
        args: Args or configs
    """

    stages = glob.glob("*best*.csv")
    try:
        substage_i = 1 + max([int(stage[0]) for stage in stages])
    except Exception:
        substage_i = 1

    best_params = Screenshot(args, n_iter_buffer)

    n_unchanged = 0
    idx_iter = 0
    for key in args.feed_dicts:
        args.feed_dicts[key].update({
            model.lr: lr_val,
            model.l1_lambda: l1_lambda,
            model.l2_lambda: l2_lambda
        })
    args.logger.log("--------- lr: {}\tl1: {}\tl2: {}\t".format(lr_val, l1_lambda, l2_lambda))

    sess.run(model.iter_monitor.initializer, feed_dict=args.feed_dicts['valid_set'])
    for idx_epoch in range(n_epoch):

        if idx_iter > n_iter or n_unchanged > n_iter_patience:
            break

        sess.run(model.iter_train.initializer, feed_dict=args.feed_dicts['train_set'])
        while True:
            if idx_iter > n_iter or n_unchanged > n_iter_patience:
                break
            t0 = time.clock()
            try:
                _, loss_train_i, loss_train_mse_i = sess.run(
                    (model.op_optimize, model.train_loss, model.train_mse_loss), feed_dict=args.feed_dicts['train_set'])
            except OutOfRangeError:  # for iter_train
                break

            # record training
            loss_valid_i, loss_valid_mse_i = sess.run(
                (model.monitor_loss, model.monitor_mse_loss), feed_dict=args.feed_dicts['valid_set'])
            new_loss = best_params.avg_n_iters_loss(loss_valid_i)
            if args.export_verbose > 0:
                print(("Substage:{}\tEpoch:{}/{}\tIteration: {}/{}" + "\tloss (train):{:1.6f}\tloss (buffer on valid):"
                       "{:1.6f}" + "\tbest:{:1.6f}\tTolerance: {}/{}").format(substage_i, idx_epoch, n_epoch, idx_iter,
                                                                              n_iter, loss_train_i, new_loss,
                                                                              best_params.loss_min, n_unchanged,
                                                                              n_iter_patience))

            sess.run(model.iter_eval.initializer, feed_dict=args.feed_dicts['test_set'])
            loss_test_mse_i = eval_model(sess, model.iter_eval, (model.eval_loss, model.eval_mse_loss),
                                         args.feed_dicts['test_set'], n_batches_eval=args.n_batches_eval)

            append_record("record_eval.csv",
                          [idx_epoch, idx_iter, loss_train_i, loss_valid_i, loss_train_mse_i,
                           loss_valid_mse_i, loss_test_mse_i, time.clock() - t0])
            
            # early stopping
            idx_iter += 1
            if new_loss < best_params.loss_min:
                n_unchanged = 0
                best_params.screenshot(sess, model, substage_i, args=args,
                                       node_index=args.dataset['node_index'], loss_min=new_loss)
            else:
                n_unchanged += 1

    # Evaluation on valid set
    t0 = time.clock()
    sess.run(model.iter_eval.initializer, feed_dict=args.feed_dicts['valid_set'])
    loss_valid_i, loss_valid_mse_i = eval_model(sess, model.iter_eval, (model.eval_loss, model.eval_mse_loss),
                                                args.feed_dicts['valid_set'], n_batches_eval=args.n_batches_eval)
    append_record("record_eval.csv", [-1, None, None, loss_valid_i, None, loss_valid_mse_i, None, time.clock() - t0])

    # Evaluation on test set
    t0 = time.clock()
    sess.run(model.iter_eval.initializer, feed_dict=args.feed_dicts['test_set'])
    loss_test_mse = eval_model(sess, model.iter_eval, model.eval_mse_loss,
                               args.feed_dicts['test_set'], n_batches_eval=args.n_batches_eval)
    append_record("record_eval.csv", [-1, None, None, None, None, None, loss_test_mse, time.clock() - t0])

    best_params.save()
    args.logger.log("------------------ Substage {} finished!-------------------".format(substage_i))
    save_model(args.saver, sess, './' + args.ckpt_name)


def append_record(filename, contents):
    """define function for appending training record"""
    with open(filename, 'a') as f:
        for content in contents:
            f.write('{},'.format(content))
        f.write('\n')


def eval_model(sess, eval_iter, obj_fn, eval_dict, return_avg=True, n_batches_eval=None):
    """simulate the model for prediction"""
    sess.run(eval_iter.initializer, feed_dict=eval_dict)
    counter = 0
    eval_results = []
    while True:
        try:
            eval_results.append(sess.run(obj_fn, feed_dict=eval_dict))
        except OutOfRangeError:
            break
        counter += 1
        if n_batches_eval is not None and counter > n_batches_eval:
            break
    if return_avg:
        return np.mean(np.array(eval_results), axis=0)
    return np.vstack(eval_results)


def train_model(model, args):
    """Train the model"""
    args.logger = TimeLogger(time_logger_step=1, hierachy=2)

    # Check if all variables in scope
    # TODO: put variables under appropriate scopes
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='initialization'):
        print(i)

    # Initialization
    args.saver = tf.compat.v1.train.Saver()
    from tensorflow.core.protobuf import rewriter_config_pb2
    config = tf.compat.v1.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = off

    # Launching session
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    try:
        args.saver.restore(sess, './' + args.ckpt_name)
        print('Load existing model at {}...'.format(args.ckpt_name))
    except Exception:
        print('Create new model at {}...'.format(args.ckpt_name))

    # Training
    for substage in args.sub_stages:
        n_iter_buffer = substage['n_iter_buffer'] if 'n_iter_buffer' in substage else args.n_iter_buffer
        n_iter = substage['n_iter'] if 'n_iter' in substage else args.n_iter
        n_iter_patience = substage['n_iter_patience'] if 'n_iter_patience' in substage else args.n_iter_patience
        n_epoch = substage['n_epoch'] if 'n_epoch' in substage else args.n_epoch
        l1 = substage['l1lambda'] if 'l1lambda' in substage else args.l1lambda if hasattr(args, 'l1lambda') else 0
        l2 = substage['l2lambda'] if 'l2lambda' in substage else args.l2lambda if hasattr(args, 'l2lambda') else 0
        train_substage(model, sess, substage['lr_val'], l1_lambda=l1, l2_lambda=l2, n_epoch=n_epoch,
                       n_iter=n_iter, n_iter_buffer=n_iter_buffer, n_iter_patience=n_iter_patience, args=args)

    # Terminate session
    sess.close()
    tf.compat.v1.reset_default_graph()


def save_model(saver, sess, path):
    """save model"""
    # Save the variables to disk.
    tmp = saver.save(sess, path)
    print("Model saved in path: %s" % tmp)


class Screenshot(dict):
    """summarize the model"""
    def __init__(self, args, n_iter_buffer):
        # initialize loss_min
        super().__init__()
        self.loss_min = 1000
        # initialize tuning_metric
        self.saved_losses = [self.loss_min]
        self.n_iter_buffer = n_iter_buffer
        # initialize verbose
        self.summary = {}
        self.summary = {}
        self.substage_i = []
        self.export_verbose = args.export_verbose

    def avg_n_iters_loss(self, new_loss):
        """average the last few losses"""
        self.saved_losses = self.saved_losses + [new_loss]
        self.saved_losses = self.saved_losses[-self.n_iter_buffer:]
        return sum(self.saved_losses) / len(self.saved_losses)

    def screenshot(self, sess, model, substage_i, node_index, loss_min, args):
        """evaluate models"""
        self.substage_i = substage_i
        self.loss_min = loss_min
        # Save the variables to disk.
        if self.export_verbose > 0:
            params = sess.run(model.params)
            for item in params:
                try:
                    params[item] = pd.DataFrame(params[item], index=node_index[0])
                except Exception:
                    params[item] = pd.DataFrame(params[item])
            self.update(params)

        if self.export_verbose > 1 or self.export_verbose == -1:  # no params but y_hat
            sess.run(model.iter_eval.initializer, feed_dict=model.args.feed_dicts['test_set'])
            y_hat = eval_model(sess, model.iter_eval, model.eval_yhat, args.feed_dicts['test_set'], return_avg=False)
            y_hat = pd.DataFrame(y_hat, columns=node_index[0])
            self.update({'y_hat': y_hat})

        if self.export_verbose > 2:
            try:
                # TODO: not yet support data iterators
                summary_train = sess.run(model.convergence_metric,
                                         feed_dict={model.in_pert: args.dataset['pert_train']})
                summary_test = sess.run(model.convergence_metric, feed_dict={model.in_pert: args.dataset['pert_test']})
                summary_valid = sess.run(model.convergence_metric,
                                         feed_dict={model.in_pert: args.dataset['pert_valid']})
                summary_train = pd.DataFrame(summary_train, columns=[node_index.values + '_mean', node_index.values +
                                                                     '_sd', node_index.values + '_dxdt'])
                summary_test = pd.DataFrame(summary_test, columns=[node_index.values + '_mean', node_index.values +
                                                                   '_sd', node_index.values + '_dxdt'])
                summary_valid = pd.DataFrame(summary_valid, columns=[node_index.values + '_mean', node_index.values +
                                                                     '_sd', node_index.values + '_dxdt'])
                self.update(
                    {'summary_train': summary_train, 'summary_test': summary_test, 'summary_valid': summary_valid}
                )
            except Exception:
                pass

    def save(self):
        """save model parameters"""
        for file in glob.glob(str(self.substage_i) + "_best.*.csv"):
            os.remove(file)
        for key in self:
            self[key].to_csv("{}_best.{}.loss.{}.csv".format(self.substage_i, key, self.loss_min))
