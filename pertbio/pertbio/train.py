import os
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
from pertbio.model import CellBox
from pertbio.utils import time_logger
import time

def train_substage(model, dataset, sess, lr_val, l1lamda, iterations, n_iter_buffer, args):
    """
    Training function that does one stage of training. The stage training can be repeated and modified to give better training result.

    Args:
        model (CellBox): an CellBox instance
        dataset (dict): including "node_index", "pert_train", "pert_valid", "pert_test",
                        "train_data", "valid_data", "test_data"
        sess (tf.Session): current session, need reinitialization for every nT
        lr_val (float): learning rate (read in from config file)
        l1lamda (float): l1 regularization weight
        iterations (int): number of iterations
        n_iter_buffer (int): training tolerance
    """

    stages = glob.glob("*best*.csv")
    try:
        substage_i = 1 + max([int(stage[0]) for stage in stages])
    except:
        substage_i = 1

    loss_min = args.loss_min
    best_params = Screenshot()
    best_params.set_verbose(args)

    n_unchanged = 0
    for i in range(iterations):
        t0 = time.clock()
        drop_out_index = np.random.choice(range(dataset['train_data'].shape[0]), args.n_s1, replace = False)

        # Feeding data
        train_set = {
            model.x_gold: dataset['train_data'].iloc[drop_out_index,:],
            model.mu: dataset['pert_train'].iloc[drop_out_index,:],
            model.lr: lr_val,
            model.l1_lambda: l1lamda
        }
        valid_set = {
            model.x_gold: dataset['valid_data'],
            model.mu: dataset['pert_valid'],
            model.l1_lambda: l1lamda
        }
        test_set = {
            model.x_gold: dataset['test_data'],
            model.mu: dataset['pert_test']
        }

        # Training:
        _, loss_train_i = sess.run(model.op_optimize, feed_dict=train_set)
        loss_valid_i = sess.run(model.loss, feed_dict= valid_set)
        loss_train_mse_i = sess.run(model.loss_mse, feed_dict= train_set)
        loss_valid_mse_i = sess.run(model.loss_mse, feed_dict= valid_set)
        loss_test_mse_i = sess.run(model.loss_mse, feed_dict= test_set)

        # Evaluation
        if loss_valid_i < loss_min:
            loss_min = loss_valid_i
            n_unchanged = 0
            best_params.screenshot(sess, model, substage_i, args = args,
                             node_index = args.dataset['node_index'], loss_min = loss_min)

        elif n_unchanged < n_iter_buffer:
            n_unchanged+=1
        else:
            break
        append_record("record_eval.csv", [i, loss_train_i, loss_valid_i, loss_train_mse_i, loss_valid_mse_i, loss_test_mse_i, time.clock() - t0])

    best_params.save()
    args.logger.log("------------------ Substage {} finished!-------------------".format(substage_i))
    save_model(args.saver, model, sess, './'+args.ckpt_name)

def append_record(filename, contents):
    with open(filename, 'a') as f:
        for content in contents:
            f.write('{},'.format(content))
        f.write('\n')


def train_model(args):
    args.logger = time_logger(time_logger_step = 1, hierachy = 2)
    ### Constructing model
    cellbox = CellBox(args)
    # DEBUGGING: See all variables in scope
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='initialization'):
        print(i)

    ### Prepare for model training
    args.n_s1 = int(args.dataset['pert_train'].shape[0]*args.dropout_percent)

    ### Launching session
    opt_op = cellbox.op_optimize
    args.saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    try:
        saver.restore(sess, args.ckpt_name)
        print('Load existing model at {}...'.format(args.ckpt_name))
    except:
        print('Create new model at {}...'.format(args.ckpt_name))

    ### Training
    for substage_i, substage in enumerate(args.sub_stages):
        try:
            n_iter_buffer = substage['n_iter_buffer']
        except:
            n_iter_buffer = args.n_iter_buffer
        try:
            n_iter = substage['n_iter']
        except:
            n_iter = args.iterations
        train_substage(cellbox, args.dataset, sess, substage['lr_val'], substage['l1lamda'],
                    iterations = n_iter, n_iter_buffer = n_iter_buffer, args = args)

    ### Terminate session
    sess.close()
    tf.reset_default_graph()

def save_model(saver, model, sess, path):
    # Save the variables to disk.
    tmp = saver.save(sess, path)
    print("Model saved in path: %s" % tmp)

class Screenshot(dict):

    def set_verbose(self, args):
        try:
            self.verbose = args.verbose # 0: no output, 1: params only, 2: params + prediction
        else:
            self.verbose = 2 # default verbose: 2

    def screenshot(self, sess, model, substage_i, node_index, loss_min, args):

        self.substage_i = substage_i
        self.loss_min = loss_min
        # Save the variables to disk.
        if self.verbose > 0:
            W_screenshot, alpha_screenshot, eps_screenshot = sess.run(model.get_params())
            w_values = pd.DataFrame(W_screenshot, columns=node_index[0], index=node_index[0])
            alpha_values = pd.DataFrame(alpha_screenshot, index=node_index[0])
            eps_values = pd.DataFrame(eps_screenshot, index=node_index[0])
            self.update({'W': w_values, 'alpha': alpha_values, 'eps_values': eps_values})

        if self.verbose > 1:
            y_hat = sess.run(model.xhat, feed_dict = {model.mu: args.dataset['pert_test']})
            y_hat = pd.DataFrame(y_hat, columns=node_index[0])
            self.update({'y_hat': y_hat})
            # convergence test - last model
            summary_train = sess.run(model.convergence_metric, feed_dict = {model.mu: args.dataset['pert_train']})
            summary_test = sess.run(model.convergence_metric, feed_dict = {model.mu: args.dataset['pert_test']})
            summary_valid = sess.run(model.convergence_metric, feed_dict = {model.mu: args.dataset['pert_valid']})
            summary_train = pd.DataFrame(summary_train, columns=np.vstack([node_index.values+'_mean', node_index.values+'_sd', node_index.values+'_dxdt']))
            summary_test = pd.DataFrame(summary_test, columns=np.vstack([node_index.values+'_mean', node_index.values+'_sd', node_index.values+'_dxdt']))
            summary_valid = pd.DataFrame(summary_valid, columns=np.vstack([node_index.values+'_mean', node_index.values+'_sd', node_index.values+'_dxdt']))
            self.update({'summary_train': summary_train, 'summary_valid': summary_valid, 'summary_test': summary_test})

    def save(self):
        for file in glob.glob(str(self.substage_i) + "_best.*.csv"):
            os.remove(file)
        for key in self:
            self[key].to_csv("{}_best.{}.loss.{}.csv".format(self.substage_i, key, self.loss_min))
