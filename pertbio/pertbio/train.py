import os
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import pertbio
from pertbio.utils import time_logger
import time

def train_substage(model, dataset, sess, lr_val, l1lamda, iterations, n_iter_buffer, n_iter_patience, args):
    """
    Training function that does one stage of training. The stage training can be repeated and modified to give better training result.

    Args:
        model (CellBox): an CellBox instance
        dataset (dict): including "node_index", "pert_train", "pert_valid", "pert_test",
                        "train_data", "valid_data", "test_data"
        sess (tf.Session): current session, need reinitialization for every nT
        lr_val (float): learning rate (read in from config file)
        l1lamda (float): l1 regularization weight
        n_iter (int): number of iterations
        n_iter_buffer (int): training loss window
        n_iter_patience (int): training loss tolerance
    """

    stages = glob.glob("*best*.csv")
    try:
        substage_i = 1 + max([int(stage[0]) for stage in stages])
    except:
        substage_i = 1

    best_params = Screenshot(args, n_iter_buffer)

    n_unchanged = 0
    for i in range(iterations):
        t0 = time.clock()

        # batch_index = np.random.choice(range(dataset['train_data'].shape[0]), args.n_s1, replace = False)
        # by args.dropout_percent
        batch_index = np.random.choice(range(dataset['train_data'].shape[0]), args.batchsize, replace = False)
        # by batchsize

        # Feeding data
        train_set = {
            model.x_gold: dataset['train_data'].iloc[batch_index,:],
            model.mu: dataset['pert_train'].iloc[batch_index,:],
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
        new_loss = best_params.avg_n_iters_loss(loss_valid_i)
        if args.export_verbose>=3:
            print("Iteration: {}/{}\ttrain:{}\tvalid:{}\tbest:{}\tTolerance: {}/{}".format(
                        i, iterations, loss_train_i, new_loss, best_params.loss_min,
                        n_unchanged, n_iter_patience
            ))
        if new_loss < best_params.loss_min:
            n_unchanged = 0
            best_params.screenshot(sess, model, substage_i, args = args,
                             node_index = args.dataset['node_index'], loss_min = new_loss)

        elif n_unchanged < n_iter_patience:
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
    model = pertbio.model.factory(args)
    # DEBUGGING: See all variables in scope
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='initialization'):
        print(i)

    ### Prepare for model training
    args.n_s1 = int(args.dataset['pert_train'].shape[0]*args.dropout_percent)

    ### Launching session
    opt_op = model.op_optimize
    args.saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    try:
        args.saver.restore(sess, './'+args.ckpt_name)
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
            n_iter = args.n_iter
        try:
            n_iter_patience = substage['n_iter_patience']
        except:
            n_iter_patience = args.n_iter_patience
        train_substage(model, args.dataset, sess, substage['lr_val'], substage['l1lamda'],
                    iterations = n_iter, n_iter_buffer = n_iter_buffer,
                    n_iter_patience = n_iter_patience, args = args)

    ### Terminate session
    sess.close()
    tf.compat.v1.reset_default_graph()

def save_model(saver, model, sess, path):
    # Save the variables to disk.
    tmp = saver.save(sess, path)
    print("Model saved in path: %s" % tmp)

class Screenshot(dict):

    def __init__(self, args, n_iter_buffer):
        # initialize loss_min
        self.loss_min = 1000
        # initialize tuning_metric
        self.saved_losses = [self.loss_min]
        self.n_iter_buffer = n_iter_buffer
        # initialize verbose
        self.summary = {}
        self.summary = {}
        try:
            self.export_verbose = args.export_verbose
            # 0: no output, 1: params only, 2: params + prediction, 3: output for each iteration
        except:
            print("Undefined verbose. Using default: 2.")
            self.export_verbose = 2 # default verbose: 2

    def avg_n_iters_loss(self, new_loss):
        self.saved_losses = self.saved_losses + [new_loss]
        self.saved_losses = self.saved_losses[-self.n_iter_buffer:]
        return sum(self.saved_losses)/len(self.saved_losses)


    def screenshot(self, sess, model, substage_i, node_index, loss_min, args):

        self.substage_i = substage_i
        self.loss_min = loss_min
        # Save the variables to disk.
        if self.export_verbose > 0:
            params = sess.run(model.params)
            for item in params:
                try:
                    params[item] = pd.DataFrame(params[item], index=node_index[0])
                except:
                    params[item] = pd.DataFrame(params[item])
            self.update(params)

        if self.export_verbose > 1 or self.export_verbose == -1: #no params but y_hat
            y_hat = sess.run(model.xhat, feed_dict = {model.mu: args.dataset['pert_test']})
            y_hat = pd.DataFrame(y_hat, columns=node_index[0])
            self.update({'y_hat': y_hat})

        if self.export_verbose > 2:
            try:
                summary_train = sess.run(model.convergence_metric, feed_dict = {model.mu: args.dataset['pert_train']})
                summary_test = sess.run(model.convergence_metric, feed_dict = {model.mu: args.dataset['pert_test']})
                summary_valid = sess.run(model.convergence_metric, feed_dict = {model.mu: args.dataset['pert_valid']})
                summary_train = pd.DataFrame(summary_train, columns=np.vstack([node_index.values+'_mean', node_index.values+'_sd', node_index.values+'_dxdt']))
                summary_test = pd.DataFrame(summary_test, columns=np.vstack([node_index.values+'_mean', node_index.values+'_sd', node_index.values+'_dxdt']))
                summary_valid = pd.DataFrame(summary_valid, columns=np.vstack([node_index.values+'_mean', node_index.values+'_sd', node_index.values+'_dxdt']))
                self.update({'summary_train': summary_train, 'summary_test': summary_train, 'summary_valid':summary_valid})
            except:
                pass

    def save(self):
        for file in glob.glob(str(self.substage_i) + "_best.*.csv"):
            os.remove(file)
        for key in self:
            self[key].to_csv("{}_best.{}.loss.{}.csv".format(self.substage_i, key, self.loss_min))
