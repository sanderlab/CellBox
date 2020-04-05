import os
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import pertbio
from pertbio.utils import TimeLogger
import time


def train_substage(model, dataset, sess, lr_val, l1lamda, iterations, n_iter_buffer, n_iter_patience, args):
    """
    Training function that does one stage of training. The stage training can be repeated and modified to give better
    training result.

    Args:
        model (CellBox): an CellBox instance
        dataset (dict): including "node_index", "pert_train", "pert_valid", "pert_test",
                        "train_data", "valid_data", "test_data"
        sess (tf.Session): current session, need reinitialization for every nT
        lr_val (float): learning rate (read in from config file)
        l1lamda (float): l1 regularization weight
        iterations (int): number of iterations
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
    for i in range(iterations):
        t0 = time.clock()

        if hasattr(args, 'batch_size') and hasattr(args, 'n_s1'):
            raise Exception("Ambiguous definition of batch size.")
        else:
            if hasattr(args, 'batchsize'):
                # by batchsize
                batch_index = np.random.choice(range(dataset['train_data'].shape[0]), args.batchsize, replace=False)
            elif hasattr(args, 'n_s1'):
                # by dropout_percent
                batch_index = np.random.choice(range(dataset['train_data'].shape[0]), args.n_s1, replace=False)
            else:
                raise Exception("Missing definition of batch size.")

        # Feeding data
        train_set = {
            model.x_gold: dataset['train_data'].iloc[batch_index, :],
            model.mu: dataset['pert_train'].iloc[batch_index, :],
            model.lr: lr_val,
            model.l1_lambda: l1lamda,
            model.idx: dataset['train_pos'][batch_index]
        }
        valid_set = {
            model.x_gold: dataset['valid_data'],
            model.mu: dataset['pert_valid'],
            model.l1_lambda: l1lamda,
            model.idx: dataset['valid_pos']
        }
        test_set = {
            model.x_gold: dataset['test_data'],
            model.mu: dataset['pert_test'],
            model.l1_lambda: l1lamda,
            model.idx: dataset['test_pos']
        }
        # Training:
        _, loss_train_i = sess.run(model.op_optimize, feed_dict=train_set)
        loss_valid_i = sess.run(model.loss, feed_dict=valid_set)
        loss_train_mse_i = sess.run(model.loss_mse, feed_dict=train_set)
        loss_valid_mse_i = sess.run(model.loss_mse, feed_dict=valid_set)
        loss_test_mse_i = sess.run(model.loss_mse, feed_dict=test_set)

        # Evaluation
        new_loss = best_params.avg_n_iters_loss(loss_valid_i)
        if args.export_verbose >= 3:
            print("Iteration: {}/{}\tnew_loss:{}\tbuffer_loss:{}\tbest:{}\tTolerance: {}/{}".format(
                i, iterations, loss_train_i, new_loss, best_params.loss_min,
                n_unchanged, n_iter_patience
            ))
        if new_loss < best_params.loss_min:
            n_unchanged = 0
            best_params.screenshot(sess, model, substage_i, args=args,
                                   node_index=args.dataset['node_index'], loss_min=new_loss)

        elif n_unchanged < n_iter_patience:
            n_unchanged += 1
        else:
            break
        append_record("record_eval.csv",
                      [i, loss_train_i, loss_valid_i, loss_train_mse_i, loss_valid_mse_i, loss_test_mse_i,
                       time.clock() - t0])

    best_params.save()
    args.logger.log("------------------ Substage {} finished!-------------------".format(substage_i))
    save_model(args.saver, sess, './' + args.ckpt_name)


def append_record(filename, contents):
    with open(filename, 'a') as f:
        for content in contents:
            f.write('{},'.format(content))
        f.write('\n')


def train_model(args):
    args.logger = TimeLogger(time_logger_step=1, hierachy=2)
    # Constructing model
    model = pertbio.model.factory(args)
    # Check if all variables in scope
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='initialization'):
        print(i)

    # Prepare for model training
    args.n_s1 = int(args.dataset['pert_train'].shape[0] * args.dropout_percent)

    # Launching session
    args.saver = tf.compat.v1.train.Saver()
    from tensorflow.core.protobuf import rewriter_config_pb2
    config = tf.compat.v1.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = off
    sess = tf.compat.v1.Session(config=config)

    sess.run(tf.compat.v1.global_variables_initializer())
    try:
        args.saver.restore(sess, './' + args.ckpt_name)
        print('Load existing model at {}...'.format(args.ckpt_name))
    except Exception:
        print('Create new model at {}...'.format(args.ckpt_name))

    # Training
    for substage_i, substage in enumerate(args.sub_stages):
        n_iter_buffer = substage['n_iter_buffer'] if 'n_iter_buffer' in substage else args.n_iter_buffer
        n_iter = substage['n_iter'] if 'n_iter' in substage else args.n_iter
        n_iter_patience = substage['n_iter_patience'] if 'n_iter_patience' in substage else args.n_iter_patience
        train_substage(model, args.dataset, sess, substage['lr_val'], substage['l1lamda'],
                       iterations=n_iter, n_iter_buffer=n_iter_buffer,
                       n_iter_patience=n_iter_patience, args=args)

    # Terminate session
    sess.close()
    tf.compat.v1.reset_default_graph()


def simu_model(config_path, working_index, mu):
    cfg = pertbio.config.Config(config_path)
    cfg.ckpt_path_full = os.path.join('./', cfg.ckpt_name)
    md5 = pertbio.utils.md5(str(vars(cfg)))

    experiment_path = 'results/{}_{}'.format(cfg.experiment_id, md5)
    working_index = cfg.model_prefix + "_" + str(working_index).zfill(3)

    tf.reset_default_graph()
    args = cfg

    args.sub_stages = args.stages[-1]['sub_stages']
    args.n_T = args.stages[-1]['nT']
    model = pertbio.model.factory(args)
    saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()

    file = experiment_path + '/' + working_index + '/' + cfg.ckpt_name
    try:
        saver.restore(sess, file)

        mu = np.float32(mu)
        if args.model == 'CellBox':
            convergence, xhat = sess.run(model.forward(mu))
        else:
            xhat = sess.run(model.forward(mu))
    except Exception:
        print('Model does not exist!')
        xhat = None

    return xhat


def save_model(saver, sess, path):
    # Save the variables to disk.
    tmp = saver.save(sess, path)
    print("Model saved in path: %s" % tmp)


class Screenshot(dict):

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
        try:
            self.export_verbose = args.export_verbose
            # 0: no output, 1: params only, 2: params + prediction, 3: output for each iteration
        except Exception:
            print("Undefined verbose. Using default: 2.")
            self.export_verbose = 2  # default verbose: 2

    def avg_n_iters_loss(self, new_loss):
        self.saved_losses = self.saved_losses + [new_loss]
        self.saved_losses = self.saved_losses[-self.n_iter_buffer:]
        return sum(self.saved_losses) / len(self.saved_losses)

    def screenshot(self, sess, model, substage_i, node_index, loss_min, args):

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
            test_set = {
                model.x_gold: args.dataset['test_data'],
                model.mu: args.dataset['pert_test'],
                model.idx: args.dataset['test_pos']
            }
            y_hat = sess.run(model.xhat, feed_dict=test_set)
            y_hat = pd.DataFrame(y_hat, columns=node_index[0])
            self.update({'y_hat': y_hat})

        if self.export_verbose > 2:
            try:
                summary_train = sess.run(model.convergence_metric, feed_dict={model.mu: args.dataset['pert_train']})
                summary_test = sess.run(model.convergence_metric, feed_dict={model.mu: args.dataset['pert_test']})
                summary_valid = sess.run(model.convergence_metric, feed_dict={model.mu: args.dataset['pert_valid']})
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
        for file in glob.glob(str(self.substage_i) + "_best.*.csv"):
            os.remove(file)
        for key in self:
            self[key].to_csv("{}_best.{}.loss.{}.csv".format(self.substage_i, key, self.loss_min))
