import numpy as np
import torch
import torch.nn as nn
import cellbox.kernel
from cellbox.utils import loss, optimize


def factory(args):
    """define model type based on configuration input"""
    #if args.model == 'CellBox':
    #    return CellBox(args).build()
    # Deprecated for now, use scikit-learn instead
    # TODO: update the co-expression models
    # if args.model == 'CoExp':
    #     return CoExp(args).build()
    # if args.model == 'CoExp_nonlinear':
    #     return CoExpNonlinear(args).build()
    if args.model == 'LinReg':
        return LinReg(args)
    #if args.model == 'NN':
    #    return NN(args).build()
    # TODO: baysian model
    # if args.model == 'Bayesian':
    #     return BN(args).build()


class PertBio(nn.Module):
    """define abstract perturbation model"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_x = args.n_x
        self.pert_in, self.expr_out = args.pert_in, args.expr_out
        self.iter_train, self.iter_monitor, self.iter_eval = args.iter_train, args.iter_monitor, args.iter_eval
        #self.train_x, self.train_y = self.iter_train.get_next()
        #self.monitor_x, self.monitor_y = self.iter_monitor.get_next()
        #self.eval_x, self.eval_y = self.iter_eval.get_next()
        self.l1_lambda, self.l2_lambda = self.args.l1_lambda_placeholder, self.args.l2_lambda_placeholder
        self.lr = self.args.lr

    def get_ops(self):
        """get operators for tensorflow"""
        # Do we need this at all for Pytorch?
        pass
        #if self.args.weight_loss == 'expr':
        #    self.train_loss, self.train_mse_loss = loss(self.train_y, self.train_yhat, self.params['W'],
        #                                                self.l1_lambda, self.l2_lambda, weight=self.train_y)
        #    self.monitor_loss, self.monitor_mse_loss = loss(self.monitor_y, self.monitor_yhat, self.params['W'],
        #                                                    self.l1_lambda, self.l2_lambda, weight=self.monitor_y)
        #    self.eval_loss, self.eval_mse_loss = loss(self.eval_y, self.eval_yhat, self.params['W'],
        #                                              self.l1_lambda, self.l2_lambda, weight=self.eval_y)
        #elif self.args.weight_loss == 'None':
        #    self.train_loss, self.train_mse_loss = loss(self.train_y, self.train_yhat, self.params['W'],
        #                                                self.l1_lambda, self.l2_lambda)
        #    self.monitor_loss, self.monitor_mse_loss = loss(self.monitor_y, self.monitor_yhat, self.params['W'],
        #                                                    self.l1_lambda, self.l2_lambda)
        #    self.eval_loss, self.eval_mse_loss = loss(self.eval_y, self.eval_yhat, self.params['W'],
        #                                              self.l1_lambda, self.l2_lambda)
        #self.op_optimize = optimize(self.train_loss, self.lr)

    def get_variables(self):
        """get model parameters (overwritten by model configuration)"""
        raise NotImplementedError

    def forward(self, x, mu):
        """forward propagation (overwritten by model configuration)"""
        raise NotImplementedError

    #def build(self):
    #    """build model"""
    #    # Do we need this at all for Pytorch?
    #    self.params = {}
    #    self.get_variables()
    #    self.train_yhat = self.forward(self.train_y0, self.train_x)
    #    self.monitor_yhat = self.forward(self.monitor_y0, self.monitor_x)
    #    self.eval_yhat = self.forward(self.eval_y0, self.train_x)
    #    self.get_ops()
    #    return self
    

class LinReg(PertBio):
    """linear regression model"""
    def __init__(self):
        super().__init__()
        self.get_variables()

    def get_variables(self):
        self.W = nn.Linear(
            in_features=self.n_x,
            out_features=self.n_x,
            bias=True
        )

    def forward(self, x, mu):
        return self.W(mu)
