import tensorflow as tf
import numpy as np
import pertbio.kernel
from pertbio.utils import loss, optimize

class CellBox:
    def __init__(self, args):
        super(CellBox, self).__init__()
        """
        CellBox construction

        Args:
            args: Config() instance

        Returns:
            dXdt (float): time derivative of input x: dxdt(t)

        """
        # Initialization
        self.args = args
        self.n_x = args.n_x
        self.x_0 = tf.constant(np.zeros((self.n_x, 1)), name="x_init", dtype=tf.float32)
        self.W, self.alpha, self.eps, self.psi = get_variables(
                    self.n_x, args.n_protein_nodes, args.n_activity_nodes)
        self.mu = tf.placeholder(tf.float32, [None, self.n_x])
        self.x_gold = tf.placeholder(tf.float32, [None, self.n_x])
        # Feed forward
        self.envelop = pertbio.kernel.get_envelop(args)
        self.envelop.W, self.envelop.alpha, self.envelop.eps, self.envelop.psi = self.get_params()

        self.ode_solver = pertbio.kernel.get_ode_solver(args)
        self._dXdt = pertbio.kernel.get_dXdt(args, self.envelop)
        self.convergence_metric , self.xhat = self.forward(self.mu)
        # Loss and training ops
        self.l1_lambda = tf.placeholder(tf.float32)
        self.loss, self.loss_mse = loss(self.x_gold, self.xhat,
                                        self.l1_lambda, self.W)
        self.lr = tf.placeholder(tf.float32)
        self.op_optimize = optimize(self.loss, self.lr, optimizer=tf.train.AdamOptimizer)

    def get_params(self):
        return self.W, self.alpha, self.eps, self.psi


    def _simu(self, t_mu):
        t_mu = tf.reshape(t_mu, [self.n_x, 1])
        xs = self.ode_solver(self.x_0, t_mu, self.args.dT, self.args.n_T,
                            self.envelop, self._dXdt, self.args)

        tail_iters = self.args.n_iter_tail
        n_last_iters = int(self.args.n_T - tail_iters)  # the last 20% iters were used for convergence test
        xs = tf.reshape(xs, [-1, self.n_x])[-n_last_iters:]
        mean, sd = tf.nn.moments(xs, axes = 0)
        dxdt = tf.reshape(
                        self._dXdt(tf.reshape(xs[-1], [-1,1]), t_mu, self.envelop)
                      , [-1])
        return tf.reshape(xs[-1], [self.n_x]), tf.concat([mean, sd, dxdt], axis = 0)

    def forward(self, mu):
        xhat, convergence_metric = tf.map_fn(fn = (lambda mu_i: self._simu(mu_i[0])),
                                       elems = (mu, mu), dtype=(tf.float32, tf.float32))
        return convergence_metric, xhat


def get_variables(n_x, n_protein_nodes, n_activity_nodes):
    '''
    Initialize parameters in the Hopfield equation
    Args:
        self.n_x (int): number of all nodes
        n_protein_nodes (int): number of protein (antibodies measured) nodes
        n_activity_nodes (int): number of non-drug nodes (protein + phenotypic nodes)

    Returns:
        W (tf.Variable): interaction matrix with constraints enforced, , shape: [n_x, n_x]
        alpha (tf.Variable): alpha, shape: [n_x, 1]
        eps (tf.Variable): eps, shape: [n_x, 1]
    '''

    with tf.variable_scope("initialization", reuse=True):
        # TODO: check to see if it make sense to change tf.Variables to tf.get_variables
        W = tf.Variable(np.random.normal(0.01, size=(n_x, n_x)), name="W", dtype=tf.float32)
        eps = tf.Variable(np.ones((n_x, 1)), name="eps", dtype=tf.float32)
        alpha = tf.Variable(np.ones((n_x, 1)), name="alpha", dtype=tf.float32)
        psi = tf.Variable(np.ones((n_x, 1)), name="psi", dtype=tf.float32)

        '''Enforce constraints  (i: recipient)
           no self regulation wii=0
           ingoing wij for drug nodes (88th to 99th) = 0 [n_activity_nodes 87: ]
                            w [87:99,_] = 0
           outgoing wij for phenotypic nodes (83th to 87th) [n_protein_nodes 82 : n_activity_nodes 87]
                            w [_, 82:87] = 0
           ingoing wij for phenotypic nodes from drug ndoes (direct) [n_protein_nodes 82 : n_activity_nodes 87]
                            w [82:87, 87:99] = 0
        '''

        W_mask = (1.0 - np.diag(np.ones([n_x])))
        W_mask[n_activity_nodes:, :] = np.zeros([n_x-n_activity_nodes, n_x])
        W_mask[:, n_protein_nodes:n_activity_nodes] = np.zeros([n_x, n_activity_nodes-n_protein_nodes])
        W_mask[n_protein_nodes:n_activity_nodes, n_activity_nodes:] = np.zeros([n_activity_nodes-n_protein_nodes,
										n_x-n_activity_nodes])
        W = W_mask * W

        alpha = tf.nn.softplus(alpha)
        eps = tf.nn.softplus(eps)
        psi = tf.nn.softplus(psi)
    return W, alpha, eps
