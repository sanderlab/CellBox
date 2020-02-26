import tensorflow as tf
import numpy as np
import pertbio.kernel
from pertbio.utils import loss, optimize
# import tensorflow_probability as tfp

def factory(args):
    if args.model == 'CellBox':
        return CellBox(args).build()
    elif args.model == 'CoExp':
        return CoExp(args).build()
    elif args.model == 'CoExp_nonlinear':
        return CoExp_nonlinear(args).build()
    elif args.model == 'LinReg':
        return LinReg(args).build()
    elif args.model == 'NN':
        return NN(args).build()
    elif args.model == 'Bayesian':
        return BN(args).build()
    else:
        raise Exception("Illegal model name. Choose from [{}]".format(
                        'CellBox, CoExp, LinReg, NN, CoExp_nonlinear, Bayesian'
                    ))

class PertBio:
    def __init__(self, args):
        self.args = args
        self.n_x = args.n_x
        self.mu = tf.compat.v1.placeholder(tf.float32, [None, self.n_x])
        self.x_gold = tf.compat.v1.placeholder(tf.float32, [None, self.n_x])
        self.idx = tf.compat.v1.placeholder(tf.int32, [None])

    def get_ops(self):
        self.l1_lambda = tf.compat.v1.placeholder(tf.float32)
        self.loss, self.loss_mse = loss(self.x_gold, self.xhat,
                                        self.l1_lambda, self.params['W'])
        self.lr = tf.compat.v1.placeholder(tf.float32)
        self.op_optimize = optimize(self.loss, self.lr)

    def build(self):
        self.params = {}
        self.get_variables()
        self.xhat = self.forward(self.mu)
        self.get_ops()
        return self


class CoExp(PertBio):
    def __init__(self, args):
        super(CoExp, self).__init__(args)
        self.mu_full = tf.constant(self.args.dataset['pert_full'], dtype=tf.float32)
        self.idx_full = tf.map_fn(fn=self.get_idx_pair, elems=self.mu_full, dtype=tf.int32)

    @tf.function
    def get_idx_pair(self, mu_tensor):
        idx = tf.compat.v2.where(tf.not_equal(mu_tensor, 0))[:,0]
        idx = tf.stack([idx[0], idx[-1]])
             # if idx = [i, j], use params[j][i]
             # if idx = [i], use params[i][i]
        return tf.compat.v2.cast(idx, dtype=tf.int32)

    def get_variables(self):
        with tf.compat.v1.variable_scope("initialization", reuse=True):
            Ws = tf.Variable(np.zeros([self.args.n_x, self.args.n_x
                            , self.n_x, self.n_x]), dtype=tf.float32)
            bs = tf.Variable(np.zeros([self.args.n_x, self.args.n_x
                            , self.n_x, 1]), dtype=tf.float32)
        self.params.update({'Ws': Ws, 'bs': bs})

    def forward(self, training):
        # during training, use mu_full, while during testing use mu
        idx = tf.map_fn(fn=self.get_idx_pair, elems=self.mu, dtype=tf.int32) # if not training else self.idx_full

        Ws = tf.gather_nd(self.params['Ws'], self.idx_full) # full_mu_size x [Params,]
        bs = tf.gather_nd(self.params['bs'], self.idx_full)
        xhats = tf.tensordot(Ws, tf.transpose(self.mu), axes=1) + bs # mu_idx_size x [xhat] x batch_size
        xhats_transposed = tf.transpose(xhats, perm=[2,0,1] )# batch_size x mu_idx_size  x [xhat]

        # mask the models for prediction
        import pdb; pdb.set_trace()
        idx_full_tile = tf.tile(tf.expand_dims(self.mu_full, axis=0), [tf.shape(self.mu)[0], 1, 1])
        idx_tile = tf.tile(tf.expand_dims(idx, axis=1), [1, self.idx_full.shape[0], 1])
        idx_equal = tf.equal(idx_full_tile, idx_tile)
        mask = tf.compat.v2.where(tf.math.logical_and(idx_equal[:,:,0], idx_equal[:,:,1]))
        xhats_masked = tf.gather_nd(xhats_transposed, mask)

        # x[5]: mask is wrong
        self.tmp_list = [xhats, xhats_transposed, idx_full_tile, idx_tile, idx_equal,  mask, xhats_masked]
        tmp = sess.run(model.tmp_list, feed_dict=valid_set)
        mask = tf.compat.v2.where(tf.math.logical_and(idx_equal[:,:,0], idx_equal[:,:,1]))

        return xhats_masked

    def get_ops(self):
        self.l1_lambda = tf.compat.v1.placeholder(tf.float32)
        self.loss_mse = tf.reduce_mean(tf.square((self.x_gold - self.xhat)))
        self.loss = tf.reduce_mean(tf.square((self.x_gold - self.xhat_training)))
        self.lr = tf.compat.v1.placeholder(tf.float32)
        self.op_optimize = optimize(self.loss, self.lr, var_list = None)

    def build(self):
        self.params = {}
        self.get_variables()
        self.xhat_training = self.forward(training=True)
        self.xhat = self.forward(training=False)
        self.get_ops()
        return self


class CoExp_nonlinear(CoExp):

    def get_variables(self):
        with tf.compat.v1.variable_scope("initialization", reuse=True):
            Ws = tf.Variable(np.zeros([self.args.n_x, self.args.n_x
                            , self.n_x, self.n_x]), dtype=tf.float32)
            bs = tf.Variable(np.zeros([self.args.n_x, self.args.n_x
                            , self.n_x, 1]), dtype=tf.float32)
            W = tf.Variable(np.zeros([self.n_x, 1]), dtype=tf.float32)
            b = tf.Variable(np.zeros([self.n_x, 1]), dtype=tf.float32)
        self.params.update({'Ws': Ws, 'bs': bs, 'W': W, 'b': b})

    def forward(self, mu):
        # during training, use mu_full, while during testing use mu
        idx = tf.map_fn(fn=self.get_idx_pair, elems=mu, dtype=tf.int32)
        # mask the models for prediction
        Ws = tf.gather_nd(self.params['Ws'], idx) # batch_size x [Params,]
        bs = tf.gather_nd(self.params['bs'], idx)
        hidden = tf.tensordot(Ws, tf.transpose(x_gold), axes=1) + bs # batch_size x [Params,] x batch_size
        hidden_transposed = tf.transpose(hidden, perm=[0,2,1])
        hidden_masked = tf.gather_nd(hidden_transposed, tf.compat.v2.where(tf.eye(tf.shape(mu)[0])))
        xhat = tf.matmul(tf.tanh(hidden_masked), self.params['W']) + tf.reshape(self.params['b'], [1, -1])
        return xhat

class LinReg(PertBio):
    def get_variables(self):
        with tf.compat.v1.variable_scope("initialization", reuse=True):
            self.params.update({
                'W': tf.Variable(np.random.normal(0.01, size=(self.n_x, self.n_x)), name="W", dtype=tf.float32),
                'b' : tf.Variable(np.random.normal(0.01, size=(self.n_x, 1)), name="b", dtype=tf.float32)
            })

    def forward(self, mu):
        xhat = tf.matmul(mu, self.params['W']) + tf.reshape(self.params['b'], [1, -1])
        return xhat

class NN(LinReg):
    def get_variables(self):
        with tf.compat.v1.variable_scope("initialization", reuse=True):
            self.params.update({
                'W_h': tf.Variable(np.random.normal(0.01, size=(self.n_x, self.args.n_hidden)), name="Wh", dtype=tf.float32),
                'b_h' : tf.Variable(np.random.normal(0.01, size=(self.args.n_hidden, 1)), name="bh", dtype=tf.float32),
                'W': tf.Variable(np.random.normal(0.01, size=(self.args.n_hidden, self.n_x)), name="Wo", dtype=tf.float32),
                'b' : tf.Variable(np.random.normal(0.01, size=(self.n_x, 1)), name="bo", dtype=tf.float32)
            })

    def forward(self, mu):
        hidden = tf.tanh(tf.matmul(mu, self.params['W_h']) + tf.reshape(self.params['b_h'], [1, -1]))
        xhat = tf.matmul(hidden, self.params['W']) + tf.reshape(self.params['b'], [1, -1])
        return xhat

class CellBox(PertBio):
    def build(self):
        self.params = {}
        self.get_variables()
        self.x_0 = tf.constant(np.zeros((self.n_x, 1)), name="x_init", dtype=tf.float32)
        self.envelop = pertbio.kernel.get_envelop(self.args)
        self.ode_solver = pertbio.kernel.get_ode_solver(self.args)
        self._dXdt = pertbio.kernel.get_dXdt(self.args, self.envelop, self.params)
        self.convergence_metric , self.xhat = self.forward(self.mu)
        self.get_ops()

    def _simu(self, t_mu):
        t_mu = tf.reshape(t_mu, [self.n_x, 1])
        xs = self.ode_solver(self.x_0, t_mu, self.args.dT, self.args.n_T,
                            self.envelop, self._dXdt, self.args)

        tail_iters = self.args.n_iter_tail
        xs = tf.reshape(xs, [-1, self.n_x])[-tail_iters:]
        mean, sd = tf.nn.moments(xs, axes = 0)
        dxdt = tf.reshape(
                        self._dXdt(tf.reshape(xs[-1], [-1,1]), t_mu, self.envelop)
                      , [-1])
        return tf.reshape(xs[-1], [self.n_x]), tf.concat([mean, sd, dxdt], axis = 0)

    def forward(self, mu):
        xhat, convergence_metric = tf.map_fn(fn = (lambda mu_i: self._simu(mu_i[0])),
                                       elems = (mu, mu), dtype=(tf.float32, tf.float32))
        return convergence_metric, xhat

    def get_variables(self):
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
        n_x, n_protein_nodes, n_activity_nodes = self.n_x, self.args.n_protein_nodes, self.args.n_activity_nodes
        with tf.compat.v1.variable_scope("initialization", reuse=True):
            # TODO: check to see if it make sense to change tf.Variables to tf.get_variables
            '''Enforce constraints  (i: recipient)
               no self regulation wii=0
               ingoing wij for drug nodes (88th to 99th) = 0 [n_activity_nodes 87: ]
                                w [87:99,_] = 0
               outgoing wij for phenotypic nodes (83th to 87th) [n_protein_nodes 82 : n_activity_nodes 87]
                                w [_, 82:87] = 0
               ingoing wij for phenotypic nodes from drug ndoes (direct) [n_protein_nodes 82 : n_activity_nodes 87]
                                w [82:87, 87:99] = 0
            '''
            W = tf.Variable(np.random.normal(0.01, size=(n_x, n_x)), name="W", dtype=tf.float32)
            W_mask = (1.0 - np.diag(np.ones([n_x])))
            W_mask[n_activity_nodes:, :] = np.zeros([n_x-n_activity_nodes, n_x])
            W_mask[:, n_protein_nodes:n_activity_nodes] = np.zeros([n_x, n_activity_nodes-n_protein_nodes])
            W_mask[n_protein_nodes:n_activity_nodes, n_activity_nodes:] = np.zeros([n_activity_nodes-n_protein_nodes,
    										n_x-n_activity_nodes])
            self.params['W'] = W_mask * W

            eps = tf.Variable(np.ones((n_x, 1)), name="eps", dtype=tf.float32)
            alpha = tf.Variable(np.ones((n_x, 1)), name="alpha", dtype=tf.float32)
            self.params['alpha'] = tf.nn.softplus(alpha)
            self.params['eps'] = tf.nn.softplus(eps)

            if self.args.envelop == 2:
                psi = tf.Variable(np.ones((n_x, 1)), name="psi", dtype=tf.float32)
                self.params['psi'] = tf.nn.softplus(psi)
