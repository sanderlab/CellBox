"""
This module defines the ODE formulations, including the choices of ODE solvers,
degree of ODEs, and the envelope forms
"""

import tensorflow as tf


def get_envelop(args):
    """get the envelope form based on the given argument"""
    if args.envelop_form == 'tanh':
        args.envelop_fn = tf.tanh
    elif args.envelop_form == 'polynomial':
        k = args.polynomial_k
        assert k > 1, "Hill coefficient has to be k>2."
        if k % 2 == 1:  # odd order polynomial equation
            args.envelop_fn = lambda x: x ** k / (1 + tf.abs(x) ** k)
        else:  # even order polynomial equation
            args.envelop_fn = lambda x: x**k/(1+x**k)*tf.sign(x)
    elif args.envelop_form == 'hill':
        k = args.polynomial_k
        assert k > 1, "Hill coefficient has to be k>=2."
        args.envelop_fn = lambda x: 2*(1-1/(1+tf.nn.relu(x+1)**k))-1
    elif args.envelop_form == 'linear':
        args.envelop_fn = lambda x: x
    elif args.envelop_form == 'clip linear':
        args.envelop_fn = lambda x: tf.clip_by_value(x, clip_value_min=-1, clip_value_max=1)
    else:
        raise Exception("Illegal envelop function. Choose from [tanh, polynomial/hill]")
    return args.envelop_fn


def get_dxdt(args, params):
    """calculate the derivatives dx/dt in the ODEs"""
    if args.ode_degree == 1:
        def weighted_sum(x):
            return tf.matmul(params['W'], x)

    elif args.ode_degree == 2:
        def weighted_sum(x):
            return tf.matmul(params['W'], x) + tf.reshape(tf.reduce_sum(params['W'], axis=1), [args.n_x, 1]) * x

    else:
        raise Exception("Illegal ODE degree. Choose from [1,2].")

    if args.envelop == 0:
        # epsilon*phi(Sigma+u)-alpha*x
        return lambda x, t_mu: params['eps'] * args.envelop_fn(weighted_sum(x) + t_mu) - params['alpha'] * x
    if args.envelop == 1:
        # epsilon*[phi(Sigma)+u]-alpha*x
        return lambda x, t_mu: params['eps'] * (args.envelop_fn(weighted_sum(x)) + t_mu) - params['alpha'] * x
    if args.envelop == 2:
        # epsilon*phi(Sigma)+psi*u-alpha*x
        return lambda x, t_mu: params['eps'] * args.envelop_fn(weighted_sum(x)) + params['psi'] * t_mu - \
                               params['alpha'] * x
    raise Exception("Illegal envelop type. Choose from [0,1,2].")


def get_ode_solver(args):
    """get the ODE solver based on the given argument"""
    if args.ode_solver == 'heun':
        return heun_solver
    if args.ode_solver == 'euler':
        return euler_solver
    if args.ode_solver == 'rk4':
        return rk4_solver
    if args.ode_solver == 'midpoint':
        return midpoint_solver
    raise Exception("Illegal ODE solver. Use [heun, euler, rk4, midpoint]")


def heun_solver(x, t_mu, dT, n_T, _dXdt, n_activity_nodes=None):
    """Heun's ODE solver"""
    xs = []
    n_x = t_mu.shape[0]
    n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
    dxdt_mask = tf.pad(tf.ones((n_activity_nodes, 1)), [[0, n_x - n_activity_nodes], [0, 0]])
    for _ in range(n_T):
        dxdt_current = _dXdt(x, t_mu)
        dxdt_next = _dXdt(x + dT * dxdt_current, t_mu)
        x = x + dT * 0.5 * (dxdt_current + dxdt_next) * dxdt_mask
        xs.append(x)
    xs = tf.stack(xs, axis=0)
    return xs


def euler_solver(x, t_mu, dT, n_T, _dXdt):
    """Euler's method"""
    xs = []
    for _ in range(n_T):
        dxdt_current = _dXdt(x, t_mu)
        x = x + dT * dxdt_current
        xs.append(x)
    xs = tf.stack(xs, axis=0)
    return xs


def midpoint_solver(x, t_mu, dT, n_T, _dXdt):
    """Midpoint method"""
    xs = []
    for _ in range(n_T):
        dxdt_current = _dXdt(x, t_mu)
        dxdt_midpoint = _dXdt(x + 0.5 * dT * dxdt_current, t_mu)
        x = x + dT * dxdt_midpoint
        xs.append(x)
    xs = tf.stack(xs, axis=0)
    return xs


def rk4_solver(x, t_mu, dT, n_T, _dXdt):
    """Runge-Kutta method"""
    xs = []
    for _ in range(n_T):
        k1 = _dXdt(x, t_mu)
        k2 = _dXdt(x + 0.5*dT*k1, t_mu)
        k3 = _dXdt(x + 0.5*dT*k2, t_mu)
        k4 = _dXdt(x + dT*k3, t_mu)
        x = x + dT * (1/6*k1+1/3*k2+1/3*k3+1/6*k4)
        xs.append(x)
    xs = tf.stack(xs, axis=0)
    return xs
