"""
This module defines the ODE formulations, including the choices of ODE solvers,
degree of ODEs, and the envelope forms
"""

import tensorflow.compat.v1 as tf
from typing import Callable, Mapping, Any
tf.disable_v2_behavior()


def get_envelope(args) -> Callable[[tf.Tensor], tf.Tensor]:
    """Gets the envelope form based on the given argument.
    
    Returns:
        A function that takes in a tensor and returns a tensor with the same shape.
            This function should apply a specific transformation such as the Hill's
            equation.
    """
    if args.envelope_form == 'tanh':
        args.envelope_fn = tf.tanh
    elif args.envelope_form == 'polynomial':
        k = args.polynomial_k
        assert k > 1, "Hill coefficient has to be k>2."
        if k % 2 == 1:  # odd order polynomial equation
            args.envelope_fn = lambda x: x ** k / (1 + tf.abs(x) ** k)
        else:  # even order polynomial equation
            args.envelope_fn = lambda x: x**k/(1+x**k)*tf.sign(x)
    elif args.envelope_form == 'hill':
        k = args.polynomial_k
        assert k > 1, "Hill coefficient has to be k>=2."
        args.envelope_fn = lambda x: 2*(1-1/(1+tf.nn.relu(x+1)**k))-1
    elif args.envelope_form == 'linear':
        args.envelope_fn = lambda x: x
    elif args.envelope_form == 'clip linear':
        args.envelope_fn = lambda x: tf.clip_by_value(x, clip_value_min=-1, clip_value_max=1)
    else:
        raise Exception("Illegal envelope function. Choose from [tanh, polynomial/hill]")
    return args.envelope_fn


def get_dxdt(
    args, params: Mapping[str, tf.Tensor]) -> Callable[[tf.Tensor], tf.Tensor]:
    """Calculates the derivatives dx/dt in the ODEs.
    
    Returns:
        A function that takes in a tensor and returns a tensor with the same shape.
            This function should apply an envelope function with given params, i.e.,
            f(params, x).
    """
    if args.ode_degree == 1:
        def weighted_sum(x):
            return tf.matmul(params['W'], x)
    elif args.ode_degree == 2:
        def weighted_sum(x):
            return tf.matmul(params['W'], x) + tf.reshape(tf.reduce_sum(params['W'], axis=1), [args.n_x, 1]) * x
    else:
        raise Exception("Illegal ODE degree. Choose from [1,2].")

    if args.envelope == 0:
        # epsilon*phi(Sigma+u)-alpha*x
        return lambda x, t_mu: params['eps'] * args.envelope_fn(weighted_sum(x) + t_mu) - params['alpha'] * x
    if args.envelope == 1:
        # epsilon*[phi(Sigma)+u]-alpha*x
        return lambda x, t_mu: params['eps'] * (args.envelope_fn(weighted_sum(x)) + t_mu) - params['alpha'] * x
    if args.envelope == 2:
        # epsilon*phi(Sigma)+psi*u-alpha*x
        return lambda x, t_mu: params['eps'] * args.envelope_fn(weighted_sum(x)) + params['psi'] * t_mu - \
                               params['alpha'] * x
    raise Exception("Illegal envelope type. Choose from [0,1,2].")


#def get_ode_solver(args) -> Callable[Any, tf.Tensor]:
def get_ode_solver(args):
    """Gets the ODE solver based on the given argument."""
    if args.ode_solver == 'heun':
        return heun_solver
    if args.ode_solver == 'euler':
        return euler_solver
    if args.ode_solver == 'rk4':
        return rk4_solver
    if args.ode_solver == 'midpoint':
        return midpoint_solver
    raise Exception("Illegal ODE solver. Use [heun, euler, rk4, midpoint]")


def heun_solver(x, t_mu, dT, n_T, _dXdt, n_activity_nodes=None) -> tf.Tensor:
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


def euler_solver(x, t_mu, dT, n_T, _dXdt, n_activity_nodes=None) -> tf.Tensor:
    """Euler's method"""
    xs = []
    n_x = t_mu.shape[0]
    n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
    dxdt_mask = tf.pad(tf.ones((n_activity_nodes, 1)), [[0, n_x - n_activity_nodes], [0, 0]])
    for _ in range(n_T):
        dxdt_current = _dXdt(x, t_mu)
        x = x + dT * dxdt_current * dxdt_mask
        xs.append(x)
    xs = tf.stack(xs, axis=0)
    return xs


def midpoint_solver(x, t_mu, dT, n_T, _dXdt, n_activity_nodes=None) -> tf.Tensor:
    """Midpoint method"""
    xs = []
    n_x = t_mu.shape[0]
    n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
    dxdt_mask = tf.pad(tf.ones((n_activity_nodes, 1)), [[0, n_x - n_activity_nodes], [0, 0]])
    for _ in range(n_T):
        dxdt_current = _dXdt(x, t_mu)
        dxdt_midpoint = _dXdt(x + 0.5 * dT * dxdt_current, t_mu)
        x = x + dT * dxdt_midpoint * dxdt_mask
        xs.append(x)
    xs = tf.stack(xs, axis=0)
    return xs


def rk4_solver(x, t_mu, dT, n_T, _dXdt, n_activity_nodes=None) -> tf.Tensor:
    """Runge-Kutta method"""
    xs = []
    n_x = t_mu.shape[0]
    n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
    dxdt_mask = tf.pad(tf.ones((n_activity_nodes, 1)), [[0, n_x - n_activity_nodes], [0, 0]])
    for _ in range(n_T):
        k1 = _dXdt(x, t_mu)
        k2 = _dXdt(x + 0.5*dT*k1, t_mu)
        k3 = _dXdt(x + 0.5*dT*k2, t_mu)
        k4 = _dXdt(x + dT*k3, t_mu)
        x = x + dT * (1/6*k1+1/3*k2+1/3*k3+1/6*k4) * dxdt_mask
        xs.append(x)
    xs = tf.stack(xs, axis=0)
    return xs
