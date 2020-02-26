import tensorflow as tf
import numpy as np


def get_envelop(args):
    if args.envelop_form == 'tanh':
        return tf.tanh
    elif args.envelop_form == 'polynomial':
        k = args.polynomial_k
        assert k>1, "Hill coefficient has to be k>2."
        if k%2 == 1: # odd order polynomial equation
            return lambda x: x**k/(1+tf.abs(x)**k)
        else: # even order polynomial equation
            return lambda x: x**k/(1+x**k)*tf.sign(x)
    elif args.envelop_form == 'hill':
        k = args.polynomial_k
        assert k>1, "Hill coefficient has to be k>2."
        return lambda x: 2*(1-1/(1+tf.nn.relu(x+1)**k))-1
    else:
        raise Exception("Illegal envelop function. Choose from [tanh, polynomial/hill]")

def get_dXdt(args, envelop, params):

    if args.ode_degree == 1:
        weighted_sum = lambda x: tf.matmul(params['W'], x)
    elif args.ode_degree == 2:
        weighted_sum = lambda x: tf.matmul(params['W'], x) + tf.reshape(tf.reduce_sum(params['W'], axis=1), [args.n_x, 1]) * x
    else:
        raise Exception("Illegal ODE degree. Choose from [1,2].")

    if args.envelop == 0:
        # epsilon*phi(Sigma+u)-alpha*x
        return lambda x, t_mu, envelop: params['eps'] * envelop(weighted_sum(x) + t_mu) - params['alpha'] * x
    elif args.envelop == 1:
        # epsilon*[phi(Sigma)+u]-alpha*x
        return lambda x, t_mu, envelop: params['eps'] * (envelop(weighted_sum(x)) + t_mu) - params['alpha'] * x
    elif args.envelop == 2:
        # epsilon*phi(Sigma)+psi*u-alpha*x
        return lambda x, t_mu, envelop: params['eps'] * envelop(weighted_sum(x)) + params['psi']*t_mu - params['alpha'] * x
    else:
        raise Exception("Illegal envelop type. Choose from [0,1,2].")

def get_ode_solver(args):
    if args.ode_solver == 'heun':
        return heun_solver
    elif args.ode_solver == 'euler':
        return euler_solver
    elif args.ode_solver == 'rk4':
        return rk4_solver
    elif args.ode_solver == 'midpoint':
        return midpoint_solver
    else:
        raise Exception("Illegal ODE solver. Use [heun, euler, rk4, midpoint]")

def heun_solver(x, t_mu, dT, n_T, envelop, _dXdt, args):
    xs = [ tf.tile(x, [1, tf.shape(t_mu)[1]]) ]
    for i in range(n_T):
        dXdt_current = _dXdt(x, t_mu, envelop)
        dXdt_next = _dXdt(x + dT * dXdt_current, t_mu, envelop)
        x = x + dT * 0.5 * (dXdt_current + dXdt_next)
        xs.append(x)
    xs = tf.stack(xs, axis = 0)
    return xs

def euler_solver(x, t_mu, dT, n_T, envelop, _dXdt, args):
    xs = [x]
    for i in range(n_T):
        dXdt_current = _dXdt(x, t_mu, envelop)
        x = x + dT * dXdt_current
        xs.append(x)
    xs = tf.stack(xs, axis = 0)
    return xs

def midpoint_solver(x, t_mu, dT, n_T, envelop, _dXdt, args):
    xs = [x]
    for i in range(n_T):
        dXdt_current = _dXdt(x, t_mu, envelop)
        dXdt_midpoint = _dXdt(x + 0.5 * dT * dXdt_current, t_mu, envelop)
        x = x + dT * dXdt_midpoint
        xs.append(x)
    xs = tf.stack(xs, axis = 0)
    return xs

def rk4_solver(x, t_mu, dT, n_T, envelop, _dXdt, args):
    xs = [x]
    for i in range(n_T):
        k1 = _dXdt(x, t_mu, envelop)
        k2 = _dXdt(x + 0.5*dT*k1, t_mu, envelop)
        k3 = _dXdt(x + 0.5*dT*k2, t_mu, envelop)
        k4 = _dXdt(x + dT*k3, t_mu, envelop)
        x = x + dT * (1/6*k1+1/3*k2+1/3*k3+1/6*k4)
        xs.append(x)
    xs = tf.stack(xs, axis = 0)
    return xs
