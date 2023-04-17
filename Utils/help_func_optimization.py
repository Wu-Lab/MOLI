import numpy as np
from scipy.optimize import minimize
import networkx as nx
from Utils.help_func import MOLI


def gradient(W_old, W_new, x, beta = 1):
    W_old = np.array(W_old)
    W_new = np.array(W_new)

    n = W_old.shape[0]
    IE, IN = indicator_mat(W_old, W_new)
    W = MOLI(W_old, x)
    W1 = MOLI(W_old, [1, 0])
    W2 = MOLI(W_old, [0, 1])
    W = W + 1e-50
    W1 = W1 + 1e-50
    W2 = W2 + 1e-50

    result = 0
    for i in range(n):
        for j in range(n):
            if IE[i, j] == 1:
                term1 = W1[i, j] / (x[0] * W1[i, j] + x[1] * W2[i, j])
                if np.isnan(np.log(W[i, j])):
                    print(i, j)
                result = result + beta * term1  # maybe W[i, j] == 0
            elif IN[i, j] == 1:
                term2 = (-1) * W1[i, j] / (1 - x[0] * W1[i, j] - x[1] * W2[i, j])
                if np.isnan(np.log(1 - W[i, j])):
                    print(i, j)
                result = result + term2  # maybe W[i, j] == 1
    return result


# IE: new edges in new net
# IN: no edges in new net
# I=IE+IN: missing edges

def indicator_mat(W_old, W_new):
    W_old = np.array(W_old)
    W_new = np.array(W_new)

    W_old_I = (W_old > 0) * 1
    W_new_I = (W_new > 0) * 1
    IE = np.multiply(1 - W_old_I, W_new_I)
    IN = np.multiply(1 - W_old_I, 1 - W_new_I)
    IN = IN - np.diag(np.diag(IN))
    return IE, IN


def likeihood(W_old, W_new, x, beta = 1):
    W_old = np.array(W_old)
    W_new = np.array(W_new)

    n = W_old.shape[0]
    IE, IN = indicator_mat(W_old, W_new)
    W = MOLI(W_old, x)
    W = np.multiply(W, IE + IN)
    W = W / W.sum()
    W = W + 1e-50
    # to make sure that np.log(W[i, j]) or np.log(1 - W[i, j]) won't go wrong

    result = 0
    for i in range(n):
        for j in range(n):
            if IE[i, j] == 1:
                if np.isnan(np.log(W[i, j])):
                    print(i, j)
                result = result + beta * np.log(W[i, j])  # maybe W[i, j] == 0
            elif IN[i, j] == 1:
                if np.isnan(np.log(1 - W[i, j])):
                    print(i, j)
                result = result + np.log(1 - W[i, j])  # maybe W[i, j] == 1
    return result


def scipy_constraint_likeihood(W_old, W_new,
                               d = 2,
                               eps = 1e-20,
                               x0 = None,
                               iteration = 1000,
                               beta = 1):

    if x0 is None:
        x0 = np.random.random(d)
        x0 = x0 / x0.sum()

    func = lambda x: (-1) * likeihood(W_old, W_new, x, beta = beta)
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})

    bnds = np.tile((0, None), d).reshape([-1, 2])

    # tol： Tolerance for termination.
    # maxiter : Maximum number of iterations to perform.
    # ftol: Precision goal for the value of f in the stopping criterion.
    res = minimize(fun = func,
                   x0 = x0,
                   method = 'SLSQP',
                   constraints = cons,
                   tol = eps,
                   bounds = bnds,
                   options = {'maxiter': iteration,
                              'ftol': eps})
    print('*********************************************************************')
    print('Optimum：', (-1) * res.fun)
    print('Optimizer：', res.x)
    print('Termination was successful：', res.success)
    print('Reasons for iteration termination：', res.message)
    print('Number of iterations：', res.nit)
    print('*********************************************************************')
    return res.x, (-1) * res.fun

