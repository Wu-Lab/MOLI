import linkpred as lp
import numpy as np
from numpy.linalg import *
from scipy.linalg import null_space


def abbr2fullname(abbr):
    namedict = {'katz': 'Katz',
                'RPR': 'RootedPageRank',
                'CN': 'CommonNeighbours',
                'JS': 'Jaccard',
                'AA': 'AdamicAdar',
                'SR': 'SimRank',
                'AS': 'AssociationStrength',
                'DP': 'DegreeProduct',
                'RA': 'ResourceAllocation',
                'L3': 'L3',
                'NR': 'NetworkRefinement'}

    return namedict[abbr]


'''
########################################################################################
#   Link prediction methods from linkpred
########################################################################################
'''


# use methods from package linkpred and return a dictionary

def predictor(G, method_name):
    method_score = eval('lp.predictors.' + method_name)(G, excluded = G.edges())
    score = {}
    for index, value in method_score.predict().items():
        score[tuple(index)] = value
    return score



'''
########################################################################################
#   Link prediction methods from me
########################################################################################
'''


# CommonNeighbours
def CN(A):
    A = np.array(A)
    A = (A > 0) * 1
    out_A = A.dot(A)
    out_A = out_A - np.diag(np.diag(out_A))
    return out_A


#  Jaccard similarity
def JS(A):
    A = np.array(A)
    A = (A > 0) * 1
    n = A.shape[0]
    score = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1, n):
            score[i, j] = ((A[i, ] & A[j, ]).sum()) / ((A[i, ] | A[j, ]).sum())
            score[j, i] = score[i, j]

    return score


# AdamicAdar
def AA(A):
    A = np.array(A)
    A = (A > 0) * 1
    degree = A.sum(1)
    n = A.shape[0]
    score = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            common_neighbour = A[i,] & A[j,]
            Nk = np.where(common_neighbour != 0)[0]
            score[i, j] = (1 / np.log(degree[Nk])).sum()
            score[j, i] = score[i, j]

    return score


# The default value of the parameter beta is set to 0.01 according to
# Link prediction techniques, applications, and performance: A survey, Ajay Kumar etc., 2020
# katz index
def katz(A, beta = 0.01):
    A = np.array(A)
    A = (A > 0) * 1
    n = A.shape[0]
    I = np.identity(n)  # create identity matrix
    eigenvalue, eigenvector = np.linalg.eig(A)
    thre = 1 / eigenvalue.max().real
    if beta < thre:
        katz_score = inv(I - A * beta) - I
    else:
        beta = thre / 2
        katz_score = inv(I - A * beta) - I
        print("The given beta does not satisfy the convergence requirement!")
        print("So we take half of the inverse of the maximum eigenvalue of A as the value of beta.")
    return katz_score


# ResourceAllocation
def RA(A):
    A = np.array(A)
    A = (A > 0) * 1
    neighbour_size = np.multiply(A, A).sum(1)
    A_left = np.multiply(A, (1 / np.sqrt(neighbour_size)).reshape(1, -1))
    A_right = np.multiply(A, (1 / np.sqrt(neighbour_size)).reshape(-1, 1))
    result = np.dot(A_left, A_right)
    result = result - np.diag(np.diag(result))
    return result

# def RA(A):
#     A = np.array(A)
#     A = (A > 0) * 1
#     degree = A.sum(1)
#     n = A.shape[0]
#     score = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i + 1, n):
#             common_neighbour = A[i,] & A[j,]
#             Nk = np.where(common_neighbour != 0)[0]
#             score[i, j] = (1 / degree[Nk]).sum()
#             score[j, i] = score[i, j]
#
#     return score


# DegreeProduct
def DP(A):
    A = np.array(A)
    n = A.shape[1]
    neighbour_size = np.multiply(A, A).sum(1)
    score = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            score[i, j] = neighbour_size[i] * neighbour_size[j]
            score[j, i] = score[i, j]

    return score

# def DP(A):
#     A = np.array(A)
#     A = (A > 0) * 1
#     n = A.shape[0]
#     degree = A.sum(1)
#     score = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i + 1, n):
#             score[i, j] = degree[i] * degree[j]
#             score[j, i] = score[i, j]
#
#     return score


# AssociationStrength
def AS(A):
    A = np.array(A)
    common_neighbour_size = A.dot(A)
    neighbour_size = np.multiply(A, A).sum(1)
    n = A.shape[1]
    score = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            score[i, j] = common_neighbour_size[i, j] / (neighbour_size[i] * neighbour_size[j])
            score[j, i] = score[i, j]
    return score

# def AS(A):
#     A = np.array(A)
#     A = (A > 0) * 1
#     degree = A.sum(1)
#     n = A.shape[0]
#     score = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i + 1, n):
#             common_neighbour = A[i, ] & A[j, ]
#             Nk = np.where(common_neighbour != 0)[0]
#             score[i, j] = Nk.shape[0] * (1 / (degree[i] * degree[j]))
#             score[j, i] = score[i, j]
#
#     return score


#  the degree normalized L3 metric proposed by Kovacs et al. (2019)
def L3(A):
    A = np.array(A)
    A = (A > 0) * 1
    row_nor = inv(np.diag(np.sqrt(A.sum(1))))
    B = A @ row_nor @ A @ row_nor @ A
    B = B - np.diag(np.diag(B))
    return B


def NR(W, m = 2, alpha = 1):
    W = np.array(W)
    n_N = W.shape[0]
    p1 = W.copy() / W.sum(1).reshape(n_N, 1)
    p2 = (m - 1) * p1.dot(inv(m * np.eye(n_N) - p1))  # row stochastic matrix
    stationary_d = null_space((p2 - np.eye(n_N)).T)
    stationary_d = alpha * stationary_d / stationary_d.sum()
    w_out = np.diag(abs(stationary_d.T)[0]).dot(p2)
    return w_out


def MOLI(W, x, alpha = 1):
    W = np.array(W)
    n_N = W.shape[1]
    d = len(x)
    p1 = W.copy() / W.sum(1).reshape(n_N, 1)
    p_x = np.zeros([n_N, n_N])
    for i in range(d):
        p_x = p_x + x[i] * matrix_power(p1, i + 2)

    p_x = p_x / p_x.sum(1).reshape(n_N, 1)
    # stationary_d = null_space((p_x - np.eye(n_N)).T)
    stationary_d = W.sum(1)
    stationary_d = alpha * stationary_d / stationary_d.sum()
    W_out = np.diag(stationary_d).dot(p_x)
    W_out = W_out + W_out.T

    return W_out
