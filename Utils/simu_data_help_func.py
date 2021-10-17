import numpy as np
import random
from Utils.help_func import MOLI



'''
# add weighted edge based on our model

def add_edge_weight(W, x, sample_edge):
    n = W.shape[0]
    W_my = my(W, x)
    W_sample = np.multiply(np.triu(W_my, k = 1), (1 - (W > 0) * 1))  
    p_sample = W_sample[np.triu_indices(n, k = 1)]
    p_sample = p_sample / p_sample.sum()

    choice_index = np.random.choice(range(p_sample.shape[0]),
                                    size = sample_edge,
                                    replace = False, p = p_sample)
    choice_triu = np.zeros(p_sample.shape)
    choice_triu[choice_index] = 1

    W_out = np.zeros((n, n))
    W_out[np.triu_indices(n, k = 1)] = choice_triu
    W_out = W_out + W_out.T
    W_out = np.multiply((W_out + (W > 0) * 1), W_my)

    return W_out
'''




'''
# add unweighted edge based on our model
'''


def add_edge(W, x, sample_edge):
    W = np.array(W)
    n = W.shape[0]
    W_my = MOLI(W, x)
    W_sample = np.multiply(np.triu(W_my, k = 1), (1 - (W > 0) * 1))
    p_sample = W_sample[np.triu_indices(n, k = 1)]
    p_sample = p_sample / p_sample.sum()
    p_sample = p_sample.tolist()


    choice_index = np.random.choice(range(len(p_sample)),
                                    size = sample_edge,
                                    replace = False, p = p_sample)
    choice_triu = np.zeros(len(p_sample))
    choice_triu[choice_index] = 1

    W_out = np.zeros((n, n))
    W_out[np.triu_indices(n, k = 1)] = choice_triu
    W_out = W_out + W_out.T
    W_out = W_out + W

    return W_out
