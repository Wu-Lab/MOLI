import numpy as np
from Utils.help_func import edgelist2mat
import networkx as nx


# edgelist has three columns: node_i, node_j, t
# Split the edgelist into 'timesplit' temporal networks
def edgelist2split_mat(edgelist, timesplit):
    # Check if the record number of edgelist is divisible by 'timesplit'
    check = edgelist.shape[0] % timesplit
    if check != 0:
        edgelist = np.delete(edgelist, list(range(-check, 0)), axis = 0)
    num_nodes = int(edgelist[:, 0:2].max()) + 1

    # Slice the original edgelist into 'timesplit' edgelist
    edgelist = edgelist[np.lexsort(edgelist.T)]  # sort by time
    splited_edgelist = np.split(edgelist, timesplit)
    splited_mat = {}

    for t in range(timesplit):
        splited_mat[t] = edgelist2mat(splited_edgelist[t], num_nodes)

    return splited_mat


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

data_3col = np.loadtxt('./data/CollegeMsg.txt')

timesplit = 3
splited_mat = edgelist2split_mat(data_3col, timesplit)

for i in range(timesplit):
    splited_mat[i] = splited_mat[i] - np.diag(np.diag(splited_mat[i]))

G = nx.from_numpy_array(splited_mat[0])
largest_cc = max(nx.connected_components(G), key = len)
delete_index = list(set(range(splited_mat[0].shape[0])) - largest_cc)

for i in range(timesplit):
    splited_mat[i] = np.delete(splited_mat[i], list(delete_index), axis = 1)
    splited_mat[i] = np.delete(splited_mat[i], list(delete_index), axis = 0)
    np.savetxt('./data/splited_mat' + str(i) + '.txt',
               splited_mat[i],
               fmt = '%d')
