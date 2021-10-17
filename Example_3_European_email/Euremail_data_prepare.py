import numpy as np
from Utils.help_func import edgelist2mat


# Split into ’timesplit‘ temporal network datasets
def edgelist2split_mat(edgelist, timesplit):
    num_nodes = int(edgelist[:, 0:2].max()) + 1

    # Slice the original edgelist into ’timesplit‘ edgelist
    edgelist = edgelist[np.lexsort(edgelist.T)]
    splited_edgelist = np.split(edgelist, timesplit)
    splited_mat = {}

    for t in range(timesplit):
        temp_edgelist = splited_edgelist[t]
        splited_mat[t] = edgelist2mat(temp_edgelist, num_nodes)

    return splited_mat
    


data_3col = np.loadtxt('./email data/email-Eu-core-temporal.txt')


# 1th_col: id of the source node (a user), starting from 0
# 2th_col: id of the target node (a user), starting from 0
# 3th_ol: timestamp (in seconds), starting from 0


timesplit = 3
splited_mat = edgelist2split_mat(data_3col, timesplit)

for i in range(timesplit):
    exec("zeros_index%s=set(np.where(splited_mat[i].sum(1)==0)[0].tolist())" % i)

delete_index = list(eval('zeros_index0'))
for i in range(timesplit):
    splited_mat[i] = np.delete(splited_mat[i], list(delete_index), axis = 1)
    splited_mat[i] = np.delete(splited_mat[i], list(delete_index), axis = 0)
    splited_mat[i] = (splited_mat[i] > 0) * 1
    np.savetxt('./email data/splited_mat' + str(i) + '.txt',
               splited_mat[i],
               fmt = '%d')
