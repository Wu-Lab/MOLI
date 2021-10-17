from ogb.linkproppred import LinkPropPredDataset
import pandas as pd
from Utils.help_func import *
from Utils.help_func_optimization import *
from Utils.predictors import *




# #####################################################################################################
#                       import dataset
# #####################################################################################################
dataset = LinkPropPredDataset(name = 'ogbl-ddi', root = './dataset')
split_edge = dataset.get_edge_split()
train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
num_nodes = dataset.graph['num_nodes']
print('Dataset loading completed!!!!\n')




# #####################################################################################################
#                     dataset splitted
# #####################################################################################################
train_mat = edgelist2mat(train_edge['edge'], num_nodes)
valid_mat_pos = edgelist2mat(valid_edge['edge'], num_nodes)
test_mat_pos = edgelist2mat(test_edge['edge'], num_nodes)
print('Dataset splitting completed!!!!\n')




# #####################################################################################################
#                     solve the optimization problem
# #####################################################################################################
eps = 1e-50
d = 2
iter_max = 1000

# x0 = np.random.random(d)
# x0 = x0 / x0.sum()
# x0 = [1, 0]
x0 = [0.99, 0.01]
x_opt, value_opt = scipy_constraint_likeihood(W_old = train_mat,
                                              W_new = train_mat + valid_mat_pos,
                                              d = d,
                                              eps = eps,
                                              x0 = x0,
                                              iteration = iter_max)

print('Optimal solution computation completed!!!!\n')
print('The optimal solution is: ', x_opt)




# #####################################################################################################
#                             test the performance
# #####################################################################################################
net_now = train_mat + valid_mat_pos
net_next = net_now + test_mat_pos
G = nx.from_numpy_array(net_now)

#  Compare with 11 methods:
methods_name = ['CN', 'JS', 'AA', 'AS', 'DP', 'RA',
                'L3', 'katz', 'NR', 'SR', 'RPR', 'MOLI']

# ------------------------------------------------------------------------------------

ref_net_nonedges = (net_next == 0) * (-1)
ref_net_edges = ((net_next != 0) & (net_now == 0)) * 1
ref_label = ref_net_nonedges + ref_net_edges
ref_label = ref_label - np.diag(np.diag(ref_label))

for meth in methods_name[0:-3]:
    print('evaluation of ' + abbr2fullname(meth) +
          ' algorithm:...........\n')
    auroc, aupr = evaluation(ref_label, eval(meth)(net_now))
    exec("AUROC_%s = auroc" % meth)
    exec("AUPR_%s = aupr" % meth)

print('evaluation of MOLI algorithm :...........\n')
AUROC_MOLI, AUPR_MOLI = evaluation(ref_label, MOLI(net_now, x_opt))

print('evaluation of simrank algorithm :...........\n')
AUROC_SR, AUPR_SR = evaluation(ref_label,
                               edgedict2mat(predictor(G, 'SimRank'),
                                            nx.number_of_nodes(G)))

print('evaluation of Rooted Page Rank algorithm :...........\n')
AUROC_RPR, AUPR_RPR = evaluation(ref_label,
                                 edgedict2mat(predictor(G, 'RootedPageRank'),
                                              nx.number_of_nodes(G)))




# #####################################################################################################
#                                      save the result
# #####################################################################################################

score_aupr = np.array([eval('AUPR_' + meth)
                       for meth in methods_name])
score_auroc = np.array([eval('AUROC_' + meth)
                        for meth in methods_name])

AUPR = pd.DataFrame({'score': score_aupr,
                     'methods': methods_name})
AUROC = pd.DataFrame({'score': score_auroc,
                      'methods': methods_name})

# save results

AUPR.to_csv("../Results/OGB_DDI_result/AUPR.csv")
AUROC.to_csv("../Results/OGB_DDI_result/AUROC.csv")

file = open("../Results/OGB_DDI_result/x_opt.txt", 'w')
file.write('x_opt:\n' + str(x_opt))
file.close()
