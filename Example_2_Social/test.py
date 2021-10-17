from Utils.help_func import *
from Utils.help_func_optimization import *
from Utils.predictors import *
import pandas as pd


# #####################################################################################################
#                                       import dataset
# #####################################################################################################
net0 = np.loadtxt('./data/splited_mat0.txt')
net1 = np.loadtxt('./data/splited_mat1.txt')
net2 = np.loadtxt('./data/splited_mat2.txt')
print('****************Dataset loading completed!!!!******************************\n')

net_now = net0 + net1
net_new = net0 + net1 + net2

net0 = (net0 > 0)*1
net_now = (net_now > 0)*1
net_new = (net_new > 0)*1
# #####################################################################################################
#                     solve the optimization problem
# #####################################################################################################
# x0 = np.random.random(2)
# x0 = x0 / x0.sum()
# x0 = [0.99, 0.01]
x0 = [0.01, 0.99]
iter_max = 1000

x_opt, value = scipy_constraint_likeihood(W_old = net0,
                                          W_new = net_now,
                                          d = 2,
                                          eps = 1e-30,
                                          x0 = x0,
                                          iteration = iter_max)


print('Optimal solution computation completed!!!!\n')
print('The optimal solution is: ', x_opt)




# #####################################################################################################
#                                     test the performance
# #####################################################################################################
G = nx.from_numpy_array(net_now)

#  Compare with 11 methods:
methods_name = ['CN', 'JS', 'AA', 'AS', 'DP', 'RA',
                'L3', 'katz', 'NR', 'SR', 'RPR', 'MOLI']


ref_net_nonedges = (net_new == 0) * (-1)
ref_net_edges = ((net_new != 0) & (net_now == 0)) * 1
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

AUPR.to_csv("../Results/Social_results/AUPR.csv")
AUROC.to_csv("../Results/Social_results/AUROC.csv")

file = open("../Results/Social_results/x_opt.txt", 'w')
file.write('x_opt:\n' + str(x_opt))
file.close()





