from Utils.help_func import *
import pandas as pd
import seaborn as sns
from Utils.help_func_optimization import *
from Utils.predictors import *
import networkx as nx




# ######################################################################################################
#                                       import data
# ######################################################################################################
# input network parameters
'''
dataname: 'BA', 'ER', 'lesmis', 'cerebral_cortex'
settings: 'x1','x2'
add_edge_prop = 0.1
'''

dataname = 'BA'
settings = 'x1'
add_edge_prop = 0.1

# experiment settings
Loops = 30  # train data size
test_loop = 70  # test data size
d = 2
iter_max = 1000

train_data_file_name = './' + dataname + '_data/' + dataname + '.txt'

if dataname == 'cerebral_cortex':
    ref_net = data_loader_cere()
else:
    ref_net = np.loadtxt(train_data_file_name)

G = nx.from_numpy_array(ref_net)






# #####################################################################################################
#                                           Training step
# #####################################################################################################
# The optimal coefficients of each order of neighborhood information
# are obtained by solving the optimization problem


x_d = np.zeros([Loops, d])

for loop in range(1, Loops):
    test_data_file_name = './' + dataname + '_data/add_edge/' \
                          + settings + '_' + str(loop) + '_addedge' + \
                          str(add_edge_prop) + '_' + dataname + '.txt'
    net_new = np.loadtxt(test_data_file_name)


    x0 = np.random.random(d)
    x0 = x0 / x0.sum()
    x_temp, value_temp = scipy_constraint_likeihood(W_old = ref_net,
                                                    W_new = net_new,
                                                    d = d,
                                                    eps = 1e-30,
                                                    x0 = x0,
                                                    iteration = iter_max)
    x_d[loop, ] = x_temp.ravel()

x_opt = np.average(x_d, axis = 0)
x_opt = x_opt / x_opt.sum()
print('**************************************************************************')
print('The optimal coefficients of each order of neighborhood information: ')
print(x_opt)
print('\n')






# #####################################################################################################
#                                           Testing step
# #####################################################################################################

#  Compare with 11 methods:
methods_name = ['CN', 'JS', 'AA', 'AS', 'DP', 'RA',
                'L3', 'katz', 'NR', 'SR', 'RPR', 'MOLI']

for meth in methods_name:
    exec("AUPR_%s = np.zeros([1, test_loop])" % meth)
    exec("AUROC_%s = np.zeros([1, test_loop])" % meth)

for loop in range(Loops, Loops + test_loop):
    test_data_file_name = './' + dataname + '_data/add_edge/' \
                          + settings + '_' + str(loop) + '_addedge' + \
                          str(add_edge_prop) + '_' + dataname + '.txt'
    net_new = np.loadtxt(test_data_file_name)


    ref_net_nonedges = (net_new == 0) * (-1)
    ref_net_edges = ((net_new != 0) & (ref_net == 0)) * 1
    ref_label = ref_net_nonedges + ref_net_edges
    ref_label = ref_label - np.diag(np.diag(ref_label))

    for meth in methods_name[0:-3]:
        print('evaluation of ' + abbr2fullname(meth) +
              ' algorithm (loops = ' + str(loop) + '):...........\n')
        auroc, aupr = evaluation(ref_label, eval(meth)(ref_net))
        exec("AUROC_%s[0, loop - Loops] = auroc" % meth)
        exec("AUPR_%s[0, loop - Loops] = aupr" % meth)

    print('evaluation of MOLI algorithm (loops = ' + str(loop) + '):...........\n')
    auroc, aupr = evaluation(ref_label, MOLI(ref_net, x_opt))
    eval('AUROC_MOLI')[0, loop - Loops] = auroc
    eval('AUPR_MOLI')[0, loop - Loops] = aupr

    print('evaluation of simrank algorithm (loops = ' + str(loop) + '):...........\n')
    auroc, aupr = evaluation(ref_label,
                             edgedict2mat(predictor(G, 'SimRank'),
                                          nx.number_of_nodes(G)))
    eval('AUPR_SR')[0, loop - Loops] = aupr
    eval('AUROC_SR')[0, loop - Loops] = auroc


    
    print('evaluation of Rooted Page Rank algorithm (loops = ' + str(loop) + '):...........\n')
    auroc, aupr = evaluation(ref_label,
                             edgedict2mat(predictor(G, 'RootedPageRank'),
                                          nx.number_of_nodes(G)))
    eval('AUPR_RPR')[0, loop - Loops] = aupr
    eval('AUROC_RPR')[0, loop - Loops] = auroc



# #####################################################################################################
#                                      plot the result
# #####################################################################################################

Methods_name = np.array(methods_name).repeat(test_loop)

score_aupr = np.vstack([eval('AUPR_' + meth)
                        for meth in methods_name]).ravel()
score_auroc = np.vstack([eval('AUROC_' + meth)
                         for meth in methods_name]).ravel()

AUPR = pd.DataFrame({'score': score_aupr,
                     'methods': Methods_name})
AUROC = pd.DataFrame({'score': score_auroc,
                      'methods': Methods_name})


plt.figure(1)
sns.boxplot(x = 'methods', y = 'score',
            data = AUPR, linewidth = 2, width = 0.6,
            saturation = 0,
            color = 'white')
sns.stripplot(x = 'methods', y = 'score',
              data = AUPR, jitter = True,
              edgecolor = "gray", size = 3)

plt.grid(linestyle = "--", alpha = 0.3)
plt.title("AUPR ( add_edge_prop: " + str(add_edge_prop) +
          '  setting: ' + settings + '  ' + ")")
plt.axhline(y = np.median(eval('AUPR_NR')), ls = "--", c = "#828282")
plt.savefig("../Results/simulated_data_results/" + dataname + "_result/" + dataname + "_add_edge_" +
            str(add_edge_prop) + '_' + settings + "_AUPR.svg")
plt.show()

plt.figure(2)
sns.boxplot(x = 'methods', y = 'score',
            data = AUROC, linewidth = 2,
            width = 0.6, saturation = 0,
            color = 'white')
sns.stripplot(x = 'methods', y = 'score',
              data = AUROC, jitter = True,
              edgecolor = "gray", size = 3)
plt.grid(linestyle = "--", alpha = 0.3)
plt.title("AUROC ( add_edge_prop: " + str(add_edge_prop) +
          '  setting: ' + settings + '  ' + ")")
plt.axhline(y = np.median(eval('AUROC_NR')), ls = "--", c = "#828282")
plt.savefig("../Results/simulated_data_results/" + dataname + "_result/" + dataname + "_add_edge_" +
            str(add_edge_prop) + '_' + settings + "_AUROC.svg")
plt.show()

# save results
save_ave_AUPR = pd.DataFrame({'score': np.array([np.average(eval('AUPR_' + meth))
                                                 for meth in methods_name]),
                              'methods': methods_name})

save_ave_AUROC = pd.DataFrame({'score': np.array([np.average(eval('AUROC_' + meth))
                                                  for meth in methods_name]),
                               'methods': methods_name})

save_sd_AUPR = pd.DataFrame({'score': np.array([np.var(eval('AUPR_' + meth))
                                                for meth in methods_name]),
                             'methods': methods_name})

save_sd_AUROC = pd.DataFrame({'score': np.array([np.var(eval('AUROC_' + meth))
                                                 for meth in methods_name]),
                              'methods': methods_name})

save_ave_AUROC.to_csv('../Results/simulated_data_results/' + dataname + '_result/' +
                      dataname + '_add_edge_' +
                      str(add_edge_prop) + '_'
                      + settings + '_ave_auroc.csv')
save_ave_AUPR.to_csv('../Results/simulated_data_results/' + dataname + '_result/' +
                     dataname + '_add_edge_' +
                     str(add_edge_prop) + '_'
                     + settings + '_ave_aupr.csv')
save_sd_AUROC.to_csv('../Results/simulated_data_results/' + dataname + '_result/' +
                     dataname + '_add_edge_' +
                     str(add_edge_prop) + '_'
                     + settings + '_sd_auroc.csv')
save_sd_AUPR.to_csv('../Results/simulated_data_results/' + dataname + '_result/' +
                    dataname + '_add_edge_' +
                    str(add_edge_prop) + '_'
                    + settings + '_sd_aupr.csv')


file = open('../Results/simulated_data_results/' + dataname + '_result/' +
            dataname + '_add_edge_' +
            str(add_edge_prop) + '_'
            + settings + "_x_opt.txt", 'w')
file.write('x_opt:\n' + str(x_opt))
file.close()
