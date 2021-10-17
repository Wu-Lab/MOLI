from Example_5_Biogrid_PPI.BioGRID_helpfunc import *
from Utils.help_func import *
from Utils.help_func_optimization import *


# #####################################################################################################
#                       import dataset
# #####################################################################################################
versionT1 = '3.1.80'
versionT2 = '3.2.106'
versionT3 = '3.4.157'

print("***************************************Import Data and preprocessing*******************************************")
data_T1 = np.loadtxt('./BioGRID processed data/BioGRID_' + versionT1 + '.txt', dtype = str)
data_T2 = np.loadtxt('./BioGRID processed data/BioGRID_' + versionT2 + '.txt', dtype = str)
data_T3 = np.loadtxt('./BioGRID processed data/BioGRID_' + versionT3 + '.txt', dtype = str)

e_T1 = data_T1.shape[0]
e_T2 = data_T2.shape[0]
e_T3 = data_T3.shape[0]

print('number of edges:')
print("T1: " + str(e_T1))
print("T2: " + str(e_T2))
print("T3: " + str(e_T3))

prot_T1 = set(data_T1[:, 0].tolist()) | set(data_T1[:, 1].tolist())
prot_T2 = set(data_T2[:, 0].tolist()) | set(data_T2[:, 1].tolist())
prot_T3 = set(data_T3[:, 0].tolist()) | set(data_T3[:, 1].tolist())

print('number of proteins:')
print("T1: " + str(len(prot_T1)))
print("T2: " + str(len(prot_T2)))
print("T3: " + str(len(prot_T3)))

all_proteins = prot_T1 & prot_T2 & prot_T3

print('number of common proteins:')
print(len(all_proteins))



print('\n**********filtering***********')
protein_2_num = dict(zip(all_proteins, range(len(all_proteins))))
dataT1_V1 = protein3col_2_num3col(data_T1, protein_2_num)
dataT2_V1 = protein3col_2_num3col(data_T2, protein_2_num)
dataT3_V1 = protein3col_2_num3col(data_T3, protein_2_num)
e_T1_V1 = dataT1_V1.shape[0]
e_T2_V1 = dataT2_V1.shape[0]
e_T3_V1 = dataT3_V1.shape[0]

print('number of edges now:')
print("T1: " + str(e_T1_V1))
print("T2: " + str(e_T2_V1))
print("T3: " + str(e_T3_V1))



print('\n*********changing into matrix***********')
n = int(dataT1_V1.max()) + 1
net_T1 = np.zeros([n, n])
net_T2 = np.zeros([n, n])
net_T3 = np.zeros([n, n])

for i in range(e_T1_V1) :
    net_T1[int(dataT1_V1[i, 0]), int(dataT1_V1[i, 1])] = dataT1_V1[i, 2]

for i in range(e_T2_V1) :
    net_T2[int(dataT2_V1[i, 0]), int(dataT2_V1[i, 1])] = dataT2_V1[i, 2]

for i in range(e_T3_V1) :
    net_T3[int(dataT3_V1[i, 0]), int(dataT3_V1[i, 1])] = dataT3_V1[i, 2]

net_T1 = net_T1 + net_T1.T
net_T2 = net_T2 + net_T2.T
net_T3 = net_T3 + net_T3.T



del_T2_index = np.where(net_T3 - net_T2 == -1)
net_T2[del_T2_index] = 0
del_T1_index = np.where(net_T2 - net_T1 == -1)
net_T1[del_T1_index] = 0
print('After filtering the edges that appear in the previous dataset and '
      'disappear in the subsequent dataset, the number of edges in each dataset：')
print("T1: " + str((net_T1 > 0).sum() / 2))
print("T2: " + str((net_T2 > 0).sum() / 2))
print("T3: " + str((net_T3 > 0).sum() / 2))




del_nodes1 = np.where(net_T1.sum(1) == 0)
del_nodes2 = np.where(net_T2.sum(1) == 0)
del_nodes3 = np.where(net_T3.sum(1) == 0)
del_nodes = np.hstack((del_nodes1, del_nodes2, del_nodes3))
net_T1 = np.delete(net_T1, del_nodes, axis = 0)
net_T1 = np.delete(net_T1, del_nodes, axis = 1)
net_T2 = np.delete(net_T2, del_nodes, axis = 0)
net_T2 = np.delete(net_T2, del_nodes, axis = 1)
net_T3 = np.delete(net_T3, del_nodes, axis = 0)
net_T3 = np.delete(net_T3, del_nodes, axis = 1)

print('After filtering the isolated points, the number of edges per dataset：')
print("T1: " + str((net_T1 > 0).sum() / 2))
print("T2: " + str((net_T2 > 0).sum() / 2))
print("T3: " + str((net_T3 > 0).sum() / 2))

print('After filtering the isolated points, the number of nodes per dataset：')
print("T1: " + str(net_T1.shape[0]))
print("T2: " + str(net_T2.shape[0]))
print("T3: " + str(net_T3.shape[0]))

print("*********************************Import Data and preprocessing finished************************************\n")




# #####################################################################################################
#                     solve the optimization problem
# #####################################################################################################
eps = 1e-50
d = 2
iter_max = 1000

# x0 = np.random.random(d)
# x0 = x0 / x0.sum()
x0 = [0.01, 0.99]
x_opt, value_opt = scipy_constraint_likeihood(W_old = net_T1,
                                              W_new = net_T2,
                                              d = d,
                                              eps = eps,
                                              x0 = x0,
                                              iteration = iter_max)

print('Optimal solution computation completed!!!!\n')
print('The optimal solution is: ', x_opt)





# ##########################################################################################################
#                                           evaluating
# ##########################################################################################################
G = nx.from_numpy_array(net_T2)

#  Compare with 11 methods:
methods_name = ['CN', 'JS', 'AA', 'AS', 'DP', 'RA',
                'L3', 'katz', 'NR', 'SR', 'RPR', 'MOLI']


ref_net_nonedges = (net_T3 == 0) * (-1)
ref_net_edges = ((net_T3 != 0) & (net_T2 == 0)) * 1
ref_label = ref_net_nonedges + ref_net_edges
ref_label = ref_label - np.diag(np.diag(ref_label))


for meth in methods_name[0:-3]:
    print('evaluation of ' + abbr2fullname(meth) +
          ' algorithm:...........\n')
    auroc, aupr = evaluation(ref_label, eval(meth)(net_T2))
    exec("AUROC_%s = auroc" % meth)
    exec("AUPR_%s = aupr" % meth)

print('evaluation of MOLI algorithm :...........\n')
AUROC_MOLI, AUPR_MOLI = evaluation(ref_label, MOLI(net_T2, x_opt))

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

AUPR.to_csv("../Results/BioGRID_PPI_result/AUPR.csv")
AUROC.to_csv("../Results/BioGRID_PPI_result/AUROC.csv")

file = open("../Results/BioGRID_PPI_result/x_opt.txt", 'w')
file.write('x_opt:\n' + str(x_opt))
file.close()
