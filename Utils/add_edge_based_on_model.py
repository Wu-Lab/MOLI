import numpy as np
from Utils.simu_data_help_func import add_edge
from Utils.help_func import dataname2data
import networkx as nx



################################################################################################################
#                     ER graph
################################################################################################################
ref_net = np.loadtxt('./Simulated/ER_data/ER.txt')
add_edge_prop = 0.1
add_edge_num = int(np.ceil(add_edge_prop * ref_net.sum() / 2))

for x in ['x1', 'x2']:
    for loops in range(100):
        x_cof = dataname2data(x)
        new_net_unweighted = add_edge(ref_net, x_cof, add_edge_num)

        np.savetxt('./Simulated/ER_data/add_edge/' + x + '_' +
                   str(loops) + '_addedge' + str(add_edge_prop) +
                   '_ER.txt', new_net_unweighted, fmt = '%d')




################################################################################################################
#                     BA graph
################################################################################################################
# original net
ref_net = np.loadtxt('./Simulated/BA_data/BA.txt')
add_edge_prop = 0.1
add_edge_num = int(np.ceil(add_edge_prop * ref_net.sum() / 2))

for x in ['x1', 'x2']:
    for loops in range(100):
        x_cof = dataname2data(x)
        new_net_unweighted = add_edge(ref_net, x_cof, add_edge_num)

        np.savetxt('./Simulated/BA_data/add_edge/' + x + '_' +
                   str(loops) + '_addedge' + str(add_edge_prop) +
                   '_BA.txt', new_net_unweighted, fmt = '%d')





################################################################################################################
#                     lesmis graph
################################################################################################################
# original net
ref_net = np.loadtxt('./Simulated/lesmis_data/lesmis.txt')
add_edge_prop = 0.1
add_edge_num = int(np.ceil(add_edge_prop * ref_net.sum() / 2))

for x in ['x1', 'x2']:
    for loops in range(100):
        x_cof = dataname2data(x)
        new_net_unweighted = add_edge(ref_net, x_cof, add_edge_num)

        np.savetxt('./Simulated/lesmis_data/add_edge/' + x + '_' +
                   str(loops) + '_addedge' + str(add_edge_prop) +
                   '_lesmis.txt', new_net_unweighted, fmt = '%d')



################################################################################################################
#                     cerebral_cortex graph
################################################################################################################
# original net
G = nx.read_graphml('./Simulated/cerebral_cortex_data/rhesus_cerebral.cortex_1.graphml')
ref_net = np.array(nx.adjacency_matrix(G).todense())
ref_net = ref_net + ref_net.T
ref_net = (ref_net > 0) * 1

add_edge_prop = 0.1
add_edge_num = int(np.ceil(add_edge_prop * ref_net.sum() / 2))

for x in ['x1', 'x2']:
    for loops in range(100):
        x_cof = dataname2data(x)
        new_net_unweighted = add_edge(ref_net, x_cof, add_edge_num)

        np.savetxt('./Simulated/cerebral_cortex_data/add_edge/' + x + '_' +
                   str(loops) + '_addedge' + str(add_edge_prop) +
                   '_cerebral_cortex.txt', new_net_unweighted, fmt = '%d')
