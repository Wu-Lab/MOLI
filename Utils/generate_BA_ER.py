import numpy as np
import networkx as nx


n = 50
m = 3

G = nx.random_graphs.barabasi_albert_graph(n, m)
number_components = nx.number_connected_components(G)
if number_components != 1:
    while number_components != 1:
        G = nx.random_graphs.barabasi_albert_graph(n, m)
        number_components = nx.number_connected_components(G)

print('number of components:' + str(number_components))
adj_mat = nx.adjacency_matrix(G).todense()
np.savetxt('../Example_1_Simulated/BA_data/BA.txt', adj_mat, fmt = '%d')




n = 50
p = 0.3

G = nx.random_graphs.erdos_renyi_graph(n, p)
number_components = nx.number_connected_components(G)

if number_components != 1:
    while number_components != 1:
        G = nx.random_graphs.erdos_renyi_graph(n, p)
        number_components = nx.number_connected_components(G)

print('number of components:' + str(number_components))
adj_mat = nx.adjacency_matrix(G).todense()
np.savetxt('../Example_1_Simulated/ER_data/ER.txt', adj_mat, fmt = '%d')
