from __future__ import division
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from Utils.predictors import *
import networkx as nx

'''
########################################################################################
#   evaluation functions
########################################################################################

'''


def evaluation(ref_label, net_predict):
    T = ref_label[ref_label != 0]
    P = net_predict[ref_label != 0]
    fpr, tpr, threshold = roc_curve(T, P)
    AUROC = auc(fpr, tpr)

    precision, tpr1, threshold1 = precision_recall_curve(T, P)
    AUPR = auc(tpr1, precision)

    return AUROC, AUPR


def edgedict2mat(edgelist_dict, num_nodes):
    mat = np.zeros([num_nodes, num_nodes])
    for key, value in edgelist_dict.items():
        i_index = int(key[0])
        j_index = int(key[1])
        mat[i_index, j_index] = value

    mat = mat + mat.T
    return mat


'''
###############################################################################################################
# Help to complete the conversion between matrix and graph, 
# calculate some properties of the graph, such as average degree, clustering coefficient, diameter, etc.
###############################################################################################################
'''


# Generate a weighted undirected graph G from a weighted matrix
def graph_generator(W):
    G = nx.Graph()
    edgelist = np.array(np.where((np.triu(W, k = 1) > 0) * 1)).T
    edgelist = tuple(tuple([y for y in x]) for x in edgelist)
    G.add_edges_from(edgelist)
    G = nx.to_undirected(G)
    return G


def average_degree(G):
    ad = np.average(np.array([G.degree[x] for x in list(G.nodes)]))
    return ad


def max_degree(G):
    ad = np.max(np.array([G.degree[x] for x in list(G.nodes)]))
    return ad


def graph_properties(G):
    V = G.number_of_nodes()
    E = G.number_of_edges()
    dia = nx.diameter(G)
    clu = nx.average_clustering(G)
    av_deg = average_degree(G)
    max_deg = max_degree(G)

    return [V, E, dia, av_deg, max_deg, clu]


def plot_degree(G, name):
    x_ = range(1, len(nx.degree_histogram(G)[1:]) + 1)
    y_ = nx.degree_histogram(G)[1:]
    y_ = np.array(y_) / sum(y_)
    plt.bar(x_, y_)
    plt.xlabel('degree')
    plt.ylabel('Probability')
    plt.title(name)
    plt.show()


def dataname2data(name):
    if name == 'x1':
        return [1]
    elif name == 'x2':
        return [0, 1]
    elif name == 'x12':
        return [0.5, 0.5]
    elif name == 'x3':
        return [0, 0, 1]
    elif name == 'x4':
        return [0, 0, 0, 1]
    elif name == 'x5':
        return [0, 0, 0, 0, 1]


def data_loader_cere():
    G1 = nx.read_graphml('./cerebral_cortex_data/rhesus_cerebral.cortex_1.graphml')
    ref_net = np.array(nx.adjacency_matrix(G1).todense())
    ref_net = ref_net + ref_net.T
    ref_net = (ref_net > 0) * 1

    return ref_net


def edgelist2mat(edgelist, num_nodes):
    mat = np.zeros([num_nodes, num_nodes])
    n = edgelist.shape[0]
    for i in range(n):
        i_index = int(edgelist[i, 0])
        j_index = int(edgelist[i, 1])
        mat[i_index, j_index] = mat[i_index, j_index] + 1

    mat = mat + mat.T
    return mat


def weight_edgelist2mat(edgelist, num_nodes):
    mat = np.zeros([num_nodes, num_nodes])
    n = edgelist.shape[0]
    for i in range(n):
        i_index = int(edgelist[i, 0])
        j_index = int(edgelist[i, 1])
        mat[i_index, j_index] = mat[i_index, j_index] + edgelist[i, 2]

    mat = mat + mat.T
    return mat
