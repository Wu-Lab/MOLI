import numpy as np
import pandas as pd


def mat_3col(mat):
    n = mat.shape[0]
    m = int(mat.sum() / 2)
    k = 0
    data3col = np.zeros([m, 3])
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] != 0:
                data3col[k, 0] = i
                data3col[k, 1] = j
                data3col[k, 2] = 1
                k = k + 1

    return data3col


def data_3col2net(all_proteins, Biogrid_3col, proteins_name_2_num):
    n = all_proteins.shape[0]
    net = np.zeros([n, n])

    for key, value in Biogrid_3col.items():
        i_index = proteins_name_2_num[key[0]]
        j_index = proteins_name_2_num[key[1]]
        net[i_index, j_index] = value

    return net


def pure_score_table(num_exp, exp_type):
    if exp_type == 'Low':
        if num_exp == 0:
            score = 0
        elif num_exp == 1:
            score = 0.8
        elif num_exp == 2:
            score = 0.9
        else:
            score = 0.95
    elif exp_type == 'High':
        if num_exp == 0:
            score = 0
        elif num_exp == 1:
            score = 0.25
        elif num_exp == 2:
            score = 0.50
        elif num_exp == 3:
            score = 0.75
        else:
            score = 0.85

    return score


def mix_score_table(num_exp, high_num_exp):
    score = high_num_exp * (pure_score_table(num_exp - high_num_exp, 'Low') + 0.05)
    if score < 0.95:
        return np.around(score, 2)
    else:
        return 0.95





"""
# calculating BioGRID PPI scores according to : Cao et al., New directions for diffusion-based network prediction 
  of protein function: incorporating pathways with confidence
# If there are at least 100 PPIs associated with a particular publication, we classify
# that publication’s endorsements as high-throughput and otherwise low-throughput.
# We merge A-B and B-A edges, 

* input data preprocessed by 'data_loader' function
"""


def BioGROD_weight_cal(data_file):
    print('\n********************************calculating edge weight***************************************')
    print('Importing data..............................\n')
    ppu = pd.read_table(data_file, header = None, sep = ' ')
    ppu.columns = ['ProteinA', 'ProteinB', 'PubMed_id']
    ID_stat = pd.value_counts(ppu['PubMed_id'].tolist())
    high_throughput_index = np.where(np.array(ID_stat) > 100)[0]
    high_throughput_id = np.array(ID_stat.keys().tolist())[high_throughput_index]

    # Adjust the ordering of the protein names at the ends of the edges so that A-B and B-A are the same edges
    print('Adjust the ordering of the protein names..............................\n')
    for i in range(ppu.shape[0]):
        if ppu.iloc[i, 0] < ppu.iloc[i, 1]:
            temp = ppu.iloc[i, 0]
            ppu.iloc[i, 0] = ppu.iloc[i, 1]
            ppu.iloc[i, 1] = temp

    print('Calculating the scores..............................\n')
    weight_dict = dict.fromkeys(zip(ppu['ProteinA'], ppu['ProteinB']), [])
    for i in range(len(ppu)):
        weight_dict[(ppu.iloc[i, 0], ppu.iloc[i, 1])] = \
            weight_dict[(ppu.iloc[i, 0], ppu.iloc[i, 1])] + \
            [ppu.iloc[i, 2]]

    for key in weight_dict.keys():
        high_throu_result = np.array([(i in high_throughput_id) for i in np.array(weight_dict[key])])
        high_num = high_throu_result.sum()
        all_num = high_throu_result.shape[0]
        if high_num == all_num:  # high
            score = pure_score_table(high_num, 'High')
        elif high_num == 0:  # low
            score = pure_score_table(all_num, 'Low')
        else:  # mix
            score = mix_score_table(all_num, high_num)

        weight_dict[key] = score

    print('Possible edge weights:')
    print(set(list(weight_dict.values())))
    print('\n')

    print('Number of edges:')
    print(len(weight_dict))
    print('\n')

    data_3col = np.zeros([len(weight_dict), 3]).astype('str')
    i = 0
    for key, value in weight_dict.items():
        data_3col[i, 0] = key[0]
        data_3col[i, 1] = key[1]
        data_3col[i, 2] = value
        i = i + 1

    print('Number of nodes:')
    print(len(set(data_3col[:, 0].tolist()) |
              set(data_3col[:, 1].tolist())))
    print('\n')

    print('\n********************************calculating edge weight finished!!'
          '***************************************')
    return data_3col








"""
# Given a dict 'protein_num': proteins for key, id number for value
# We turn a three-column array of (protein name A, protein name B, score) into
#         a three-column array of (protein id A, protein id B, score)
"""


def protein3col_2_num3col(data, protein_num):
    out_data = np.zeros([data.shape[0], 3])
    for i in range(data.shape[0]):
        if ((data[i, 0] in protein_num.keys()) &
                (data[i, 1] in protein_num.keys())):
            out_data[i, 0] = int(protein_num[data[i, 0]])
            out_data[i, 1] = int(protein_num[data[i, 1]])
            out_data[i, 2] = float(data[i, 2])

    del_index = np.where(out_data.sum(1) == 0)
    out_data = np.delete(out_data, del_index[0], axis = 0)
    return out_data








"""
Raw data from https://downloads.thebiogrid.org/BioGRID/Release-Archive
Only the proteins that appear in SGD are retained,
and only the data with the interaction type physical are retained.
Returns a three-column array of (protein name A, protein name B, Pubmed ID) 
"""


def data_loader(file, proteins):
    print('***************************************data loader****************************************************')
    Biogrid = pd.read_table(file, dtype = str)

    # Only the proteins that appear in SGD are retained,
    # and only the data with the interaction type physical are retained.
    keep_index = []
    for i in range(Biogrid.shape[0]):
        if ((Biogrid['Systematic Name Interactor A'][i] in proteins) &
                (Biogrid['Systematic Name Interactor B'][i] in proteins) &
                (Biogrid['Experimental System Type'][i] == 'physical') &
                (Biogrid['Systematic Name Interactor A'][i] !=
                 Biogrid['Systematic Name Interactor B'][i])):
            keep_index.append(i)

    Biogrid_V1 = Biogrid.iloc[keep_index, ]

    print('The numbers of proteins retained:')
    print(len(set(Biogrid_V1['Systematic Name Interactor A'].tolist()) |
              set(Biogrid_V1['Systematic Name Interactor B'].tolist())))

    ppu = np.zeros([Biogrid_V1.shape[0], 3]).astype(str)
    Interactor_A = Biogrid_V1['Systematic Name Interactor A'].tolist()
    Interactor_B = Biogrid_V1['Systematic Name Interactor B'].tolist()
    pubme_id = Biogrid_V1['Pubmed ID'].tolist()

    for i in range(Biogrid_V1.shape[0]):
        ppu[i, 0] = Interactor_A[i]
        ppu[i, 1] = Interactor_B[i]
        ppu[i, 2] = str(pubme_id[i])

    print('***************************************data loader '
          'finished!!!****************************************************')
    return ppu
