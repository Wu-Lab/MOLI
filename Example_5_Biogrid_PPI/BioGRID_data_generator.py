from Example_5_Biogrid_PPI.BioGRID_helpfunc import *


# ####################################################################################################
#                       input the version of three datasets
# ####################################################################################################
versionT1 = '3.1.80'
versionT2 = '3.2.106'
versionT3 = '3.4.157'


# ####################################################################################################
#                            import the proteins we need
# ####################################################################################################
all_proteins = np.loadtxt('./BioGRID raw data/Proteins.txt', dtype = str)
print('Number of all proteins in SGD： ' + str(all_proteins.shape[0]))


# ####################################################################################################
#                               load  BioGRID datasets
# ####################################################################################################
# Create your own raw data file !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
file_T1 = './BioGRID raw data/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-' + versionT1 + '.tab2.txt'
file_T2 = './BioGRID raw data/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-' + versionT2 + '.tab2.txt'
file_T3 = './BioGRID raw data/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-' + versionT3 + '.tab2.txt'

print('\nLoading dataset-T1.....................')
Biogrid_T1 = data_loader(file_T1, all_proteins)
print('\nLoading dataset-T2.....................')
Biogrid_T2 = data_loader(file_T2, all_proteins)
print('\nLoading dataset-T3.....................')
Biogrid_T3 = data_loader(file_T3, all_proteins)


print('\n*******************************************************************')
print('The number of edges per dataset after filtering the raw Biogrid data (proteins and interaction type)：')
print('T1:' + str(Biogrid_T1.shape))
print('T2:' + str(Biogrid_T2.shape))
print('T3:' + str(Biogrid_T3.shape))

np.savetxt('./BioGRID processed data/ppu_' + versionT1 + '_id.txt', Biogrid_T1, fmt = '%s')
np.savetxt('./BioGRID processed data/ppu_' + versionT2 + '_id.txt', Biogrid_T2, fmt = '%s')
np.savetxt('./BioGRID processed data/ppu_' + versionT3 + '_id.txt', Biogrid_T3, fmt = '%s')


# ####################################################################################################
#                     calculating scores between two proteins
# ####################################################################################################
BioGRID_nameversion_T1 = BioGROD_weight_cal('./BioGRID processed data/ppu_' + versionT1 + '_id.txt')
BioGRID_nameversion_T2 = BioGROD_weight_cal('./BioGRID processed data/ppu_' + versionT2 + '_id.txt')
BioGRID_nameversion_T3 = BioGROD_weight_cal('./BioGRID processed data/ppu_' + versionT3 + '_id.txt')

np.savetxt('./BioGRID processed data/BioGRID_' + versionT1 + '.txt', BioGRID_nameversion_T1, fmt = '%s')
np.savetxt('./BioGRID processed data/BioGRID_' + versionT2 + '.txt', BioGRID_nameversion_T2, fmt = '%s')
np.savetxt('./BioGRID processed data/BioGRID_' + versionT3 + '.txt', BioGRID_nameversion_T3, fmt = '%s')
