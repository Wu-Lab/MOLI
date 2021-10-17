


"""

Download the raw BioGRID data from https://downloads.thebiogrid.org/BioGRID/Release-Archive
you can follow the instruction of BioGRID_data_generator.py to process these data:

1. Only the proteins that appear in SGD and the data interaction type == 'physical' are retained.
    run:
        ppu_version_name = data_loader(file_name, proteins)
    input:
        * proteins can be imported from ./Proteins.txt
        * file_name is the dataname you have downloaded from the BioGRID website
    output:
        ppu_version_name is a three-column array of (protein name A, protein name B, Pubmed ID)

2.calculating the confidence scores between two proteins (nodes)
    BioGRID_nameversion_T = BioGROD_weight_cal('./ppu_' + versionT + '_id.txt')

3.To test the performance of link prediction, run :
    test_performance.py to compute the AUPR and AUROC scores

"""
