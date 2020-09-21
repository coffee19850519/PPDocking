import os, json
import numpy as np
#import pandas as pd
from Bio.PDB import PDBParser
from ssbio.protein.structure.properties.dssp import get_dssp_df
from math import radians

def ss_to_index(input):


    ss_symbols = ['H','B','E','G','I','T','S','-']
    ss_index = ss_symbols.index(input)
    #ss includes 8-type symbols
    #one_hot = np.eye(len(ss_symbols))[ss_index]
    del ss_symbols
    return ss_index





def encode(df):

    node_features = {}
    #new_df = pd.DataFrame(columns=cols)
    for index, row in df.iterrows():

        node_id = row['chain']+ str(row['resnum']) +row['aa_three']

        #features = ss_decode(row['ss'])
        # features.append(row['exposure_rsa'])
        # features.append(radians(row['phi']))
        # features.append(radians(row['psi']))
        # features.append(row['NH_O_1_energy'])
        # features.append(row['O_NH_1_energy'])
        # features.append(row['NH_O_2_energy'])
        # features.append(row['O_NH_2_energy'])

        node_features.update({node_id : [ss_to_index(row['ss']), row['exposure_rsa'],radians(row['phi']), radians(row['psi']), row['NH_O_1_energy'], row['O_NH_1_energy'], row['NH_O_2_energy'], row['O_NH_2_energy']]})
        # new_df = new_df.append([{'ss': ss, 'exposure_rsa': exposure_rsa, 'phi': phi, 'NH_O_1_energy': NH_O_1_energy,
        #                          'O_NH_1_energy': O_NH_1_energy, 'NH_O_2_energy': NH_O_2_energy,
        #                          'O_NH_2_energy': O_NH_2_energy}], ignore_index=True)
    return node_features

def create_ss_feature_files(pdb_folder, pdb_filename, complex_name,save_folder):

    p = PDBParser()
    structure = p.get_structure("1", os.path.join(pdb_folder, pdb_filename))
    model = structure[0]
    df = get_dssp_df(model, os.path.join(pdb_folder, pdb_filename), outfile=os.path.splitext(pdb_filename)[0], outdir=pdb_folder, outext='_dssp.df')
    node_features = encode(df)
    #save node features as json file
    #make directory
    save_folder = os.path.join(save_folder, complex_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    with open(os.path.join(save_folder, os.path.splitext(pdb_filename)[0] + r'.json'),'w') as json_fp:
        json.dump(node_features, json_fp)
    os.remove(os.path.join(pdb_folder,os.path.splitext(pdb_filename)[0]+ '_dssp.df'))

if __name__ == '__main__':
    # pdb_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/3_pdb/'
    pdb_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/1_bm4_dataset/2_modify_pdb_result/'
    cluster_file_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/'
    save_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/7_node_feature/native_node_feature/'
    with open("/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/caseID.lst", 'r') as list_fp:
        complex_list = list_fp.read().splitlines()
    for complex in complex_list:
        #1.Compute DSSP of decoy pdb
        #get selected positive and negative lists for this complex
        # with open(os.path.join(cluster_file_folder, complex + '_cluster_decoy_name.txt'),'r') as negative_fp:
        #     negative_list = negative_fp.read().splitlines()
        # with open(os.path.join(cluster_file_folder, complex + '_positive_decoy_name.txt'),'r') as positive_fp:
        #     positive_list = positive_fp.read().splitlines()
        #
        # for decoy in os.listdir(os.path.join(pdb_folder, complex)):
        #     #only pick the decoys in selected list
        #     if os.path.splitext(decoy)[1] == '.pdb' and (decoy in negative_list or decoy in positive_list):
        #         if not os.path.exists(os.path.join(save_folder, complex, os.path.splitext(decoy)[0] + r'.json')):
        #             print('generating dssp for {:s}:{:s}\n'.format(complex, decoy))
        #             create_ss_feature_files(os.path.join(pdb_folder, complex), decoy, complex, save_folder)
        # del negative_list, positive_list


        ##2.compute DSSP of native pdb
        decoy = complex + ".pdb"
        create_ss_feature_files(pdb_folder, decoy, complex, save_folder)

