from util_function import read_txt
import os
import pandas as pd
from argparse import ArgumentParser
from scipy.spatial.distance import euclidean, pdist, squareform, cdist
import numpy as np
from tqdm import tqdm
import json

def form_dataframe(decoy_path):
    """
    :param the path of decoy pdb
    :return: transform pdb into dataframe
    """
    r_atomic_data = []
    l_atomic_data = []
    with open(decoy_path, 'r') as file:
        line = file.readline().rstrip('\n')  # call readline()
        while line:
            if (line[0:6].strip(" ") == 'ATOM'):
                data = dict()
                ####the position of the pdb columns has been double checked
                data["record_name"] = line[0:6].strip(" ")
                data["serial_number"] = int(line[6:11].strip(" "))
                data["atom"] = line[12:16].strip(" ")
                data["resi_name"] = line[17:20].strip(" ")

                ########repalce special residues using normal residues??????????????????????????????????????????????????
                ###???????????????????????????????
                ### cheque
                #################################
                if data["resi_name"] == "MSE":
                    data["resi_name"] = "MET"
                elif data["resi_name"] == "DVA":
                    data["resi_name"] = "VAL"
                elif data["resi_name"] == "SOC":
                    data["resi_name"] = "CYS"
                elif data["resi_name"] == "XXX":
                    data["resi_name"] = "ALA"
                elif data["resi_name"] == "FME":
                    data["resi_name"] = "MET"
                elif data["resi_name"] == "PCA":
                    data["resi_name"] = "GLU"
                elif data["resi_name"] == "5HP":
                    data["resi_name"] = "GLU"
                elif data["resi_name"] == "SAC":
                    data["resi_name"] = "SER"
                elif data["resi_name"] == "CCS":
                    data["resi_name"] = "CYS"

                data["chain_id"] = line[21].strip(" ")
                data["resi_num"] = int(line[22:26].strip(" "))
                data["x"] = float(line[30:38].strip(" "))
                data["y"] = float(line[38:46].strip(" "))
                data["z"] = float(line[46:54].strip(" "))

                data["node_id"] = data["chain_id"] + str(data["resi_num"]) + data["resi_name"]

                if data["chain_id"] == 'A':
                    r_atomic_data.append(data)
                elif data["chain_id"] == 'B':
                    l_atomic_data.append(data)

            line = file.readline()


    return r_atomic_data, l_atomic_data


def compute_distmat_between_different_proteins(dataframe1, dataframe2, atom_list):
    """
    Computes the pairwise euclidean distances between every atom.

    Design choice: passed in a DataFrame to enable easier testing on
    dummy data.
    """

    #1_compute the distance of any two atoms from dataframe1 and dataframe2.
    if atom_list!=None:
        dataframe1 = dataframe1[dataframe1['atom'].isin(atom_list)]
        dataframe2 = dataframe2[dataframe2['atom'].isin(atom_list)]

    pairse_distmat = cdist(dataframe1[["x", "y", "z"]], dataframe2[["x", "y", "z"]], metric="euclidean")
    pairse_distmat = pd.DataFrame(pairse_distmat)
    pairse_distmat.index = dataframe1.index
    pairse_distmat.columns = dataframe2.index

    return pairse_distmat


def get_interface_nodes(dataframe1, dataframe2, args):
    '''
    :param dataframe1: receptor dataframe
    :param dataframe2: ligand dataframe
    :param args:
    :return: edge tuple and edge feature list
    '''
    edge_tuple_list = []
    edge_feature_list = []
    cut_off_index = []
    cut_off_column = []

    pairse_distmat = compute_distmat_between_different_proteins(dataframe1, dataframe2, args.atomlist)

    #2_obtain residue nodeid by atom index
    if args.atomlist != None:
        ############if compute distance only using atom in the atomlist
        for col in pairse_distmat.columns:
            target_index = pairse_distmat[pairse_distmat[col] <= args.cutoff].index.tolist()
            target_column = [int(col)] * len(target_index)
            cut_off_index += target_index
            cut_off_column += target_column

        resi1 = dataframe1.loc[cut_off_index]["node_id"].values
        resi2 = dataframe2.loc[cut_off_column]["node_id"].values
    else:
        ############if compute distance using all atom
        interacting_atoms = np.where(pairse_distmat <= args.cutoff)  # the distance of any atom is less than 5A
        # interacting_atoms = get_interacting_atoms_(args.cutoff, distmat)
        resi1 = dataframe1.loc[interacting_atoms[0]]["node_id"].values
        resi2 = dataframe2.loc[interacting_atoms[1]]["node_id"].values

    interacting_resis = list(set(resi1.tolist())) + list(set(resi2.tolist()))

    # interacting_resis = set(list(zip(resi1, resi2)) + list(zip(resi2, resi1)))   ###because we need (i1,i2) and (i2,i1) both

    # for index, (i1, i2) in enumerate(interacting_resis):
    #     if i1 != i2:
    #         if i1[0] == 'A':
    #             d1 = get_new_atom_dataframe(dataframe1, i1)
    #             d2 = get_new_atom_dataframe(dataframe2, i2)  ###atom order:C, CA, N, O, SC
    #         elif i2[0] =='A':
    #             d1 = get_new_atom_dataframe(dataframe2, i1)
    #             d2 = get_new_atom_dataframe(dataframe1, i2)  ###atom order:C, CA, N, O, SC
    #
    #         # Computes the pairwise euclidean distances between every atom.
    #         pairse_distmat = cdist(d1[["x", "y", "z"]], d2[["x", "y", "z"]], metric="euclidean")
    #         edge_tuple_list.append((i1, i2))
    #         edge_feature_list.append(pairse_distmat.reshape(-1).tolist())

    return interacting_resis


if __name__ == '__main__':
    parser = ArgumentParser(description="Generating graph for protein-protein docking models")
    parser.add_argument('-atomlist',  default=None, help='atom list used to get edges')
    parser.add_argument('-cutoff', default=5, help='atom distance used to get edges')
    parser.add_argument('-datascale', default="all", help='cluster_result or all')

    args = parser.parse_args()
    # args.atomlist = ['N', 'C', 'CA', 'O']

    file_dir = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    # pdb_folder = os.path.join(file_dir, "3_pdb")
    pdb_folder = "/run/media/fei/easystore/5_new_pdb_6deg/3_pdb/"
    target_name_path = os.path.join(file_dir, "caseID_part2.lst")
    decoy_name_folder = os.path.join(file_dir, "5_decoy_name")
    interface_count_folder = os.path.join(file_dir, "8_interface_residue_count")

    target_name_list = read_txt(target_name_path)
    for target_name in target_name_list:
        print(target_name)
        residue_interface_count = {}
        if args.datascale =="cluster_result":
            decoy_name_list = read_txt(os.path.join(decoy_name_folder, target_name + "_positive_decoy_name.txt"))\
                          + read_txt(os.path.join(decoy_name_folder, target_name + "_cluster_decoy_name.txt"))

            for decoy_name in tqdm(decoy_name_list):
                decoy_path = os.path.join(pdb_folder, target_name, decoy_name)
                r_atomic_data, l_atomic_data = form_dataframe(decoy_path)
                rdataframe = pd.DataFrame(r_atomic_data)
                ldataframe = pd.DataFrame(l_atomic_data)

                interacting_resis = get_interface_nodes(rdataframe, ldataframe, args)
                for resis in interacting_resis:
                    if resis not in residue_interface_count.keys():
                        residue_interface_count[resis] = 1
                    else:
                        residue_interface_count[resis] +=1
        elif args.datascale == "all":
            for i in tqdm(range(54000)):
                decoy_path = os.path.join(pdb_folder, target_name, "complex." + str(i+1)+".pdb")
                r_atomic_data, l_atomic_data = form_dataframe(decoy_path)
                rdataframe = pd.DataFrame(r_atomic_data)
                ldataframe = pd.DataFrame(l_atomic_data)

                interacting_resis = get_interface_nodes(rdataframe, ldataframe, args)
                for resis in interacting_resis:
                    if resis not in residue_interface_count.keys():
                        residue_interface_count[resis] = 1
                    else:
                        residue_interface_count[resis] += 1

        print(target_name + " has finished!")
        with open(os.path.join(interface_count_folder, target_name + '.json'), 'w') as f:
            json.dump(residue_interface_count, f)






