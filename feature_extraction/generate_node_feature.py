import numpy as np
import csv
import networkx as nx
import matplotlib.pyplot as plt
import os
import os.path
from tqdm import tqdm
import random
import torch.optim as optim
import torch
import math
from sklearn.model_selection import KFold, train_test_split
from pprint import pformat
from collections import defaultdict
# from software.pytorch_DGCNN.util import parse_args, GNNGraph
import json
import pandas as pd
from util_function import read_txt

def convert_residue_to_one_letter(residue):
    """
    Convertd residue to one letter residue

    """

    One2ThreeDict = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
        'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
        'B': 'ASX', 'U': 'SEC', 'Z': 'GLX'
    }

    Three2OneDict = {v: k for k, v in One2ThreeDict.items()}
    AA = Three2OneDict[residue]
    return AA

def convert_residue_to_AAIndex(residue):
    """
    Convertd the raw data to physico-chemical property

    """
    letterDict = {}
    letterDict["A"] = [0.008, 0.134, -0.475, -0.039, 0.181]
    letterDict["R"] = [0.171, -0.361, 0.107, -0.258, -0.364]
    letterDict["N"] = [0.255, 0.038, 0.117, 0.118, -0.055]
    letterDict["D"] = [0.303, -0.057, -0.014, 0.225, 0.156]
    letterDict["C"] = [-0.132, 0.174, 0.070, 0.565, -0.374]
    letterDict["Q"] = [0.149, -0.184, -0.030, 0.035, -0.112]
    letterDict["E"] = [0.221, -0.280, -0.315, 0.157, 0.303]
    letterDict["G"] = [0.218, 0.562, -0.024, 0.018, 0.106]
    letterDict["H"] = [0.023, -0.177, 0.041, 0.280, -0.021]
    letterDict["I"] = [-0.353, 0.071, -0.088, -0.195, -0.107]
    letterDict["L"] = [-0.267, 0.018, -0.265, -0.274, 0.206]
    letterDict["K"] = [0.243, -0.339, -0.044, -0.325, -0.027]
    letterDict["M"] = [-0.239, -0.141, -0.155, 0.321, 0.077]
    letterDict["F"] = [-0.329, -0.023, 0.072, -0.002, 0.208]
    letterDict["P"] = [0.173, 0.286, 0.407, -0.215, 0.384]
    letterDict["S"] = [0.199, 0.238, -0.015, -0.068, -0.196]
    letterDict["T"] = [0.068, 0.147, -0.015, -0.132, -0.274]
    letterDict["W"] = [-0.296, -0.186, 0.389, 0.083, 0.297]
    letterDict["Y"] = [-0.141, -0.057, 0.425, -0.096, -0.091]
    letterDict["V"] = [-0.274, 0.136, -0.187, -0.196, -0.299]
    letterDict["X"] = [0, 0, 0, 0, 0]



    AACategoryLen = 5
    # probMatr = np.zeros(AACategoryLen)
    AA = convert_residue_to_one_letter(residue)

    if AA in letterDict:
        probMatr = np.array(letterDict[AA])

    return probMatr


def get_all_residue_PSSM(complex_name, PSSM_path):
    """
    :param complex name
    :return: read PSSM from PSSM_path and save them into a dictionary
    """
    PSSM_r_path = os.path.join(PSSM_path, complex_name + "_r_u", "pssm")
    PSSM_l_path = os.path.join(PSSM_path, complex_name + "_l_u", "pssm")

    r_chain_name = []
    l_chain_name = []

    for file in os.listdir(PSSM_r_path):
        r_chain_name.append(file.split(".")[1])

    for file in os.listdir(PSSM_l_path):
        l_chain_name.append(file.split(".")[1])

    chain_pos_pssm = defaultdict(dict)
    pssm_len = 21

    for chain_name in r_chain_name:
        path = os.path.join(PSSM_r_path, complex_name + "_r_u." + chain_name + ".pdb.pssm")
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                if line.split()[0] != 'pdbresi':
                    data = line.split()
                    pdbresi = data[0]
                    pdbresn = data[1]

                    pssm = np.zeros(pssm_len)
                    for j in range(pssm_len-1):
                        pssm[j] = 1 / (1 + (2.7182) ** (-int(data[j + 4])))
                    pssm[pssm_len-1] = data[pssm_len + 3]

                    if pdbresi + pdbresn not in chain_pos_pssm[chain_name].keys():
                        chain_pos_pssm[chain_name][pdbresi + pdbresn] = pssm
                line = f.readline()

    for chain_name in l_chain_name:
        path = os.path.join(PSSM_l_path, complex_name + "_l_u." + chain_name + ".pdb.pssm")
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                if line.split()[0] != 'pdbresi':
                    data = line.split()
                    pdbresi = data[0]
                    pdbresn = data[1]

                    pssm = np.zeros(pssm_len)
                    for j in range(pssm_len-1):
                        pssm[j] = 1 / (1 + (2.7182) ** (-int(data[j + 4])))
                    pssm[pssm_len - 1] = data[pssm_len + 3]
                    chain_pos_pssm[chain_name][pdbresi + pdbresn] = pssm
                line = f.readline()

    return chain_pos_pssm


def form_dataframe(decoy_path):
    """
    :param the path of decoy pdb
    :return: transform pdb into dataframe
    """
    atomic_data = []
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
                atomic_data.append(data)

            line = file.readline()

    return atomic_data


if __name__ == '__main__':
    fir_dir = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad"
    pdb_path = "/run/media/fei/easystore/5_new_pdb_6deg/3_pdb/"
    PSSM_path = os.path.join(fir_dir,"2_PSSM")
    node_feature_path = os.path.join(fir_dir,'7_node_feature')
    target_name = os.path.join(fir_dir, 'caseID_part2.lst')

    all_target_name = read_txt(target_name)
    # with open(target_name, "r") as f:
    #     line = f.readline()
    #     while line:
    #         target_in = line.split()
    #         all_target_name.append(target_in[0])
    #         line = f.readline()

    for target in all_target_name:
        chain_pos_pssm = get_all_residue_PSSM(target, PSSM_path)

        decoy_path = os.path.join(pdb_path, target, 'complex.1.pdb')
        atomic_data = form_dataframe(decoy_path)
        dataframe = pd.DataFrame(atomic_data)

        nodeid_initial_list = dataframe["node_id"].tolist()
        nodeid_list = list(set(nodeid_initial_list))
        nodeid_list.sort(key=nodeid_initial_list.index)

        node_features_dic = {}
        for nodeid in nodeid_list:

            chain_id = nodeid[0]
            resi_num = nodeid[1:(len(nodeid)-3)]
            resi_name = nodeid[(len(nodeid)-3):len(nodeid)]

            resi_one_name = convert_residue_to_one_letter(resi_name)

            if str(resi_num) + resi_one_name in chain_pos_pssm[chain_id].keys():
                node_feature_PSSM = chain_pos_pssm[chain_id][str(resi_num) + resi_one_name]
            else:
                node_feature_PSSM = np.zeros(21)
                print("Pssm cannot be found!")

            node_feature_AAIdex = convert_residue_to_AAIndex(resi_name)
            node_feature = np.concatenate([node_feature_AAIdex, node_feature_PSSM], axis=0)

            node_features_dic[nodeid] = node_feature.tolist()

        with open(os.path.join(node_feature_path, target + '.json'), 'w') as f:
            json.dump(node_features_dic, f)

