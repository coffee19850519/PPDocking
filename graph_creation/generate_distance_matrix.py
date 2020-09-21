import sys
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import networkx as nx
import tarfile
import os
import os.path
import shutil
from tqdm import tqdm
from argparse import ArgumentParser
from util_function import read_txt
import json
from torch_geometric.data import DataLoader
from torch_geometric.data import (InMemoryDataset, Data, download_url, Dataset,
                                  extract_zip)
import torch
from itertools import product
import os.path as osp
from multiprocessing import cpu_count, Pool
from graph_creation.generate_graph import generate_graph_info, get_interface_edge
# from software.pytorch_DGCNN.main import *
from util_function import write_txt

def form_dataframe(decoy_path, chain_name):
    """
    :param the path of decoy pdb
    :return: transform pdb into dataframe
    """
    atomic_data = []
    with open(decoy_path, 'r') as file:
        line = file.readline().rstrip('\n')  # call readline()
        while line:
            if (line[0:6].strip(" ") == 'ATOM' and chain_name ==line[21].strip(" ")):
                data = dict()
                ####the position of the pdb columns has been double checked
                # data["record_name"] = line[0:6].strip(" ")
                # data["serial_number"] = int(line[6:11].strip(" "))
                data["atom"] = line[12:16].strip(" ")
                data["resi_name"] = line[17:20].strip(" ")


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

                # data["chain_id"] = line[21].strip(" ")
                # data["resi_num"] = int(line[22:26].strip(" "))
                data["x"] = float(line[30:38].strip(" "))
                data["y"] = float(line[38:46].strip(" "))
                data["z"] = float(line[46:54].strip(" "))

                data["node_id"] = line[21].strip(" ") + line[22:26].strip(" ") + line[17:20].strip(" ")
                atomic_data.append(data)
                del data

            line = file.readline()


    dataframe = pd.DataFrame(atomic_data)
    del atomic_data, line

    return dataframe


def ss_to_index(dssp_code):

    #original encoding code
    #ss_symbols = ['H','B','E','G','I','T','S','-']
    #helix (G, H and I), strand (E and B) and loop (S, T, and -)
    #new encoding code: helix-0, strand-1, loop-2
    code_dict = {0:0, 1:1, 2:1, 3:0, 4:0, 5:2, 6:2, 7:2}

    #ss includes 8-type symbols
    ss_one_hot = np.eye(3)[code_dict[dssp_code[0]]]
    dssp_code = ss_one_hot.tolist() + dssp_code[1:]
    del code_dict, ss_one_hot
    return dssp_code


def get_node_feature(folder, node_list, target_name, decoy_number):
    '''
    :param:
    :return: the list of node feature
    '''
    node_features = []
    with open(os.path.join(folder, target_name + '.json')) as json_file:
        node_complex_feature_dic = json.load(json_file)          ###AAidex + PSSM

    if decoy_number == "native":
        decoy_json_name = target_name + ".json"
    else:
        decoy_json_name = os.path.splitext(decoy_number)[0]+ ".json"

    with open(os.path.join(folder, target_name, decoy_json_name)) as json_file:
        node_decoy_feature_dic = json.load(json_file)         ###DSSP

    #todo: encode secondary structure categories in dssp features

    for n in node_list:
        if n not in node_decoy_feature_dic.keys():
            for key in node_decoy_feature_dic.keys():
                if key[0:(len(key)-3)] == n[0:(len(n)-3)]:
                    node_feature = node_complex_feature_dic[n] + ss_to_index(node_decoy_feature_dic[key])
                    node_features.append(node_feature)
        else:
            try:
                node_feature = node_complex_feature_dic[n] + ss_to_index(node_decoy_feature_dic[n])
                node_features.append(node_feature)
            except Exception:
                print(target_name + " " + decoy_number + " " + n + "is not in the DSSP or PSSM feature")

    del node_complex_feature_dic, node_decoy_feature_dic, node_list
    return node_features


def read_GLY_side_chain(GLY_side_chain_path):
    """
    :param the path of GLY_side_chain
    :return: GLY_side_chain dataframe
    """
    atomic_data = []
    with open(GLY_side_chain_path, 'r') as file:
        line = file.readline().rstrip('\n')  # call readline()
        while line:
            if (line[0:6].strip(" ") == 'ATOM'):
                data = dict()
                ####the position of the pdb columns has been double checked
                # data["record_name"] = line[0:6].strip(" ")
                # data["serial_number"] = int(line[6:11].strip(" "))
                data["atom"] = "SC"
                # data["chain_id"] = line[21].strip(" ")
                # data["resi_num"] = int(line[22:26].strip(" "))
                data["x"] = float(line[30:38].strip(" "))
                data["y"] = float(line[38:46].strip(" "))
                data["z"] = float(line[46:54].strip(" "))

                data["node_id"] = line[21].strip(" ") + line[22:26].strip(" ") + line[17:20].strip(" ")
                atomic_data.append(data)
                del data

            line = file.readline()

    dataframe = pd.DataFrame(atomic_data)
    del atomic_data, line
    return dataframe


def get_receptor_graph(receptor_graph_path):
    r_node_list = []
    r_edge_tuple_list = []
    r_edge_feature_list = []
    with open(os.path.join(receptor_graph_path, "node_list.txt"), 'r') as f:
        for line in f:
            r_node_list.append(line.strip("\n"))

    with open(os.path.join(receptor_graph_path, "edge_tuple_list.txt"), 'r') as f:
        for line in f:
            edge_tuple = (line.strip("\n").split(" ")[0], line.strip("\n").split(" ")[1])
            r_edge_tuple_list.append(edge_tuple)
            del edge_tuple

    with open(os.path.join(receptor_graph_path, "edge_feature_list"), 'r') as f:
        for line in f:
            feature_list = line.strip(" \n").split(" ")
            feature_list_float = [float(feature) for feature in feature_list]
            r_edge_feature_list.append(feature_list_float)
            del feature_list_float

    return r_node_list, r_edge_tuple_list, r_edge_feature_list


def process_classification(pdb_folder, GLY_side_chain_folder, interface_count_folder,  node_feature_folder,
                           receptor_graph_folder,  decoy_name, decoy_label):
    involve_atom_list = None
    dis_cut_off = 8.5
    full_decoy_name = decoy_name

    target_name = decoy_name.split(".")[0]
    decoy_name = decoy_name[5:]

    decoy_file_path = os.path.join(pdb_folder, target_name)

    #read receptor graph
    receptor_graph_path = os.path.join(receptor_graph_folder, target_name)
    r_node_list, r_edge_tuple_list, r_edge_feature_list = get_receptor_graph(receptor_graph_path)


    ###################################
    #2.generate graph of ligand

    decoy_complex_path = os.path.join(decoy_file_path, decoy_name)

    rdataframe = form_dataframe(decoy_complex_path, 'A')
    ldataframe = form_dataframe(decoy_complex_path, 'B') ##liand B chain

    ####read GLY SC x, y,z
    GLY_side_chain_path = os.path.join(GLY_side_chain_folder, target_name, "GLY." + decoy_name)
    GLY_dataframe = read_GLY_side_chain(GLY_side_chain_path)

    # r_node_list, r_edge_tuple_list, r_edge_feature_list = generate_graph_info(rdataframe, involve_atom_list, dis_cut_off, GLY_dataframe)

    l_node_list, l_edge_tuple_list, l_edge_feature_list = generate_graph_info(ldataframe, involve_atom_list, dis_cut_off,  GLY_dataframe)
    interface_edge_tuple_list, interface_feature_list = get_interface_edge(rdataframe, ldataframe,involve_atom_list, dis_cut_off, GLY_dataframe)######## compute interface edges

    #generate interface position
    node_list = r_node_list + l_node_list
    interface_pos_list = [False for i in range(len(node_list))]
    for edge_tuple in interface_edge_tuple_list:    ##generate interface position
        i1 = node_list.index(edge_tuple[0])
        i2 = node_list.index(edge_tuple[1])
        interface_pos_list[i1] = True
        interface_pos_list[i2] = True
    # generate interface position

    #generate interface count
    '''
    interface_count_list = []
    interface_count_path = os.path.join(interface_count_folder, target_name + ".json")
    a_have = 0
    b_not_have = 0
    with open(interface_count_path) as json_file:
        interface_count_dic = json.load(json_file)  ###DSSP
    for node in node_list:
        if node in interface_count_dic.keys():
            interface_count_list.append(interface_count_dic[node])
            a_have +=1
        else:
            interface_count_list.append(0)
            b_not_have +=1
    
    # print(str(a_have) + "'s interface count is more than 0; " + str(b_not_have) + "'s interface count equals to 0; ")
    '''
    # generate interface count

    #change node index into integer
    edge_tuple_list = r_edge_tuple_list + l_edge_tuple_list + interface_edge_tuple_list
    edge_feature_list = r_edge_feature_list + l_edge_feature_list + interface_feature_list

    new_edge_tuple_list = []       ####change the nodeid of edge tuple into integer
    for edge_tuple in edge_tuple_list:
        i1 = node_list.index(edge_tuple[0])
        i2 = node_list.index(edge_tuple[1])
        new_edge_tuple_list.append((i1, i2))

    # change node index into integer

    #extract node feature
    # node_feature_list = get_node_feature(node_feature_folder, node_list, target_name, decoy_name)
    # extract node feature

    edge_index = torch.tensor(new_edge_tuple_list).t().contiguous()

    # if len(node_feature_list) != 0:
    #     node_features = np.stack(node_feature_list)
    # else:
    #     print(target_name + decoy_name + " has no feature.")

    data = Data(edge_index=edge_index,
                x=None,
                y=None,
                edge_attr=torch.from_numpy(np.array(edge_feature_list)).to(torch.float),
                # interface_count= torch.from_numpy(np.array(interface_count_list)).to(torch.int),
                # add some specific augments
                interface_pos=torch.tensor(interface_pos_list, dtype=torch.bool),
                complex_name=target_name,
                decoy_name=decoy_name)

    if not os.path.exists(r'/home/fei/Research/distance_matrix/{:s}'.format(str(dis_cut_off))):
        os.mkdir(r'/home/fei/Research/distance_matrix/{:s}'.format(str(dis_cut_off)))

    torch.save(data, r'/home/fei/Research/distance_matrix/{:s}/{:s}.pt'.format(str(dis_cut_off), full_decoy_name))
    print(full_decoy_name)
    del data, r_node_list, r_edge_tuple_list, r_edge_feature_list,interface_edge_tuple_list, \
        interface_feature_list,node_list,interface_pos_list,edge_tuple_list,\
        edge_feature_list,new_edge_tuple_list,edge_index, decoy_label, \
        target_name, decoy_name,rdataframe, ldataframe, GLY_dataframe
    # data_list.append(data)

    # print("Decoy pdb has been processed!")


        #####2.generate node and edge info of native pdb
        # native_r_path = os.path.join(native_pdb_folder, target_name + "_r_b.pdb")
        # native_l_path = os.path.join(native_pdb_folder, target_name + "_l_b.pdb")
        # native_label = "1"
        #
        # rdataframe = form_dataframe(native_r_path, 'A')
        #
        # ####read GLY SC x, y,z
        # GLY_side_chain_path = os.path.join(GLY_side_chain_folder, "native_GLY_sidechain", target_name, "GLY." + target_name + '.pdb')
        # GLY_dataframe = read_GLY_side_chain(GLY_side_chain_path)
        #
        # r_node_list, r_edge_tuple_list, r_edge_feature_list = generate_graph_info(rdataframe,involve_atom_list, dis_cut_off,  GLY_dataframe)
        #
        # ldataframe = form_dataframe(native_l_path, 'B')
        #
        # l_node_list, l_edge_tuple_list, l_edge_feature_list = generate_graph_info(ldataframe, involve_atom_list, dis_cut_off,  GLY_dataframe)
        #
        # interface_edge_tuple_list, interface_feature_list = get_interface_edge(rdataframe, ldataframe,
        #                                                                        involve_atom_list, dis_cut_off,  GLY_dataframe)  ######## compute interface edges
        #
        # node_list = r_node_list + l_node_list
        #
        # interface_pos_list = [False for i in range(len(node_list))]
        # for edge_tuple in interface_edge_tuple_list:
        #     i1 = node_list.index(edge_tuple[0])
        #     i2 = node_list.index(edge_tuple[1])
        #     interface_pos_list[i1] = True
        #     interface_pos_list[i2] = True
        #
        # # generate interface count
        # interface_count_list = []
        # a_have = 0
        # b_not_have = 0
        # # interface_count_path = os.path.join(interface_count_folder, target_name + ".json")
        # # with open(interface_count_path) as json_file:
        # #     interface_count_dic = json.load(json_file)  ###DSSP
        # for node in node_list:
        #     if node in interface_count_dic.keys():
        #         interface_count_list.append(interface_count_dic[node])
        #         a_have +=1
        #     else:
        #         interface_count_list.append(0)
        #         b_not_have +=1
        # print(str(a_have) + "'s interface count is more than 0; " + str(b_not_have) +"'s interface count equals to 0; ")
        # # generate interface count
        #
        #
        # edge_tuple_list = r_edge_tuple_list + l_edge_tuple_list + interface_edge_tuple_list
        # edge_feature_list = r_edge_feature_list + l_edge_feature_list + interface_feature_list
        #
        # # node_index_list = list(node_dic.keys())
        # new_edge_tuple_list = []
        # for edge_tuple in edge_tuple_list:
        #     i1 = node_list.index(edge_tuple[0])
        #     i2 = node_list.index(edge_tuple[1])
        #     new_edge_tuple_list.append((i1, i2))
        # del edge_tuple_list
        #
        # ###3 node feature extraction
        # naive_node_feature_folder = os.path.join(file_dir, "7_node_feature", "native_node_feature")
        # decoy_name="native"
        # node_feature_list = get_node_feature(naive_node_feature_folder, node_list, target_name, decoy_name)
        # print("Native pdb has been processed!")
        #
        # edge_index = torch.tensor(new_edge_tuple_list).t().contiguous()
        #
        # if len(node_feature_list) != 0:
        #     node_features = np.stack(node_feature_list)
        # else:
        #     print(target_name + item + " has no feature.")
        #     continue
        #
        # data = Data(edge_index=edge_index,
        #             x=torch.from_numpy(node_features).to(torch.float),
        #             y=torch.tensor(float(native_label), dtype=torch.float).view(1, -1),
        #             edge_attr=torch.from_numpy(np.array(edge_feature_list)).to(torch.float),
        #             interface_count=torch.from_numpy(np.array(interface_count_list)).to(torch.int),
        #             # num_nodes = len(interface_g.nodes),
        #             # add some specific augments
        #             interface_pos=torch.tensor(interface_pos_list, dtype=torch.bool),
        #             complex_name=target_name,
        #             decoy_name=decoy_name)
        #
        # data_list.append(data)


def read_target_name(target_name_path):

    if isinstance(target_name_path, list):
        all_target_name = target_name_path
    else:
        with open(target_name_path, "r") as f:
            all_target_name = f.read().splitlines()
    return all_target_name


def read_decoys_and_categories(label_folder, target_names, subset_number):
    decoy_names = []
    labels = []

    existing_decoy_names = []
    for file in os.listdir(r'/home/fei/Research/distance_matrix/8.5/'):
        existing_decoy_names.append(os.path.splitext(file)[0])

    for target in target_names:
        regression_decoy_name_path = os.path.join(label_folder, target + "_regression_score_label.csv")
        regression_decoy_name_dataframe = pd.read_csv(regression_decoy_name_path)

        decoy_name_list = list(regression_decoy_name_dataframe["No"])
        CAPRI_label_list = list(regression_decoy_name_dataframe["CAPRI"])
        # decoy_name_list = read_txt(regression_decoy_name_path)   ###get decoy name from positive and negtive txt

        # previous_decoy_name_list = []
        # if subset_number > 1:
        #     for i in range(subset_number-1):
        #         decoy_name_path = os.path.join(label_folder, target, "label." + str(i+1) + ".txt")
        #         previous = read_txt(decoy_name_path)
        #         previous_decoy_name_list.extend(previous)
        #         previous_decoy_name_list = list(set(previous_decoy_name_list))

        # inter_list = list(set(decoy_name_list) & set(previous_decoy_name_list))
        # new_decoy_list = set(decoy_name_list).difference(set(previous_decoy_name_list))
        #new_decoy_list = list(set(decoy_name_list).difference(set(previous_decoy_name_list)))
        if decoy_name_list!="":
            for i in range(len(decoy_name_list)):
                decoy_name = decoy_name_list[i]
                decoy_label = CAPRI_label_list[i]

                if target + "." + decoy_name not in existing_decoy_names:
                    decoy_names.append(target + "." + decoy_name)
                    labels.append(int(decoy_label))
        del regression_decoy_name_dataframe

    return decoy_names, labels


def process_classification_patch(fir_dir, selected_labels, selected_decoy_names):


    # decoy_file_folder = r'/run/media/fei/Project/docking/data/pickle/'

    #label_data_folder = osp.join(fir_dir, "5_decoy_name")

    node_feature_folder = osp.join(fir_dir, "7_node_feature/")

    pdb_folder = osp.join(fir_dir, "3_pdb")

    # decoy_name_folder = os.path.join(file_dir, "5_decoy_name")
    native_pdb_folder = osp.join(fir_dir, "1_bm4_dataset", "2_modify_pdb_result")
    GLY_side_chain_folder = osp.join(fir_dir, "9_GLY_sidechain")
    interface_count_folder = osp.join(fir_dir, "8_interface_residue_count")
    receptor_graph_folder = osp.join(fir_dir, "12_receptor_graph")

    # read decoy details and create
    # start multi-process to accelerate
    # declare process pool
    core_num = cpu_count()
    pool = Pool(core_num)

    # selected_labels = [0]
    # selected_decoy_names = ['7CEI.complex.17314.pdb']

    for selected_label, selected_decoy_name in zip(selected_labels, selected_decoy_names):
        pool.apply_async(process_classification, (pdb_folder, GLY_side_chain_folder,
                        interface_count_folder,  node_feature_folder, receptor_graph_folder,
                                selected_decoy_name, selected_label))

    pool.close()
    pool.join()
    del pool



class PPDocking_large_dataset(Dataset):
    def __init__(self, root,
                 target_name_folder,
                 #patch_size,
                 fold,
                 subset_number,
                 mode,
                 split='train', transform=None, pre_transform=None):

        self.fold = fold
        self.subset_number = subset_number
        self.mode = mode
        self.split = split
        self.target_name_folder = target_name_folder
        #self.patch_size = patch_size
        self.root = root
        #self.save_dir = os.path.join(root,  'processed_all_test')

        #generate file list according to split, mode, current_fold
        #target_names = read_target_name(os.path.join(self.target_name_folder, str(self.fold), str(self.split) + ".txt"))

        # negative_ratio is 0 means extracting all nagetives
        # self.selected_decoy_names, self.selected_labels = read_decoys_and_categories(target_name_folder, target_names, subset_number)

        super(PPDocking_large_dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return

    @property
    def processed_file_names(self):
        pass

    @property
    def raw_paths(self):
       pass

    @property
    def raw_dir(self):
        pass

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed', self.mode)
        #return

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        return ['train.pt', 'val.pt', 'test.pt']

    def process(self):
        #read all decoy_names and their labels

        # read all complex names
        # with open(os.path.join(self.root, "caseID.lst"), 'r') as complex_name_fp:
        #     all_complex_names = complex_name_fp.read().splitlines()

        all_complex_names = ['1T6B', '1EWY', '2VDB', '1AY7', '1ACB', '1XQS', '1GHQ', '1JIW', '1FFW','1SYX','1I2M']
        label_folder = os.path.join(self.root, "5_decoy_name")
        all_decoy_names, all_labels = read_decoys_and_categories(label_folder, all_complex_names, self.subset_number)



        # read their selected decoy names
        # all_decoy_names = []
        # all_labels = []
        # for complex_name in all_complex_names:
        #     with open(r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/{:s}_cluster_decoy_name.txt'.format(complex_name), 'r') as negative_fp:
        #         negatives = negative_fp.read().splitlines()
        #         for negative in negatives:
        #             all_decoy_names.append(complex_name + '.' +  negative)
        #         all_labels.extend(np.zeros(len(negatives), np.int))
        #     with open(r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/{:s}_positive_decoy_name.txt'.format(complex_name), 'r') as positive_fp:
        #         positives = positive_fp.read().splitlines()
        #         for positive in positives:
        #             all_decoy_names.append(complex_name + '.' + positive)
        #         all_labels.extend(np.ones(len(positives), np.int))

        process_classification_patch(self.root, all_labels, all_decoy_names)

        del all_complex_names, all_labels,all_decoy_names
        # all_file_list = self.raw_file_names
        # slice, last = divmod(len(all_file_list), self.patch_size)
        #
        # #this loop controls data batch split for fitting memory size
        # for patch_idx in range(0, slice + 1):
        #     targets_in_patch = all_file_list[patch_idx * self.patch_size : (patch_idx + 1) * self.patch_size]
        #
        #     #make the process in parallel
        #
        #     # Read data from `raw_path`.
        #     data = process_classification_patch(self.root, targets_in_patch, self.subset_number)
        #     torch.save(data, self.save_dir + r'/{:s}_{:s}_fold{:d}_{:d}.pt'.format(self.mode,
        #                                     self.split, self.fold, int(patch_idx / self.patch_size)))
        #     del data, targets_in_patch
        #
        # # deal with last patch
        # if len(all_file_list) > self.patch_size and last != 0:
        #     #pick last patch targets
        #     targets_in_patch = all_file_list[ -last :]
        #     # Read data from `raw_path`.
        #     data = process_classification_patch(targets_in_patch)
        #     torch.save(data, self.save_dir + r'/{:s}_{:s}_fold{:d}_{:d}.pt'.format(self.mode,
        #                                     self.split, self.fold, 1 + int(len(all_file_list) / self.patch_size)))
        #     del data, targets_in_patch
        # del all_file_list

    def len(self):
        return len(self.selected_labels)

    def get(self, idx):
        data = torch.load(self.processed_dir + r'/{:s}.pt'.format(self.selected_decoy_names[idx]))
        return data


if __name__ == '__main__':
    ######################
    ###generate pt with different cutoff
    ######################

    # n = 5
    #subset_number = 1 # start from 1
    # mode = "classification"
    # train = PPDocking(root="/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/", split='train', fold= n, mode= mode, subset_number = subset_number)
    subset_number=1
    train_dataset = PPDocking_large_dataset(root=r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/',
                                        target_name_folder=r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/11_dataset_splitting/new_dataset_split_result/',
                                        # patch_size= 3,
                                        fold=0,
                                        subset_number=subset_number,
                                        split='train',
                                        mode='classification')
    del train_dataset


    #process_test()


    # subset_number = 1    ##1...100
    # n = 5
    # for i in range(0, n):
    #     train_dataset = PPDocking_large_dataset(root= r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/',
    #                                            target_name_folder=r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/11_dataset_splitting/new_dataset_split_result/',
    #                                            #patch_size= 3,
    #                                            fold= i,
    #                                            subset_number= subset_number,
    #                                            split= 'train',
    #                                            mode= 'classification')
    #     del train_dataset


    # current_fold = 1
    # process_classification(current_fold, "train")
    # process_classification(current_fold, "val")
    # process_classification(current_fold, "test")
