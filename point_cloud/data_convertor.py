import os
#import Bio.PDB as bio
from rdkit.Chem import MolFromPDBFile
from itertools import product
import numpy as np
import torch, json
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T
import rdkit.Chem as Chem
import pandas as pd
#from tqdm import tqdm
from multiprocessing import cpu_count, Pool

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
CHAIN_LIST = ['A', 'B']

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return fbond + fstereo

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def read_target_name(target_name_path):

    if isinstance(target_name_path, list):
        all_target_name = target_name_path
    else:
        with open(target_name_path, "r") as f:
            all_target_name = f.read().splitlines()
    return all_target_name

def atom_featurization(atom):
    return onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) + \
           onek_encoding_unk(atom.GetExplicitValence(), [0, 1, 2, 3, 4, 5])+ \
           onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) + \
           onek_encoding_unk(atom.GetMonomerInfo().GetChainId(), CHAIN_LIST) +\
           [1 if atom.GetIsAromatic() else 0]

def convert_single_pdb_to_point_cloud(pdb_folder, target, decoy_pdb):
    atom_features = []
    atom_coordinates = []

    decoy = MolFromPDBFile(os.path.join(pdb_folder, target, decoy_pdb))
    if decoy is None:
        raise Exception('no interface exists in ' + decoy_pdb)

    atom_coordinates.extend(decoy.GetConformers()[0].GetPositions())

    for atom_idx in range(0, decoy.GetNumAtoms()):
        atom_features.append(atom_featurization(decoy.GetAtomWithIdx(atom_idx)))

    del decoy
    return atom_coordinates, atom_features



def get_node_feature(folder, node_list, target_name, decoy_number):
    '''
    :param:
    :return: the list of node feature
    '''
    node_features = []
    with open(os.path.join(folder, target_name + '.json')) as json_file:
        node_complex_feature_dic = json.load(json_file)          ###AAidex + PSSM

    # if decoy_number == "native":
    #     decoy_json_name = target_name + ".json"
    # else:
    #     decoy_json_name = os.path.splitext(decoy_number)[0]+ ".json"
    #
    # with open(os.path.join(folder, target_name, decoy_json_name)) as json_file:
    #     node_decoy_feature_dic = json.load(json_file)         ###DSSP

    # for n in node_list:
    #     if n not in node_decoy_feature_dic.keys():
    #         for key in node_decoy_feature_dic.keys():
    #             if key[0:(len(key)-3)] == n[0:(len(n)-3)]:
    #                 node_feature = node_complex_feature_dic[n] + ss_to_index(node_decoy_feature_dic[key])
    #                 node_features.append(node_feature)
    #     else:
    #         try:
    #             node_feature = node_complex_feature_dic[n] + ss_to_index(node_decoy_feature_dic[n])
    #             node_features.append(node_feature)
    #         except Exception:
    #             print(target_name + " " + decoy_number + " " + n + "is not in the DSSP or PSSM feature")
    #
    # del node_complex_feature_dic, node_decoy_feature_dic, node_list
    return node_features


def convert_surface_pdb_to_point_cloud(surface_folder,node_feature_folder, target_name, pdb_file):
    atom_features = []
    atom_coordinates = []
    edge_index = []
    edge_attr = []
    residue_num_list = []
    chain_list = []
    node_features = {}

    exceptions = {'1AVX':{'A':0, 'B':500}, '1BVN':{'A':0,'B':800}, '1CLV':{'A':0,'B':500}}

    with open(os.path.join(node_feature_folder, target_name + '.json')) as json_file:
        node_complex_feature_dic = json.load(json_file)

    for chain_id in ['A', 'B']:
        decoy = MolFromPDBFile(os.path.join(surface_folder, target_name, os.path.splitext(pdb_file)[0]+ chain_id + '.pdb'))
        if chain_id == 'A':
            chainA_atom_num = decoy.GetNumAtoms()
            A_coordinates = decoy.GetConformers()[0].GetPositions()
            atom_coordinates.extend(A_coordinates)
        else:
            B_coordiantes = decoy.GetConformers()[0].GetPositions()
            atom_coordinates.extend(B_coordiantes)
        if decoy is None:
            raise Exception('no interface exists in ' + pdb_file)



        chain_list.append(target_name + '.' + os.path.splitext(pdb_file)[0] + '.' + chain_id)

        # try:
        for atom_idx in range(0, decoy.GetNumAtoms()):
            atom_features.append(atom_featurization(decoy.GetAtomWithIdx(atom_idx)))
            residue_info = decoy.GetAtomWithIdx(atom_idx).GetPDBResidueInfo()
            residue_num = residue_info.GetResidueNumber()
            if target_name in exceptions.keys():
                residue_num = residue_num - exceptions[target_name][chain_id]

            residue_num_list.append(target_name + '.' + os.path.splitext(pdb_file)[0] + '.' + chain_id + '.' + str(residue_info.GetResidueNumber()))
            #if meeting a new residue here, load its
            node_features.update( {target_name + '.' + os.path.splitext(pdb_file)[0] + '.' + chain_id + '.' + str(residue_info.GetResidueNumber()): \
                                    node_complex_feature_dic[chain_id + str(residue_num) + residue_info.GetResidueName()]})
            del residue_info
        # except Exception as e:
        #     print(target_name+ ' '+ str(e))

        for bond in decoy.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if chain_id == 'A':
                edge_index.append((a1, a2))
                edge_attr.append(bond_features(bond))
                edge_index.append((a2, a1))
                edge_attr.append(bond_features(bond))
            else:
                edge_index.append((a1 + chainA_atom_num, a2 + chainA_atom_num))
                edge_attr.append(bond_features(bond))
                edge_index.append((a2 + chainA_atom_num, a1 + chainA_atom_num))
                edge_attr.append(bond_features(bond))
        del decoy

    del node_complex_feature_dic

    return chainA_atom_num, atom_features, edge_index, edge_attr, residue_num_list, atom_coordinates, chain_list, list(node_features.values())

def read_decoy_labels_and_correspondence(label_folder, target_name):

    decoy_name_path = os.path.join(label_folder, target_name + "_regression_score_label.csv")
    decoy_name_df = pd.read_csv(decoy_name_path)  ###get decoy name from positive and negtive txt
    with open(os.path.join(os.path.splitext(decoy_name_path)[0] + '_correspondence.json'), 'r') as fp:
        correspondence_gt = json.load(fp)

    return decoy_name_df, correspondence_gt



# def convert_dataset_to_h5(pdb_folder, fold_num, data_split_folder, save_h5_folder):
#
#     #read fold split results
#     target_name_folder = os.path.join(data_split_folder, "resplit_result")
#
#     for fold_idx in range(0, fold_num):
#         for split in ['train', 'val', 'test']:
#             # read targets
#             # target_names = read_target_name(os.path.join(target_name_folder, fold_idx, dataset+'.txt'))
#             target_names = ['1ACB']
#             h = h5py.File(os.path.join(save_h5_folder, split+str(fold_idx)+'.h5'), mode='w')
#             for target in target_names:
#                 target_gp = h.create_group(target)
#                 # all_atom_coordinates = []
#                 # all_atom_features = []
#                 for decoy_pdb in tqdm(os.listdir(os.path.join(pdb_folder, target))):
#                     if os.path.splitext(decoy_pdb)[1] != '.pdb':
#                         continue
#
#                     target_gp.create_group(target+'.'+ decoy_pdb)
#
#                     atom_coordinates, atom_features = convert_single_pdb_to_point_cloud(pdb_folder, target, decoy_pdb)
#
#                     # all_atom_coordinates.append(atom_coordinates)
#                     # all_atom_features.append(atom_features)
#                     target_gp.create_dataset(name=target+'.'+ decoy_pdb + '.pos', data=atom_coordinates,)
#                     target_gp.create_dataset(name=target+'.'+ decoy_pdb + '.x', data=atom_features)
#                     del atom_coordinates, atom_features

def convert_decoys_in_target_to_pt(target,label_folder, pdb_folder,node_feature_folder, pre_transform, interface_folder = None):

    print(target + ": start!")
    # read decoy labels
    target_lables, correspondence_gt = read_decoy_labels_and_correspondence(label_folder, target)
    data_list = []

    # for decoy_pdb in ['complex.42758.pdb']:
    # for decoy_pdb in tqdm(os.listdir(os.path.join(pdb_folder, target))):
    for decoy_pdb in target_lables.No.tolist():
        try:
            # atom_coordinates, atom_features = convert_single_pdb_to_point_cloud(r'/home/fei/Desktop/', target, decoy_pdb)
            if interface_folder is not None:
                chainA_atom_num, atom_features, edge_index, edge_attr, residue_num_list, atom_coordinates, chain_list, residue_features = \
                    convert_surface_pdb_to_point_cloud(interface_folder, node_feature_folder, target, decoy_pdb)
            else:
                atom_coordinates, atom_features = convert_single_pdb_to_point_cloud(pdb_folder, target,
                                                                                     decoy_pdb)
            decoy_labels = target_lables[target_lables['No'] == decoy_pdb].iloc[0].to_dict()

            category = 1.0 if decoy_labels['CAPRI'] != 0.0 else 0.0

            # A_data = Data(pos=torch.from_numpy(np.array(A_pos, dtype=np.float)))
            # B_data = Data(pos=torch.from_numpy(np.array(B_pos, dtype=np.float)))

            # normalize point coordinates
            # if pre_transform is not None:
            #     A_data = pre_transform(A_data)
            #     B_data = pre_transform(B_data)



                #preprocess correspondence
            full_name_correspondence = []
            for current_correspondence in  correspondence_gt[decoy_pdb]:
                full_name_correspondence.append([
                    target + '.' + os.path.splitext(decoy_pdb)[0] + '.' + current_correspondence[0],
                    target + '.' + os.path.splitext(decoy_pdb)[0] + '.' + current_correspondence[1],
                    current_correspondence[2]
                ])

            data = Data(x=torch.from_numpy(np.array(atom_features, dtype=np.float)),
                        y=torch.tensor(category, dtype=torch.float).view(1, -1),
                        pos=torch.from_numpy(np.array(atom_coordinates, dtype=np.float)),
                        # face= torch.cat((A_data.face, chainA_atom_num + B_data.face),dim= 1),
                        # normal = torch.cat((A_data.norm, B_data.norm), dim= 0),
                        edge_index = torch.tensor(edge_index).t().contiguous(),
                        edge_attr= torch.tensor(edge_attr).to(torch.float),
                        # num_nodes=len(interface_g.nodes),
                        residue_index_list = residue_num_list,
                        dockq=torch.tensor(decoy_labels['DockQ'], dtype=torch.float).view(1, -1),
                        fnat=torch.tensor(decoy_labels['fnat'], dtype=torch.float).view(1, -1),
                        correspondence = full_name_correspondence,
                        # fnat=torch.tensor(decoy_labels['fnat'], dtype=torch.float).view(1, -1),
                        # interface_pos=torch.tensor(interface_pos, dtype=torch.bool),
                        # category=selected_categories[i],
                        residue_features = torch.from_numpy(np.array(residue_features, dtype= np.float)),
                        chain = chain_list,
                        complex_name=target,
                        decoy_name=target + '.' + os.path.splitext(decoy_pdb)[0])

            if pre_transform is not None:
                data = pre_transform(data)
            data_list.append(data)
            del atom_features, data, decoy_labels, edge_index, edge_attr, residue_num_list, atom_coordinates, full_name_correspondence, chain_list, residue_features
        except Exception as e:
            # skip the samples with irregular coordinate format
            print(target + '.' + decoy_pdb + ' has irregular position format error message: ' + str(e))
            # print('bug:' + target)
            continue
            # return []
    del target_lables, correspondence_gt
    print(target + ": end!")
    return data_list

def convert_dataset_to_pt(pdb_folder, label_folder, target_name_folder, node_feature_folder, fold_idx, dataset, interface_folder,
                          pre_transform = None):
    # read targets
    target_names = read_target_name(os.path.join(target_name_folder, str(fold_idx), dataset+'.txt'))
    # target_names = ['1ACB']
    data_list = []
    # target_names = ['1ACB']
    # declare process pool
    # core_num = cpu_count()
    # pool = Pool(core_num)
    # async_results = []
    # h = h5py.File(os.path.join(save_h5_folder, dataset + str(fold_idx) + '.h5'))
    for target in target_names:
        if target == '1BUH' or target == '1JTG':
            continue
        # async_results.append(pool.apply_async(convert_decoys_in_target_to_pt,
        #                                  (target,label_folder, pdb_folder, pre_transform, interface_folder)))
        data_list.extend(convert_decoys_in_target_to_pt(target,label_folder, pdb_folder, node_feature_folder, pre_transform, interface_folder))

    # for res in async_results:
    #     data_list.extend(res.get())
    #
    # del async_results
    # pool.close()
    # pool.join()
    # del pool
    # data, slices = dataset.collate(data_list)
    # torch.save((data, slices),
    #            os.path.join(save_pt_folder, 'fold_{:d}_{:s}_data.pt'.format(fold_idx, split)),_use_new_zipfile_serialization=True)
    # del data_list, data, slices
    return data_list


class PPDocking_point_cloud(InMemoryDataset):
    def __init__(self,
                 root,
                 fold,
                 split,
                 pdb_folder = None,
                 data_split_folder=None,
                 label_folder = None,
                 node_feature_folder = None,
                 interface_folder =None,
                 transform=None,
                 pre_transform= None,
                 pre_filter=None):

        assert split in ['train', 'val', 'test']
        self.current_fold = fold
        self.data_split_folder = data_split_folder
        self.label_folder = label_folder
        self.pdb_folder = pdb_folder
        self.interface_folder = interface_folder
        self.node_feature_folder = node_feature_folder
        self.pre_transform = pre_transform
        super(PPDocking_point_cloud, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(os.path.join(root, self.processed_paths, 'fold_{:d}_{:s}_data.pt'.format(fold, split)))
        # if split == 'train':
        #     path = self.processed_paths[0]
        # elif split == 'val':
        #     path = self.processed_paths[1]
        # elif split == 'test':
        #     path = self.processed_paths[2]
        # elif split == 'trainval':
        #     path = self.processed_paths[3]
        # else:
        #     raise ValueError((f'Split {split} found, but expected either '
        #                       'train, val, trainval or test'))
        # self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['raw.pt']

    @property
    def processed_file_names(self):
        splits = ['train','val', 'test']
        folds = np.arange(0, 1)
        return ['fold_{:s}_{:d}_data.pt'.format(s, f) for s, f in product(splits, folds)]

    @property
    def processed_paths(self):
        return 'processed'

    def download(self):
        pass

    def process(self):
        #declare process pool
        # core_num = cpu_count()
        # pool = Pool(5)

        for fold_idx in range(0, 1):
            # for split in ['val']:
            for split in ['train','val', 'test']:
                # pool.apply_async(self.generate_point_cloud_pt, (fold_idx, split))
                self.generate_point_cloud_pt(fold_idx, split)

        # pool.close()
        # pool.join()
        # del pool

    def generate_point_cloud_pt(self, fold_idx, split):
        if not os.path.exists(os.path.join(self.root, 'processed', 'fold_{:d}_{:s}_data.pt'.format(fold_idx, split))):
            print('start generating fold_{:d}_{:s}_data.pt'.format(fold_idx, split))
            data_list = convert_dataset_to_pt(self.pdb_folder, self.label_folder, self.data_split_folder,self.node_feature_folder,
                                              fold_idx, split, self.interface_folder, self.pre_transform)
            data, slices = self.collate(data_list)
            torch.save((data, slices),
                       os.path.join(self.root, 'processed', 'fold_{:d}_{:s}_data.pt'.format(fold_idx, split)),
                       _use_new_zipfile_serialization=True)
            del data_list, data, slices
            print('fold_{:d}_{:s}_data.pt generated'.format(fold_idx, split))




if __name__ == '__main__':
    pdb_folder = r'/home/fei/Desktop/'
    # data_split_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/11_dataset_splitting/resplit_result/'
    data_split_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/11_dataset_splitting/toy_split/'
    label_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/'
    root = r'/run/media/fei/Windows/hanye/point_cloud_two_chain_in_one/'
    pdb_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/3_pdb/'
    interface_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/14_interface_info/'

    node_feature_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/7_node_feature/'
    # convert_dataset_to_h5(pdb_folder, 1, data_split_folder, save_h5_folder)
    # convert_dataset_to_pt(pdb_folder, 1, data_split_folder, label_folder, save_h5_folder)
    #data, slices = torch.load(os.path.join(save_h5_folder, 'fold_{:d}_{:s}_data.pt'.format(0, 'train')))
    # dataset = PPDocking_point_cloud(root, 1, 'train')

    pre_transforms = T.Compose([
        T.Delaunay(),
        T.GenerateMeshNormals()
    ])

    dataset = PPDocking_point_cloud(root, 0, 'train',
                                    pdb_folder,
                                    data_split_folder,
                                    label_folder,
                                    node_feature_folder,
                                    interface_folder,
                                    pre_transform= pre_transforms)
