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
           [atom.GetIsAromatic()]

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

def convert_chain_surface_pdb_to_point_cloud(surface_folder, target_name, pdb_file):

    atom_features = []
    atom_coordinates = []
    edge_index = []
    edge_attr = []
    residue_num_list = []
    chain_list = []
    for chain_id in ['A', 'B']:

        decoy = MolFromPDBFile(
            os.path.join(surface_folder, target_name, os.path.splitext(pdb_file)[0] + chain_id + '.pdb'))

        if decoy is None:
            raise Exception('no interface exists in ' + pdb_file)
        if chain_id == 'A':
            chainA_atom_num = decoy.GetNumAtoms()
            A_coordinates = decoy.GetConformers()[0].GetPositions()
            atom_coordinates.extend(A_coordinates)
        else:
            B_coordiantes = decoy.GetConformers()[0].GetPositions()
            atom_coordinates.extend(B_coordiantes)



        chain_list.append(target_name + '.' + os.path.splitext(pdb_file)[0] + '.' + chain_id)

        # try:
        for atom_idx in range(0, decoy.GetNumAtoms()):
            atom_features.append(atom_featurization(decoy.GetAtomWithIdx(atom_idx)))
            residue_info = decoy.GetAtomWithIdx(atom_idx).GetPDBResidueInfo()
            residue_num = residue_info.GetResidueNumber()

            residue_num_list.append(target_name + '.' + os.path.splitext(pdb_file)[0] + '.' + chain_id + '.' + str(residue_num))
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

    return atom_coordinates, atom_features, edge_index, edge_attr, residue_num_list, chain_list, chainA_atom_num


def read_decoy_labels_and_correspondence(label_folder, target_name):

    decoy_name_path = os.path.join(label_folder, target_name + "_regression_score_label.csv")
    decoy_name_df = pd.read_csv(decoy_name_path)  ###get decoy name from positive and negtive txt
    decoy_name_df = decoy_name_df[decoy_name_df['fnat'] > 0.05]

    with open(os.path.join(os.path.splitext(decoy_name_path)[0] + '_triple_correspondence.json'), 'r') as fp:
        correspondence_triple = json.load(fp)
    # with open(os.path.join(os.path.splitext(decoy_name_path)[0] + '_positive_correspondence.json'), 'r') as fp:
    #     correspondence_pos = json.load(fp)
    # with open(os.path.join(os.path.splitext(decoy_name_path)[0] + '_negative_correspondence.json'), 'r') as fp:
    #     correspondence_neg = json.load(fp)
    return decoy_name_df, correspondence_triple#, correspondence_pos, correspondence_neg


class PairData(Data):
    def __inc__(self, key, value):
        if key == 'edge_index_A':
            return self.x_A.size(0)
        if key == 'edge_index_B':
            return self.x_B.size(0)
        if key == 'face_A':
            return self.x_A.size(0)
        if key == 'face_B':
            return self.x_B.size(0)
        else:
            return super(PairData, self).__inc__(key, value)

    def __init__(self, edge_index_A = None, x_A= None, y= None, pos_A= None, edge_attr_A= None,residue_num_list_A= None,chain_list_A = None, face_A= None,norm_A= None,
                 edge_index_B= None, x_B= None, pos_B= None,edge_attr_B= None, residue_num_list_B= None, chain_list_B = None, face_B= None, norm_B= None,
                 dockq= None, fnat= None, correspondence= None, complex_name= None, decoy_name= None):
        super(PairData, self).__init__()
        self.edge_index_A = edge_index_A
        self.x_A = x_A
        self.edge_index_B = edge_index_B
        self.x_B = x_B
        self.y = y
        self.pos_A = pos_A
        self.pos_B = pos_B
        self.edge_attr_A = edge_attr_A
        self.edge_attr_B = edge_attr_B
        self.residue_num_list_A = residue_num_list_A
        self.residue_num_list_B = residue_num_list_B
        self.face_A = face_A
        self.face_B = face_B
        self.norm_A = norm_A
        self.norm_B = norm_B
        self.dockq = dockq
        self.fnat = fnat
        self.correspondence = correspondence
        self.complex_name = complex_name
        self.decoy_name = decoy_name
        self.chain_list_A = chain_list_A
        self.chain_list_B = chain_list_B



def convert_decoys_in_target_to_pt(target,label_folder, pdb_folder, pre_transform, interface_folder = None):

    print(target + ": start!")
    # read decoy labels
    target_labels, correspondence_triple= read_decoy_labels_and_correspondence(label_folder, target)


    pair_data_list = []
    # for decoy_pdb in ['complex.42758.pdb']:
    # for decoy_pdb in tqdm(os.listdir(os.path.join(pdb_folder, target))):

    #read native data
    native_atom_coordinates, native_atom_features, native_edge_index, native_edge_attr, native_residue_num_list, native_chain_list, _ = \
        convert_chain_surface_pdb_to_point_cloud(interface_folder, target, 'native')
    native_data = Data(x=torch.from_numpy(np.array(native_atom_features, dtype=np.float)),
                      # y=torch.tensor(category, dtype=torch.float).view(1, -1),
                      pos=torch.from_numpy(np.array(native_atom_coordinates, dtype=np.float)),
                      edge_index=torch.tensor(native_edge_index).t().contiguous(),
                      edge_attr=torch.tensor(native_edge_attr).to(torch.float),
                      residue_num_list=native_residue_num_list,
                      chain_list = native_chain_list)
    if pre_transform is not None:
        native_data = pre_transform(native_data)
    del native_atom_coordinates, native_atom_features, native_edge_index, native_edge_attr, native_residue_num_list, native_chain_list
    for decoy_pdb in target_labels.No.tolist():
        #if there is no correspondence for current decoy, means no overlap between the decoy and native
        if len(correspondence_triple[decoy_pdb]) == 0:
            continue

        try:
            # atom_coordinates, atom_features = convert_single_pdb_to_point_cloud(r'/home/fei/Desktop/', target, decoy_pdb)

            # decoy_labels = target_lables[target_lables['No'] == decoy_pdb].iloc[0].to_dict()
            # for chain_id in ['A', 'B']:
            assert interface_folder is not None

            decoy_atom_coordinates, decoy_atom_features, decoy_edge_index, decoy_edge_attr, \
            decoy_residue_num_list, decoy_chain_list, _ = \
                convert_chain_surface_pdb_to_point_cloud(interface_folder, target, decoy_pdb)

            # category = 1.0 if decoy_labels['CAPRI'] != 0.0 else 0.0
            decoy_data = Data(x=torch.from_numpy(np.array(decoy_atom_features, dtype=np.float)),
                        # y=torch.tensor(category, dtype=torch.float).view(1, -1),
                        pos=torch.from_numpy(np.array(decoy_atom_coordinates, dtype=np.float)),
                        edge_index = torch.tensor(decoy_edge_index).t().contiguous(),
                        edge_attr= torch.tensor(decoy_edge_attr).to(torch.float),
                        residue_num_list = decoy_residue_num_list,
                        chain_list = decoy_chain_list)

            # normalize point coordinates
            if pre_transform is not None:
                decoy_data = pre_transform(decoy_data)

            del decoy_atom_coordinates, decoy_atom_features, decoy_edge_index, decoy_edge_attr, decoy_residue_num_list, decoy_chain_list

            #preprocess correspondence
            full_name_correspondence = []
            for current_correspondence in correspondence_triple[decoy_pdb]:
                full_name_correspondence.append([
                    target + '.' + 'native' + '.' + current_correspondence[0][0],
                    target + '.' + 'native' + '.' + current_correspondence[0][1],
                    target + '.' + os.path.splitext(decoy_pdb)[0] + '.' + current_correspondence[0][0],
                    target + '.' + os.path.splitext(decoy_pdb)[0] + '.' + current_correspondence[0][1],
                    target + '.' + os.path.splitext(decoy_pdb)[0] + '.' + current_correspondence[1][0],
                    target + '.' + os.path.splitext(decoy_pdb)[0] + '.' + current_correspondence[1][1]
                ])

            #generate pair_point_cloud data
            pair_data = PairData(
                edge_index_A = native_data.edge_index, x_A = native_data.x,
                pos_A = native_data.pos, edge_attr_A = native_data.edge_attr,
                chain_list_A= native_data.chain_list,
                residue_num_list_A =  native_data.residue_num_list, face_A = native_data.face, norm_A = native_data.norm,
                edge_index_B = decoy_data.edge_index, x_B = decoy_data.x, pos_B = decoy_data.pos, edge_attr_B = decoy_data.edge_attr,
                chain_list_B = decoy_data.chain_list,
                residue_num_list_B = decoy_data.residue_num_list, face_B = decoy_data.face, norm_B = decoy_data.norm,
                correspondence = full_name_correspondence,
                complex_name = target,
                decoy_name = target + '.' + decoy_pdb)
            pair_data_list.append(pair_data)
            del pair_data, decoy_data, full_name_correspondence

        except Exception as e:
            # skip the samples with irregular coordinate format
            print(target + '.' + decoy_pdb + ' has irregular position format error message: ' + str(e))
            continue
    del target_labels, correspondence_triple, native_data
    print(target + ": end!")
    return pair_data_list

def convert_dataset_to_pt(pdb_folder, label_folder, target_name_folder, fold_idx, dataset, interface_folder,
                          pre_transform = None):
    # read targets
    target_names = read_target_name(os.path.join(target_name_folder, str(fold_idx), dataset+'.txt'))
    data_list = []
    # target_names = ['1ACB']
    # declare process pool
    # core_num = cpu_count()
    # pool = Pool(core_num)
    # async_results = []
    # h = h5py.File(os.path.join(save_h5_folder, dataset + str(fold_idx) + '.h5'))
    for target in target_names:

        # async_results.append(pool.apply_async(convert_decoys_in_target_to_pt,
        #                                  (target,label_folder, pdb_folder, pre_transform, interface_folder)))
        data_list.extend(convert_decoys_in_target_to_pt(target,label_folder, pdb_folder, pre_transform, interface_folder))

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
            # for split in ['val', 'test']:
            for split in ['train','val', 'test']:
                # pool.apply_async(self.generate_point_cloud_pt, (fold_idx, split))
                self.generate_point_cloud_pt(fold_idx, split)

        # pool.close()
        # pool.join()
        # del pool

    def generate_point_cloud_pt(self, fold_idx, split):
        if not os.path.exists(os.path.join(self.root, 'processed', 'fold_{:d}_{:s}_data.pt'.format(fold_idx, split))):
            print('start generating fold_{:d}_{:s}_data.pt'.format(fold_idx, split))
            data_list = convert_dataset_to_pt(self.pdb_folder, self.label_folder, self.data_split_folder,
                                              fold_idx, split, self.interface_folder, self.pre_transform)
            data, slices = self.collate(data_list)
            torch.save((data, slices),
                       os.path.join(self.root, 'processed', 'fold_{:d}_{:s}_data.pt'.format(fold_idx, split)),
                       _use_new_zipfile_serialization=True)
            del data_list, data, slices
            print('fold_{:d}_{:s}_data.pt generated'.format(fold_idx, split))




if __name__ == '__main__':
    # pdb_folder = r'/home/fei/Desktop/'
    data_split_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/11_dataset_splitting/resplit_result/'
    # data_split_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/11_dataset_splitting/toy_split/'
    label_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/'
    root = r'/run/media/fei/Windows/hanye/point_cloud_format/'
    pdb_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/3_pdb/'
    interface_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/14_interface_info/'
    # convert_dataset_to_h5(pdb_folder, 1, data_split_folder, save_h5_folder)
    # convert_dataset_to_pt(pdb_folder, 1, data_split_folder, label_folder, save_h5_folder)
    #data, slices = torch.load(os.path.join(save_h5_folder, 'fold_{:d}_{:s}_data.pt'.format(0, 'train')))
    # dataset = PPDocking_point_cloud(root, 1, 'train')

    pre_transforms = T.Compose([
        # T.NormalizeScale(),
        T.Delaunay(),
        T.GenerateMeshNormals()
    ])

    dataset = PPDocking_point_cloud(root, 0, 'train', pdb_folder,data_split_folder,label_folder, interface_folder, pre_transform= pre_transforms)

    pass