#from ..PyG_Experiments.util_function import read_txt
import os
import random
import math
from tqdm import tqdm
import torch
from itertools import product
import numpy as np
import pandas as pd
from torch_geometric.data import (InMemoryDataset, Data)

class PPDocking(InMemoryDataset):
    def __init__(self,
                 root,
                 fold,
                 patch,
                 mode,
                 split,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        assert split in ['train', 'val', 'test']

        self.fold = fold
        super(PPDocking, self).__init__(root, transform, pre_transform, pre_filter)
        # self.data, self.slices = torch.load(os.path.join(load_folder, '{:d}_subset_{:d}_fold_{:d}_patch_{:s}_data.pt'.format(subset, fold, patch, split)))


    @property
    def raw_file_names(self):
        splits = ['train', 'valid', 'test']
        files = ['feats.npy', 'graph_id.npy', 'graph.json', 'labels.npy']
        return ['{}_{}'.format(s, f) for s, f in product(splits, files)]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_paths(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        pass

    def process(self):

        # for meta_idx in range(200, 300):
        #     ensemble_data_folder = os.path.join(r'/run/media/fei/Project/docking/data/ensemble_data/', str(meta_idx))
        #     if not os.path.exists(ensemble_data_folder):
        #         os.mkdir(ensemble_data_folder)
        #     for i in range(0, 5):
        #         self.data, self.slices = self.collate(
        #             process_classification(r'/run/media/fei/Project/docking/data/8_split_sample/{:d}_train'.format(i)))
        #         torch.save((self.data, self.slices),
        #                    os.path.join(ensemble_data_folder, r'{:d}_train_data.pt'.format(i)))
        #         self.data, self.slices = self.collate(
        #             process_classification(r'/run/media/fei/Project/docking/data/8_split_sample/{:d}_val'.format(i)))
        #         torch.save((self.data, self.slices), os.path.join(ensemble_data_folder, r'{:d}_val_data.pt'.format(i)))
        # for meta_idx in range(301, 500):
        #     ensemble_data_folder = os.path.join(r'/storage/htc/joshilab/hefe/docking/ensemble_data/', str(meta_idx))
        #     if not os.path.exists(ensemble_data_folder):
        #       os.mkdir(ensemble_data_folder)
        #     for i in range(0, 5):
        #         self.data, self.slices = self.collate(process_classification(r'/storage/htc/joshilab/hefe/docking/8_split_sample/{:d}_train'.format(i)))
        #         torch.save((self.data, self.slices), os.path.join(ensemble_data_folder, r'{:d}_train_data.pt'.format(i)))
        #         self.data, self.slices = self.collate(process_classification(r'/storage/htc/joshilab/hefe/docking/8_split_sample/{:d}_val'.format(i)))
        #         torch.save((self.data, self.slices), os.path.join(ensemble_data_folder, r'{:d}_val_data.pt'.format(i)))
                # self.data, self.slices = self.collate(process_classification(r'/home/fei/Desktop/docking/8_split_sample/{:d}_test'.format(i)))
                # torch.save((self.data, self.slices), r'/home/fei/Desktop/docking/data/{:d}_test_data.pt'.format(i))


        # for i in range(0, 1):
        #     self.data, self.slices = self.collate(process_classification(r'/run/media/fei/Project/docking/data/8_split_sample/{:d}_train'.format(i)))
        #     torch.save((self.data, self.slices), r'/run/media/fei/Project/docking/data/classification_hop-1/edge_attr/{:d}_train_data.pt'.format(i))
        #     self.data, self.slices = self.collate(process_classification(r'/run/media/fei/Project/docking/data/8_split_sample/{:d}_val'.format(i)))
        #     torch.save((self.data, self.slices), r'/run/media/fei/Project/docking/data/classification_hop-1/edge_attr/{:d}_val_data.pt'.format(i))
        #     self.data, self.slices = self.collate(process_classification(r'/run/media/fei/Project/docking/data/8_split_sample/{:d}_test'.format(i)))
        #     torch.save((self.data, self.slices), r'/run/media/fei/Project/docking/data/classification_hop-1/edge_attr/{:d}_test_data.pt'.format(i))

        # self.data, self.slices = self.collate(create_test_graph())
        # torch.save((self.data, self.slices), r'/run/media/fei/Project/docking/data/test/test.pt')

        pass


def read_decoys_and_categories(label_folder, target_names):
    decoy_all_list = []
    for target in target_names:

        decoy_name_path = os.path.join(label_folder, target+ "_regression_score_label.csv")
        decoy_name_df = pd.read_csv(decoy_name_path)   ###get decoy name from positive and negtive txt
        for index, rows in decoy_name_df.iterrows():
            decoy_all = {}
            decoy_all['No'] = target + "." + rows["No"]
            decoy_all['iRMS'] = rows["iRMS"]
            decoy_all['LRMS'] = rows["LRMS"]
            decoy_all['fnat'] = rows["fnat"]
            decoy_all['fnonnat'] = rows["fnonnat"]
            decoy_all['normalized_IOU'] = rows["normalized_IOU"]
            decoy_all['IOU'] = rows["IOU"]
            decoy_all['CAPRI'] = rows["CAPRI"]
            decoy_all['DockQ'] = rows["DockQ"]
            decoy_all['IOU_rank'] = rows["IOU_rank"]
            decoy_all['DockQ_rank'] = rows["DockQ_rank"]
            decoy_all['normalized_DockQ'] = rows["normalized_DockQ"]
            decoy_all_list.append(decoy_all)
            del decoy_all
        del decoy_name_df,decoy_name_path

    return decoy_all_list

def read_txt(path):
    list = []
    with open(path,'r') as f:
        for line in f:
            list.append(line.strip())
    return list

def read_topk_decoys_and_categories(label_folder, label_file_suffix, target_names, k):
    decoy_all_list = []
    for target in target_names:

        decoy_name_path = os.path.join(label_folder, target + label_file_suffix)
        decoy_name_df = pd.read_csv(decoy_name_path)  ###get decoy name from positive and negtive txt
        decoy_name_df['num'] = None
        decoy_name_df.num = decoy_name_df.No.map(lambda x: int(str(x).split('.')[1]))
        decoy_name_df.sort_values(by = ['num'], ascending=True, inplace= True)
        #assert k <= len(decoy_name_df)
        for index, rows in decoy_name_df[:k].iterrows():
            decoy_all = {}
            decoy_all['No'] = target + "." + rows["No"]
            decoy_all['iRMS'] = rows["iRMS"]
            decoy_all['LRMS'] = rows["LRMS"]
            decoy_all['fnat'] = rows["fnat"]
            decoy_all['fnonnat'] = rows["fnonnat"]
            decoy_all['IOU'] = rows["IOU"]
            decoy_all['CAPRI'] = rows["CAPRI"]
            decoy_all['DockQ'] = rows["DockQ"]
            decoy_all["normalized_IOU"] = rows ["normalized_IOU"]
            decoy_all["normalized_DockQ"] = rows["normalized_DockQ"]
            decoy_all["IOU_rank"] = rows["IOU_rank"]
            decoy_all['DockQ_rank'] = rows['DockQ_rank']
            decoy_all_list.append(decoy_all)
            del decoy_all
        del decoy_name_df, decoy_name_path
    return decoy_all_list

def merge_train_and_val(bsize, split, save_merge_pt_folder):
    for fold in range(0, 5):
        target_names = read_txt(os.path.join(target_name_folder, str(fold), split + ".txt"))
        # decoy_all_list = read_decoys_and_categories(label_folder, target_names)
        decoy_all_list = read_topk_decoys_and_categories(label_folder, "_regression_score_label.csv",target_names, 500)
        random.shuffle(decoy_all_list)

        total_iters = math.ceil(float(len(decoy_all_list)) / bsize)
        print("fold {:d}: patch size is {:d}".format(fold, total_iters))
        dataset = PPDocking(root=root,
                            fold=fold,
                            patch=0,
                            mode='regression',
                            split=split
                            )

        for pos in range(total_iters):
            selected_idx = decoy_all_list[pos * bsize: (pos + 1) * bsize]
            data_list = []
            for decoy_all in tqdm(selected_idx):
                if os.path.exists(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_all['No']))):
                    try:
                        data = torch.load(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_all['No'])))
                    except:
                        print('{:s}.pt might be corrupted!'.format(decoy_all['No']))
                        with open(os.path.join(save_merge_pt_folder, 'exception.txt'), 'a') as fp:
                            fp.write('{:s}.pt \n'.format(decoy_all['No']))
                        continue

                    new_data = Data(edge_index=data.edge_index,
                                x=data.x,
                                y=data.y,
                                edge_attr=data.edge_attr,
                                interface_count=data.interface_count,
                                # add some specific augments
                                interface_pos=data.interface_pos,
                                complex_name=data.complex_name,
                                decoy_name=data.decoy_name,
                                iRMS = decoy_all["iRMS"],
                                LRMS = decoy_all["LRMS"],
                                fnat = decoy_all["fnat"],
                                fnonnat = decoy_all["fnonnat"],
                                IOU= decoy_all["IOU"],
				                normalized_IOU = decoy_all["normalized_IOU"],
				                normalized_DockQ = decoy_all["normalized_DockQ"],
                                IOU_rank = decoy_all["IOU_rank"],
				                DockQ_rank = decoy_all["DockQ_rank"],
				                CAPRI = decoy_all["CAPRI"],
                                DockQ = decoy_all["DockQ"]
                                    )

                    further_process_features(data)
                    interface_pos = new_data.interface_pos.numpy()
                    if np.sum(interface_pos == True) == 0:
                        print(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_all['No'])) + " has no interface")
                        os.remove(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_all['No'])))
                    else:
                        data_list.append(new_data)
                    del data, new_data
                else:
                    print(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_all['No'])) + "does not exist!")

            data, slices = dataset.collate(data_list)
            del data_list
            torch.save((data, slices),
                       os.path.join(save_merge_pt_folder, '{:d}_fold_{:d}_patch_{:s}_data.pt'.
                       format(fold, pos, split)))
            del data, slices


def read_1_to_n_label_txt(label_folder, target_names, max_subset_number):

    all_decoy_name = []
    for target_name in target_names:
        decoy_name_list = []
        for i in range(1,  max_subset_number + 1):
            label_path = os.path.join(label_folder, target_name, "label.{:d}.txt".format(i))
            label_list = read_txt(label_path)
            for label in label_list:
                decoy_name_list.append(target_name + "." + label.split(" ")[0])
            del label_list
        decoy_name_list = list(set(decoy_name_list))
        all_decoy_name.extend(decoy_name_list)
        del decoy_name_list

    return all_decoy_name


def merge_test(max_subset_number, split, bsize, save_merge_pt_folder):
    for fold in range(0, 5):  #todo remeber to fix
        target_names = read_txt(os.path.join(target_name_folder, str(fold), split + ".txt"))
        selected_decoy_names= read_1_to_n_label_txt(label_folder, target_names, max_subset_number)
        random.shuffle(selected_decoy_names)

        total_iters = math.ceil(float(len(selected_decoy_names)) / bsize)

        print("fold {:d}: patch size is {:d}".format(fold, total_iters))
        dataset = PPDocking(root=root,
                            subset=max_subset_number,
                            fold=fold,
                            patch=0,
                            mode='classification',
                            split=split
                            )

        for pos in range(total_iters):
            selected_idx = selected_decoy_names[pos * bsize: (pos + 1) * bsize]
            data_list = []
            for decoy_name in tqdm(selected_idx):
                if os.path.exists(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_name))):
                    try:
                        data = torch.load(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_name)))
                    except Exception:
                        print(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_name)) + " has an IO error.")

                    # further_process_features(data)
                    interface_pos = data.interface_pos.numpy()
                    if np.sum(interface_pos == True) == 0:
                        print(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_name)) + " has no interface")
                        os.remove(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_name)))
                    else:
                        data_list.append(data)
                    del data

            data, slices = dataset.collate(data_list)
            torch.save((data, slices),
                       os.path.join(save_merge_pt_folder, '{:d}_subset_{:d}_fold_{:d}_patch_{:s}_data.pt'.format(max_subset_number, fold, pos, split)))
            del data, slices, data_list

def normalize_radius(x):
    x = x / (2 * np.pi)
    return x

def further_process_features(data):
    node_feature = data.x.numpy()
    # node_feature = np.delete(node_feature, 25, axis =1)
    # node_feature[:, 29] = normalize_radius(node_feature[:,29])
    # node_feature[:, 30] = normalize_radius(node_feature[:,30])

    node_feature[:, 30] = normalize_radius(node_feature[:,30])
    node_feature[:, 31] = normalize_radius(node_feature[:,31])
    data.x = torch.from_numpy(node_feature)
    data.edge_attr =  1.0 / data.edge_attr
    data.interface_count = data.interface_count.float()/torch.max(data.interface_count)

if __name__ == '__main__':

    root = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    target_name_folder = os.path.join(root, "11_dataset_splitting", "resplit_result")
    independence_pt_dir = "/home/fei/Research/processed_all_data/classification/"
    label_folder = os.path.join(root, "5_decoy_name")
    bsize = 150000
    # split = "train"
    # save_merge_pt_folder = "/storage/htc/joshilab/hefe/docking/regression_patch/"
    # merge_train_and_val(bsize, split, save_merge_pt_folder)

    # bsize = 60000
    split = "val"
    save_merge_pt_folder = "/run/media/fei/Windows/hanye/all_small_regression_patch/"
    merge_train_and_val(bsize, split, save_merge_pt_folder)
    #
    # bsize = 60000
    split = "test"
    save_merge_pt_folder = "/run/media/fei/Windows/hanye/all_small_regression_patch/"
    merge_train_and_val(bsize, split, save_merge_pt_folder)    ###test can use merge_train_and_val too

    # selected_idx = ["2C0L.complex.39950.pdb"]
    # for decoy_name in tqdm(selected_idx):
    #     if os.path.exists(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_name))):
    #         try:
    #             data = torch.load(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_name)))
    #         except Exception:
    #             print(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_name)) + " has an IO error.")









