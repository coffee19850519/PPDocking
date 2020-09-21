from PyG_Experiments.util_function import read_txt
import os
import random
import math
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset
from itertools import product
import numpy as np
class PPDocking(InMemoryDataset):
    def __init__(self,
                 root,
                 subset,
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
def read_decoys_and_categories(label_folder, target_names, subset_number):
    decoy_names = []
    labels = []
    for target in target_names:
        decoy_name_path = os.path.join(label_folder, target, "label." + str(subset_number) + ".txt")
        decoy_name_list = read_txt(decoy_name_path)   ###get decoy name from positive and negtive txt

        if decoy_name_list!="":
            for item in decoy_name_list:
                decoy_name, decoy_label = item.split(" ")
                decoy_names.append(target + "." + decoy_name)
                labels.append(int(decoy_label))
        del decoy_name_list
    return decoy_names, labels


def merge_train_and_val(subset_number, bsize, split, save_merge_pt_folder):
    for fold in range(0, 5):
        target_names = read_txt(os.path.join(target_name_folder, str(fold), split + ".txt"))
        selected_decoy_names, _= read_decoys_and_categories(label_folder, target_names, subset_number)
        random.shuffle(selected_decoy_names)

        total_iters = math.ceil(float(len(selected_decoy_names)) / bsize)

        print("fold {:d}: patch size is {:d}".format(fold, total_iters))
        dataset = PPDocking(root=root,
                            subset=subset_number,
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
                    data = torch.load(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_name)))

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
                       os.path.join(save_merge_pt_folder, '{:d}_subset_{:d}_fold_{:d}_patch_{:s}_data.pt'.
                       format(subset_number, fold, pos, split)))
            del data, slices, data_list

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
    # node_feature = data.x.numpy()
    # node_feature = np.delete(node_feature, 25, axis =1)
    # node_feature[:, 29] = normalize_radius(node_feature[:,29])
    # node_feature[:, 30] = normalize_radius(node_feature[:,30])
    # data.x = torch.from_numpy(node_feature)
    data.edge_attr /= 5.0
    data.interface_count = data.interface_count.float()/torch.max(data.interface_count)


if __name__ == '__main__':
    root = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    target_name_folder = os.path.join(root, "11_dataset_splitting", "new_dataset_split_result")
    independence_pt_dir = "/home/fei/Research/processed_all_data/classification/"
    label_folder = os.path.join(root, "5_decoy_name")

    bsize = 80000
    # split = "train"
    # for subset_number in range(0,5):
    #     merge_train_and_val(subset_number, bsize, split)
    save_merge_pt_folder = "/run/media/fei/Windows1/hanye/classification_patch/"
    for subset_number in range(12,15):
        split = "val"
        merge_train_and_val(subset_number, bsize, split, save_merge_pt_folder)

    # split = "test"
    # max_subset_number = 100
    # merge_test(max_subset_number, split,bsize)

    # selected_idx = ["2C0L.complex.39950.pdb"]
    # for decoy_name in tqdm(selected_idx):
    #     if os.path.exists(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_name))):
    #         try:
    #             data = torch.load(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_name)))
    #         except Exception:
    #             print(os.path.join(independence_pt_dir, '{:s}.pt'.format(decoy_name)) + " has an IO error.")









