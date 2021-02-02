from itertools import product
import os
import os.path as osp
import json
import pandas as pd
import torch
import numpy as np
import networkx as nx
import datetime
from glob import glob
from multiprocessing import cpu_count, Pool
from torch_geometric.data import (InMemoryDataset, Data, download_url, Dataset,
                                  extract_zip)
from torch_geometric.data.dataset import to_list
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops, to_undirected
# from PyG_Experiments.sort_pool_regression import train_classification,test_regression,test_classification, train_regression, GATSortPool
import PyG_Experiments.sort_pool_regression
from PyG_Experiments.util_function import load_checkpoint
from tqdm import tqdm
from util_function import read_txt


def process_regression(target_name_file):
    decoy_file_folder = r'/run/media/fei/Project/docking/data/pickle/'

    label_data_folder = r'/run/media/fei/Project/docking/data/label'

    node_feature_folder = r'/run/media/fei/Project/docking/data/node_feature/'

    target_name = read_target_name(target_name_file)

    selected_decoy_names, selected_ious, selected_categories, selected_dockq = read_decoys_and_labels(label_data_folder, target_name, label_cutoff= 0.23, negative_ratio= 0.05)
    #for s, split in enumerate(['train', 'val', 'test']):
    #read decoy details and create
    data_list = []
    for i in range(len(selected_decoy_names)):

        complex_name = selected_decoy_names[i].split('.')[0]

        decoy_path = os.path.join(decoy_file_folder, complex_name, selected_decoy_names[i] + ".pkl")
        graph = nx.read_gpickle(decoy_path)

        interface_g = graph
        #interface_g = get_subgraph(graph, hop= 3)

        interface_index = get_nodes_by_nodetag(interface_g, '1')
        if len(interface_index) == 0:
            print(selected_decoy_names[i] + " has no interface.")
            continue
        del interface_index

        node_features = get_node_feature(graph, complex_name, node_feature_folder)

        if len(node_features) != 0:
            node_features = np.stack(node_features)
        else:
            print(selected_decoy_names[i] + " has no feature.")
            continue

        #interface_g = nx.convert_node_labels_to_integers(interface_g)  #### trasform node name into integers
        interface_g = nx.convert_node_labels_to_integers(graph)

        edge_index = torch.tensor(list(interface_g.edges)).t().contiguous()
        edge_index = edge_index - edge_index.min()
        edge_index, _ = remove_self_loops(edge_index)

        interface_pos = get_interface_node_tag(interface_g, '1')

        data = Data(edge_index=edge_index,
                    x= torch.from_numpy(node_features).to(torch.float),
                    y= torch.tensor(selected_ious[i],dtype= torch.float).view(1, -1),
                    num_nodes = len(interface_g.nodes),
                    dockq = torch.tensor(selected_dockq[i],dtype= torch.float).view(1, -1),
                    interface_pos= torch.tensor(interface_pos,dtype= torch.bool),
                    category = selected_categories[i],
                    complex_name = complex_name,
                    decoy_name= selected_decoy_names[i])

        data_list.append(data)
        del interface_g, node_features, graph
    del selected_decoy_names, selected_ious, selected_categories, selected_dockq
    return data_list


def data_convertor_bridge(args):
    return data_convertor(args[0], args[1], args[2], args[3])

def process_classification_patch(patch_target_names):
    decoy_file_folder = r'/run/media/fei/Project/docking/data/pickle/'

    label_data_folder = r'/run/media/fei/Project/docking/data/label/'

    node_feature_folder = r'/run/media/fei/Project/docking/data/node_feature/'
    # negative_ratio is 0 means extracting all nagetives
    selected_decoy_names, selected_labels = read_decoys_and_categories(label_data_folder, patch_target_names, negative_ratio=0)

    # read decoy details and create
    # start multi-process to accelerate
    # declare process pool
    core_num = cpu_count()
    pool = Pool(core_num)
    # args = [(decoy_file_folder, selected_label, node_feature_folder, selected_decoy_name)
    #         for selected_label, selected_decoy_name in zip(selected_labels, selected_decoy_names)]

    for selected_label, selected_decoy_name in zip(selected_labels, selected_decoy_names):
        pool.apply_async(data_convertor, (decoy_file_folder, selected_label, node_feature_folder, selected_decoy_name))

    # for i, result in tqdm(enumerate(pool.imap_unordered(data_convertor_bridge, args)),desc= 'processing progress:', total= len(selected_decoy_names)):
    #     if result is not None:
    #         data_list.append(result)
    #     del result
    pool.close()
    pool.join()
    del selected_decoy_names, selected_labels, pool
    #return data_list

def data_convertor(decoy_file_folder, selected_label, node_feature_folder, decoy_name):
    #data_list = []
    #for selected_label, decoy_name in zip(selected_labels,decoy_names):
    complex_name = decoy_name.split('.')[0]

    decoy_path = os.path.join(decoy_file_folder, complex_name, decoy_name + ".pkl")
    graph = nx.read_gpickle(decoy_path)

    interface_g = graph
    # interface_g = get_subgraph(graph, hop= 3)
    interface_index = get_nodes_by_nodetag(interface_g, '1')
    if len(interface_index) == 0:
        print(decoy_name + " has no interface.")
        return
    del interface_index

    node_features = get_node_feature(interface_g, complex_name, node_feature_folder)

    if len(node_features) != 0:
        node_features = np.stack(node_features)
    else:
        print(decoy_name + " has no feature.")
        return

    # interface_g = nx.convert_node_labels_to_integers(interface_g)  #### trasform node name into integers
    interface_g = nx.convert_node_labels_to_integers(interface_g)

    edge_index = torch.tensor(list(interface_g.edges)).t().contiguous()
    edge_index = edge_index - edge_index.min()
    edge_index, _ = remove_self_loops(edge_index)
    interface_pos = get_interface_node_tag(interface_g, '1')

    # create pseudo edge attr
    edge_attr = np.random.rand(edge_index.size()[1], 3) * 5

    data = Data(edge_index=edge_index,
                x=torch.from_numpy(node_features).to(torch.float),
                y=torch.tensor(selected_label, dtype=torch.float).view(1, -1),
                edge_attr=torch.from_numpy(edge_attr).to(torch.float),
                # num_nodes = len(interface_g.nodes),
                # add some specific augments
                interface_pos=torch.tensor(interface_pos, dtype=torch.bool),
                complex_name=complex_name,
                decoy_name=decoy_name).to('cpu')
    torch.save(data, r'/run/media/fei/Entertainment/separate_all_test/{:s}.pt'.format(decoy_name))
    del interface_g, node_features, graph, interface_pos
        #pbar.update()
        #print(decoy_name + ' done! \n')

    #return data
   # return data_list

def process_classification(target_name_file):
    decoy_file_folder = r'/run/media/fei/Project/docking/data/pickle/'

    label_data_folder = r'/run/media/fei/Project/docking/data/label/'

    node_feature_folder = r'/run/media/fei/Project/docking/data/node_feature/'

    target_name = read_target_name(target_name_file)

    selected_decoy_names, selected_labels = read_decoys_and_categories(label_data_folder, target_name,negative_ratio= 1)
    #for s, split in enumerate(['train', 'val', 'test']):
    #read decoy details and create
    data_list = []
    for i, decoy_name in enumerate(tqdm(selected_decoy_names)):

        complex_name = decoy_name.split('.')[0]

        decoy_path = os.path.join(decoy_file_folder, complex_name, decoy_name + ".pkl")
        graph = nx.read_gpickle(decoy_path)

        interface_g = graph
        #interface_g = get_subgraph(graph, hop= 3)
        interface_index = get_nodes_by_nodetag(interface_g, '1')
        if len(interface_index) == 0:
            print(decoy_name + " has no interface.")
            continue
        del interface_index


        node_features = get_node_feature(interface_g, complex_name, node_feature_folder)

        if len(node_features) != 0:
            node_features = np.stack(node_features)
        else:
            print(decoy_name + " has no feature.")
            continue

        #interface_g = nx.convert_node_labels_to_integers(interface_g)  #### trasform node name into integers
        interface_g = nx.convert_node_labels_to_integers(interface_g)

        edge_index = torch.tensor(list(interface_g.edges)).t().contiguous()
        edge_index = edge_index - edge_index.min()
        edge_index, _ = remove_self_loops(edge_index)
        interface_pos = get_interface_node_tag(interface_g, '1')

        #create pseudo edge attr
        edge_attr = np.random.rand(edge_index.size()[1], 3) * 5


        data = Data(edge_index=edge_index,
                    x= torch.from_numpy(node_features).to(torch.float),
                    y= torch.tensor(selected_labels[i],dtype= torch.float).view(1, -1),
                    edge_attr= torch.from_numpy(edge_attr).to(torch.float),
                    #num_nodes = len(interface_g.nodes),
                    #add some specific augments
                    interface_pos= torch.tensor(interface_pos,dtype= torch.bool),
                    complex_name = complex_name,
                    decoy_name = decoy_name)

        data_list.append(data)
        del interface_g, node_features, graph, interface_pos
    return data_list

def create_test_graph():
    graph = nx.Graph()
    graph.add_nodes_from([0,1,2])
    graph.add_edges_from([(0,1),(1,2)])
    node_features = []
    for i in range(len(graph.nodes)):
        node_features.append(np.random.rand(2))
    node_features = np.stack(node_features)
    data_list = []

    # interface_g = nx.convert_node_labels_to_integers(interface_g)  #### trasform node name into integers
    #interface_g = nx.convert_node_labels_to_integers(interface_g)
    #graph = graph.to_undirected()
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    edge_index = edge_index - edge_index.min()
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, len(node_features))
    # create pseudo edge attr
    edge_attr = []
    edge_attr.append([0.215, 1.25, 4.56])
    edge_attr.append([0.215, 1.25, 4.56])
    edge_attr.append([2.529, 3.185, 1.187])
    edge_attr.append([2.529, 3.185, 1.187])
    edge_attr = np.stack(edge_attr)
    data = Data(edge_index= edge_index,
                x=torch.from_numpy(node_features).to(torch.float),
                y=torch.tensor(1.0, dtype=torch.float).view(1, -1),
                edge_attr=torch.from_numpy(edge_attr).to(torch.float),
                # num_nodes = len(interface_g.nodes),
                # add some specific augments
                interface_pos=torch.tensor([True, True, False], dtype=torch.bool),
                complex_name=['haha'],
                decoy_name=['haha'])


    data_list.append(data)
    del node_features, graph, edge_attr
    return data_list


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

def read_test_decoys_and_categories(label_folder, target_names):
    decoy_names = []
    labels = []
    for target in target_names:
        decoy_name_path = os.path.join(label_folder, target + ".txt")
        decoy_name_list = read_txt(decoy_name_path)   ###get decoy name from positive and negtive txt

        if decoy_name_list!="":
            for item in decoy_name_list:
                decoy_names.append(item)
        del decoy_name_list
    return decoy_names


def read_all_normalized_label_to_dict(label_folder, target_names, label_name):
    all_dict = {}
    for target_name in target_names:
        target_dict = {}
        df = pd.read_csv(os.path.join(label_folder, target_name+'_boostrapping_regression_score_label.csv'))
        for index, row in df.iterrows():
            target_dict.update({row['No']:row['normalized_' + label_name]})
        all_dict.update({target_name:target_dict})
        del target_dict
    del  target_names
    return all_dict

class PPDocking(InMemoryDataset):
    def __init__(self,
                 root,
                 subset,
                 fold,
                 patch,
                 mode,
                 split,
                 load_folder,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        assert split in ['train', 'val', 'test']

        self.fold = fold
        super(PPDocking, self).__init__(root, transform, pre_transform, pre_filter)
        # if mode == 'classification':
        #     self.data, self.slices = torch.load(os.path.join(load_folder, '{:d}_subset_{:d}_fold_{:d}_patch_{:s}_data.pt'.format(subset, fold, patch, split)))
        # else:
        #     self.data, self.slices = torch.load(os.path.join(load_folder,
        #                                                      '{:d}_fold_{:d}_patch_{:s}_data.pt'.format(
        #                                                          fold, patch, split)))
        #     pass
            # if split == 'train':
            #
            #     target_names = read_txt(os.path.join(root, '5_decoy_name','caseID.lst'))
            #     dict = read_all_normalized_label_to_dict(os.path.join(root, '5_decoy_name'), target_names, 'DockQ')
            #     for i in range(len(self.data.complex_name)):
            #         self.data.DockQ[i] = dict[self.data.complex_name[i]][self.data.decoy_name[i]]
            #     del target_names, dict
            #     torch.save((self.data, self.slices),
            #                os.path.join(load_folder,
            #                             '{:d}_fold_{:d}_patch_{:s}_data.pt'.format(
            #                                 fold, patch, split)))


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


class PPDocking_large_dataset(Dataset):
    def __init__(self, root,
                 target_name_folder,
                 label_folder,
                 fold,
                 subset_number,
                 split='train',
                 mode='classification',
                 transform=None, pre_transform=None):

        self.subset_number = subset_number
        self.fold = fold
        self.mode = mode
        self.split = split
        self.target_name_folder = target_name_folder
        self.label_folder = label_folder
        self.root = root
        self.save_dir = os.path.join("/home/fei/Research/processed_all_data/", mode)
        self.target_names = read_txt(os.path.join(self.target_name_folder, str(self.fold), self.split + ".txt"))
        self.selected_decoy_names = read_test_decoys_and_categories(self.label_folder, self.target_names)


        # self.selected_decoy_names = ['1ACB.complex.179.pdb', '1ACB.complex.214.pdb', '1ACB.complex.220.pdb', '1ACB.complex.237.pdb']
        super(PPDocking_large_dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        # target_names = read_target_name(os.path.join(self.target_name_folder, '{:d}_{:s}'.format(self.fold, self.split)))
        # selected_decoy_names, _, _, _ = read_decoys_and_labels(r'/run/media/fei/Project/docking/data/label' , target_names, label_cutoff=0.23, negative_ratio=1)
        # for i in range(0, len(selected_decoy_names)):
        #     selected_decoy_names[i] += '.pt'
        # return selected_decoy_names
        pass

    @property
    def raw_paths(self):
       pass

    @property
    def raw_dir(self):
        pass

    @property
    def processed_dir(self):
        return osp.join(self.root, 'separate_all_test')

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        return []

    def process(self):

        pass

    def len(self):

        return len(self.selected_decoy_names)
        # return len(glob(osp.join(self.save_dir + r'/{:s}_{:s}_fold{:d}_*.pt'.format(self.mode,
        #                                     self.split, self.fold))))

    def get(self, idx):
        data = torch.load(osp.join(self.save_dir, '{:s}.pt'.format(self.selected_decoy_names[idx])))
        return data


class arguments:
    def __init__(self, num_node_features, num_layers, hidden, dense_hidden, k, interface_only, hop, head, mode, batch_size, fold_num, concat):
        self.num_node_features = num_node_features
        self.num_layers = num_layers
        self.hidden = hidden
        self.dense_hidden = dense_hidden
        self.k =k
        self.interface_only = interface_only
        self.hop = hop
        self.head = head
        self.mode = mode
        self.batch_size = batch_size
        self.fold_num = fold_num
        self.concat = concat


if __name__ == '__main__':
    # process_test()
    # for i in range(0, 5):
    #     test_dataset = PPDocking_large_dataset(root= r'/run/media/fei/Entertainment/',
    #                                            target_name_folder=r'/run/media/fei/Project/docking/data/8_split_sample/',
    #                                            patch_size= 3,
    #                                            fold= i,
    #                                            split='test',
    #                                            mode= 'classification')
    #     del test_dataset
    # subset_number = 1
    # train_dataset = PPDocking_large_dataset(root=r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/',
    #                                         target_name_folder=r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/11_dataset_splitting/new_dataset_split_result/',
    #                                         # patch_size= 3,
    #                                         fold=0,
    #                                         subset_number=subset_number,
    #                                         split='train',
    #                                         mode='classification')
    # train_dataset = PPDocking_large_dataset(root=r'/run/media/fei/Entertainment/',
    #                                        target_name_folder=r'/run/media/fei/Project/docking/data/8_split_sample/',
    #                                        patch_size= 3,
    #                                        fold= 0,
    #                                        split='train',
    #                                        mode= 'classification')

    # test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=8)
    #
    #
    #
    # args = arguments(num_node_features=26, num_layers=3, hidden=32,
    #                  dense_hidden=256, k=100, interface_only=True, hop=-1, head= 2, fold_num= 5, mode= 'classification',
    #                  concat= True, batch_size=128)
    # model_save_folder = r'/run/media/fei/Project/docking/best_model_implementation/GRAT_model/'
    # nfold_results = []
    # train_summary = []
    # all_mean_std = []
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for n in range(0, 5):
        print(str(n) + ' fold starts ... ...')

        train_dataset = PPDocking(root = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/', split= 'train', fold= n,
                                  load_folder= r'/run/media/fei/Windows/hanye/boostrapping_regression_train_patch/', subset=0, patch=0,
                                  mode= 'regression')
        # train_dataset = PPDocking_large_dataset(root=r'/run/media/fei/Entertainment/',
        #                                                                                   target_name_folder=r'/run/media/fei/Project/docking/data/8_split_sample/',
        #                                                                                   patch_size= 3,
        #                                                                                   fold= n,
        #                                                                                   split='test',
        #                                                                                   mode= 'classification')
        del train_dataset
        # val_dataset = PPDocking(root=r'', split='val', fold= n, hop= args.hop, mode= args.mode )
        #
        # # val_dataset = PPDocking_large_dataset(root=r'/run/media/fei/Entertainment/',
        # #                                         target_name_folder=r'/run/media/fei/Project/docking/data/8_split_sample/',
        # #                                         patch_size=3,
        # #                                         fold=n,
        # #                                         split='test',
        # #                                         mode='classification')
        # if args.mode == 'classification':
        #     model, summary = train_classification(train_dataset, val_dataset, args, fold= n,model_save_folder= model_save_folder)#,  mean, std)
        #     train_summary.extend(summary)
        # elif args.mode == 'regression':
        #     mean = train_dataset.data.dockq.mean(dim=0, keepdim=True)
        #     std = train_dataset.data.dockq.std(dim=0, keepdim=True)
        #     train_dataset.data.dockq = (train_dataset.data.dockq - mean) / std
        #     val_dataset.data.dockq = (val_dataset.data.dockq - mean) / std
        #     mean, std = mean.item(), std.item()
        #     model, summary = train_regression(train_dataset, val_dataset, args, fold=n,mean= mean,std= std)
        #     train_summary.extend(summary)
        #     all_mean_std.append({'fold':n, 'mean': mean, 'std': std})
        #
        # else:
        #     print('unsupportive mode! Please check the args')
        #     raise Exception('unsupportive mode! Please check the args')
        #
        # del model, train_dataset, val_dataset,

        #loop to load each patch of test data and do test
        '''
        test_dataset = PPDocking_large_dataset(root= r'/run/media/fei/Entertainment/',
                                               target_name_folder=r'/run/media/fei/Project/docking/data/8_split_sample/',
                                               patch_size= 3,
                                               fold= n,
                                               split='test',
                                               mode= 'classification')
         #PPDocking(root=r'', split='test', fold= n, hop= args.hop,mode= args.mode)

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=6)

        # test_mae = test(model, device, test_loader)
        # print('test mae:' + str(test_mae))

        # load best checkpoint to do test
        model, _, _ = load_checkpoint(os.path.join(model_save_folder, '{:d}fold_{:s}_model.pt'.format(n, args.mode)), GATSortPool(args).to(device))
        if args.mode == 'classification':
            test_loss, test_auc, test_rpc, test_ef, all_complex_names,_, all_scores, all_targets = test_classification(model, device, test_loader)
            print('{:d} fold test_auc: {:.4f}, test_rpc: {:.4f}, test_ef1: {:.4f}'.format(n, test_auc, test_rpc, test_ef1))
            nfold_results.append({'fold': n, 'auc': test_auc, 'rpc':test_rpc, 'ef1': test_ef1 })
            del test_loss, test_auc, test_rpc, test_ef1, all_complex_names, all_scores, all_targets
        elif args.mode == 'regression':
            test_loss, test_mae, test_pcc, test_ef1, _,_,_,_  = test_regression(model, device, test_loader,mean, std)
            print('{:d} fold test_pcc: {:.4f}, test_mae: {:.4f}, test_ef1: {:.4f}'.format(n, test_pcc, test_mae, test_ef1))
            nfold_results.append({'fold': n, 'mae': test_mae, 'pcc': test_pcc, 'ef1': test_ef1})
            del test_loss, test_mae, test_pcc, test_ef1
        del  model, test_dataset, test_loader

    #save training details into csv file
    df_train_summary = pd.DataFrame(train_summary)
    df_train_summary.to_csv(os.path.join(model_save_folder,r'train_summary.csv'), index = False)
    del df_train_summary, train_summary

    if args.mode == 'classification':
        df_nfold_results = pd.DataFrame(nfold_results)
        mean_auc = np.mean(df_nfold_results['auc'])
        mean_rpc = np.mean(df_nfold_results['rpc'])
        mean_ef1 = np.mean(df_nfold_results['ef1'])
        print('{:d} fold test mean_auc: {:.4f}, test mean_rpc: {:.4f}, test mean_ef1: {:.4f}'.format(n+1, mean_auc, mean_rpc, mean_ef1))
        nfold_results.append({'mean_auc': mean_auc, 'mean_rpc': mean_rpc, 'mean_ef1':mean_ef1})
        with open(os.path.join(model_save_folder, '{:d}fold_hop{:d}_{:s}_results.json'.format(n+1, args.hop, args.mode)), 'w') as json_fp:
            json.dump(nfold_results, json_fp)
        del nfold_results, df_nfold_results, args
    else:
        df_nfold_results = pd.DataFrame(nfold_results)
        mean_pcc = np.mean(df_nfold_results['pcc'])
        mean_mae = np.mean(df_nfold_results['mae'])
        mean_ef1 = np.mean(df_nfold_results['ef1'])
        print('{:d} fold test mean_pcc: {:.4f}, test mean_mae: {:.4f}, test mean_ef1: {:.4f}'.format(n + 1, mean_pcc, mean_mae, mean_ef1))
        nfold_results.append({'mean_pcc': mean_pcc, 'mean_mae':mean_mae, 'mean_ef1':mean_ef1})

        #save

        with open(os.path.join(model_save_folder, '{:d}fold_hop{:d}_{:s}_results.json'.format(n+1, args.hop, args.mode)), 'w') as json_fp:
            json.dump(nfold_results, json_fp)
        del nfold_results, df_nfold_results, args

        '''