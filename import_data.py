from itertools import product
import os
import os.path as osp
import json
import pandas as pd
import torch
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from torch_geometric.data import (InMemoryDataset, Data, download_url, extract_zip)
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
from sort_pool_regression import train_classification,test_regression,test_classification, train_regression, GATSortPool
from util_function import load_checkpoint
from tqdm import tqdm

def read_target_name(target_name_path):

    if isinstance(target_name_path, list):
        all_target_name = target_name_path
    else:
        with open(target_name_path, "r") as f:
            line = f.readline()
            all_target_name = []
            while line:
                target_in = line.split()
                all_target_name.append(target_in[0])
                line = f.readline()
    return all_target_name

def read_decoys_and_labels(label_folder, target_names, label_cutoff, negative_ratio):
    selected_decoy_names = []
    selected_ious = []
    selected_categories = []
    selected_dockq = []
    for target in target_names:
        all_target_labels = pd.read_csv(os.path.join(label_folder, target + '_score_label.csv'))
        assert 'IOU' and 'No' and 'DockQ' in all_target_labels.columns
        target_decoys = all_target_labels.loc[all_target_labels['DockQ'] >= label_cutoff]
        selected_decoy_names = np.concatenate([selected_decoy_names, target_decoys['No']], axis=0)
        selected_ious = np.concatenate([selected_ious, target_decoys['IOU']], axis=0)
        selected_categories = np.concatenate([selected_categories, np.ones_like(selected_decoy_names)], axis=0)
        selected_dockq = np.concatenate([selected_dockq, target_decoys['DockQ']], axis=0)

        random_decoys = (all_target_labels.loc[all_target_labels['DockQ'] < label_cutoff]).sample(n= int(len(target_decoys) * negative_ratio), axis= 0)
        selected_decoy_names = np.concatenate([selected_decoy_names, random_decoys['No']], axis=0)
        selected_ious = np.concatenate([selected_ious, random_decoys['IOU']], axis=0)
        selected_categories = np.concatenate([selected_categories, np.zeros_like(selected_decoy_names)], axis=0)
        selected_dockq = np.concatenate([selected_dockq, random_decoys['DockQ']], axis=0)

        del target_decoys, all_target_labels,random_decoys
    return selected_decoy_names, selected_ious, selected_categories, selected_dockq

def read_decoys_and_categories(label_folder, target_names, high_cutoff = 0.23, negative_ratio = 1):
    decoy_names = []
    labels = []
    for target in target_names:
        all_target_labels = pd.read_csv(os.path.join(label_folder, target + '_score_label.csv'))
        assert 'DockQ' and 'No' in all_target_labels.columns
        positive_decoys = all_target_labels.loc[all_target_labels['DockQ'] >= high_cutoff]
        positive_decoy_names = positive_decoys['No']
        positive_labels = np.ones_like(positive_decoy_names)
        decoy_names.extend(positive_decoy_names)
        labels.extend(positive_labels)
        #negative_decoys = (all_target_labels.loc[all_target_labels['DockQ'] < high_cutoff]).sample(n= int(len(positive_decoys) * negative_ratio), axis= 0)
        negative_decoys = all_target_labels.loc[all_target_labels['DockQ'] < high_cutoff]
        negative_decoy_names = negative_decoys['No']
        nagative_labels = np.zeros_like(negative_decoy_names)
        decoy_names.extend(negative_decoy_names)
        labels.extend(nagative_labels)
        del positive_decoys, all_target_labels,negative_decoys, positive_labels,nagative_labels, positive_decoy_names, negative_decoy_names
    return decoy_names, labels


def get_nodes_by_nodetag(graph, tag):
    resis = []
    for n in graph.nodes:
        if graph.node[n]["surface_tag"] == tag:
            if n not in resis:
                resis.append(n)
    return resis

def get_subgraph(graph, hop):
    previous_node_list = get_nodes_by_nodetag(graph, "1")
    all_node_list = previous_node_list.copy()

    for i in range(hop):
        hop_node_list = []
        for node in previous_node_list:
            for node_nei in graph.neighbors(node):
                if node_nei not in hop_node_list:
                    hop_node_list.append(node_nei)
                if node_nei not in all_node_list:
                    all_node_list.append(node_nei)
        previous_node_list = hop_node_list
    subgraph = nx.subgraph(graph, all_node_list)
    #subgraph = generate_subgraph_from_graph(graph, all_node_list)
    del previous_node_list, all_node_list
    return subgraph

def get_node_feature(g, complex_name, node_feature_path):
    node_features = []

    with open(os.path.join(node_feature_path, complex_name + '.json'),) as json_file:  # todo
        node_feature_dic = json.load(json_file)

    for n in g.nodes():
        node_features.append(node_feature_dic[n])
    del node_feature_dic
    return node_features

def get_interface_node_tag(graph, tag):
    interface_tag = []
    for n in graph.nodes:
        if graph.node[n]["surface_tag"] == tag:
            interface_tag.append(True)
        else:
            interface_tag.append(False)
    return interface_tag



class PPDocking(InMemoryDataset):
    def __init__(self,
                 root,
                 fold,
                 hop,
                 mode,
                 split='train',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        assert split in ['train', 'val', 'test']

        super(PPDocking, self).__init__(root, transform, pre_transform, pre_filter)

        if split == 'train':
            self.data, self.slices = torch.load(r'./example/data/{:d}_train_data.pt'.format(fold))
        elif split == 'val':
            self.data, self.slices = torch.load(r'/example/data/{:d}_val_data.pt'.format(fold))
        elif split == 'test':
            self.data, self.slices = torch.load(r'/example/data/{:d}_classification_test.pt'.format(fold))
            # self.data, self.slices = torch.load(r'/home/chenyb/dock/classification_hop1/{:d}_test_data.pt'.format(fold))

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
        pass
        # for i in range(0, 1):
            # self.data, self.slices = self.collate(process_classification(r'/home/fei/Desktop/docking/8_split_sample/{:d}_train'.format(i)))
            # torch.save((self.data, self.slices), r'/home/fei/Desktop/docking/data/{:d}_train_data.pt'.format(i))
            # self.data, self.slices = self.collate(process_classification(r'/home/fei/Desktop/docking/8_split_sample/{:d}_val'.format(i)))
            # torch.save((self.data, self.slices), r'/home/fei/Desktop/docking/data/{:d}_val_data.pt'.format(i))
            # self.data, self.slices = self.collate(process_classification(r'/home/chenyb/dock/data/split_sample/{:d}_test'.format(i)))
            # torch.save((self.data, self.slices), r'/home/chenyb/dock/data/{:d}_test_data.pt'.format(i))

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

def process_classification(target_name_file):
    decoy_file_folder = r'/home/chenyb/dock/data/pickle/'

    label_data_folder = r'/home/chenyb/dock/data/label/'

    node_feature_folder = r'/home/chenyb/dock/data/node_feature/'

    target_name = read_target_name(target_name_file)

    selected_decoy_names, selected_labels = read_decoys_and_categories(label_data_folder, target_name, negative_ratio=15)
    #for s, split in enumerate(['train', 'val', 'test']):
    #read decoy details and create
    data_list = []
    for i, decoy_name in enumerate(tqdm(selected_decoy_names)):

        complex_name = decoy_name.split('.')[0]

        decoy_path = os.path.join(decoy_file_folder, complex_name, decoy_name + ".pkl")
        graph = nx.read_gpickle(decoy_path)

        interface_g = graph
        interface_g = get_subgraph(graph, hop= 3)
        interface_index = get_nodes_by_nodetag(interface_g, '1')
        # if len(interface_index) == 0:
        #     print(decoy_name + " has no interface.")
        #     continue
        # del interface_index


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
        # interface_pos = get_interface_node_tag(interface_g, '1')

        data = Data(edge_index=edge_index,
                    x= torch.from_numpy(node_features).to(torch.float),
                    y= torch.tensor(selected_labels[i],dtype= torch.float).view(1, -1),
                    num_nodes = len(interface_g.nodes),
                    # interface_pos= torch.tensor(interface_pos,dtype= torch.bool),
                    complex_name = complex_name,
                    decoy_name = decoy_name)

        data_list.append(data)
        # del interface_g, node_features, graph, interface_pos
        del interface_g, node_features, graph
    return data_list

class arguments:
    def __init__(self, num_node_features, num_layers, hidden, dense_hidden, k, interface_only, hop, head, mode, batch_size, fold_num):
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

if __name__ == '__main__':
    #process_test()

    args = arguments(num_node_features=36, num_layers=2, hidden=32,
                     dense_hidden=256, k=100, interface_only=True, hop=-1,
                     head=2, fold_num=1, mode='classification', batch_size=1000)
    nfold_results = []
    train_summary = []
    all_mean_std = []

    model_save_folder = r'./model'
    for n in range(0, args.fold_num):
        print(str(n) + ' fold starts ... ...')
        #
        # train_dataset = PPDocking(root=r'', split='train', fold=n, hop=args.hop, mode=args.mode)
        # val_dataset = PPDocking(root=r'', split='val', fold=n, hop=args.hop, mode=args.mode)
        #
        # if args.mode == 'classification':
        #     model, summary = train_classification(train_dataset, val_dataset, args, fold=n, model_save_folder= model_save_folder)#,  mean, std)
        #     train_summary.extend(summary)
        # elif args.mode == 'regression':
        #     mean = train_dataset.data.dockq.mean(dim=0, keepdim=True)
        #     std = train_dataset.data.dockq.std(dim=0, keepdim=True)
        #     train_dataset.data.dockq = (train_dataset.data.dockq - mean) / std
        #     val_dataset.data.dockq = (val_dataset.data.dockq - mean) / std
        #     mean, std = mean.item(), std.item()
        #     model, summary = train_regression(train_dataset, val_dataset, args, fold=n, mean=mean, std=std)
        #     train_summary.extend(summary)
        #     all_mean_std.append({'fold': n, 'mean': mean, 'std': std})
        #
        # else:
        #     print('unsupportive mode! Please check the args')
        #     raise Exception('unsupportive mode! Please check the args')
        # # save training details into csv file
        # df_train_summary = pd.DataFrame(train_summary)
        # df_train_summary.to_csv(os.path.join(model_save_folder, r'train_summary.csv'), index=False)
        # del df_train_summary, train_summary
        # del model, summary, train_dataset, val_dataset



        # start testing...
        test_dataset = PPDocking(root=r'./example/data', split='test', fold=n, hop=args.hop, mode=args.mode)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

        # load best checkpoint to do test

        model, _, _ = load_checkpoint(os.path.join(model_save_folder, '{:d}fold_{:s}_model.pt'.format(n, args.mode)), GATSortPool(args).to(device))
        if args.mode == 'classification':
            test_loss, test_auc, test_rpc, test_ef1, all_complex_names, _, all_scores, all_targets = test_classification(model, device, test_loader)
            print('{:d} fold test_auc: {:.4f}, test_rpc: {:.4f}, test_ef1: {:.4f}'.format(n, test_auc, test_rpc, test_ef1))
            nfold_results.append({'fold': n, 'auc': test_auc, 'rpc': test_rpc, 'ef1': test_ef1})
            del test_loss, test_auc, test_rpc, test_ef1, all_complex_names, all_scores, all_targets
        elif args.mode == 'regression':
            test_loss, test_mae, test_pcc, test_ef1, _, _, _, _ = test_regression(model, device, test_loader)
            print('{:d} fold test_pcc: {:.4f}, test_mae: {:.4f}, test_ef1: {:.4f}'.format(n, test_pcc, test_mae, test_ef1))
            nfold_results.append({'fold': n, 'mae': test_mae, 'pcc': test_pcc, 'ef1': test_ef1})
            del test_loss, test_mae, test_pcc, test_ef1
        del model, test_dataset, device, test_loader

    if args.mode == 'classification':
        df_nfold_results = pd.DataFrame(nfold_results)
        mean_auc = np.mean(df_nfold_results['auc'])
        mean_rpc = np.mean(df_nfold_results['rpc'])
        mean_ef1 = np.mean(df_nfold_results['ef1'])
        print('{:d} fold test mean_auc: {:.4f}, test mean_rpc: {:.4f}, test mean_ef1: {:.4f}'.format(n+1, mean_auc, mean_rpc, mean_ef1))
        nfold_results.append({'mean_auc': mean_auc, 'mean_rpc': mean_rpc, 'mean_ef1': mean_ef1})
        with open(os.path.join(model_save_folder, '{:d}fold_hop{:d}_{:s}_results.json'.format(n+1, args.hop, args.mode)), 'w') as json_fp:
            json.dump(nfold_results, json_fp)
        del nfold_results, df_nfold_results, args
    else:
        df_nfold_results = pd.DataFrame(nfold_results)
        mean_pcc = np.mean(df_nfold_results['pcc'])
        mean_mae = np.mean(df_nfold_results['mae'])
        mean_ef1 = np.mean(df_nfold_results['ef1'])
        print('{:d} fold test mean_pcc: {:.4f}, test mean_mae: {:.4f}, test mean_ef1: {:.4f}'.format(n + 1, mean_pcc, mean_mae, mean_ef1))
        nfold_results.append({'mean_pcc': mean_pcc, 'mean_mae': mean_mae, 'mean_ef1': mean_ef1})

        #save

        with open(os.path.join(model_save_folder, '{:d}fold_hop{:d}_{:s}_results.json'.format(n+1, args.hop, args.mode)), 'w') as json_fp:
            json.dump(nfold_results, json_fp)
        del nfold_results, df_nfold_results, args

