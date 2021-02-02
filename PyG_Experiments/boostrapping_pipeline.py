import os
import json
import pandas as pd
import torch
import numpy as np
import glob
import networkx as nx
import datetime
import datetime

from PyG_Experiments.CGConv_net import CGConv_by_occurrence_global_pool
from PyG_Experiments.util_function import load_checkpoint
from PyG_Experiments.compute_enrichment_factor import calculate_average_EF_from_meta_learners, \
    calculate_successful_rate, calculate_average_hit_count
from PyG_Experiments.training_and_testing import train_classification, train_regression, test_classification, \
    test_regression, val_classification
from itertools import product
from multiprocessing import cpu_count, Pool
from torch_geometric.data import (InMemoryDataset, Data, download_url, Dataset, extract_zip)
from torch_geometric.data.dataset import to_list
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops, to_undirected
from PyG_Experiments.sort_pool_regression import GATSortPool
from PyG_Experiments.sort_pool_by_occurrence import GATSortPool_by_occurrence_global_pool


class arguments:
    def __init__(self, num_node_features, num_edge_attr,
                 num_layers, hidden, dense_hidden, k, interface_only, hop, head, mode, concat,
                 batch_size, fold_num, pos_weights, learning_rate, device, savecheckpoints_mode,
                 savecheckpoints_monitor, earlystop_patience, epochs,
                 train_classification_patch_path, val_classification_patch_path, test_classification_patch_path,
                 train_subset_number, test_max_subset_number, val_max_subset_number,boostrapping_train_max_number,
                 root,model_save_name,checkpoints_folder
                 ):
        self.num_node_features = num_node_features
        self.num_layers = num_layers
        self.num_edge_attr = num_edge_attr
        self.hidden = hidden
        self.dense_hidden = dense_hidden
        self.k = k
        self.interface_only = interface_only
        self.hop = hop
        self.head = head
        self.mode = mode
        self.batch_size = batch_size
        self.fold_num = fold_num
        self.concat = concat
        self.pos_weights = pos_weights
        self.learning_rate = learning_rate
        self.device = device
        self.savecheckpoints_mode = savecheckpoints_mode
        self.savecheckpoints_monitor = savecheckpoints_monitor
        self.earlystop_patience = earlystop_patience
        self.epochs = epochs
        self.train_classification_patch_path = train_classification_patch_path
        self.val_classification_patch_path = val_classification_patch_path
        self.test_classification_patch_path = test_classification_patch_path
        self.train_subset_number = train_subset_number
        self.boostrapping_train_max_number = boostrapping_train_max_number
        self.test_max_subset_number = test_max_subset_number
        self.val_max_subset_number = val_max_subset_number
        self.root = root
        self.model_save_name=model_save_name
        self.checkpoints_folder=checkpoints_folder


if __name__ == '__main__':
    args = arguments(
        # 1.node feature and edge feature
        num_node_features=36,
        num_edge_attr=4,

        # 2.model setting
        num_layers=3,
        hidden=36,
        dense_hidden=512,
        k=100,
        interface_only=True,
        hop=-1,
        head=2,
        mode='classification',
        concat=False,

        # 3.training setting
        fold_num=5,
        batch_size=300,
        pos_weights=1.0,
        learning_rate=0.0005,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        savecheckpoints_mode='max',
        savecheckpoints_monitor="val_rpc",
        earlystop_patience=10,
        epochs=201,

        # 4.dataset setting
        train_classification_patch_path="/home/fei/Research/processed_all_data/classification_patch/",
        val_classification_patch_path="/run/media/fei/f88f3a13-5638-42ad-a600-bb7cab6b17f2/media/mount/hanye/classification_patch/",
        test_classification_patch_path="/run/media/fei/Windows/hanye/classification_patch/",

        train_subset_number=1,
        test_max_subset_number=100,
        val_max_subset_number=100,
        boostrapping_train_max_number = 4,

        # 5.others
        root=r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/',
        model_save_name = "200727_trainval_3pos_globalpool_CGCN_4edgefeature_3layers_valtest_alldata_vloss_savecheckpoint+1-4trainsubset_boostrapping",
        checkpoints_folder = r'/home/fei/Research/saved_checkpoints'
    )

    label_folder = os.path.join(args.root, "5_decoy_name")
    model_save_folder = os.path.join(args.checkpoints_folder, args.model_save_name)

    nfold_results = []
    train_summary = []
    all_mean_std = []

    all_decoy_names = []
    all_complex_names = []
    all_targets = []
    all_scores = []

    # model = SortPoolRegression(train_dataset.num_node_features,2,32).to(device)
    # model = GATRegression(train_dataset.num_node_features, 64).to(device)
    # model = TopK(train_dataset, 3, 32, dense_hidden= 256 , ratio= 0.8).to(device)

    # model = GATSortPool(args).to(device)
    # model = GATSortPool_by_occurrence(args).to(device)
    # model = CGConv_by_occurrence_global_pool(args).to(device)
    # model = CGConvSortPool(args).to(device)
    # model = CGConvSortPool_by_occurrence(args).to(device)

    for n in range(0, args.fold_num):

        print(str(n) + ' fold starts ... ...')

        val_temp = glob.glob(os.path.join(args.val_classification_patch_path,
                                          '{:d}_subset_{:d}_fold_*_patch_val_data.pt'.format(args.val_max_subset_number,
                                                                                             n)))
        test_temp = glob.glob(os.path.join(args.test_classification_patch_path,
                                           '{:d}_subset_{:d}_fold_*_patch_test_data.pt'.format(
                                               args.test_max_subset_number, n)))
        val_patch_size = len(val_temp)
        test_patch_size = len(test_temp)

        del val_temp, test_temp

        training_model = CGConv_by_occurrence_global_pool(args).to(args.device)

        for boostrapping_number in range(1, args.boostrapping_train_max_number + 1):

            train_temp = glob.glob(os.path.join(args.train_classification_patch_path,
                                                '{:d}_subset_{:d}_fold_*_patch_train_data.pt'.format(
                                                    boostrapping_number, n)))
            train_patch_size = len(train_temp)

            if args.mode == 'classification':
                args.learning_rate = args.learning_rate * 0.8
                model, summary= train_classification(args=args, model=training_model, fold=n,
                                                      model_save_folder=model_save_folder,
                                                      train_patch_size=train_patch_size, val_patch_size=val_patch_size)
                train_summary.extend(summary)

            elif args.mode == 'regression':
                mean = train_dataset.data.dockq.mean(dim=0, keepdim=True)
                std = train_dataset.data.dockq.std(dim=0, keepdim=True)
                train_dataset.data.dockq = (train_dataset.data.dockq - mean) / std
                val_dataset.data.dockq = (val_dataset.data.dockq - mean) / std
                mean, std = mean.item(), std.item()
                model, summary = train_regression(train_dataset, val_dataset, args, fold=n, mean=mean, std=std)
                train_summary.extend(summary)
                all_mean_std.append({'fold': n, 'mean': mean, 'std': std})

            else:
                print('unsupportive mode! Please check the args')
                raise Exception('unsupportive mode! Please check the args')

            del model

            # loop to load each patch of test data and do test

            # load best checkpoint to do test
            training_model, _, _ = load_checkpoint(
                os.path.join(model_save_folder, '{:d}_fold_{:s}_model.pt'.format(n, args.mode)), training_model)


        training_model, _, _ = load_checkpoint(
            os.path.join(model_save_folder, '{:d}_fold_{:s}_model.pt'.format(n, args.mode)), training_model)

        # split = "test"
        if args.mode == 'classification':
            # test_loss, test_auc, test_rpc, test_ef1, all_complex_names,_, all_scores, all_targets = test_classification(model, device, test_loader), test_rpc, test_ef
            test_loss, test_auc, complex_names, decoy_names, scores, targets = test_classification(
                args=args, model=training_model, fold=n, split="test", max_subset_number=args.test_max_subset_number,
                patch_size=test_patch_size)

            all_complex_names.extend(complex_names)
            all_decoy_names.extend(decoy_names)
            all_scores.extend(scores)
            all_targets.extend(targets)

            print('{:d} fold test_auc: {:.4f}, test_rpc: {:.4f}, test_ef: {:.4f}'.format(n, test_auc, test_rpc,
                                                                                          test_ef))

            nfold_results.append({'fold': n, 'auc': test_auc, 'rpc': test_rpc, 'ef': test_ef})
            del test_loss, test_auc, test_rpc, test_ef, complex_names, decoy_names, scores, targets
        elif args.mode == 'regression':
            test_loss, test_mae, test_pcc, test_ef, _, _, _, _ = test_regression(load_model, args.device, test_loader,
                                                                                  mean, std)
            print('{:d} fold test_pcc: {:.4f}, test_mae: {:.4f}, test_ef: {:.4f}'.format(n, test_pcc, test_mae,
                                                                                          test_ef))

            nfold_results.append({'fold': n, 'mae': test_mae, 'pcc': test_pcc, 'ef': test_ef})
            del test_loss, test_mae, test_pcc, test_ef
        del training_model

    # save training details into csv file
    df_train_summary = pd.DataFrame(train_summary)
    df_train_summary.to_csv(os.path.join(model_save_folder, r'train_summary.csv'), index=False)
    del df_train_summary, train_summary

    # # save to file at save_folder
    df_predict = pd.DataFrame({'complex_name': all_complex_names, 'decoy_no': all_decoy_names, 'label': all_targets,
                               'prediction': all_scores})
    df_predict.to_csv(os.path.join(model_save_folder, r'data_prediction.csv'), index=False)

    del df_predict, all_complex_names, all_decoy_names, all_targets, all_scores

    if args.mode == 'classification':
        df_nfold_results = pd.DataFrame(nfold_results)
        mean_auc = np.mean(df_nfold_results['auc'])
        mean_rpc = np.mean(df_nfold_results['rpc'])
        mean_ef = np.mean(df_nfold_results['ef'])
        print('{:d} fold test mean_auc: {:.4f}, test mean_rpc: {:.4f}, test mean_ef: {:.4f}'.format(n + 1, mean_auc,
                                                                                                     mean_rpc,
                                                                                                     mean_ef))

        nfold_results.append({'mean_auc': mean_auc, 'mean_rpc': mean_rpc, 'mean_ef': mean_ef})

        with open(os.path.join(model_save_folder,
                               '{:d}fold_hop{:d}_{:s}_results.json'.format(n + 1, args.hop, args.mode)),
                  'w') as json_fp:
            json.dump(nfold_results, json_fp)

        del nfold_results, df_nfold_results, args

    else:
        df_nfold_results = pd.DataFrame(nfold_results)
        mean_pcc = np.mean(df_nfold_results['pcc'])
        mean_mae = np.mean(df_nfold_results['mae'])
        mean_ef = np.mean(df_nfold_results['ef'])
        print('{:d} fold test mean_pcc: {:.4f}, test mean_mae: {:.4f}, test mean_ef: {:.4f}'.format(n + 1, mean_pcc,
                                                                                                     mean_mae,
                                                                                                     mean_ef))
        nfold_results.append({'mean_pcc': mean_pcc, 'mean_mae': mean_mae, 'mean_ef': mean_ef})

        # save
        with open(os.path.join(model_save_folder,
                               '{:d}fold_hop{:d}_{:s}_results.json'.format(n + 1, args.hop, args.mode)),
                  'w') as json_fp:
            json.dump(nfold_results, json_fp)
        del nfold_results, df_nfold_results, args

    max_count = 1000
    calculate_average_EF_from_meta_learners(model_save_folder, max_count)
    calculate_successful_rate(model_save_folder, max_count)
    calculate_average_hit_count(model_save_folder, max_count)