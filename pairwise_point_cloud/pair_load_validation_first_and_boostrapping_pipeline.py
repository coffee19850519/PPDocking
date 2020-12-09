import os
import json
import pandas as pd
import torch
import numpy as np
import glob

from PyG_Experiments.CGConv_net import CGConv_by_occurrence_global_pool
from PyG_Experiments.util_function import load_checkpoint
from PyG_Experiments.compute_enrichment_factor import calculate_average_EF_from_meta_learners, \
    calculate_successful_rate, calculate_average_hit_count
from pairwise_point_cloud.pair_load_validation_first_and_boostrapping_training_and_testing import train_classification, train_regression, test_classification, \
    test_regression, val_classification, arguments
from PyG_Experiments.util_function import EarlyStopping, compute_PCC, compute_EF, save_node_significance, compute_ratio_EF
from pairwise_point_cloud.pair_test import DockPointNet
from pairwise_point_cloud.pair_point_cloud_convertor import PairData
from PyG_Experiments.GCNE_net import GCEonv_with_global_pool
from PyG_Experiments.GATE_net import GATEConv_with_global_pool
from torch_geometric.nn import DataParallel
from multiprocessing import cpu_count, Pool
from torch_geometric.data import (InMemoryDataset, Data, download_url, Dataset, extract_zip)
from torch_geometric.data.dataset import to_list
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops, to_undirected
from PyG_Experiments.sort_pool_regression import GATSortPool
from PyG_Experiments.sort_pool_by_occurrence import GATSortPool_by_occurrence_global_pool
from PyG_Experiments.self_attention_net import SFEConv_with_global_pool


if __name__ == '__main__':

    ####validation-1:1, and load only one time;
    ####boostrapping

    args = arguments(
        # 1.node feature and edge feature
        num_node_features = 37,
        # num_node_features=36,
        num_edge_attr=25,
        regression_label_type="DockQ",

        # 2.model setting
        num_layers=2,
        hidden=64,
        dense_hidden=512,
        k=100,
        interface_only=True,
        hop=-1,
        head=1,
        mode='classification',
        concat= False,

        # 3.training setting
        fold_num=1,
        batch_size=100,
        pos_weights=3.0,
        # learning_rate=0.0005,
        learning_rate=[0.001, 0.001, 0.001, 0.001, 0.001],
        weight_decay = 0.001,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        savecheckpoints_mode='max',
        savecheckpoints_monitor="val_loss",
        earlystop_patience=10,
        epochs=201,

        # 4.dataset setting
        # train_classification_patch_path="/run/media/fei/Windows/hanye/small_regression_train_patch/",
        train_classification_patch_path="/run/media/fei/Windows/hanye/regression_patch/",
        # train_classification_patch_path= r'/run/media/fei/Windows/hanye/boostrapping_regression_train_patch/',
        val_classification_patch_path="/run/media/fei/Windows/hanye/all_small_regression_patch/",
        test_classification_patch_path="/run/media/fei/Windows/hanye/all_small_regression_patch/",

        boostrapping_train_start_number=1,
        boostrapping_train_max_number=1,
        test_max_subset_number=100,

        # 5.others
        # root=r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/',
        root=r'/run/media/fei/Windows/hanye/point_cloud_format/',
        model_save_name="201118_classification_pointcloud_toy_data_only_pos",
        checkpoints_folder=r'/home/fei/Research/saved_checkpoints/',
        pretrained_model_folder="/home/fei/Research/saved_checkpoints/",
        pretrained_model_mode="classification"
    )

    label_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/'
    model_save_folder = os.path.join(args.checkpoints_folder, args.model_save_name)

    nfold_results = []
    # train_summary = []
    all_mean_std = []

    all_decoy_names = []
    all_complex_names = []
    all_targets = []
    all_scores = []
    all_labels = []
    all_test_loss = []
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
        # learning_rate = args.learning_rate
        learning_rate = args.learning_rate[n]
        # training_model = GCEonv_with_global_pool(args)
        training_model = DockPointNet(args.num_node_features, 11,1)
        # training_model = SFEConv_with_global_pool(args)
        # training_model = GATEConv_with_global_pool(args)
        #training_model = DataParallel(training_model)

        training_model = training_model.to(args.device)

        # load classification weights
        # training_model, _, _ = load_checkpoint(
        #     os.path.join(args.pretrained_model_folder, '{:d}_fold_{:s}_model.pt'.format(n, args.pretrained_model_mode)), training_model)

        earlystop = EarlyStopping(patience=args.earlystop_patience, verbose=True, delta= 0.0000001)

        for boostrapping_number in range(args.boostrapping_train_start_number, args.boostrapping_train_max_number + 1):

            train_temp = glob.glob(os.path.join(args.train_classification_patch_path,'{:d}_fold_*_patch_train_data.pt'.format(n)))
            train_patch_size = len(train_temp)


            val_temp = glob.glob(os.path.join(args.val_classification_patch_path,'{:d}_fold_*_patch_val_data.pt'.format(n)))
            val_patch_size = len(val_temp)

            if args.mode == 'classification':

                model, learning_rate = train_classification(args=args, model=training_model, fold=n,
                                                      model_save_folder=model_save_folder,
                                                      train_patch_size=train_patch_size,
                                                      train_subset_number=boostrapping_number,
                                                      val_subset_number=boostrapping_number,
                                                      earlystop=earlystop,
                                                      learning_rate=learning_rate,
                                                      weight_decay= args.weight_decay
                                                    )
                # train_summary.extend(summary)


            elif args.mode == 'regression':
                # mean = train_dataset.data.dockq.mean(dim=0, keepdim=True)
                # std = train_dataset.data.dockq.std(dim=0, keepdim=True)
                # train_dataset.data.dockq = (train_dataset.data.dockq - mean) / std
                # val_dataset.data.dockq = (val_dataset.data.dockq - mean) / std
                # mean, std = mean.item(), std.item()
                # model, summary = train_regression(train_dataset, val_dataset, args, fold=n, mean=mean, std=std)
                # train_summary.extend(summary)
                # all_mean_std.append({'fold': n, 'mean': mean, 'std': std})
                model, summary = train_regression(args=args, model=training_model, fold=n,
                                                      model_save_folder=model_save_folder,
                                                      train_patch_size=train_patch_size,
                                                      val_patch_size=val_patch_size,
                                                      train_subset_number=boostrapping_number,
                                                      val_subset_number=boostrapping_number,
                                                      earlystop=earlystop,
                                                      learning_rate=learning_rate)

            else:
                print('unsupportive mode! Please check the args')
                raise Exception('unsupportive mode! Please check the args')

            del model

            # learning_rate = learning_rate * 0.8
            # loop to load each patch of test data and do test

            # load best checkpoint to do test
            training_model, _, _ = load_checkpoint(
                os.path.join(model_save_folder, '{:d}_fold_{:s}_model.pt'.format(n, args.mode)), training_model)

            ###RESET
            earlystop.reset()



        test_temp = glob.glob(os.path.join(args.test_classification_patch_path,
                                           '{:d}_fold_*_patch_test_data.pt'.format(n)))
        test_patch_size = len(test_temp)


        del test_temp

        # training_model, _, _ = load_checkpoint(
        #     os.path.join(model_save_folder, '{:d}_fold_{:s}_model.pt'.format(n, args.mode)), training_model)

        # split = "test"
        if args.mode == 'classification':
            # test_loss, test_auc, test_rpc, test_ef, all_complex_names,_, all_scores, all_targets = test_classification(model, device, test_loader)
            #, test_ef, test_sr, test_loss
            test_loss, all_complex_names, all_decoy_names, all_scores, all_targets, auc, auc_precision_recall, test_ef, test_sr = test_classification(
                args=args, model=training_model, fold=n, split="test", max_subset_number=args.test_max_subset_number,
                patch_size=test_patch_size)

            all_complex_names.extend(all_complex_names)
            all_decoy_names.extend(all_decoy_names)
            all_scores.extend(all_scores)
            all_targets.extend(all_targets)
            all_test_loss.append(test_loss)
            #, test_ef: {:.4f}, test_sr: {:.4f}, test_ef,test_sr, 'ef': test_ef, 'sr': test_sr, test_ef,test_loss
            print('{:d} fold test_acc: {:.4f}, test_auc:{:.4f}, test_prc:{:.4f}, test_ef: {:.4f}, test_sr: {:.4f}'.format(n,test_loss,auc, auc_precision_recall, test_ef, test_sr))

            nfold_results.append({'fold': n, 'loss': test_loss, 'auc': auc, 'prc':auc_precision_recall, 'ef': test_ef, 'sr': test_sr})
            del all_complex_names, all_decoy_names, all_scores, all_targets, all_test_loss
        elif args.mode == 'regression':
            test_loss, test_pcc, test_spcc, test_ef,test_sr, complex_names, decoy_names, scores, targets, categories= test_regression(
                args=args, model=training_model, fold=n, split="test", max_subset_number=args.test_max_subset_number,
                patch_size=test_patch_size)

            all_complex_names.extend(complex_names)
            all_decoy_names.extend(decoy_names)
            all_scores.extend(scores)
            all_targets.extend(categories)
            all_labels.extend(targets)

            print('{:d} fold test_loss:{:.4f}, test_pcc: {:.4f}, test_spcc:{:.4f}, test_ef: {:.4f}, test_sr: {:.4f}'.format(n, test_loss, test_pcc, test_spcc, test_ef, test_sr))

            nfold_results.append({'fold': n, 'loss':test_loss, 'pcc': test_pcc, 'spcc': test_spcc, 'ef': test_ef, 'sr': test_sr})
            del test_pcc, test_ef

    del training_model



    # # save to file at save_folder
    df_predict = pd.DataFrame({'complex_name': all_complex_names, 'decoy_no': all_decoy_names, 'label': all_targets,
                               'prediction': all_scores, 'regression_label': all_labels})
    df_predict.to_csv(os.path.join(model_save_folder, r'data_prediction.csv'), index=False)

    del df_predict, all_complex_names, all_decoy_names, all_targets, all_scores

    if args.mode == 'classification':



        df_nfold_results = pd.DataFrame(nfold_results)
        mean_loss = np.mean(df_nfold_results['loss'])

        print('{:d} fold test mean_loss: {:.4f}'.format(n + 1, mean_loss))

        nfold_results.append({'mean_acc': mean_loss})

        with open(os.path.join(model_save_folder,
                               '{:d}fold_hop{:d}_{:s}_results.json'.format(n + 1, args.hop, args.mode)),
                  'w') as json_fp:
            json.dump(nfold_results, json_fp)

        del nfold_results, df_nfold_results, args

    else:
        df_nfold_results = pd.DataFrame(nfold_results)
        mean_pcc = np.mean(df_nfold_results['pcc'])
        #mean_mae = np.mean(df_nfold_results['mae'])
        mean_ef = np.mean(df_nfold_results['ef'])

        print('{:d} fold test mean_pcc: {:.4f},  test mean_ef: {:.4f}'.format(n + 1, mean_pcc, mean_ef))
        nfold_results.append({'mean_pcc': mean_pcc, 'mean_ef': mean_ef})

        # save
        with open(os.path.join(model_save_folder,
                               '{:d}fold_hop{:d}_{:s}_results.json'.format(n + 1, args.hop, args.mode)),
                  'w') as json_fp:
            json.dump(nfold_results, json_fp)
        del nfold_results, df_nfold_results, args

    # max_count = 1000
    # calculate_average_EF_from_meta_learners(model_save_folder, max_count)
    # calculate_successful_rate(model_save_folder, max_count)
    # calculate_average_hit_count(model_save_folder, max_count)