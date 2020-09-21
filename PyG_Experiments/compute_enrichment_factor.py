import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from torch_geometric.data import DataLoader
from PyG_Experiments.util_function import compute_ratio_EF, compute_top_EF,load_checkpoint
from PyG_Experiments.sort_pool_regression import GATSortPool
from PyG_Experiments.import_data import PPDocking, arguments

def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)

def plot_results(complex_name,top_accuracy_list,path):

    makedirs(path)
    fig,ax = plt.subplots()

    ax.plot(top_accuracy_list, '.-')
    ax.set_ylim([0, 1.0])
    ax.grid(True)
    ax.set_xlabel('Top ranked considered')
    ax.set_ylabel('Hit Rate')
    ax.set_title(complex_name)

    plt.savefig(os.path.join(path, "Top ranked considered.pdf"))


def calculate_EF(fold_num, args, checkpoint_folder, ratio_list, top_list, mean_std_dict = None, save_prediction_folder = None):
    all_complex_names = []
    all_scores = []
    all_targets = []
    mean = 0
    std = 1
    # load normalized mean and std
    if mean_std_dict is not None:
        df_mean_std = pd.DataFrame(mean_std_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #declare model

    for n in range(fold_num):

        ##load test data
        test_dataset = PPDocking(root= r'', fold= n, hop= args.hop, mode= args.mode, split= 'test')
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                                 num_workers=6)
        #read checkpoint
        model, _, _ = load_checkpoint(
            os.path.join(checkpoint_folder,'{:d}fold_{:s}_model.pt'.format(n, args.mode)),
            GATSortPool(args).to(device))

        #predict
        if args.mode == 'classification':
            
            _, _, _,_, fold_complex_names, fold_decoy_names, fold_scores, fold_targets = \
            test_classification(model, device, test_loader)

        else:

            # mean = df_mean_std.loc[df_mean_std['fold'] == n]['mean']
            # std = df_mean_std.loc[df_mean_std['fold'] == n]['std']
            _, _, _,_, fold_scores, fold_complex_names, fold_decoy_names, fold_targets = \
            test_regression(model, device, test_loader, mean= mean, std= std)

        dict = {'decoy_no': fold_decoy_names, 'label':fold_targets, 'prediction':fold_scores}
        fold_results = pd.DataFrame(dict)
        if save_prediction_folder is not None and not os.path.exists(save_prediction_folder):
            os.makedirs(save_prediction_folder)

        fold_results.to_csv(os.path.join(save_prediction_folder, '{:d}fold_predict_results.csv'.format(n)), index=False)


        all_complex_names.extend(fold_complex_names)
        all_scores.extend(fold_scores)
        all_targets.extend(fold_targets)
        del test_dataset, test_loader, fold_complex_names, fold_scores, fold_targets, model,fold_results,dict

    compute_ratio_EF(all_scores, all_targets, all_complex_names, ratio_list, checkpoint_folder, 'ratioEF.csv')
    compute_top_EF(all_scores, all_targets, all_complex_names, top_list, checkpoint_folder)

# def calculate_average_EF_from_meta_learners(save_result_folder, ratio_list, fold_num):
#     #read meta-learners' predictions
#     all_fold_mean = []
#     all_fold_median = []
#     all_fold_Q1 = []
#     all_fold_Q3 = []
#     all_fold_label = []
#     all_fold_complex_name = []
#
#     for fold_idx in range(0, fold_num):
#         current_fold_redults = pd.read_csv(os.path.join(save_result_folder, r'{:d}fold_all_learner_results.csv'.format(fold_idx)))
#         all_fold_mean.extend(current_fold_redults['mean'])
#         all_fold_median.extend(current_fold_redults['median'])
#         all_fold_Q1.extend(current_fold_redults['Q1'])
#         all_fold_Q3.extend(current_fold_redults['Q3'])
#         all_fold_complex_name.extend(current_fold_redults['complex_name'])
#         all_fold_label.extend(current_fold_redults['label'])
#         del current_fold_redults
#
#     compute_ratio_EF(all_fold_mean, all_fold_label, all_fold_complex_name, ratio_list, save_result_folder, r'all_fold_ensemble_ratioEF_mean.csv')
#     compute_ratio_EF(all_fold_median, all_fold_label, all_fold_complex_name, ratio_list, save_result_folder, r'all_fold_ensemble_ratioEF_median.csv')
#     compute_ratio_EF(all_fold_Q1, all_fold_label, all_fold_complex_name,
#                      ratio_list, save_result_folder, r'all_fold_ensemble_ratioEF_Q1.csv')
#     compute_ratio_EF(all_fold_Q3, all_fold_label, all_fold_complex_name,
#                      ratio_list, save_result_folder, r'all_fold_ensemble_ratioEF_Q3.csv')
#
#     del all_fold_complex_name, all_fold_label, all_fold_mean, all_fold_median, all_fold_Q1, all_fold_Q3

def calculate_average_EF_from_meta_learners(save_result_folder, max_count):

    current_fold_redults = pd.read_csv(os.path.join(save_result_folder, 'data_prediction.csv'))

    compute_ratio_EF(current_fold_redults['prediction'], current_fold_redults['label'], current_fold_redults['complex_name'], max_count, save_result_folder, r'all_fold_EF.csv')

    del current_fold_redults #all_fold_Qn80, all_fold_Qn90, all_fold_Qn95,all_fold_Qn85,all_fold_Qn75,all_fold_Qn70,all_fold_Qn60

def calculate_average_hit_count(save_results_folder, top_high):

    data_full = pd.read_csv(os.path.join(save_results_folder, 'data_prediction.csv'))
    # for fold_idx in range(1, fold_num):
    #     data = pd.read_csv(os.path.join(save_results_folder, '{:d}fold_ensemble_results.csv').format(fold_idx))
    #     data_full = pd.concat([data_full, data], axis=0, sort=False).reset_index(drop=True)
    #     del data

    # get all complex names
    all_complex_names = data_full['complex_name'].unique()
    ensemble_metrics = list(data_full)
    ensemble_metrics.remove('decoy_no')
    ensemble_metrics.remove('complex_name')
    ensemble_metrics.remove('label')
    hit_count_dict_list = []
    k_list = [i for i in range(1, top_high + 1)]

    for metric_name in ensemble_metrics:
        top_accuracy_all_list = []
        for complex_name in all_complex_names:
            top_accuracy_dic = {}
            top_accuracy_dic['complex_name'] = complex_name
            df_current_complex_predictions = data_full[data_full['complex_name'] == complex_name]
            df_current_complex_predictions = df_current_complex_predictions.sort_values(by=[metric_name, 'label'],
                                                                                        ascending=False).reset_index(drop=True)
            # df = df.loc[df['prediction'] > cut_off]

            for k in range(1, top_high + 1):
                top_dataframe_at_current_metric = df_current_complex_predictions.iloc[0:k]
                top_accuracy_at_current_metric = top_dataframe_at_current_metric["label"].sum(axis=0)
                top_accuracy_dic[k] = top_accuracy_at_current_metric
                del top_dataframe_at_current_metric, top_accuracy_at_current_metric
                # top_accuracy_list.append(top_accuracy/(k+1))
                # top_accuracy_list.append(top_accuracy)
            # plot_results(complex_name, top_accuracy_list, os.path.join(fir_dir, f'fold_{i}', '20200306-1'))
            top_accuracy_all_list.append(top_accuracy_dic)
            del top_accuracy_dic

        top_dataframe = pd.DataFrame(top_accuracy_all_list, columns=['complex_name'] + k_list)

        top_average_dict = {}
        top_average_dict['metric'] = metric_name
        for k_top in k_list:
            top_average_dict[k_top] = np.mean(top_dataframe[k_top])
        hit_count_dict_list.append(top_average_dict)
        del top_average_dict,top_dataframe

    top_average_dataframe = pd.DataFrame(hit_count_dict_list)
    top_average_dataframe.to_csv(os.path.join(save_results_folder, 'top_hit_count_result.csv'),index = False)
    del top_average_dataframe, data_full,all_complex_names


def calculate_successful_rate(save_results_folder, top_high):

    success_rate_list = [i for i in range(1, top_high + 1)]

    success_rate_dic_list = []
    #merge 5-fold results into one dataframe

    data_full = pd.read_csv(os.path.join(save_results_folder, 'data_prediction.csv'))
    # for fold_idx in range(1,fold_num):
    #     data = pd.read_csv(os.path.join(save_results_folder, '{:d}fold_ensemble_results.csv').format(fold_idx))
    #     data_full = pd.concat([data_full, data], axis=0, sort=False).reset_index(drop=True)
    #     del data

    #get all complex names
    all_complex_names = data_full['complex_name'].unique()
    ensemble_metrics = list(data_full)
    ensemble_metrics.remove('decoy_no')
    ensemble_metrics.remove('complex_name')
    ensemble_metrics.remove('label')


    for metric_name in ensemble_metrics:
        top_accuracy_all_list = []
        for complex_name in all_complex_names:
            top_accuracy_dic = {}
            top_accuracy_dic['complex_name'] = complex_name
            df_current_complex_predictions = data_full[data_full['complex_name'] == complex_name]
            df_current_complex_predictions = df_current_complex_predictions.sort_values(by=[metric_name, 'label'], ascending=False).reset_index(drop=True)
            # df = df.loc[df['prediction'] > cut_off]

            for k in range(1, top_high+1):
                top_dataframe_at_current_metric = df_current_complex_predictions.iloc[0:k]
                top_accuracy_at_current_metric = top_dataframe_at_current_metric["label"].sum(axis=0)
                top_accuracy_dic[k] = top_accuracy_at_current_metric
                del  top_dataframe_at_current_metric, top_accuracy_at_current_metric
                # top_accuracy_list.append(top_accuracy/(k+1))
                # top_accuracy_list.append(top_accuracy)
            # plot_results(complex_name, top_accuracy_list, os.path.join(fir_dir, f'fold_{i}', '20200306-1'))
            top_accuracy_all_list.append(top_accuracy_dic)
            del top_accuracy_dic

        top_dataframe = pd.DataFrame(top_accuracy_all_list)

        success_rate_dic = {}
        success_rate_dic['ensemble_metrics'] = metric_name
        for s_top in success_rate_list:
            success_rate_dic[s_top] = (top_dataframe.loc[top_dataframe[s_top] != 0].shape[0]) / (
            top_dataframe.shape[0])
        success_rate_dic_list.append(success_rate_dic)
        del top_dataframe, success_rate_dic, top_accuracy_all_list

    success_rate_dataframe = pd.DataFrame(success_rate_dic_list)
    #top_dataframe = pd.concat([success_rate_dataframe, top_dataframe], axis=0, sort=False).reset_index(drop=True)

    path = os.path.join(save_results_folder, 'success_rate_results.csv')
    success_rate_dataframe.to_csv(path,index = False)

    del success_rate_list, data_full,success_rate_dataframe, success_rate_dic_list, all_complex_names

if __name__ == '__main__':
    # args = arguments(num_node_features=26, num_layers=3, hidden=32,
    #                  dense_hidden=512, k=100, interface_only=True, hop=-1, head= 2, mode= 'regression')
    # checkpoint_folder= r'/home/fei/Desktop/pp-docking/PyG_Experiments/'
    ratio_list = [0.0001, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    # top_list = [20, 50, 100, 200, 500]
    # #if args.mode == 'regression':
    #     #read mean and std
    # calculate_EF(5, args, checkpoint_folder, ratio_list, top_list, save_prediction_folder=checkpoint_folder)

    model_save_folder = r'/run/media/fei/Project/docking/ENSEMBLE_MODEL/best_GAT_ensemble/'
    calculate_average_EF_from_meta_learners(model_save_folder, ratio_list, 5)


    #calculate_successful_rate(1000, 5, model_save_folder)
    #calculate_average_hit_count(2001, 5, model_save_folder)




