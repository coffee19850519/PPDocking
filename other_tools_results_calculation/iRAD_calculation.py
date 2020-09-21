import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from util_function import read_txt
# from torch_geometric.data import DataLoader
# from PyG_Experiments.util_function import compute_ratio_EF, compute_top_EF, load_checkpoint
# from PyG_Experiments.sort_pool_regression import GATSortPool, test_classification, test_regression
# from PyG_Experiments.import_data import PPDocking, arguments


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


def plot_results(complex_name, top_accuracy_list, path):
    makedirs(path)
    fig, ax = plt.subplots()

    ax.plot(top_accuracy_list, '.-')
    ax.set_ylim([0, 1.0])
    ax.grid(True)
    ax.set_xlabel('Top ranked considered')
    ax.set_ylabel('Hit Rate')
    ax.set_title(complex_name)

    plt.savefig(os.path.join(path, "Top ranked considered.pdf"))

def compute_ratio_EF(prediction_result_folder, top_high, save_folder):

    prediction_redults = pd.read_csv(os.path.join(prediction_result_folder, 'data_prediction.csv'))
    decoy_no = prediction_redults["decoy_no"]
    decoy_no_int = decoy_no.apply(lambda x: int(x.split(".")[1]))

    dict = {'complex_name': prediction_redults["complex_name"], 'decoy_no': decoy_no_int, 'target': prediction_redults["label"]}
    source_data = pd.DataFrame(dict, columns=dict.keys())

    all_complex_names = source_data['complex_name'].unique()

    top_list = []
    for complex_name in all_complex_names:
        df = source_data.loc[source_data['complex_name'] == complex_name].sort_values(by=['decoy_no'], ascending=True).reset_index(drop=True)
        EF_max = (df.shape[0]) / (df["target"].sum(axis=0))
        top_EF_dic = {}
        top_EF_dic['complex_name'] = complex_name
        top_EF_dic['EF_max'] = EF_max

        for k in range(1, top_high + 1):
            top_dataframe = df.iloc[0:k]
            top_accuracy = (top_dataframe["target"].sum(axis=0)) / (top_dataframe.shape[0])
            top_EF = top_accuracy * EF_max
            top_EF_dic[k] = top_EF

        top_list.append(top_EF_dic)
        del top_EF_dic

    count_list = [i for i in range(1, top_high + 1)]
    top_dataframe = pd.DataFrame(top_list, columns=['complex_name'] + ['EF_max'] + count_list)

    #df_all = pd.concat([df_all, top_dataframe], axis=0, sort=False).reset_index(drop=True)

    EF_percent_dic = {}
    EF_percent_dic['complex_name'] = 'average'
    EF_percent_dic['EF_max'] = np.mean(top_dataframe['EF_max'])
    for i in range(1, top_high +1):
        EF_percent_dic[i] = np.mean(top_dataframe[i])

    df_all = top_dataframe.append(EF_percent_dic, ignore_index= True)

    #EF_percent_dataframe = pd.DataFrame([EF_percent_dic], columns=['name'] + ['EF_max'] + EF_list)

    #df_all = pd.concat([EF_percent_dataframe, df_all], axis=0, sort=False).reset_index(drop=True)
    #path = os.path.join(fir_dir, 'enrichment_result', '20200330-2', 'enrichment_factor-1.csv')
    #makedirs(path, isfile=True)
    df_all.to_csv(os.path.join(save_folder, "Enrichment_factor.csv"), index=False)

    # for key, values in EF_percent_dic.items():
    #     print(str(key) + ':' + str(values))



def calculate_average_hit_count(prediction_result_folder, top_high, save_folder):

    prediction_redults = pd.read_csv(os.path.join(prediction_result_folder, 'data_prediction.csv'))
    decoy_no = prediction_redults["decoy_no"]
    decoy_no_int = decoy_no.apply(lambda x : int(x.split(".")[1]))

    dict = {'complex_name': prediction_redults["complex_name"], 'decoy_no': decoy_no_int, 'target': prediction_redults["label"]}
    source_data = pd.DataFrame(dict, columns=dict.keys())

    all_complex_names = source_data['complex_name'].unique()

    # data_full = pd.read_csv(os.path.join(prediction_results_folder, 'data_prediction.csv'))
    # for fold_idx in range(1, fold_num):
    #     data = pd.read_csv(os.path.join(save_results_folder, '{:d}fold_ensemble_results.csv').format(fold_idx))
    #     data_full = pd.concat([data_full, data], axis=0, sort=False).reset_index(drop=True)
    #     del data

    # get all complex names
    # all_complex_names = data_full['complex_name'].unique()
    # ensemble_metrics = list(data_full)
    # ensemble_metrics.remove('decoy_no')
    # ensemble_metrics.remove('complex_name')
    # ensemble_metrics.remove('label')
    hit_count_dict_list = []
    k_list = [i for i in range(1, top_high + 1)]


    top_accuracy_all_list = []
    for complex_name in all_complex_names:
        top_accuracy_dic = {}
        top_accuracy_dic['complex_name'] = complex_name
        df_current_complex_predictions = source_data[source_data['complex_name'] == complex_name]
        df_current_complex_predictions = df_current_complex_predictions.sort_values(by=["decoy_no"],ascending=True).reset_index(drop=True)
        # df = df.loc[df['prediction'] > cut_off]

        for k in range(1, top_high + 1):
            top_dataframe_at_current_metric = df_current_complex_predictions.iloc[0:k]
            top_accuracy_at_current_metric = top_dataframe_at_current_metric["target"].sum(axis=0)
            top_accuracy_dic[k] = top_accuracy_at_current_metric
            del top_dataframe_at_current_metric, top_accuracy_at_current_metric
            # top_accuracy_list.append(top_accuracy/(k+1))
            # top_accuracy_list.append(top_accuracy)
        # plot_results(complex_name, top_accuracy_list, os.path.join(fir_dir, f'fold_{i}', '20200306-1'))
        top_accuracy_all_list.append(top_accuracy_dic)
        del top_accuracy_dic

    top_dataframe = pd.DataFrame(top_accuracy_all_list, columns=['complex_name'] + k_list)

    top_average_dict = {}
    for k_top in k_list:
        top_average_dict[k_top] = np.mean(top_dataframe[k_top])


    # top_average_dataframe = top_dataframe.append(top_average_dict, ignore_index= True)
    df_all = top_dataframe.append(top_average_dict, ignore_index=True)
    df_all.to_csv(os.path.join(save_folder, 'top_hit_count_result.csv'), index=False)
    del df_all, all_complex_names
    del top_average_dict

def calculate_successful_rate(prediction_result_folder, top_high, save_folder):
    success_rate_list = [i for i in range(1, top_high + 1)]

    success_rate_dic_list = []
    # merge 5-fold results into one dataframe

    # data_full = pd.read_csv(os.path.join(prediction_results_folder, 'data_prediction.csv'))
    # # for fold_idx in range(1,fold_num):
    # #     data = pd.read_csv(os.path.join(save_results_folder, '{:d}fold_ensemble_results.csv').format(fold_idx))
    # #     data_full = pd.concat([data_full, data], axis=0, sort=False).reset_index(drop=True)
    # #     del data
    #
    # # get all complex names
    # all_complex_names = data_full['complex_name'].unique()
    # ensemble_metrics = list(data_full)
    # ensemble_metrics.remove('decoy_no')
    # ensemble_metrics.remove('complex_name')
    # ensemble_metrics.remove('label')
    #
    # for metric_name in ensemble_metrics:
    top_accuracy_all_list = []
    prediction_redults = pd.read_csv(os.path.join(prediction_result_folder, 'data_prediction.csv'))
    decoy_no = prediction_redults["decoy_no"]
    decoy_no_int = decoy_no.apply(lambda x: int(x.split(".")[1]))

    dict = {'complex_name': prediction_redults["complex_name"], 'decoy_no': decoy_no_int,
            'target': prediction_redults["label"]}
    source_data = pd.DataFrame(dict, columns=dict.keys())

    all_complex_names = source_data['complex_name'].unique()

    top_list = []
    for complex_name in all_complex_names:
        top_accuracy_dic = {}
        top_accuracy_dic['complex_name'] = complex_name
        df_current_complex_predictions = source_data[source_data['complex_name'] == complex_name]
        df_current_complex_predictions = df_current_complex_predictions.sort_values(by=['decoy_no'],ascending=True).reset_index(drop=True)
        # df = df.loc[df['prediction'] > cut_off]

        for k in range(1, top_high + 1):
            top_dataframe_at_current_metric = df_current_complex_predictions.iloc[0:k]
            top_accuracy_at_current_metric = top_dataframe_at_current_metric["target"].sum(axis=0)
            top_accuracy_dic[k] = top_accuracy_at_current_metric
            del top_dataframe_at_current_metric, top_accuracy_at_current_metric
            # top_accuracy_list.append(top_accuracy/(k+1))
            # top_accuracy_list.append(top_accuracy)
        # plot_results(complex_name, top_accuracy_list, os.path.join(fir_dir, f'fold_{i}', '20200306-1'))
        top_accuracy_all_list.append(top_accuracy_dic)
        del top_accuracy_dic

    top_dataframe = pd.DataFrame(top_accuracy_all_list)

    success_rate_dic = {}
    for s_top in success_rate_list:
        success_rate_dic[s_top] = (top_dataframe.loc[top_dataframe[s_top] != 0].shape[0]) / (
            top_dataframe.shape[0])
    success_rate_dic_list.append(success_rate_dic)
    del top_dataframe, success_rate_dic, top_accuracy_all_list

    success_rate_dataframe = pd.DataFrame(success_rate_dic_list)
    # top_dataframe = pd.concat([success_rate_dataframe, top_dataframe], axis=0, sort=False).reset_index(drop=True)

    path = os.path.join(save_folder, 'success_rate_results.csv')
    success_rate_dataframe.to_csv(path, index=False)

    del success_rate_list, success_rate_dataframe, success_rate_dic_list, all_complex_names


if __name__ == '__main__':
    root = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    target_names = read_txt(os.path.join(root, "caseID.lst"))
    prediction_result_folder = "/home/fei/Research/saved_checkpoints/200822_regression_vsr_savecheckpoint_GCNE_3layers_max_JK_AllGlobalPooling_NormalizedLabel/"

    top_high = 1000
    zdock_save_folder = "/home/fei/Research/irad_results/"
    compute_ratio_EF(prediction_result_folder, top_high, zdock_save_folder)
    calculate_successful_rate(prediction_result_folder, top_high,zdock_save_folder)
    calculate_average_hit_count(prediction_result_folder, top_high,zdock_save_folder)




