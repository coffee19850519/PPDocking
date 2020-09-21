import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from util_function import read_txt


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


if __name__ == '__main__':
    root = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    target_names = read_txt(os.path.join(root, "caseID.lst"))

    prediction_result_folder = \
        "/home/fei/Research/saved_checkpoints/200822_regression_vsr_savecheckpoint_GCNE_3layers_max_JK_AllGlobalPooling_NormalizedLabel/"
    prediction_redults = pd.read_csv(os.path.join(prediction_result_folder, 'data_prediction.csv'))

    top_high = 1000
    zrank_save_folder = "/home/fei/Research/zrank_results/top500-test-zdock-regression_experiment_result/"

    zrank_out_folder = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/results/"

    decoy_no = prediction_redults["decoy_no"]
    decoy_no_int = decoy_no.apply(lambda x: int(x.split(".")[1]))

    dict = {'complex_name': prediction_redults["complex_name"], 'decoy_no': decoy_no_int, 'target': prediction_redults["label"]}
    source_data = pd.DataFrame(dict, columns=dict.keys())

    all_complex_names = source_data['complex_name'].unique()
    top_EF_list = []
    top_hit_count_list = []
    for complex_name in all_complex_names:
        zrank_out_path = os.path.join(zrank_out_folder, complex_name + ".zd3.0.2.irad.out.scores")
        zrank_df = pd.read_table(zrank_out_path,header=None)
        zrank_df.columns = ['irad_num', 'irad_scores','zrank_num','zrank_scores']
        # zrank_df['irad_scores'] = None
        # zrank_df['zrank_num'] = None
        # zrank_df['zrank_scores'] = None
        # for i in range(len(zrank_df)):
        #     data = zrank_df["irad_num"][i].split()
        #     zrank_df["irad_num"][i] = data[0]
        #     zrank_df['irad_scores'][i] = data[1]
        #     zrank_df['zrank_num'][i] = data[2]
        #     zrank_df['zrank_scores'][i] = data[3]

        df = source_data.loc[source_data['complex_name'] == complex_name].\
            sort_values(by=['decoy_no'],ascending=True).reset_index(drop=True)
        irad_num_list = list(df['decoy_no'])

        df['zrank_num'] = zrank_df.loc[zrank_df['irad_num'].isin(irad_num_list)].reset_index(drop=True)['zrank_num']
        df=df.sort_values(by=['zrank_num'], ascending=True).reset_index(drop=True)


        top_EF_dic = {}
        top_EF_dic['complex_name'] = complex_name
        EF_max = (df.shape[0]) / (df["target"].sum(axis=0))
        top_EF_dic['EF_max'] = EF_max

        top_hit_count_dic = {}
        top_hit_count_dic['complex_name'] = complex_name

        for k in range(1, top_high + 1):
            top_dataframe = df.iloc[0:k]
            top_accuracy = (top_dataframe["target"].sum(axis=0)) / (top_dataframe.shape[0])
            top_EF = top_accuracy * EF_max
            top_EF_dic[k] = top_EF

            top_hit_count_dic[k] = top_dataframe["target"].sum(axis=0)

        top_EF_list.append(top_EF_dic)
        top_hit_count_list.append(top_hit_count_dic)
        del top_EF_dic, top_hit_count_dic,top_dataframe

    count_list = [i for i in range(1, top_high + 1)]
    top_EF_dataframe = pd.DataFrame(top_EF_list, columns=['complex_name'] + ['EF_max'] + count_list)
    top_hit_count_dataframe = pd.DataFrame(top_hit_count_list, columns=['complex_name'] + count_list)

    EF_percent_dic = {}
    EF_percent_dic['complex_name'] = 'average'
    EF_percent_dic['EF_max'] = np.mean(top_EF_dataframe['EF_max'])
    for i in range(1, top_high + 1):
        EF_percent_dic[i] = np.mean(top_EF_dataframe[i])
    df_EF_all = top_EF_dataframe.append(EF_percent_dic, ignore_index=True)
    df_EF_all.to_csv(os.path.join(zrank_save_folder, "Enrichment_factor.csv"), index=False)

    hit_count_percent_dic = {}
    hit_count_percent_dic['complex_name'] = 'average'
    for i in range(1, top_high + 1):
        hit_count_percent_dic[i] = np.mean(top_hit_count_dataframe[i])
    df_hit_count_all = top_hit_count_dataframe.append(hit_count_percent_dic, ignore_index=True)
    df_hit_count_all.to_csv(os.path.join(zrank_save_folder, "hit_count.csv"), index=False)

    success_rate_dic = {}
    success_rate_dic_list = []
    for i in range(1, top_high + 1):
        success_rate_dic[i] = (top_hit_count_dataframe.loc[top_hit_count_dataframe[i] != 0].shape[0]) / (
            top_hit_count_dataframe.shape[0])
    success_rate_dic_list.append(success_rate_dic)

    success_rate_dataframe = pd.DataFrame(success_rate_dic_list)

    path = os.path.join(zrank_save_folder, 'success_rate_results.csv')
    success_rate_dataframe.to_csv(path, index=False)








