import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import metrics

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


if __name__ == '__main__':
    #1.sampling
    fir_dir = "/home/fei/Research/Dataset/decoy/1_decoys_bm4_zd3.0.2_15deg/exp_3/9_test_results"
    num_folds = 5   #todo check
    df_all = pd.DataFrame(columns=['name', 'auc','auc_precision_recall'])
    for i in range(num_folds):
        preds_path = os.path.join(fir_dir, f'fold_{i}', '20200330-2', 'predsLogModel.csv')
        data = pd.read_csv(preds_path)
        # data.rename(columns={' label': 'label'}, inplace=True)

        complex_name_list = []
        for j in range(data.shape[0]):
            decoy_info = data['decoy_no'][j]
            complex_name = decoy_info.split('.')[0]
            if complex_name not in complex_name_list:
                complex_name_list.append(complex_name)

        auc_list=[]
        for complex_name in complex_name_list:
            auc_dic = {}
            df = data[data['decoy_no'].str.contains(complex_name)]
            targets = np.array(df["label"])
            scores = np.array(df["prediction"])

            fpr, tpr, _ = metrics.roc_curve(targets, scores, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            precision, recall, _ = metrics.precision_recall_curve(targets, scores, pos_label=1)
            auc_precision_recall = metrics.auc(recall, precision)

            # df = df.loc[df['prediction'] > cut_off]
            auc_dic['name'] = complex_name
            auc_dic['auc'] = auc
            auc_dic['auc_precision_recall'] = auc_precision_recall

            auc_list.append(auc_dic)
            del auc_dic

        top_dataframe = pd.DataFrame(auc_list, columns=['name', 'auc','auc_precision_recall'])
        df_all = pd.concat([df_all, top_dataframe], axis = 0, sort=False).reset_index(drop=True)

    auc_average_dic= {}
    auc_average_dic['name'] = 'average'
    auc_average_dic['auc'] = np.mean(df_all['auc'])
    auc_average_dic['auc_precision_recall'] = np.mean(df_all['auc_precision_recall'])

    auc_dataframe = pd.DataFrame([auc_average_dic], columns=['name', 'auc','auc_precision_recall'])

    df_all = pd.concat([auc_dataframe, df_all], axis = 0, sort=False).reset_index(drop=True)
    path = os.path.join(fir_dir, 'auc_result', '20200330-2','auc.csv')
    makedirs(path, isfile=True)
    df_all.to_csv(path,index = False)

    for key,values in auc_average_dic.items():
        print(str(key) + ':' + str(values))










    #         for k in range(df.shape[0]):
    #             top_dataframe = df.iloc[0:k+1]
    #             top_accuracy = top_dataframe["label"].sum(axis=0)
    #             top_accuracy_list.append(top_accuracy/(k+1))
    #
    #         # plot_results(complex_name, top_accuracy_list, os.path.join(fir_dir, f'fold_{i}', '20200306-1'))
    #         top_accuracy_all[complex_name] = top_accuracy_list
    #
    #
    #     sum_top = np.zeros(top_high)
    #     for m in range(top_high):
    #         complex_num = 0
    #         for complex_name in complex_name_list:
    #             if len(top_accuracy_all[complex_name]) > m:
    #                 complex_num += 1
    #                 sum_top[m] = top_accuracy_all[complex_name][m] +sum_top[m]
    #         sum_top[m] = sum_top[m]/complex_num
    #
    #     # plot_results("average", sum_top, os.path.join(fir_dir, f'fold_{i}', '20200306-1'))
    #     top_results.append(sum_top)
    #
    # average_top = []
    # for i in range(top_high):
    #     average=0
    #     for j in range(num_folds):
    #         average +=top_results[j][i]
    #     average = average/num_folds
    #     average_top.append(average)
    # plot_results("average", average_top, os.path.join(fir_dir,'20200327-3-0.5'))
    #
    #

