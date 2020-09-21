import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    num_folds = 5
    top_high =50
    top_results = []
    cut_off = 0.7
    for i in range(num_folds):
        preds_path = os.path.join(fir_dir, f'fold_{i}', '20200327-3', 'predsLogModel.csv')
        data = pd.read_csv(preds_path)
        # data.rename(columns={' label': 'label'}, inplace=True)

        complex_name_list = []
        for j in range(data.shape[0]):
            decoy_info = data['decoy_no'][j]
            complex_name = decoy_info.split('.')[0]
            if complex_name not in complex_name_list:
                complex_name_list.append(complex_name)

        top_accuracy_all = {}
        for complex_name in complex_name_list:
            df = data[data['decoy_no'].str.contains(complex_name)]
            df = df.sort_values(by=['prediction','label'], ascending=False).reset_index(drop=True)
            df = df.loc[df['prediction'] > cut_off]

            top_accuracy_list = []
            for k in range(df.shape[0]):
                top_dataframe = df.iloc[0:k+1]
                top_accuracy = top_dataframe["label"].sum(axis=0)
                top_accuracy_list.append(top_accuracy/(k+1))

            # plot_results(complex_name, top_accuracy_list, os.path.join(fir_dir, f'fold_{i}', '20200306-1'))
            top_accuracy_all[complex_name] = top_accuracy_list


        sum_top = np.zeros(top_high)
        for m in range(top_high):
            complex_num = 0
            for complex_name in complex_name_list:
                if len(top_accuracy_all[complex_name]) > m:
                    complex_num += 1
                    sum_top[m] = top_accuracy_all[complex_name][m] +sum_top[m]
            sum_top[m] = sum_top[m]/complex_num

        # plot_results("average", sum_top, os.path.join(fir_dir, f'fold_{i}', '20200306-1'))
        top_results.append(sum_top)

    average_top = []
    for i in range(top_high):
        average=0
        for j in range(num_folds):
            average +=top_results[j][i]
        average = average/num_folds
        average_top.append(average)
    plot_results("average", average_top, os.path.join(fir_dir,'20200327-3-0.7'))



