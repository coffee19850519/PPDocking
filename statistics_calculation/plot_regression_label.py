import pandas as pd
import os
from util_function import read_txt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def read_decoys_and_delete_negatives(label_folder, target_names, label_name):
    for target in target_names:
        decoy_name_path = os.path.join(label_folder, target+ "_regression_score_label.csv")
        decoy_name_df = pd.read_csv(decoy_name_path)   ###get decoy name from positive and negtive txt
        negative_indices = decoy_name_df[(decoy_name_df[label_name] > 0) & (decoy_name_df[label_name] < 0.02) ].index
        negative_num = len(negative_indices)
        drop_indices = np.random.choice(negative_indices, int(negative_num * 0.5), replace=False)
        df_subset = decoy_name_df.drop(drop_indices)
        df_subset.to_csv(decoy_name_path, index=False)

        '''
        pt_file_names = decoy_name_df['No'].map(lambda x: target+ '.' + x + '.pt')
        # do pairing
        left_indics, right_indics = get_ranking_pairs(list(decoy_name_df['DockQ']))
        left_decoy_list.extend(pt_file_names[left_indics])
        right_decoy_list.extend(pt_file_names[right_indics])
        left_dockQ.extend(decoy_name_df['DockQ'][left_indics])
        right_dockQ.extend(decoy_name_df['DockQ'][right_indics])
        '''
def boostrapping_regression_train_data(regression_label_csv_folder, target_names_list, label_name, bins_int):
    for target_name in target_names_list:
        print(target_name)
        csv_path = os.path.join(regression_label_csv_folder, target_name + "_regression_score_label.csv")
        dataframe = pd.read_csv(csv_path)
        IOU = np.array(dataframe[label_name])

        hist, bins_array = np.histogram(IOU, bins=bins_int)

        keep_label_number = hist[2]

        dataframe_boostrapping_all = pd.DataFrame(
            columns=['No', 'iRMS', 'LRMS', 'fnat', 'fnonnat', 'IOU', 'CAPRI', 'DockQ'])
        for i in range(0, bins_int):
            dataframe_boostrapping = dataframe[
                (dataframe[label_name] >= bins_array[i]) & (dataframe[label_name] <= bins_array[i + 1])].reset_index(drop=True)
            if dataframe_boostrapping.shape[0] == 0:
                continue

            if dataframe_boostrapping.shape[0] >= keep_label_number:
                dataframe_boostrapping = dataframe_boostrapping.sample(keep_label_number)  # random sampling
                dataframe_boostrapping_all = pd.concat([dataframe_boostrapping_all, dataframe_boostrapping],
                                                       axis=0).reset_index(drop=True)
            else:
                fold = int(keep_label_number / (dataframe_boostrapping.shape[0]))
                remainder = keep_label_number % (dataframe_boostrapping.shape[0])
                dataframe_boostrapping_fold = dataframe_boostrapping
                while (fold > 1):
                    dataframe_boostrapping_fold = pd.concat([dataframe_boostrapping_fold, dataframe_boostrapping],
                                                            axis=0).reset_index(drop=True)
                    fold -= 1
                dataframe_boostrapping = pd.concat(
                    [dataframe_boostrapping_fold, dataframe_boostrapping.sample(remainder)], axis=0).reset_index(
                    drop=True)
                dataframe_boostrapping_all = pd.concat([dataframe_boostrapping_all, dataframe_boostrapping],
                                                       axis=0).reset_index(drop=True)

        dataframe_boostrapping_all = dataframe_boostrapping_all.sample(frac=1).reset_index(drop=True)
        boostrapping_csv_save_path = os.path.join(regression_label_csv_folder,
                                                  target_name + "_boostrapping_regression_score_label.csv")
        dataframe_boostrapping_all.to_csv(boostrapping_csv_save_path, index=False)

def plot_regression_label_histogram(regression_label_csv_folder, bins_int, target_name, label_name):
    csv_path = os.path.join(regression_label_csv_folder, target_name + "_boostrapping_regression_score_label.csv")
    dataframe = pd.read_csv(csv_path)
    IOU = np.array(dataframe[label_name])
    plt.hist(IOU, bins = bins_int)
    plt.xlabel("range")
    plt.ylabel(label_name)
    plt.title(target_name)
    plt.show()

if __name__ == '__main__':
    file_folder = "/home/fei/Desktop/root_folder/"

    regression_label_csv_folder = os.path.join(file_folder, "5_decoy_name")
    target_name_path = os.path.join(regression_label_csv_folder,"caseID.lst")
    target_names_list = read_txt(target_name_path)

    # read_decoys_and_delete_negatives(regression_label_csv_folder, target_names_list, 'IOU')

    bins_int = 10
    for target_name in target_names_list:
        plot_regression_label_histogram(regression_label_csv_folder, bins_int, target_name, 'IOU')

    # boostrapping_regression_train_data(regression_label_csv_folder, target_names_list, 'IOU', 10)
