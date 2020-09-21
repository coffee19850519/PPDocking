import os
import pandas as pd
import random
from util_function import read_txt,write_txt

if __name__ == '__main__':
    root = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    cluster_decoy_folder = os.path.join(root, "4_label")
    decoy_txt_folder = os.path.join(root, "5_decoy_name")

    complex_names = read_txt(os.path.join(root, "caseID.lst"))
    positive_num = 0
    negtive_num = 0
    for complex_name in complex_names:
        print(complex_name)
        positive_dataframe = pd.read_csv(os.path.join(cluster_decoy_folder, complex_name + "_positive_score_label.csv"))
        cluster_dataframe = pd.read_csv(os.path.join(cluster_decoy_folder, complex_name + "_cluster_score_label.csv"))

        # label_have_pt_list = []
        # label_path = os.path.join(decoy_txt_folder, complex_name)
        # for i in range(100):
        #     label_have_pt_list.extend(read_txt(os.path.join(label_path, "label.{:d}.txt".format(i + 1))))

        # label_have_pt_list = list(set(label_have_pt_list))
        # label_have_pt_list = [data.split(" ")[0] for data in label_have_pt_list]

        positive_list = read_txt(os.path.join(decoy_txt_folder, complex_name + "_positive_decoy_name.txt"))
        negtive_list = read_txt(os.path.join(decoy_txt_folder, complex_name + "_negtive_decoy_name.txt"))
        cluster_list = read_txt(os.path.join(decoy_txt_folder, complex_name + "_cluster_decoy_name.txt"))
        negtive_have_pt_list = read_txt(os.path.join(decoy_txt_folder, complex_name + "_negtive_classification_decoy_name.txt"))


        border_list = list(set(cluster_list) - set(negtive_list))
        border_dataframe = cluster_dataframe.loc[cluster_dataframe['No'].isin(border_list)].reset_index(drop=True)

        negtive_list = random.sample(negtive_have_pt_list, int(0.2*(len(border_list) + len(positive_list))))
        negtive_dataframe = cluster_dataframe.loc[cluster_dataframe['No'].isin(negtive_list)].reset_index(drop=True)

        regression_dataframe = pd.concat([positive_dataframe, border_dataframe, negtive_dataframe], axis = 0)

        regression_dataframe.to_csv(os.path.join(decoy_txt_folder, complex_name + "_regression_score_label.csv"), index = False)


















