import os
import random
import pandas as pd
from util_function import read_txt,write_txt

if __name__ == '__main__':
    root = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    cluster_decoy_folder = os.path.join(root, "4_label")
    decoy_txt_folder = os.path.join(root, "5_decoy_name")

    complex_names = read_txt(os.path.join(root, "caseID.lst"))
    positive_num = 0
    negtive_num = 0
    for complex_name in complex_names:
        # ###generate negtive decoys list(IOU ==0)
        # cluster_decoy_path = os.path.join(cluster_decoy_folder, complex_name + "_cluster_score_label.csv")
        # cluster_decoy_df = pd.read_csv(cluster_decoy_path)
        #
        # if cluster_decoy_df["CAPRI"].sum(axis=0) > 0:
        #     print(complex_name + "has positive samples!")
        #
        # negtive_decoy_df = cluster_decoy_df.loc[cluster_decoy_df['IOU']==0].sort_values(by=['No'], ascending=True).reset_index(drop=True)
        # negtive_pdb = list(negtive_decoy_df['No'])
        # negtive_pdb_path = os.path.join(os.path.join(root, "5_decoy_name"), complex_name + "_negtive_decoy_name.txt")
        # write_txt(negtive_pdb,negtive_pdb_path)
        positive_list = read_txt(os.path.join(decoy_txt_folder,  complex_name + "_positive_decoy_name.txt"))
        negtive_list = read_txt(os.path.join(decoy_txt_folder, complex_name + "_negtive_decoy_name.txt"))

        label_list = []
        label_path = os.path.join(decoy_txt_folder, complex_name)
        for i in range(100):
            label_list.extend(read_txt(os.path.join(label_path,"label.{:d}.txt".format(i+1))))

        label_list = list(set(label_list))
        label_list = [data.split(" ")[0] for data in label_list]
        negtive_classification_list = list(set(label_list).intersection(set(negtive_list)))

        write_txt(negtive_classification_list, os.path.join(decoy_txt_folder, complex_name + "_negtive_classification_decoy_name.txt"))

    #     # print("{:s} positive number and negtive number is {:d} and {:d}".format(complex_name,len(positive_list),len(negtive_list)))
    #     print("The ratio of {:s} positive number and negtive number is {:f}".format(complex_name, float(len(negtive_list))/len(positive_list)))
    #
    #     positive_num += len(positive_list)
    #     negtive_num += len(negtive_list)
    #
    # # print("All of positive number and negtive number is {:d} and {:d}".format(positive_num, negtive_num))
    # print("The ratio of all positive number and negtive number is {:f}".format(float(negtive_num)/positive_num))







