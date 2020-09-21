import numpy as np
import pandas as pd
from util_function import read_txt
import os

if __name__ == '__main__':

    ###according to classification prediction results selecting the top N samples from regression dataset
    path = "/home/fei/Research/saved_checkpoints/200809_dataset(classification)train1-11_val1-11_testall_vauc_savecheckpoint_GCNE_3layers_max_JK_AllGlobalPooling/all_learner_results.csv"
    dataframe = pd.read_csv(path)

    target_name_path = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/caseID.lst"
    target_name_list = read_txt(target_name_path)

    top = 500

    for target_name in target_name_list:
        dataframe_individual = dataframe[dataframe['complex_name'] == target_name].reset_index(drop=True)
        dataframe_individual = dataframe_individual.sort_values(by=['max'], ascending=False).reset_index(drop=True)
        individual_decoy_name_list = list(dataframe_individual["decoy_no"])

        regression_path = os.path.join("/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/",target_name + "_regression_score_label.csv")
        dataframe_regression = pd.read_csv(regression_path)
        dataframe_regression = dataframe_regression.drop(columns=['Unnamed: 0'])

        regression_name_list = list(dataframe_regression["No"])

        new_regression_dataframe = pd.DataFrame(columns = ["No","iRMS","LRMS","fnat","fnonnat","IOU","CAPRI","DockQ","normalized_IOU","normalized_DockQ"])
        i=0
        for decoy_name in individual_decoy_name_list:
            if decoy_name in regression_name_list:
                new_regression_dataframe = pd.concat([new_regression_dataframe, dataframe_regression[dataframe_regression["No"]==decoy_name].reset_index(drop=True)], axis=0, sort=False).reset_index(drop=True)
                i+=1

            if i>=top:
                break
            else:
                continue

        new_regression_dataframe.to_csv(os.path.join("/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/",target_name + "_new_regression_score_label.csv"), index=False)







