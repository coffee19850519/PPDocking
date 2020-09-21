import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from util_function import read_txt
import os

def normalize_label(df, column_name):
    scaler = MinMaxScaler()
    df['normalized_' + column_name] = scaler.fit_transform(X= np.array(df[column_name], np.float).reshape((-1,1)))

def normalize_all_decoys_in_target(label_file):
    df = pd.read_csv(label_file)
    normalize_label(df, 'IOU')
    normalize_label(df, 'DockQ')
    df.to_csv(label_file)



if __name__ == '__main__':
    label_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/'
    target_names = read_txt(os.path.join(label_folder, 'caseID.lst'))
    for target_name in target_names:
        normalize_all_decoys_in_target(os.path.join(label_folder, target_name+'_boostrapping_regression_score_label.csv'))

    # dict = read_all_normalized_label_to_dict(label_folder, target_names, 'IOU')
    # pass
    # mini_batch = df['normalized_IOU'].sample(16)
    # sigmoid = expit(mini_batch)
    # sm = softmax(mini_batch)
    # pass