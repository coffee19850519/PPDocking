import os
import random
import pandas as pd
from util_function import read_txt

def get_decoys_name():
    ###get decoys which have zdock iRMSD less than 6 and save in a txt file

    file_folder = "/run/media/fei/easystore/5_new_pdb_6deg/3_pdb/"
    irmsd_folder = '/run/media/fei/easystore/5_new_pdb_6deg/results/'
    decoy_name_folder = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/"
    target_name_path = os.path.join(file_folder,"caseID_part2.lst")
    target_name_list = []
    with open(target_name_path, 'r') as f:
        for line in f:
            target_name_list.append(line.strip())

    for target_name in target_name_list:
        decoy_positive_name_list= []
        irmsd_path = os.path.join(irmsd_folder,target_name + ".zd3.0.2.irad.out.rmsds")
        with open(irmsd_path, 'r') as f:
            for line in f:
                line_list = line.rstrip("\n").split("\t")
                if float(line_list[1])<=6:
                    decoy_positive_name_list.append("complex." + line_list[0])

        decoy_name_path = os.path.join(decoy_name_folder,target_name+"_positive_decoy_name.txt")
        with open(decoy_name_path,'w') as f:
            f.write("\n".join(decoy_positive_name_list))
            f.write("\n")

def prepare_positive_decoy():
    ####get the true positive decoys and save in the original csv
    file_folder = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    label_folder = '/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/4_label/'
    # decoy_name_folder = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/"
    target_name_path = os.path.join(file_folder,"caseID.lst")
    target_name_list = []
    with open(target_name_path, 'r') as f:
        for line in f:
            target_name_list.append(line.strip())

    for target_name in target_name_list:
        label_path = os.path.join(label_folder, target_name + '_positive_score_label.csv')
        data = pd.read_csv(label_path)
        data = data.loc[data['CAPRI'] > 0].reset_index(drop=True)
        data.to_csv(label_path, index=False)

def return_positive_decoy():
    file_folder = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    label_folder = '/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/4_label/'
    decoy_name_folder = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/"
    target_name_path = os.path.join(file_folder,"caseID.lst")
    target_name_list = read_txt(target_name_path)

    for target_name in target_name_list:
        label_path = os.path.join(label_folder, target_name + "_positive_score_label.csv")
        label_dataframe = pd.read_csv(label_path)
        decoy_name_list = label_dataframe['No'].tolist()
        decoy_name_path = os.path.join(decoy_name_folder, target_name + "_positive_decoy_name.txt")

        f = open(decoy_name_path, 'w')
        for decoy_name in decoy_name_list:
            f.write(decoy_name)
            f.write('\n')
        f.close()


def get_regression_data_number():
    decoy_name_folder = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/"
    file_folder = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    target_name_path = os.path.join(file_folder, "caseID.lst")
    target_name_list = read_txt(target_name_path)
    sum = 0
    for target_name in target_name_list:
        label_path = os.path.join(decoy_name_folder, target_name + "_regression_score_label.csv")
        label_dataframe = pd.read_csv(label_path)
        print(target_name + ":" + str(label_dataframe.shape[0]))
        sum += label_dataframe.shape[0]
        print("sum:{:d}".format(sum))


def get_positive_data_number():
    decoy_name_folder = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/"
    file_folder = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    target_name_path = os.path.join(file_folder, "caseID.lst")
    target_name_list = read_txt(target_name_path)
    for target_name in target_name_list:
        label_path = os.path.join(decoy_name_folder, target_name + "_positive_decoy_name.txt")
        label_list = read_txt(label_path)
        print(target_name + ":" + str(len(label_list)))


if __name__ == '__main__':
    # prepare_positive_decoy()
    #return_positive_decoy()

    get_regression_data_number()
    # get_positive_data_number()