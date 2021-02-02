from multiprocessing import cpu_count, Pool
import pandas as pd
import numpy as np
import os, torch
from util_function import read_txt


def calculate_atom_count_in_residual(data):
    total = 0
    for idx in range(0, len(data)):
        if (data[idx, 0:5] == np.array([0.008, 0.134, -0.475, -0.039, 0.181], np.float32)).all():
            total += 115 * data[idx,-7] #ALA
        elif (data[idx, 0:5] == np.array([0.171, -0.361, 0.107, -0.258, -0.364], np.float32)).all():
            total += 225 * data[idx,-7] #ARG
        elif (data[idx, 0:5] == np.array([0.255, 0.038, 0.117, 0.118, -0.055], np.float32)).all():
            total += 160 * data[idx,-7] #ASN
        elif (data[idx, 0:5] == np.array([0.303, -0.057, -0.014, 0.225, 0.156], np.float32)).all():
            total += 150 * data[idx,-7] #ASP
        elif (data[idx, 0:5] == np.array([-0.132, 0.174, 0.070, 0.565, -0.374], np.float32)).all():
            total += 135 * data[idx,-7] #CYS
        elif (data[idx, 0:5] == np.array([0.149, -0.184, -0.030, 0.035, -0.112], np.float32)).all():
            total += 180 * data[idx,-7] #GLN
        elif (data[idx, 0:5] == np.array([0.221, -0.280, -0.315, 0.157, 0.303], np.float32)).all():
            total += 190 * data[idx,-7] #GLU
        elif (data[idx, 0:5] == np.array([0.218, 0.562, -0.024, 0.018, 0.106], np.float32)).all():
            total += 75 * data[idx,-7] #GLY
        elif (data[idx, 0:5] == np.array([0.023, -0.177, 0.041, 0.280, -0.021], np.float32)).all():
            total += 195 * data[idx,-7] #HIS
        elif (data[idx, 0:5] == np.array([-0.353, 0.071, -0.088, -0.195, -0.107], np.float32)).all():
            total += 175 * data[idx,-7] #ILE
        elif (data[idx, 0:5] == np.array([-0.267, 0.018, -0.265, -0.274, 0.206], np.float32)).all():
            total += 170 * data[idx,-7] #LEU
        elif (data[idx, 0:5] == np.array([0.243, -0.339, -0.044, -0.325, -0.027], np.float32)).all():
            total += 200 * data[idx,-7] #LYS
        elif (data[idx, 0:5] == np.array([-0.239, -0.141, -0.155, 0.321, 0.077], np.float32)).all():
            total += 185 * data[idx,-7] #MET
        elif (data[idx, 0:5] == np.array([-0.329, -0.023, 0.072, -0.002, 0.208], np.float32)).all():
            total += 210 * data[idx,-7] #PHE
        elif (data[idx, 0:5] == np.array([0.173, 0.286, 0.407, -0.215, 0.384], np.float32)).all():
            total += 145 * data[idx,-7] #PRO
        elif (data[idx, 0:5] == np.array([0.199, 0.238, -0.015, -0.068, -0.196], np.float32)).all():
            total += 115 * data[idx,-7] #SER
        elif (data[idx, 0:5] == np.array([0.068, 0.147, -0.015, -0.132, -0.274], np.float32)).all():
            total += 140 * data[idx,-7] #THR
        elif (data[idx, 0:5] == np.array([-0.296, -0.186, 0.389, 0.083, 0.297], np.float32)).all():
            total += 255 * data[idx,-7] #TRP
        elif (data[idx, 0:5] == np.array([-0.141, -0.057, 0.425, -0.096, -0.091], np.float32)).all():
            total += 230 * data[idx,-7] #TYR
        elif (data[idx, 0:5] == np.array([-0.274, 0.136, -0.187, -0.196, -0.299], np.float32)).all():
            total += 155 * data[idx,-7] #VAL
        elif (data[idx, 0:5] == np.array([0, 0, 0, 0, 0], np.float32)).all():
            total += 0 #ASX/SEC/GLX
    return total

def calculate_gt_for_decoy(decoy_name, target_name, pt_folder):
    try:
        data = torch.load(os.path.join(pt_folder, target_name + '.' + decoy_name + '.pt'))
        x = data['x'][data['interface_pos']].detach().cpu().numpy()
        interface_atom_num = calculate_atom_count_in_residual(x)
        del data, x
        return interface_atom_num
    except Exception as e:
        print(e)
        return 0


def calculate_gt_for_target(label_folder, pt_folder, target_name):
    print(target_name + ": start!")

    decoy_list = []
    positive_list = read_txt(os.path.join(label_folder, target_name + "_positive_decoy_name.txt"))
    category_list = np.ones_like(positive_list, np.int).tolist()
    negative_list = read_txt(os.path.join(label_folder, target_name + "_cluster_decoy_name.txt"))
    category_list.extend(np.zeros_like(negative_list, np.int).tolist())
    decoy_list.extend(positive_list)
    decoy_list.extend(negative_list)
    # scores_data_frame['interface_node_num'] = scores_data_frame.apply(lambda row: calculate_gt_for_decoy(row['No'],target_name, pt_folder), axis=1)
    decoy_interface_area = []
    for decoy in decoy_list:
        decoy_interface_area.append(calculate_gt_for_decoy(decoy,target_name, pt_folder))

    print('all decoys in target {:s} done, will save the results to csv file'.format(target_name))
    dict = {'decoy_name' : decoy_list, 'decoy_interface_area': decoy_interface_area, 'category': category_list}

    scores_data_frame = pd.DataFrame(dict)
    label_path = os.path.join(label_folder, target_name + "_decoy_asa.csv")
    scores_data_frame.to_csv(label_path, index=False)
    del scores_data_frame, dict, decoy_list, decoy_interface_area, positive_list, negative_list, category_list
    print(target_name + ": end!")



def calulate_ground_truth(target_name_list, label_folder,pt_folder):

    # declare process pool
    core_num = cpu_count()
    pool = Pool(core_num)


    for target_name in target_name_list:
        pool.apply_async(calculate_gt_for_target, (label_folder, pt_folder, target_name))
        # calculate_gt_for_target(label_folder, pt_folder, target_name)

    pool.close()
    pool.join()
    del pool


if __name__ == '__main__':
    label_folder = r"/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/"
    pt_folder = r'/run/media/fei/easystore/classification/'
    target_name_path = r"/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/caseID_temp.lst"
    target_name_list = []
    with open(target_name_path,'r') as f:
        for line in f:
            target_name_list.append(line.strip())
    calulate_ground_truth(target_name_list, label_folder,pt_folder)