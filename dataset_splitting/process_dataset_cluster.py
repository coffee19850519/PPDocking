import os
from util_function import read_txt, write_txt

def modify_fold(i, initial_list, target_dic, style):
    new_list = initial_list.copy()
    for target in initial_list:
        if target in target_dic.keys():
            new_list.extend(target_dic[target])

    new_path = os.path.join(data_split_folder,"new_dataset_split_result",i)
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    write_txt(new_list, os.path.join(new_path,style + ".txt"))


if __name__ == '__main__':
    fir_dir = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    data_split_folder = os.path.join(fir_dir, "11_dataset_splitting")
    cluster_result = os.path.join(data_split_folder,"Fw_Re_108_u_pdb", "binding_cdhit_o.txt")

    target_dic = {}
    with open(cluster_result, 'r') as file:
        line = file.readline() # call readline()
        while line:
            if (line[5] == ':'):
                center_target = line.rstrip("\n")[1:5]
                other_target = line.rstrip("\n")[6:10]

                if center_target in target_dic.keys():
                    target_dic[center_target].append(other_target)
                else:
                    target_dic[center_target]=[]
                    target_dic[center_target].append(other_target)
            line = file.readline()
    print("Done!")

    initial_split_folder = os.path.join(data_split_folder, "Fw_Re_108_u_pdb", "data_b_name")
    for i in range(5):
        initial_split_fold_folder = os.path.join(initial_split_folder, str(i))
        # initial_split_fold_train_path = os.path.join(initial_split_fold_folder, "train.txt")
        # initial_split_fold_val_path = os.path.join(initial_split_fold_folder, "val.txt")
        initial_split_fold_test_path = os.path.join(initial_split_fold_folder, "test.txt")

        # modify_fold(str(i), read_txt(initial_split_fold_train_path), target_dic, "train")
        # modify_fold(str(i), read_txt(initial_split_fold_val_path), target_dic, "val")
        modify_fold(str(i), read_txt(initial_split_fold_test_path), target_dic, "test")


