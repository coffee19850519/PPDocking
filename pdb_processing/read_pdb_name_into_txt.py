import os
import os.path as osp
from util_function import read_txt,write_txt

def read_pdb_name():
    filePath = "/home/fei/Research/processed_all_data/classification/"
    for pdb_name in os.listdir(filePath):
        complex_name = pdb_name.split(".")[0]

        with open(os.path.join("/home/fei/Research/processed_all_data", complex_name + ".txt"), 'a') as f:
            f.write(pdb_name)
            f.write('\n')

def read_1_to_n_label_txt():
    fir_dir = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    decoy_name_folder = osp.join(fir_dir, "5_decoy_name")
    complex_list = read_txt(osp.join(fir_dir, "caseID.lst"))
    max_label = 23

    for complex_name in complex_list:
        decoy_name_list = []
        for i in range(1, max_label+1):
            label_path = osp.join(decoy_name_folder, complex_name, "label.{:d}.txt".format(i))
            label_list = read_txt(label_path)
            for label in label_list:
                decoy_name_list.append(complex_name + "." + label.split(" ")[0])
        decoy_name_list = list(set(decoy_name_list))

        label_1_to_n_path =osp.join("/home/fei/Research/label_1-n/", complex_name +".txt")
        write_txt(decoy_name_list, label_1_to_n_path)
        del decoy_name_list, label_list

if __name__ == '__main__':
    # read_pdb_name()
    read_1_to_n_label_txt()

