import os
from util_function import read_txt
from tqdm import tqdm

if __name__ == '__main__':
    fir_dir = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    pdb_folder = os.path.join(fir_dir, "3_pdb")
    decoy_name_folder = os.path.join(fir_dir, "5_decoy_name")

    target_name_path = os.path.join(fir_dir, "caseID_test.lst")
    target_name_list = read_txt(target_name_path) # read the complex name

    for target_name in target_name_list:
        print(target_name)
        decoy_name_list = read_txt(os.path.join(decoy_name_folder, target_name + "_positive_decoy_name.txt")) \
                          + read_txt(os.path.join(decoy_name_folder, target_name + "_cluster_decoy_name.txt"))

        target_pdb_path = os.path.join(pdb_folder, target_name)
        for item in tqdm(os.listdir(target_pdb_path)):
            if os.path.splitext(item)[1] == ".pdb":
                if item not in decoy_name_list:
                    os.remove(os.path.join(target_pdb_path, item))


