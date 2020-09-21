import os.path as osp
from util_function import read_txt
if __name__ == '__main__':
    file_dir = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    decoy_name_folder= osp.join(file_dir, "5_decoy_name")
    complex_name_list = read_txt(osp.join(file_dir, "caseID.lst"))

    positive_sum = 0
    negtive_sum = 0
    for complex_name in complex_name_list:
        positive_list = read_txt(osp.join(decoy_name_folder, complex_name + "_positive_decoy_name.txt"))
        negtive_list = read_txt(osp.join(decoy_name_folder, complex_name + "_cluster_decoy_name.txt"))
        if list(set(positive_list) & set(negtive_list)):
            print("There is intersection in the {:s}".format(complex_name))
        print(complex_name + " " + str(len(positive_list)) + " " + str(len(negtive_list)) )
        positive_sum+=len(positive_list)
        negtive_sum+=len(negtive_list)

    print("The positive number is {:d}".format(positive_sum))
    print("The negtive number is {:d}".format(negtive_sum))
