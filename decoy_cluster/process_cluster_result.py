import os
import random
from shutil import copyfile
from util_function import read_txt, write_txt


def choose_random_from_cluster(target_name_list):
    ###random select decoys from cluster results
    positive_name_folder = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/"

    for target_name in target_name_list:
        print(target_name)
        decoy_name_list = []
        positive_name_path = os.path.join(positive_name_folder, target_name + "_positive_decoy_name.txt")
        positive_name_list = read_txt(positive_name_path)

        cluster_result_path = os.path.join("/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/6_cluster_result/",target_name+"_final_result_cluster.out")
        with open(cluster_result_path, 'r') as f:
            for line in f:
                line_list = line.rstrip("\n").split(" ")[3:]
                random_decoy = random.sample(line_list,1)
                while random_decoy[0] in positive_name_list:
                    print(random_decoy + "is a positive decoy, which should be avoided")
                    random_decoy = random.sample(line_list, 1)

                decoy_name_list.append(random_decoy[0] + ".pdb")

        decoy_name_path = os.path.join(decoy_name_folder, target_name+"_cluster_decoy_name.txt")
        with open(decoy_name_path,'w') as f:
            f.write("\n".join(decoy_name_list))
            f.write("\n")


def move_cluster_result(target_name_list):
    cluster_result_folder = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/6_cluster_result/"
    for target_name in target_name_list:
        cluster_result_path = os.path.join(pdb_folder,target_name,"final_result_cluster.out")
        dist = os.path.join(cluster_result_folder,target_name +"_final_result_cluster.out")
        copyfile(cluster_result_path, dist)


# def split_cluster_negtive_result(target_name_list):
#     ##split cluster result into different parts which have equal items witn positive decoys
#     for target_name in target_name_list:
#         positive_path = os.path.join(decoy_name_folder, target_name + "_positive_decoy_name.txt")
#         cluster_path = os.path.join(decoy_name_folder, target_name + "_cluster_decoy_name.txt")
#
#         positive_list = read_txt(positive_path)
#         cluster_list = read_txt(cluster_path)
#         # print(target_name + ":" + str(len(list(set(positive_list) & set(cluster_list)))))
#
#         folder = os.path.join(decoy_name_folder, target_name)
#         if not os.path.exists(folder):
#             os.mkdir(folder)
#
#         positive_list_label = [item + " 1" for item in positive_list]
#         for i in range(100):
#             all_decoy_path = os.path.join(folder, "label." + str(i + 1) + ".txt")
#             random_negtive_list = random.sample(cluster_list, len(positive_list))
#
#             if not list(set(positive_list) & set(random_negtive_list)):
#
#                 random_negtive_list_label = [item + " 0" for item in random_negtive_list]
#                 all_decoy_list = positive_list_label + random_negtive_list_label
#
#             write_txt(all_decoy_list, all_decoy_path)
#
#
#         # print("target name: " + target_name)
#         # print("positive length:" + str(len(positive_list)))
#         # print("cluster length:" + str(len(cluster_list)))
#         # print("ratio:" + str(int(len(cluster_list)/len(positive_list))))

def split_cluster_negtive_result(target_name_list, subset_num):
    ##split cluster result into different parts which have equal items witn positive decoys
    for target_name in target_name_list:
        print(target_name)
        positive_path = os.path.join(decoy_name_folder, target_name + "_positive_decoy_name.txt")
        negtive_classification_path = os.path.join(decoy_name_folder, target_name + "_negtive_classification_decoy_name.txt")

        positive_list = read_txt(positive_path)
        negtive_classification_list = read_txt(negtive_classification_path)
        # print(target_name + ":" + str(len(list(set(positive_list) & set(cluster_list)))))

        folder = os.path.join(decoy_name_folder,"classification_label_subset_name",target_name)
        if not os.path.exists(folder):
            os.mkdir(folder)

        positive_num = len(positive_list)
        negtive_classification_num = len(negtive_classification_list)
        positive_list_label = [item + " 1" for item in positive_list]

        times = float(negtive_classification_num/positive_num)

        start = 0
        end = positive_num

        for i in range(subset_num):
            all_decoy_path = os.path.join(folder, "label." + str(i + 1) + ".txt")

            if times >= subset_num:
                random_negtive_list = negtive_classification_list[start:end]
                start = end
                end = end + positive_num

            elif times<1:
                random_negtive_list = negtive_classification_list + random.sample(negtive_classification_list, positive_num-negtive_classification_num)
            else:
                random_negtive_list = random.sample(negtive_classification_list, len(positive_list))

            if not list(set(positive_list) & set(random_negtive_list)):
                random_negtive_list_label = [item + " 0" for item in random_negtive_list]
                all_decoy_list = positive_list_label + random_negtive_list_label

            write_txt(all_decoy_list, all_decoy_path)

        # print("target name: " + target_name)
        # print("positive length:" + str(len(positive_list)))
        # print("cluster length:" + str(len(cluster_list)))
        # print("ratio:" + str(int(len(cluster_list)/len(positive_list))))

def check_inter_of_results(target_name_list):
    for target_name in target_name_list:
        print(target_name)
        positive_path = os.path.join(decoy_name_folder, target_name + "_positive_decoy_name.txt")
        cluster_path = os.path.join(decoy_name_folder, target_name + "_cluster_decoy_name.txt")

        positive_list = read_txt(positive_path)
        cluster_list = read_txt(cluster_path)

        inter_list = list(set(positive_list) & set(cluster_list))

        if inter_list:
            cluster_result_path = os.path.join(
                "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/6_cluster_result/",
                target_name + "_final_result_cluster.out")

            for inter_item in inter_list:
                with open(cluster_result_path, 'r') as f:
                    for line in f:
                        line_list = line.rstrip("\n").split(" ")[3:]
                        line_list = [item + ".pdb" for item in line_list]   ###keep the same format with positive_list and cluster_list

                        if inter_item in line_list:
                            line_list = list(set(line_list).difference(set(positive_list)))
                            if len(line_list) > 0:
                                random_decoy = random.sample(line_list, 1)
                                while random_decoy[0] in positive_list:
                                    print(random_decoy[0] + " is a positive decoy, which should be avoided")
                                    random_decoy = random.sample(line_list, 1)
                                else:
                                    cluster_list.remove(inter_item)
                                    cluster_list.append(random_decoy[0])
                            else:
                                cluster_list.remove(inter_item)
            if not list(set(positive_list) & set(cluster_list)):
                write_txt(cluster_list, cluster_path)
                print("Saving!")
            else:
                print("The result is not correct!")


if __name__ == '__main__':

    file_folder = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    pdb_folder = os.path.join(file_folder, "3_pdb")
    decoy_name_folder = os.path.join(file_folder, "5_decoy_name")
    subset_num = 25

    target_name_path = os.path.join(file_folder,"caseID.lst")
    target_name_list = []
    with open(target_name_path, 'r') as f:
        for line in f:
            target_name_list.append(line.strip())

    # choose_random_from_cluster(target_name_list)
    # move_cluster_result(target_name_list)
    # check_inter_of_results(target_name_list)

    split_cluster_negtive_result(target_name_list, subset_num)




