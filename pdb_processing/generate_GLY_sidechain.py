from util_function import read_txt
import os
import sys
import pandas as pd
from argparse import ArgumentParser
from scipy.spatial.distance import euclidean, pdist, squareform, cdist
import numpy as np
from tqdm import tqdm
import json

if __name__ == '__main__':
    file_dir = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    pdb_folder = "/run/media/fei/easystore/5_new_pdb_6deg/3_pdb"
    target_name_path = os.path.join(file_dir, "caseID_part2.lst")
    decoy_name_folder = os.path.join(file_dir, "5_decoy_name")
    GLY_sidechain_folder = os.path.join(file_dir, "9_GLY_sidechain")
    exec_path = os.path.join(file_dir, "9_GLY_sidechain","reduce-master", "reduce_src", "reduce")

    target_name_list = read_txt(target_name_path)
    for target_name in target_name_list:
        ###1.decoy pdb
        decoy_name_list = read_txt(os.path.join(decoy_name_folder, target_name + "_positive_decoy_name.txt"))\
                          + read_txt(os.path.join(decoy_name_folder, target_name + "_cluster_decoy_name.txt"))
        for decoy_name in tqdm(decoy_name_list):
            decoy_path = os.path.join(pdb_folder, target_name, decoy_name)
            decoy_H_path = os.path.join(GLY_sidechain_folder, target_name, "H." + decoy_name)
            GLY_pdb = os.path.join(GLY_sidechain_folder, target_name, "GLY." + decoy_name)
            if not os.path.exists(os.path.join(GLY_sidechain_folder, target_name)):
                os.mkdir(os.path.join(GLY_sidechain_folder, target_name))

            if not os.path.exists(GLY_pdb):
                # "./reduce -NOFLIP /home/fei/Desktop/complex.4.pdb > /home/fei/Desktop/complexH.4.pdb"
                cmd = exec_path + ' -NOFLIP ' + decoy_path + " > " + decoy_H_path
                os.system(cmd)
                pdb_lines = []
                with open(decoy_H_path, 'r') as file:
                    line = file.readline().rstrip('\n')  # call readline()
                    while line:
                        if (line[0:6].strip(" ") == 'ATOM' and line[17:20].strip(" ") == 'GLY' and line[12:16].strip(" ") == 'HA2'):
                            pdb_lines.append(line)
                        line = file.readline().rstrip('\n')
                os.remove(decoy_H_path)
                f = open(GLY_pdb, 'w')
                f.write("\n".join(pdb_lines))
                f.write("\n")
                f.close()

        ##2.native DSSP
        # decoy_path = os.path.join(pdb_folder, target_name+".pdb")
        # decoy_H_path = os.path.join(GLY_sidechain_folder, target_name, "H." + target_name+".pdb")
        # GLY_pdb = os.path.join(GLY_sidechain_folder, target_name, "GLY."+ target_name+".pdb")
        # if not os.path.exists(os.path.join(GLY_sidechain_folder, target_name)):
        #     os.mkdir(os.path.join(GLY_sidechain_folder, target_name))
        # # "./reduce -NOFLIP /home/fei/Desktop/complex.4.pdb > /home/fei/Desktop/complexH.4.pdb"
        # cmd = exec_path + ' -NOFLIP ' + decoy_path + " > " + decoy_H_path
        # os.system(cmd)
        # pdb_lines = []
        # with open(decoy_H_path, 'r') as file:
        #     line = file.readline().rstrip('\n')  # call readline()
        #     while line:
        #         if (line[0:6].strip(" ") == 'ATOM' and line[17:20].strip(" ") == 'GLY' and line[12:16].strip(
        #                 " ") == 'HA2'):
        #             pdb_lines.append(line)
        #         line = file.readline().rstrip('\n')
        # os.remove(decoy_H_path)
        # f = open(GLY_pdb, 'w')
        # f.write("\n".join(pdb_lines))
        # f.write("\n")
        # f.close()
