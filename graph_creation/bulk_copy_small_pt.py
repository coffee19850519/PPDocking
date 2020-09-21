import os
from shutil import copy
from  multiprocessing import Pool, cpu_count
import pandas as pd
import torch
from util_function import read_txt

def check_pt_quality(folder, file_name, except_file_path):
    # modify_time = os.path.getmtime(os.path.join(folder, file_name))
    # modify_date = time.strftime("%Y-%m-%d", time.localtime(modify_time))
    # now_date = datetime.datetime.now().date()
    # if modify_date == now_date:
    try:
        print('checking file {:s}'.format(file_name))
        pt_file = torch.load(os.path.join(folder, file_name))
        del pt_file
    except:
        print('{:s} file might be corrupted!'.format(file_name))
        with open(except_file_path, 'a') as pf:
            pf.write(file_name + '\n')

def copy_file(src_folder, dst_folder, file_name, except_file_path):
    if os.path.isdir(src_folder) and os.path.isdir(dst_folder):
        print('copying file {:s}'.format(file_name))
        src = os.path.join(src_folder, file_name)
        dst = os.path.join(dst_folder, file_name)
        copy(src, dst)
        # double check copied pt
        try:
            pt = torch.load(dst)
            del pt
        except:
            print('{:s} file might be corrupted!'.format(file_name))
            with open(except_file_path, 'a') as pf:
                pf.write(file_name + '\n')
    else:
        print('please check the folder setting!')

def bulk_copy_different_files(src_folder, dst_folder, except_file_path):
    #src_folder = r'/run/media/coffee/Windows/classification/'
    #dst_folder = r'/run/media/coffee/easystore/classification/'

    diff_file_set = set(os.listdir(src_folder)).difference(set(os.listdir(dst_folder)))

    print('{:d} files will be copied from {:s} to {:s}'.format(len(diff_file_set), src_folder, dst_folder))

    core_num = cpu_count()
    pool = Pool(core_num)

    for file_name in diff_file_set:
        pool.apply_async(copy_file, (src_folder, dst_folder, file_name, except_file_path))

    pool.close()
    pool.join()
    del pool, diff_file_set

def bulk_copy_different_files_by_list(list, src_folder, dst_folder, except_file_path):

    core_num = cpu_count()
    pool = Pool(core_num)

    for file_name in list:
        pool.apply_async(copy_file, (src_folder, dst_folder, file_name, except_file_path))

    pool.close()
    pool.join()
    del pool, list


def read_all_decoys(label_folder, target_list_file):
    target_names = read_txt(target_list_file)
    all_decoy_names = []
    for target in target_names:
        decoy_name_path = os.path.join(label_folder, target+ "_boostrapping_regression_score_label.csv")
        decoy_name_df = pd.read_csv(decoy_name_path)   ###get decoy name from positive and negtive txt
        pt_file_names = decoy_name_df['No'].map(lambda x: target+ '.' + x + '.pt')
        all_decoy_names.extend(pt_file_names)
        del decoy_name_df,decoy_name_path, pt_file_names
    return all_decoy_names

if __name__ == '__main__':
    #folder = r'/run/media/coffee/easystore/classification/'
    # for file in os.listdir(folder):
    #     check_pt_quality(folder, file)

    src_folder = r'/home/fei/Research/processed_all_data/classification/'
    dst_folder = r'/run/media/fei/Windows/hanye/regression_individual_pt/'
    except_file_path = r'/home/fei/Desktop/exception.txt'
    label_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/'
    target_list_file = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/caseID.lst'
    #read all involved regression data into list
    decoy_file_list = read_all_decoys(label_folder, target_list_file)

    bulk_copy_different_files_by_list(decoy_file_list, src_folder, dst_folder, except_file_path)