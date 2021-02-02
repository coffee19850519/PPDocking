import os
from pairwise_point_cloud.triple_loss_point_cloud_convertor import read_target_name

interface_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/14_interface_info/'
complex_list = read_target_name (r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/caseID.lst')


for complex in complex_list:
    native_file_folder = os.path.join(interface_folder, complex)
    os.rename(
        os.path.join(native_file_folder, 'native_A.pdb'),
        os.path.join(native_file_folder, 'nativeA.pdb')
    )
    os.rename(
        os.path.join(native_file_folder, 'native_B.pdb'),
        os.path.join(native_file_folder, 'nativeB.pdb')
    )