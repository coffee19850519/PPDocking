import os
from util_function import read_txt, write_txt
from graph_creation.extract_node_and_edge import form_dataframe, read_GLY_side_chain, generate_graph_info
from tqdm import tqdm
if __name__ == '__main__':
    involve_atom_list = None
    dis_cut_off = 8.5
    # label_number = 1

    file_dir = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
    pdb_folder = os.path.join(file_dir, "3_pdb")
    decoy_name_folder = os.path.join(file_dir, "5_decoy_name")
    node_feature_folder = os.path.join(file_dir, "7_node_feature")
    native_pdb_folder = os.path.join(file_dir, "1_bm4_dataset", "2_modify_pdb_result")
    GLY_side_chain_folder = os.path.join(file_dir, "9_GLY_sidechain")
    interface_count_folder = os.path.join(file_dir, "8_interface_residue_count")
    receptor_graph_folder = os.path.join(file_dir, "12_receptor_graph", str(dis_cut_off))

    target_name_path = os.path.join(file_dir, "caseID.lst")
    target_name_list = read_txt(target_name_path) # read the complex name

    # data_list = []

    for target_name in tqdm(target_name_list):
        print(target_name)
        decoy_file_path = os.path.join(pdb_folder, target_name)

        ###################################
        # 1.generate graph of receptor
        decoy_r_u_path = os.path.join(decoy_file_path, target_name + '_r_u.pdb.ms')
        rdataframe = form_dataframe(decoy_r_u_path, 'A')  ###get dataframe of chain A of pdb

        ####read GLY SC x, y,z
        GLY_side_chain_path = os.path.join(GLY_side_chain_folder, target_name)
        GLY_dataframe = read_GLY_side_chain(os.path.join(GLY_side_chain_path, os.listdir(GLY_side_chain_path)[0]))

        r_node_list, r_edge_tuple_list, r_edge_feature_list = generate_graph_info(rdataframe, involve_atom_list,
                                                                                  dis_cut_off, GLY_dataframe)
        del GLY_dataframe

        if not os.path.exists(os.path.join(receptor_graph_folder, target_name)):
            os.mkdir(os.path.join(receptor_graph_folder, target_name))



        write_txt(r_node_list, os.path.join(receptor_graph_folder, target_name, "node_list.txt"))


        # write_txt(r_edge_tuple_list, os.path.join(receptor_graph_folder, target_name, "edge_tuple_list.txt"))
        f = open(os.path.join(receptor_graph_folder, target_name, "edge_tuple_list.txt"), 'w')
        for item in r_edge_tuple_list:
            str_item = item[0] + " " + item[1]
            f.write(str_item)
            f.write('\n')
        f.close()



        f = open(os.path.join(receptor_graph_folder, target_name, "edge_feature_list"), 'w')
        for item in r_edge_feature_list:
            str_item = ""
            for feature in item:
                str_item = str_item + str(feature) + " "
            f.write(str_item)
            f.write('\n')
        f.close()
