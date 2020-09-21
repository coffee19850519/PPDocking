import Bio.PDB
import os.path
from tqdm import tqdm
from pdb_processing.renumber_pdb import renumber_residue_number
import numpy as np
import pandas as pd

def get_pdb_chains(pdb):
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)
    pdb_struct = pdb_parser.get_structure("reference", pdb)[0]
    chain=[]
    for c in pdb_struct:
        chain_name = c.id
        if chain_name.strip()!="":
            chain.append(chain_name.strip())

    return chain


def change_chain(pdb_string,chain):
    new_str=[]
    for line in pdb_string:
        s=list(line.rstrip("\n"))
        s[21]=chain
        new_str.append("".join(s))
    return "\n".join(new_str)

def correct_chain_name(pdb, chain_name):
    pdb_lines= []
    f = open(pdb,'r')
    for line in f.readlines():
        if line[0:4].strip() == "ATOM":
            pdb_lines.append(line)
    f.close()

    f = open(pdb, 'w')
    f.write(change_chain(pdb_lines, chain_name))
    f.write("\n")
    print(pdb.split('/')[-1] + " is been corrected!")
    f.write("END\n")
    f.close()


def get_rid_of_nonatom(pdb):

    ####################################
    #GET RID OF HETATM, TER, CONNECT
    ####################################

    new_file_text = ""

    with open(pdb, "r") as f:
        #for line in f.readlines():
        for line in filter(None, f.readlines()):
            line = line.rstrip("\n")
            if line[0:6].strip() != "HETATM" and line[0:6].strip() != "CONECT" and line[0:6].strip() != "TER":
                new_file_text += line + '\n'

    ####Write new pdb text into pdb file
    f = open(pdb, 'w')
    f.write(new_file_text)
    f.close()
    print(pdb + " has been corrected!")

def check_line_length(pdb):
    with open(pdb, "r") as f:
        for line in filter(None, f.readlines()):
            line = line.rstrip("\n")
            if len(line) > 82:
                print(pdb + " has line longer than 82!")



def check_chain_number(pdb):
    chain_name = get_pdb_chains(pdb)
    if len(chain_name) ==1:
        # print(pdb.split('/')[-1] + " has " + str(len(chain_name)) + " chains")
        # print(pdb.split('/')[-1])
        return True

    return False


def check_residue_number(pdb):
    old_number = -100
    with open(pdb, "r") as f:
        for line in filter(None, f.readlines()):
            line = line.rstrip("\n")
            if line[0:4].strip() == "ATOM":
                if line[26].strip()!='':
                    print(pdb.split('/')[-1])
                    return True
                residue_number = int(line[22:26].strip())
                if residue_number < old_number:
                    print(pdb.split('/')[-1])
                    return True
                old_number = residue_number
    return False


def form_dataframe(decoy_path):
    """
    :param the path of decoy pdb
    :return: transform pdb into dataframe
    """
    atomic_data = []
    with open(decoy_path, 'r') as file:
        line = file.readline().rstrip('\n')  # call readline()
        while line:
            if (line[0:6].strip(" ") == 'ATOM'):
                data = dict()
                ####the position of the pdb columns has been double checked
                data["record_name"] = line[0:6].strip(" ")
                data["serial_number"] = int(line[6:11].strip(" "))
                data["atom"] = line[12:16].strip(" ")
                data["resi_name"] = line[17:20].strip(" ")

                ########repalce special residues using normal residues??????????????????????????????????????????????????
                ###???????????????????????????????
                ### cheque
                #################################
                if data["resi_name"] == "MSE":
                    data["resi_name"] = "MET"
                elif data["resi_name"] == "DVA":
                    data["resi_name"] = "VAL"
                elif data["resi_name"] == "SOC":
                    data["resi_name"] = "CYS"
                elif data["resi_name"] == "XXX":
                    data["resi_name"] = "ALA"
                elif data["resi_name"] == "FME":
                    data["resi_name"] = "MET"
                elif data["resi_name"] == "PCA":
                    data["resi_name"] = "GLU"
                elif data["resi_name"] == "5HP":
                    data["resi_name"] = "GLU"
                elif data["resi_name"] == "SAC":
                    data["resi_name"] = "SER"
                elif data["resi_name"] == "CCS":
                    data["resi_name"] = "CYS"

                data["chain_id"] = line[21].strip(" ")
                data["resi_num"] = int(line[22:26].strip(" "))
                data["x"] = float(line[30:38].strip(" "))
                data["y"] = float(line[38:46].strip(" "))
                data["z"] = float(line[46:54].strip(" "))


                data["node_id"] = data["chain_id"] + str(data["resi_num"]) + data["resi_name"]
                atomic_data.append(data)

            line = file.readline()


    return atomic_data

def check_missing_residue(pdb):

    atomic_data = form_dataframe(pdb)
    dataframe = pd.DataFrame(atomic_data)

    for g, d in dataframe.query("record_name == 'ATOM'").groupby(  # add the CA of all residue
            ["node_id", "chain_id", "resi_num", "resi_name"]
    ):
        node_id, chain_id, resi_num, resi_name = g
        # print(node_id)

        if 'N' not in d["atom"].values:
            print(pdb.split("/")[-1])
            print(node_id + " does not have N atom!")
        if 'CA' not in d["atom"].values:
            print(pdb.split("/")[-1])
            print(node_id + " does not have CA atom!")
        if 'C' not in d["atom"].values:
            print(pdb.split("/")[-1])
            print(node_id + " does not have C atom!")
        if 'O' not in d["atom"].values:
            print(pdb.split("/")[-1])
            print(node_id + " does not have O atom!")


        # try:
        #     df_CA = d.query("atom == 'CA'")
        #     if df_CA.empty != True:
        #         x = df_CA["x"].values[0]
        #         y = df_CA["y"].values[0]
        #         z = df_CA["z"].values[0]
        #
        #         self.add_node(
        #             node_id,
        #             chain_id=chain_id,
        #             resi_num=resi_num,
        #             resi_name=resi_name,
        #             x=x,
        #             y=y,
        #             z=z,
        #             features=None
        #         )
        #     else:
        #         self.dataframe = self.dataframe.drop(d.index, axis=0)
        # except:
        #     raise ValueError("Cannot find CA in " + node_id)

def generate_case_ID(file_path):
    target_name_list = []
    for target_name in os.listdir(file_path):
        target_name_list.append(target_name.split("_")[0])

    target_name_list = list(set(target_name_list))
    return target_name_list




if __name__ == '__main__':

    # file_path = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/1_bm4_dataset/2_modify_pdb_result/"
    file_path = "/home/fei/Research/Dataset/benchmark_PDB/complex_have_two_chains/"
    # target_name_list = generate_case_ID(file_path)
    target_name_path = "/home/fei/Research/Dataset/benchmark_PDB/complex_have_two_chains.txt"
    target_name_list = []
    with open(target_name_path, 'r') as f:
        for line in f:
            target_name_list.append(line.strip())
    f.close()

    for target_name in target_name_list:
        native1 = os.path.join(file_path, target_name + "_b1.pdb")
        native2 = os.path.join(file_path, target_name + "_b2.pdb")
        unbound1 = os.path.join(file_path, target_name + "_u1.pdb")
        unbound2 = os.path.join(file_path, target_name + "_u2.pdb")

        ############1_get rid of non atom #######################
        # get_rid_of_nonatom(native1)
        # get_rid_of_nonatom(native2)
        # get_rid_of_nonatom(unbound1)
        # get_rid_of_nonatom(unbound2)
        #############1_get rid of non atom#######################

        #############2_check line length#######################
        # check_line_length(native1)
        # check_line_length(native2)
        # check_line_length(unbound1)
        # check_line_length(unbound2)
        #############2_correct chain name#######################

        #############3_check chain number, if number more than 1, remove extra chain############
        # if check_chain_number(native1) and check_chain_number(native2) and check_chain_number(unbound1) and check_chain_number(unbound2):
        #     print((native1.split("/")[-1]).split("_")[0])

        #############4_correct chain name#######################
        #change chain name of receptor into 'A' and ligand into 'B'
        #
        # correct_chain_name(native1, 'A')
        # correct_chain_name(native2, 'B')
        # correct_chain_name(unbound1, 'A')
        # correct_chain_name(unbound2, 'B')

        #############4_correct chain name#######################

        #############2_check line length#######################
        # check_line_length(native1)
        # check_line_length(native2)
        # check_line_length(unbound1)
        # check_line_length(unbound2)
        #############2_correct chain name#######################

        #############5_renumber residue name#######################
        if check_residue_number(native1):
            renumber_residue_number(native1, True, False)
        if check_residue_number(native2):
            renumber_residue_number(native2, True, False)
        if check_residue_number(unbound1):
            renumber_residue_number(unbound1, True, False)
        if check_residue_number(unbound2):
            renumber_residue_number(unbound2, True, False)
        #############5_renumber residue name#######################

        #############6_check_missing_residue#######################
        # check_missing_residue(native1)
        # check_missing_residue(native2)
        # check_missing_residue(unbound1)
        # check_missing_residue(unbound2)













