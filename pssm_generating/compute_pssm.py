from pssmgen.pssm import PSSM
import os.path
import Bio.PDB
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from tqdm import tqdm


def get_pdb_chains(pdb):
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)
    pdb_struct = pdb_parser.get_structure("reference", pdb)[0]
    chain=[]
    for c in pdb_struct:
        chain_name = c.id
        if chain_name.strip() != "":
            chain.append(chain_name.strip())

    return chain

def compute_gssm(pdb_name,decoy_name):
    gen = PSSM(work_dir=pdb_name)
    # set psiblast executable, database and other psiblast parameters (here shows the defaults)
    gen.configure(blast_exe='/home/fei/Research/tools/ncbi-blast-2.10.0+/bin/psiblast',
                  database='/home/fei/Research/Dataset/pssm_dataset/swissprot/swissprot',
                  num_threads=4, evalue=0.0001, comp_based_stats='T',
                  max_target_seqs=2000, num_iterations=3, outfmt=7,
                  save_each_pssm=False, save_pssm_after_last_round=True)
    # gen.configure(blast_exe='/home/fei/Research/tools/ncbi-blast-2.10.0+/bin/psiblast',
    #               database='/home/fei/Research/Dataset/pssm_dataset/swissprot/swissprot',
    #                              num_threads=4, evalue=0.0001, comp_based_stats='T',
    #                              max_target_seqs=2000, num_iterations=0, outfmt=7,
    #                              save_each_pssm=False, save_pssm_after_last_round=True)
    ####read chain name#######
    pdb_path = pdb_name + "/pdb/" + decoy_name + ".pdb"
    chain_list = get_pdb_chains(pdb_path)

    ####read chain name#######

    # generates FASTA files
    gen.get_fasta(pdb_dir='pdb', chain=chain_list, out_dir='fasta')

    # generates PSSM
    gen.get_pssm(fasta_dir='fasta', out_dir='pssm_raw', run=True)

    # map PSSM and PDB to get consisitent files
    gen.map_pssm(pssm_dir='pssm_raw', pdb_dir='pdb', out_dir='pssm', chain=chain_list)

    # write consistent files and move
    gen.get_mapped_pdb(pdbpssm_dir='pssm', pdb_dir='pdb', pdbnonmatch_dir='pdb_nonmatch')


if __name__ == '__main__':
    file_path = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/2.1_native_PSSM/"
    target_name = "/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/caseID.lst"
    with open(target_name) as f:
        data = [line.strip("\r\n ") for line in f]

    for i in range(len(data)):
        pdb_name = os.path.join(file_path, data[i] + "_l_u")
        if os.path.exists(pdb_name):
            print(pdb_name)
            compute_gssm(pdb_name, data[i] + "_l_u")
        pdb_name = os.path.join(file_path, data[i] + "_r_u")
        if os.path.exists(pdb_name):
            print(pdb_name)
            compute_gssm(pdb_name, data[i] + "_r_u")



        # initiate the PSSM object
        # gen = PSSM(work_dir='/home/fei/Research/Dataset/decoy/1_decoys_bm4_zd3.0.2_15deg/pdb_file/complex_output_file/1_pssm/1BJ1')
        # pdb_name = "1A2K_r_u"
