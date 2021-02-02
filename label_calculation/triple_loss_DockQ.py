# !/usr/bin/env python3
import csv
import Bio.PDB
import json
# warnings.simplefilter('ignore', BiopythonWarning)
import sys
import os
import re
import tempfile
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from argparse import ArgumentParser
import itertools
import subprocess
from multiprocessing import cpu_count, Pool
from util_function import write_txt
import pandas as pd

def parse_fnat(fnat_out):
    fnat=-1;
    nat_correct=-1
    nat_total=-1
    fnonnat=-1
    nonnat_count=-1
    model_total=-1
    inter=[]
    chain_A_interface = []
    chain_B_interface = []
    native_chain_A_interface = []
    native_chain_B_interface = []

    correspndence = []
    overlap = []
    for line in fnat_out.split("\n"):
#        print line
        line=line.rstrip('\n')
        match=re.search(r'NATIVE: (\d+)(\w) (\d+)(\w)',line)
        model_interface = re.search(r'MODEL: (\d+)(\w) (\d+)(\w)',line)
        model_native_overlap = re.search(r'Overlap (\d+)(\w) (\d+)(\w)',line)
        if(re.search(r'^Fnat',line)):
            list=line.split(' ')
            fnat=float(list[3])
            nat_correct=int(list[1])
            nat_total=int(list[2])
        elif(re.search(r'^Fnonnat',line)):
            list=line.split(' ')
            fnonnat=float(list[3])
            nonnat_count=int(list[1])
            model_total=int(list[2])
        elif(match):
            #print line
            res1=match.group(1)
            chain1=match.group(2)
            res2=match.group(3)
            chain2=match.group(4)
           # print res1 + ' ' + chain1 + ' ' + res2 + ' ' + chain2
           #  inter.append((chain1 + '.' + res1, chain2 + '.' + res2))
            inter.append(res1 + chain1)
            inter.append(res2 + chain2)
            if chain1 == 'A':
                native_chain_A_interface.append(int(res1))
            else:
                native_chain_B_interface.append(int(res1))
            if chain2 == 'B':
                native_chain_B_interface.append(int(res2))
            else:
                native_chain_A_interface.append(int(res2))
        elif(model_interface):
            # print line
            res1 = model_interface.group(1)
            chain1 = model_interface.group(2)
            res2 = model_interface.group(3)
            chain2 = model_interface.group(4)
            # print res1 + ' ' + chain1 + ' ' + res2 + ' ' + chain2
            if int(res1) < 9000 and int(res2) < 9000:
                #otherwise, those residues were masked when aligned to native
                if chain1 == 'A':
                    chain_A_interface.append(int(res1))
                else:
                    chain_B_interface.append(int(res1))
                if chain2 == 'B':
                    chain_B_interface.append(int(res2))
                else:
                    chain_A_interface.append(int(res2))
                #build the coupling residues from two individuals
                correspndence.append((chain1 + '.' + res1, chain2 + '.' + res2))
        elif(model_native_overlap):
            res1 = model_native_overlap.group(1)
            chain1 = model_native_overlap.group(2)
            res2 = model_native_overlap.group(3)
            chain2 = model_native_overlap.group(4)
            overlap.append((chain1 + '.' + res1, chain2 + '.' + res2))


    return (fnat,nat_correct,nat_total,fnonnat,nonnat_count,model_total,inter,
            {'A':{}.fromkeys(native_chain_A_interface).keys(), 'B': {}.fromkeys(native_chain_B_interface).keys()},
            {'A': {}.fromkeys(chain_A_interface).keys(), 'B': {}.fromkeys(chain_B_interface).keys()},
            correspndence,overlap)

def capri_class(fnat,iRMS,LRMS,capri_peptide=False):


    if capri_peptide:
               
        if(fnat < 0.2 or (LRMS > 5.0 and iRMS > 2.0)):
            return 'Incorrect'
        elif((fnat >= 0.2 and fnat < 0.5) and (LRMS <= 5.0 or iRMS <= 2.0) or (fnat >= 0.5 and LRMS > 2.0 and iRMS > 1.0)):
            return 'Acceptable'
        elif((fnat >= 0.5 and fnat < 0.8) and (LRMS <= 2.0 or iRMS <= 1.0) or (fnat >= 0.8 and LRMS > 1.0 and iRMS > 0.5)):
            return 'Medium'
        elif(fnat >= 0.8 and (LRMS <= 1.0 or iRMS <= 0.5)):
            return 'High'
        else:
            return 'Undef'
    else:

        if(fnat < 0.1 or (LRMS > 10.0 and iRMS > 4.0)):
            return 'Incorrect'
        elif((fnat >= 0.1 and fnat < 0.3) and (LRMS <= 10.0 or iRMS <= 4.0) or (fnat >= 0.3 and LRMS > 5.0 and iRMS > 2.0)):
            return 'Acceptable'
        elif((fnat >= 0.3 and fnat < 0.5) and (LRMS <= 5.0 or iRMS <= 2.0) or (fnat >= 0.5 and LRMS > 1.0 and iRMS > 1.0)):
            return 'Medium'
        elif(fnat >= 0.5 and (LRMS <= 1.0 or iRMS <= 1.0)):
            return 'High'
        else:
            return 'Undef'


def capri_class_DockQ(DockQ,capri_peptide=False):

    if capri_peptide:
        return 'Undef for capri_peptides'
    
    (c1,c2,c3)=(0.23,0.49,0.80)
    if(DockQ < c1):
        return 'Incorrect'
    elif(DockQ >= c1 and DockQ < c2):
        return 'Acceptable'
    elif(DockQ >= c2 and DockQ < c3):
        return 'Medium'
    elif(DockQ >= c3):
        return 'High'
    else:
        return 'Undef'


def calc_DockQ(model,native,use_CA_only=False,capri_peptide=False):
    
    exec_path=os.path.dirname(os.path.abspath(sys.argv[0]))    
    atom_for_sup=['CA','C','N','O']
    if(use_CA_only):
        atom_for_sup=['CA']

    cmd_fnat=exec_path + '/fnat ' + model + ' ' + native + ' 5 -all'
    cmd_interface=exec_path + '/fnat ' + model + ' ' + native + ' 10 -all'

    if capri_peptide:
        cmd_fnat=exec_path + '/fnat ' + model + ' ' + native + ' 4 -all'
        cmd_interface=exec_path + '/fnat ' + model + ' ' + native + ' 8 -cb' 

    #fnat_out = os.popen(cmd_fnat).readlines()
    fnat_out = subprocess.getoutput(cmd_fnat)
#    sys.exit()
    (fnat,nat_correct,nat_total,fnonnat,nonnat_count,model_total,interface5A, native_interface5A_dict ,decoy_interface5A_dict, correspondence5A, overlap5A)=parse_fnat(fnat_out)
    del fnat_out
    assert fnat!=-1, "Error running cmd: %s\n" % (cmd_fnat)
#    inter_out = os.popen(cmd_interface).readlines()
    inter_out = subprocess.getoutput(cmd_interface)
    (fnat_bb,nat_correct_bb,nat_total_bb,fnonnat_bb,nonnat_count_bb,model_total_bb,interface, native_interface_dict, decoy_interface_dict, correspondence, overlap)=parse_fnat(inter_out)

    del inter_out
    assert fnat_bb!=-1, "Error running cmd: %s\n" % (cmd_interface)

    #print fnat
    #Use same interface as for fnat for iRMS
    #interface=interface5A


    # Start the parser
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)

    # Get the structures
    ref_structure = pdb_parser.get_structure("reference", native)
    sample_structure = pdb_parser.get_structure("model", model)

    # Use the first model in the pdb-files for alignment
    # Change the number 0 if you want to align to another structure
    ref_model    = ref_structure[0]
    sample_model = sample_structure[0]



    # Make a list of the atoms (in the structures) you wish to align.
    # In this case we use CA atoms whose index is in the specified range
    ref_atoms = []
    sample_atoms = []

    common_interface=[]

    chain_res={}


    #find atoms common in both sample and native
    atoms_def_sample=[]
    atoms_def_in_both=[]
    #first read in sample
    for sample_chain in sample_model:
#        print sample_chain
        chain=sample_chain.id
#        print chain
        for sample_res in sample_chain:
           # print sample_res
            if sample_res.get_id()[0] != ' ': #Skip hetatm.
                continue
            resname=sample_res.get_id()[1]
            key=str(resname) + chain
            for a in atom_for_sup:
                atom_key=key + '.' + a
                if a in sample_res:
                    if atom_key in atoms_def_sample:
                        print(atom_key + ' already added (MODEL)!!!')
                    atoms_def_sample.append(atom_key)

    #then read in native also present in sample
    for ref_chain in ref_model:
        chain=ref_chain.id
        for ref_res in ref_chain:
            #print ref_res
            if ref_res.get_id()[0] != ' ': #Skip hetatm.
#                print ref_res.get_id()
                continue
            resname=ref_res.get_id()[1]
            key=str(resname) + chain
            for a in atom_for_sup:
                atom_key=key + '.' + a
                if a in ref_res and atom_key in atoms_def_sample:
                    if atom_key in atoms_def_in_both:
                        print(atom_key + ' already added (Native)!!!')
                    atoms_def_in_both.append(atom_key)


#    print atoms_def_in_both
    for sample_chain in sample_model:
        chain=sample_chain.id
        if chain not in chain_res.keys():
            chain_res[chain]=[]
        for sample_res in sample_chain:
            if sample_res.get_id()[0] != ' ': #Skip hetatm.
                continue
            resname=sample_res.get_id()[1]
            key=str(resname) + chain
            chain_res[chain].append(key)
            if key in interface:
                for a in atom_for_sup:
                    atom_key=key + '.' + a
                    if a in sample_res and atom_key in atoms_def_in_both:
                        sample_atoms.append(sample_res[a])
                common_interface.append(key)

    #print inter_pairs

    chain_ref={}
    common_residues=[]



    # Iterate of all chains in the model in order to find all residues
    for ref_chain in ref_model:
        # Iterate of all residues in each model in order to find proper atoms
        #  print dir(ref_chain)
        chain=ref_chain.id
        if chain not in chain_ref.keys():
            chain_ref[chain]=[]
        for ref_res in ref_chain:
            if ref_res.get_id()[0] != ' ': #Skip hetatm.
                continue
            resname=ref_res.get_id()[1]
            key=str(resname) + chain

            #print ref_res
            #      print key
            # print chain_res.values()
            if key in chain_res[chain]: # if key is present in sample
                #print key
                for a in atom_for_sup:
                    atom_key=key + '.' + a
                    if a in ref_res and atom_key in atoms_def_in_both:
                        chain_ref[chain].append(ref_res[a])
                        common_residues.append(key)
                      #chain_sample.append((ref_res['CA'])
            if key in common_interface:
              # Check if residue number ( .get_id() ) is in the list
              # Append CA atom to list
                #print key  
                for a in atom_for_sup:
                    atom_key=key + '.' + a
                    #print atom_key
                    if a in ref_res and atom_key in atoms_def_in_both:
                        ref_atoms.append(ref_res[a])



    #get the ones that are present in native        
    chain_sample={}
    for sample_chain in sample_model:
        chain=sample_chain.id
        if chain not in chain_sample.keys():
            chain_sample[chain]=[]
        for sample_res in sample_chain:
            if sample_res.get_id()[0] != ' ': #Skip hetatm.
                continue
            resname=sample_res.get_id()[1]
            key=str(resname) + chain
            if key in common_residues:
                for a in atom_for_sup:
                    atom_key=key + '.' + a
                    if a in sample_res and atom_key in atoms_def_in_both:
                        chain_sample[chain].append(sample_res[a])

        #if key in common_residues:
        #     print key  
        #sample_atoms.append(sample_res['CA'])
        #common_interface.append(key)


    assert len(ref_atoms)!=0, "length of native is zero"
    assert len(sample_atoms)!=0, "length of model is zero"
    assert len(ref_atoms)==len(sample_atoms), "Different number of atoms in native and model %d %d\n" % (len(ref_atoms),len(sample_atoms))

    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms, sample_atoms)
    super_imposer.apply(sample_model.get_atoms())

    # Print RMSD:
    irms=super_imposer.rms

    (chain1,chain2)=chain_sample.keys()

    ligand_chain=chain1
    receptor_chain=chain2
    len1=len(chain_res[chain1])
    len2=len(chain_res[chain2])

    assert len1!=0, "%s chain has zero length!\n" % chain1
    assert len2!=0, "%s chain has zero length!\n" % chain2

    class1='ligand'
    class2='receptor'
    if(len(chain_sample[chain1]) > len(chain_sample[chain2])):
        receptor_chain=chain1
        ligand_chain=chain2
        class1='receptor'
        class2='ligand'



    #print len1
    #print len2
    #print chain_sample.keys()

    #Set to align on receptor
    assert len(chain_ref[receptor_chain])==len(chain_sample[receptor_chain]), "Different number of atoms in native and model receptor (chain %c) %d %d\n" % (receptor_chain,len(chain_ref[receptor_chain]),len(chain_sample[receptor_chain]))

    super_imposer.set_atoms(chain_ref[receptor_chain], chain_sample[receptor_chain])
    super_imposer.apply(sample_model.get_atoms())
    receptor_chain_rms=super_imposer.rms
    #print receptor_chain_rms
    #print dir(super_imposer)
    #print chain1_rms

    #Grep out the transformed ligand coords

    #print ligand_chain

    #print chain_ref[ligand_chain]
    #print chain_sample[ligand_chain]
    #l1=len(chain_ref[ligand_chain])
    #l2=len(chain_sample[ligand_chain])




    assert len(chain_ref[ligand_chain])!=0 or len(chain_sample[ligand_chain])!=0, "Zero number of equivalent atoms in native and model ligand (chain %s) %d %d.\nCheck that the residue numbers in model and native is consistent\n" % (ligand_chain,len(chain_ref[ligand_chain]),len(chain_sample[ligand_chain]))


    assert len(chain_ref[ligand_chain])==len(chain_sample[ligand_chain]), "Different number of atoms in native and model ligand (chain %c) %d %d\n" % (ligand_chain,len(chain_ref[ligand_chain]),len(chain_sample[ligand_chain]))

    coord1=np.array([atom.coord for atom in chain_ref[ligand_chain]])
    coord2=np.array([atom.coord for atom in chain_sample[ligand_chain]])

    #coord1=np.array([atom.coord for atom in chain_ref[receptor_chain]])
    #coord2=np.array([atom.coord for atom in chain_sample[receptor_chain]])

    #print len(coord1)
    #print len(coord2)

    sup=SVDSuperimposer()
    Lrms = sup._rms(coord1,coord2) #using the private _rms function which does not superimpose





    del chain_ref, common_residues, ref_atoms, sample_atoms, common_interface, chain_res, sup, atoms_def_sample, atoms_def_in_both, chain_sample
    del ref_structure,sample_structure, decoy_interface_dict, native_interface_dict, correspondence, overlap
    del ref_model, sample_model,coord1, coord2, interface, interface5A, super_imposer, sample_res, sample_chain, ref_res, ref_chain, pdb_parser
    #super_imposer.set_atoms(chain_ref[ligand_chain], chain_sample[ligand_chain])
    #super_imposer.apply(sample_model.get_atoms())
    #coord1=np.array([atom.coord for atom in chain_ref[receptor_chain]])
    #coord2=np.array([atom.coord for atom in chain_sample[receptor_chain]])
    #Rrms= sup._rms(coord1,coord2)
    #should give same result as above line
    #diff = coord1-coord2
    #l = len(diff) #number of atoms
    #from math import sqrt
    #print sqrt(sum(sum(diff*diff))/l)
    #print np.sqrt(np.sum(diff**2)/l)
    DockQ=(float(fnat) + 1/(1+(irms/1.5)*(irms/1.5)) + 1/(1+(Lrms/8.5)*(Lrms/8.5)))/3
    info={}
    info['DockQ']=DockQ
    info['irms']=irms
    info['Lrms']=Lrms
    info['fnat']=fnat
    info['nat_correct']=nat_correct
    info['nat_total']=nat_total

    info['fnonnat']=fnonnat
    info['nonnat_count']=nonnat_count
    info['model_total']=model_total
    
    info['chain1']=chain1
    info['chain2']=chain2
    info['len1']=len1
    info['len2']=len2
    info['class1']=class1
    info['class2']=class2
    
    return info, native_interface5A_dict, decoy_interface5A_dict,correspondence5A, overlap5A

def get_pdb_chains(pdb):
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)
    pdb_struct = pdb_parser.get_structure("reference", pdb)[0]
    chain=[]
    for c in pdb_struct:
        chain.append(c.id)

    del pdb_struct, pdb_parser
    return chain
#ATOM   3312  CA
#ATOM   3315  CB  ALA H 126     -21.802  31.674  73.597  1.00 58.05           C  

def make_two_chain_pdb(pdb,group1,group2): #renumber from 1
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)
    pdb_struct = pdb_parser.get_structure("reference", pdb)[0]
    for c in pdb_struct:
        if c.id in group1:
            c.id='A'
        if c.id in group2:
            c.id='B'
    (code,outfile)=tempfile.mkstemp()
    io=Bio.PDB.PDBIO()
    io.set_structure(pdb_struct)
    io.save(outfile)
    exec_path=os.path.dirname(os.path.abspath(sys.argv[0]))    
    cmd=exec_path + '/scripts/renumber_pdb.pl ' + outfile
    os.system(cmd)
    os.remove(outfile)

    del pdb_struct
    return outfile +'.renum'

def change_chain(pdb_string,chain):
    new_str=[];
    for line in pdb_string:
        s=list(line);
        s[21]=chain;
        new_str.append("".join(s));

    return "\n".join(new_str);

def make_two_chain_pdb_perm(pdb,group1,group2): #not yet ready
    pdb_chains={}
    with open(pdb, "r") as f:
        for line in f.readlines():
            if line[0:4] == "ATOM":
                #       s=list(line);
                #print line
                chain=line[21]
                atom=line[13:16]
                resnum=int(line[22:26])
                #        print atom + ':' + str(resnum) +':'
                if chain not in pdb_chains:
                    pdb_chains[chain]=[]
                pdb_chains[chain].append(line)
     #       print line
     #       s[21]='B'
     #       print "".join(s)
    #        print chain


    #sys.exit()
    (code,outfile)=tempfile.mkstemp()

    with open(outfile, "w") as f1:
        for c in group1:
         #   print pdb_chains[c]
            f1.write(change_chain(pdb_chains[c],"A"))
        f1.write("TER\n")
        for c in group2:
            f1.write(change_chain(pdb_chains[c],"B"))


    #print outfile
    exec_path=os.path.dirname(os.path.abspath(sys.argv[0]))    
    cmd=exec_path + '/scripts/renumber_pdb.pl ' + outfile
    os.system(cmd)
    os.remove(outfile)
    del pdb_chains
    return outfile +'.renum'
    
def cal_similarity(model, native, args):

    #bio_ver=1.64
    # bio_ver=1.61
    # if(float(Bio.__version__) < bio_ver):
    #     print("Biopython version (%s) is too old need at least >=%.2f" % (Bio.__version__,bio_ver))
    #     sys.exit()

#    if(len(sys.argv)!=3):
#        print "Usage: ./Dock.py <model> <native>"
#        sys.exit()

#    print args
#    print args.model[0]
#    sys.exit()
#    model=sys.argv[1]
#    native=sys.argv[2]

    exec_path=os.path.dirname(os.path.abspath(sys.argv[0]))
    fix_numbering=exec_path + '/scripts/fix_numbering.pl'
    # model=args.model
    model_in=model
    # native=args.native
    native_in=native
    use_CA_only=args.useCA
    capri_peptide=args.capri_peptide

    model_chains=[]
    native_chains=[]
    best_info=''
    if(not args.skip_check):
        model_chains=get_pdb_chains(model)
        native_chains=get_pdb_chains(native)
    files_to_clean=[]

#    print native_chains
    if((len(model_chains) > 2 or len(native_chains) > 2) and
       (args.model_chain1 == None and args.native_chain1 == None)):
        print("Multi-chain model need sets of chains to group\nuse -native_chain1 and/or -model_chain1 if you want a different mapping than 1-1" )
        print("Model chains  : " + str(model_chains))
        print("Native chains : " + str(native_chains))
        sys.exit()
    if not args.skip_check and (len(model_chains) < 2 or len(native_chains)< 2):
        print("Need at least two chains in the two inputs\n")
        sys.exit()
        
    if len(model_chains) > 2 or len(native_chains)> 2:
        group1=model_chains[0]
        group2=model_chains[1]
        nat_group1=native_chains[0]
        nat_group2=native_chains[1]
        if(args.model_chain1 != None):
            group1=args.model_chain1
            nat_group1=group1
            if(args.model_chain2 != None):
                group2=args.model_chain2
            else:
                #will use the complement from group1
                group2=[]
                for c in model_chains:
                    if c not in group1:
                        group2.append(c)
            nat_group1=group1
            nat_group2=group2
            

        if(args.native_chain1 != None):
            nat_group1=args.native_chain1
            if(args.native_chain2 != None):
                nat_group2=args.native_chain2
            else:
                #will use the complement from group1
                nat_group2=[]
                for c in native_chains:
                    if c not in nat_group1:
                        nat_group2.append(c)
                        
        if(args.model_chain1 == None):
            group1=nat_group1
            group2=nat_group2

        #print group1
        #print group2

        #print "native"
        #print nat_group1
        #print nat_group2
        if(args.verbose):
            print('Merging ' + ''.join(group1) + ' -> ' + ''.join(nat_group1) + ' to chain A')
            print('Merging ' + ''.join(group2) + ' -> ' + ''.join(nat_group2) + ' to chain B')
        native=make_two_chain_pdb_perm(native,nat_group1,nat_group2)
        files_to_clean.append(native)
        pe=0
        if args.perm1 or args.perm2:
            best_DockQ=-1;
            best_g1=[]
            best_g2=[]
            
            iter_perm1=itertools.combinations(group1,len(group1))
            iter_perm2=itertools.combinations(group2,len(group2))
            if args.perm1:
                iter_perm1=itertools.permutations(group1)
            if args.perm2:
                iter_perm2=itertools.permutations(group2)
               
            combos1=[];
            combos2=[];
            for g1 in iter_perm1:#_temp:
                combos1.append(g1)
            for g2 in iter_perm2:
                combos2.append(g2)

            for g1 in combos1:
                for g2 in combos2:
                    pe=pe+1
                   # print str(g1)+' '+str(g2)
#            print pe
#            print group1
#            print group2
            pe_tot=pe
            pe=1
            #sys.exit()
            if args.verbose:
                print('Starting chain order permutation search (number of permutations: ' + str(pe_tot) + ')')
            for g1 in combos1:
                for g2 in combos2:
                #g2=group2    
                    model_renum=make_two_chain_pdb_perm(model_in,g1,g2)
                    model_fixed=model_renum
                    if not args.no_needle:
                        fix_numbering_cmd=fix_numbering + ' ' + model_renum + ' ' + native + ' > /dev/null'
                        model_fixed=model_renum + ".fixed"
                        #                print fix_numbering_cmd
                        os.system(fix_numbering_cmd)
                        os.remove(model_renum)
                        if not os.path.exists(model_fixed):
                            print('If you are sure the residues are identical you can use the options -no_needle')
                            sys.exit()
                    test_dict, native_interface, decoy_interface, correspondence, overlap =calc_DockQ(model_fixed,native,use_CA_only)
                    # os.remove(model_fixed)
                    if not args.quiet:
                        print(str(pe)+'/'+str(pe_tot) + ' ' + ''.join(g1) + ' -> ' + ''.join(g2) + ' ' + str(test_dict['DockQ']))
                    if(test_dict['DockQ'] > best_DockQ):
                        best_DockQ=test_dict['DockQ'];
                        info=test_dict
                        best_g1=g1
                        best_g2=g2
                        best_info='Best score ( ' + str(best_DockQ) +' ) found for model -> native, chain1:' + ''.join(best_g1) + ' -> ' + ''.join(nat_group1) + ' chain2:' + ''.join(best_g2) + ' -> ' + ''.join(nat_group2)
                        
                        if args.verbose:
                            print(best_info)
                        if not args.quiet:    
                            print("Current best " + str(best_DockQ))
                    pe=pe+1
            if not args.quiet:
                print(best_info)
#            print 'Best score ( ' + str(best_DockQ) +' ) found for ' + str(best_g1) + ' ' + str(best_g2)
        else:
            model_renum=make_two_chain_pdb_perm(model,group1,group2)
            model_fixed=model_renum
            if not args.no_needle:
                fix_numbering_cmd=fix_numbering + ' ' + model_renum + ' ' + native + ' > /dev/null'
                model_fixed=model_renum + ".fixed"
#                print fix_numbering_cmd
                os.system(fix_numbering_cmd)
                os.remove(model_renum)
                if not os.path.exists(model_fixed):
                    print('If you are sure the residues are identical you can use the options -no_needle')
                    sys.exit()
            info,native_interface_dict, decoy_interface_dict, correspondence, overlap =calc_DockQ(model_fixed,native,use_CA_only)

            #os.system('cp ' + native + ' native_multichain.pdb')
            #os.system('cp ' + model_fixed + ' .')

            # os.remove(model_fixed)
    else:
        if not args.no_needle:
            fix_numbering_cmd = fix_numbering + ' ' + model + ' ' + native + ' > /dev/null'
            model_fixed = model + ".fixed"
            #                print fix_numbering_cmd
            os.system(fix_numbering_cmd)
            # os.remove(model)
            if not os.path.exists(model_fixed):
                print('If you are sure the residues are identical you can use the options -no_needle')
                sys.exit()
        info, native_interface_dict, decoy_interface_dict, correspondence, overlap = calc_DockQ(model_fixed, native, use_CA_only)
        # os.remove(model_fixed)

        # info=calc_DockQ(model,native,use_CA_only=use_CA_only,capri_peptide=capri_peptide) #False):
#        info=calc_DockQ(model,native,use_CA_only=)
    
    irms=info['irms']
    Lrms=info['Lrms']
    fnat=info['fnat']
    DockQ=info['DockQ']
    fnonnat=info['fnonnat']
    IOU = info['nat_correct']/(info['nat_total'] + info['model_total'] - info['nat_correct'])
    
    if(args.short):
        if capri_peptide:
            if args.print == True:
                print("DockQ-capri_peptide %.3f Fnat %.3f iRMS %.3f LRMS %.3f Fnonnat %.3f IOU %.3f %s %s %s" % (DockQ,fnat,irms,Lrms,fnonnat,IOU, model_in,native_in,best_info))
        else:
            if args.print == True:
                print("DockQ %.3f Fnat %.3f iRMS %.3f LRMS %.3f Fnonnat %.3f IOU %.3f %s %s %s" % (DockQ,fnat,irms,Lrms,fnonnat,IOU, model_in,native_in,best_info))

    else:
        if capri_peptide:
            print( '****************************************************************')
            print( '*                DockQ-CAPRI peptide                           *')
            print( '*   Do not trust any thing you read....                        *')
            print( '*   OBS THE DEFINITION OF Fnat and iRMS are different for      *')
            print( '*   peptides in CAPRI                                          *')
            print( '*                                                              *')
            print( '*   For the record:                                            *')
            print( '*   Definition of contact <4A all heavy atoms (Fnat)           *')
            print( '*   Definition of interface <8A CB (iRMS)                      *')
            print( '*   For comments, please email: bjorn.wallner@.liu.se          *')
            print( '****************************************************************')
        else:
            print( '****************************************************************')
            print( '*                       DockQ                                  *')
            print( '*   Scoring function for protein-protein docking models        *')
            print( '*   Statistics on CAPRI data:                                  *')
            print( '*    0.00 <= DockQ <  0.23 - Incorrect                         *')
            print( '*    0.23 <= DockQ <  0.49 - Acceptable quality                *')
            print( '*    0.49 <= DockQ <  0.80 - Medium quality                    *')
            print( '*            DockQ >= 0.80 - High quality                      *')
            print( '*   Reference: Sankar Basu and Bjorn Wallner, DockQ: A quality *')
            print( '*   measure for protein-protein docking models, submitted      *')
            print( '*                                                              *')
            print( '*   For the record:                                            *')
            print( '*   Definition of contact <5A (Fnat)                           *')
            print( '*   Definition of interface <10A all heavy atoms (iRMS)        *')
            print( '*   For comments, please email: bjorn.wallner@.liu.se          *')
            print( '*                                                              *')
            print( '****************************************************************')
        print("Model  : %s" % model_in)
        print("Native : %s" % native_in)
        if len(best_info):
            print(best_info)
        print('Number of equivalent residues in chain ' + info['chain1'] + ' ' + str(info['len1']) + ' (' + info['class1'] + ')')
        print('Number of equivalent residues in chain ' + info['chain2'] + ' ' + str(info['len2']) + ' (' + info['class2'] + ')')
        print("Fnat %.3f %d correct of %d native contacts" % (info['fnat'],info['nat_correct'],info['nat_total']))
        print("Fnonnat %.3f %d non-native of %d model contacts" % (info['fnonnat'],info['nonnat_count'],info['model_total']))
        print("iRMS %.3f" % irms)
        print("LRMS %.3f" % Lrms)
       # print 'CAPRI ' + capri_class(fnat,irms,Lrms,capri_peptide=capri_peptide)
        peptide_suffix=''
        if capri_peptide:
            peptide_suffix='_peptide'
        print('CAPRI{} {}'.format(peptide_suffix,capri_class(fnat,irms,Lrms,capri_peptide=capri_peptide)))
        print( 'DockQ_CAPRI ' + capri_class_DockQ(DockQ,capri_peptide=capri_peptide))
        peptide_disclaimer=''
        if capri_peptide:
            peptide_disclaimer='DockQ not reoptimized for CAPRI peptide evaluation'
        print("DockQ {:.3f} {}".format(DockQ,peptide_disclaimer))

    for f in files_to_clean:
        os.remove(f)

    CAPRI = capri_class(fnat,irms,Lrms,capri_peptide=capri_peptide)
    if CAPRI =='Incorrect':
        CAPRI_interger = 0
    elif CAPRI =='Acceptable':
        CAPRI_interger = 1
    elif CAPRI =='Medium':
        CAPRI_interger = 2
    elif CAPRI =='High':
        CAPRI_interger = 3

    return DockQ, fnat,irms, Lrms, fnonnat, IOU, model_in,native_in,best_info,CAPRI_interger, native_interface_dict, decoy_interface_dict, correspondence,overlap


def read_chain_name(pdb_path):
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)
    pdb_struct = pdb_parser.get_structure("reference", pdb_path)[0]
    chain=[]
    for c in pdb_struct:
        chain_name = c.id
        if chain_name.strip()!="":
            chain.append(chain_name.strip())

    return chain


class interface_extraction(Bio.PDB.Select):
    def __init__(self, interface_dict, current_chain):
        self.interface_dict = interface_dict
        self.current_chain = current_chain

    def accept_residue(self, residue):
        if residue.get_parent().id == self.current_chain and residue.id[1] in self.interface_dict[self.current_chain]:
            return 1
        else:
            return 0

def calculate_gt_for_decoy(decoy_name, native, target_name, args, decoy_path, save_decoy_interface_folder):
    label_dic = {}
    positive_gt = []
    negative_gt = []
    triple_correspondence = []
    model = os.path.join(decoy_path, target_name, decoy_name)
    args.model_chain1 = 'A'
    args.model_chain2 = 'B'

    DockQ, fnat, irms, Lrms, fnonnat, IOU, _, _, _, CAPRI_interger, native_interface_index, decoy_interface_index, correspondence, overlap = cal_similarity(model, native, args)

    label_dic['No'] = decoy_name
    label_dic['iRMS'] = irms
    label_dic['LRMS'] = Lrms
    label_dic['fnat'] = fnat
    label_dic['fnonnat'] = fnonnat
    label_dic['IOU'] = IOU
    label_dic['CAPRI'] = CAPRI_interger
    label_dic['DockQ'] = DockQ

    #save decoy_interface pdb

    # read interface residual info
    parser = Bio.PDB.PDBParser(QUIET= True)
    decoy_structure = parser.get_structure(decoy_name, os.path.join(decoy_path, target_name, decoy_name + '.fixed'))
    io = Bio.PDB.PDBIO()
    io.set_structure(decoy_structure)
    if not os.path.exists(os.path.join(save_decoy_interface_folder,target_name)):
        os.mkdir(os.path.join(save_decoy_interface_folder,target_name))
    io.save(os.path.join(save_decoy_interface_folder, target_name, os.path.splitext(decoy_name)[0] + 'A.pdb'), interface_extraction(decoy_interface_index, 'A'))
    io.save(os.path.join(save_decoy_interface_folder, target_name, os.path.splitext(decoy_name)[0] + 'B.pdb'), interface_extraction(decoy_interface_index, 'B'))

    #read and save the native interface
    if os.path.exists(os.path.join(save_decoy_interface_folder, target_name, 'native_A.pdb')) is not True or \
            os.path.exists(os.path.join(save_decoy_interface_folder, target_name, 'native_B.pdb')) is not True:
        native_structure = parser.get_structure('native', native)
        io.set_structure(native_structure)
        io.save(os.path.join(save_decoy_interface_folder, target_name, 'native_A.pdb'),
                interface_extraction(native_interface_index, 'A'))
        io.save(os.path.join(save_decoy_interface_folder, target_name, 'native_B.pdb'),
                interface_extraction(native_interface_index, 'B'))

    #parse the corresponding residues at model interface and their
    for couple_residues in correspondence:
        if (couple_residues in overlap) or \
                ((couple_residues[1], couple_residues[0]) in overlap):
            # positive binding
            positive_gt.append((couple_residues[0], couple_residues[1]))
        else:
            # negative binding
            negative_gt.append((couple_residues[0], couple_residues[1]))

    for pos_neg_pair in itertools.product(positive_gt, negative_gt):
        triple_correspondence.append(pos_neg_pair)

    del parser, decoy_structure, io, correspondence, overlap, model

    return label_dic, {decoy_name : triple_correspondence}, {decoy_name: positive_gt}, {decoy_name: negative_gt}

def calculate_gt_for_target(decoy_name_folder, file_path, target_name, args, decoy_path, save_decoy_interface_folder):
    print(target_name + ": start!")
    native = os.path.join(file_path, "all_pdb", target_name + ".pdb")
    args.native_chain1 = 'A'
    args.native_chain2 = 'B'

    label_result_path = os.path.join(decoy_name_folder, target_name + "_regression_score_label.csv")
    label_dataframe = pd.read_csv(label_result_path)
    #
    #
    # with open(positive_decoy_name_path, 'r') as f:
    # scores_data_frame = pd.read_csv(positive_decoy_name_path)
    # decoy_name_list = list(scores_data_frame['No'])

    # for line in decoy_name_list:
    #     decoy_number = line.rstrip("\n").split(".")[1]
    #     decoy_number_list.append(decoy_number)

    label_list = []
    triple_correspondence_gt_list = {}
    positive_gt_list = {}
    negative_gt_list = {}

    for decoy_name in label_dataframe.No.tolist():
        if os.path.splitext(decoy_name)[1] == '.pdb':
            labels, triple_correspondence_gt, positive_gt, negative_gt  = calculate_gt_for_decoy(decoy_name, native, target_name, args, decoy_path, save_decoy_interface_folder)
            label_list.append(labels)
            triple_correspondence_gt_list.update(triple_correspondence_gt)
            positive_gt_list.update(positive_gt)
            negative_gt_list.update(negative_gt)
            del labels, triple_correspondence_gt, positive_gt, negative_gt
    print('all decoys in target {:s} done, will save the results to csv file'.format(target_name))
    label_dataframe = pd.DataFrame(label_list)
    # label_path = os.path.join(file_path, target_name + "_regression_score_label.csv")
    label_dataframe.to_csv(label_result_path, index=False)

    with open(os.path.splitext(label_result_path)[0] + '_triple_correspondence.json', 'w') as fp:
        json.dump(triple_correspondence_gt_list, fp)


    with open(os.path.splitext(label_result_path)[0] + '_positive_correspondence.json', 'w') as fp:
        json.dump(positive_gt_list, fp)

    with open(os.path.splitext(label_result_path)[0] + '_negative_correspondence.json', 'w') as fp:
        json.dump(negative_gt_list, fp)

    del label_dataframe, label_list,triple_correspondence_gt_list, positive_gt_list, negative_gt_list
    print(target_name + ": end!")


def calulate_ground_truth(target_name_list, file_path, args, decoy_path,decoy_name_folder, save_decoy_interface_folder):

    # declare process pool
    core_num = cpu_count()
    pool = Pool(core_num)

    # selected_labels = [0]
    # selected_decoy_names = ['7CEI.complex.17314.pdb']
    # target_name_list = ['1ACB'] # residue start from 1
    # target_name_list = ['1AK4']  # residue donot start from 1
    # target_name_list = ['1KXP']  # residue start from 2000+

    for target_name in target_name_list:
        pool.apply_async(calculate_gt_for_target, (decoy_name_folder, file_path, target_name, args, decoy_path,save_decoy_interface_folder))
        # calculate_gt_for_target(decoy_name_folder, file_path, target_name, args, decoy_path, save_decoy_interface_folder)

    pool.close()
    pool.join()
    del pool




if __name__ == '__main__':
    parser = ArgumentParser(description="DockQ - Quality measure for protein-protein docking models")
    parser.add_argument('-model', type=str, nargs=1, help='path to model file')
    parser.add_argument('-native', type=str, nargs=1, help='path to native file')
    parser.add_argument('-capri_peptide', default=False, action='store_true',
                        help='use version for capri_peptide (DockQ cannot not be trusted for this setting)')
    parser.add_argument('-short', default=False, action='store_true', help='short output')
    parser.add_argument('-verbose', default=False, action='store_true', help='talk a lot!')
    parser.add_argument('-quiet', default=False, action='store_true', help='keep quiet!')
    parser.add_argument('-useCA', default=False, action='store_true', help='use CA instead of backbone')
    parser.add_argument('-skip_check', default=False, action='store_true',
                        help='skip initial check fo speed up on two chain examples')
    parser.add_argument('-no_needle', default=False, action='store_true',
                        help='do not use global alignment to fix residue numbering between native and model during chain permutation (use only in case needle is not installed, and the residues between the chains are identical')
    parser.add_argument('-perm1', default=False, action='store_true',
                        help='use all chain1 permutations to find maximum DockQ (number of comparisons is n! = 24, if combined with -perm2 there will be n!*m! combinations')
    parser.add_argument('-perm2', default=False, action='store_true',
                        help='use all chain2 permutations to find maximum DockQ (number of comparisons is n! = 24, if combined with -perm1 there will be n!*m! combinations')
    #    parser.add_argument('-comb',default=False,action='store_true',help='use all cyclicchain permutations to find maximum DockQ (number of comparisons is n!*m! = 24*24 = 576 for two tetramers interacting')
    parser.add_argument('-model_chain1', metavar='model_chain1', type=str, nargs='+',
                        help='pdb chain order to group together partner 1')
    parser.add_argument('-model_chain2', metavar='model_chain2', type=str, nargs='+',
                        help='pdb chain order to group together partner 2 (complement to partner 1 if undef)')
    parser.add_argument('-native_chain1', metavar='native_chain1', type=str, nargs='+',
                        help='pdb chain order to group together from native partner 1')
    parser.add_argument('-native_chain2', metavar='native_chain2', type=str, nargs='+',
                        help='pdb chain order to group together from native partner 2 (complement to partner 1 if undef)')
    parser.add_argument('-print', type=bool)
    args = parser.parse_args()
    # args.model = 'examples/complex.4.pdb'
    # args.native = 'examples/1AHW.pdb'
    # args.no_needle = True
    args.short = True
    # args.useCA=True
    args.print =False
    args.skip_check = True

    file_path = r"/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/4_label"
    decoy_path = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/3_pdb'
    target_name_path = r"/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/caseID.lst"
    decoy_name_folder = r"/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name"
    decoy_interface_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/14_interface_info/'
    target_name_list = []
    with open(target_name_path,'r') as f:
        for line in f:
            target_name_list.append(line.strip())
    calulate_ground_truth(target_name_list, file_path,args,decoy_path, decoy_name_folder, decoy_interface_folder)
    # queue = []
    # nproc=4
    # while 1:
    #
    #     target_name = target_name_list.pop()
    #     # Create Process for arg
    #     p = Process(target=calulate_ground_truth, args=(target_name,))
    #     queue.append(p)
    #
    #     if (len(queue) == nproc) or (not target_name_list and len(queue)):
    #
    #         for job in queue:
    #             job.start()
    #         for job in queue: # Waiting for job to finish
    #             job.join()
    #         queue = []
    #
    #     if not target_name_list and not queue:
    #         break
    #
    # print("Finished")




    # threads = []
    # for i in range(0, len(target_name_list)-5,20):
    #     target_names = target_name_list[i:i+20]
    #     t = threading.Thread(target=calulate_ground_truth, args = (target_names,))
    #     threads.append(t)
    #
    # target_names = target_name_list[80:85]
    # t = threading.Thread(target=calulate_ground_truth, args=(target_names,))
    # threads.append(t)
    #
    # for t in threads:
    #     t.setDaemon(True)
    #     t.start()
    #
    # t.join()
    # print("all over %s",ctime())











