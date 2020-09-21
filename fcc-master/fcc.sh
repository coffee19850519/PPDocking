#!/bin/bash

file_path="/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/"
input=$file_path"caseID_test.lst"

pdb_path="/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/3_pdb/"


while IFS= read -r line
do
    echo $line
    cd $pdb_path$line
    
    # Make a file list with all your PDB files
    ls *.pdb > pdb.list
    
    # Ensure all PDB models have segID identifiers
    # Convert chainIDs to segIDs if necessary using scripts/pdb_chainxseg.py
    
    echo "The first step starts!"
    for pdb in $( cat pdb.list ); 
    do /home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/protein_quality_assessment_code/fcc-master/scripts/pdb_chainxseg.py $pdb > temp; 
    mv temp $pdb; 
    done
    echo "The first step ends!"
    
    echo "The second step starts!"
    # Generate contact files for all PDB files in pdb.list
    # using 4 cores on this machine.
    python /home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/protein_quality_assessment_code/fcc-master/scripts/make_contacts.py -f pdb.list -n 4 -p $pdb_path$line
    echo "The second step ends!"

    # Create a file listing the names of the contact files
    # Use file.list to maintain order in the cluster output
    sed -e 's/pdb/contacts/' pdb.list | sed -e '/^$/d' > pdb.contacts
    
    echo "The third step starts!"
    # Calculate the similarity matrix
    python /home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/protein_quality_assessment_code/fcc-master/scripts/calc_fcc_matrix.py -f pdb.contacts -o fcc_matrix.out
    echo "The third step ends!"
    
    
    echo "The fourth step starts!"
    # Cluster the similarity matrix using a threshold of 0.75 (75% contacts in common)
    python /home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/protein_quality_assessment_code/fcc-master/scripts/cluster_fcc.py fcc_matrix.out 0.75 -o clusters_0.75.out -c 1
    echo "The fourth step ends!"
    
    echo "The fifth step starts!"
    # Use ppretty_clusters.py to output meaningful names instead of model indexes
    python /home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/protein_quality_assessment_code/fcc-master/scripts/ppretty_clusters.py clusters_0.75.out pdb.list
    echo "The fifth step ends!"
    
    rm -rf pdb.list pdb.contacts fcc_matrix.out
    find . -name '*.contacts' -type f -exec rm -rf {} \;
    echo "All steps end!"

done < "$input"
