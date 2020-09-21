from scipy.spatial.distance import euclidean, pdist, squareform, cdist
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

def compute_distmat(dataframe, atom_list):
    """
    Computes the pairwise euclidean distances between every atom.

    Design choice: passed in a DataFrame to enable easier testing on
    dummy data.
    atom_list: involve atom list
    """
    if atom_list!=None:
        dataframe = dataframe[dataframe['atom'].isin(atom_list)]

    eucl_dists = pdist(dataframe[["x", "y", "z"]], metric="euclidean")
    euclidean_dists = pd.DataFrame(squareform(eucl_dists))
    euclidean_dists.index = dataframe.index
    euclidean_dists.columns = dataframe.index
    del eucl_dists

    return euclidean_dists


def add_node_info(dataframe):
    """
    get nodeid from pdb dataframe and save it in list
    """
    nodeid_initial_list = dataframe["node_id"].tolist()
    nodeid_list = list(set(nodeid_initial_list))
    nodeid_list.sort(key=nodeid_initial_list.index)   ##remove redundancy

    del nodeid_initial_list

    return nodeid_list


def get_new_atom_dataframe(dataframe, i1,GLY_dataframe):
    '''
    :param i1: redisue node id
    :return: dataframe including SC
    '''
    Backbone_Atom = ['N', 'CA', 'C', 'O']
    dataframe = dataframe[dataframe["node_id"] == i1][["atom", "x", "y", "z"]].reset_index(drop=True)

    # print(i1)
    assert 'N' in dataframe["atom"].values
    assert 'CA' in dataframe["atom"].values
    assert 'C' in dataframe["atom"].values
    assert 'O' in dataframe["atom"].values
    ###check the residue order
    side_chain_df = dataframe[(~dataframe["atom"].isin(Backbone_Atom))]
    backbone_df = dataframe[dataframe["atom"].isin(Backbone_Atom)]


    if len(side_chain_df) != 0:
        side_chain = side_chain_df.mean(axis=0)
        backbone_df = backbone_df.append({"atom": "SC", 'x': side_chain['x'], 'y': side_chain['y'], 'z': side_chain['z']}, ignore_index=True)
    else:
        try:
            GLY_side_chain = GLY_dataframe[GLY_dataframe['node_id']==i1]
            backbone_df = backbone_df.append({"atom": "SC", 'x': float(GLY_side_chain['x']), 'y': float(GLY_side_chain['y']), 'z': float(GLY_side_chain['z'])}, ignore_index=True)
        except Exception:
            print("The lack of side chain is not GLY: " + i1)

    del side_chain_df, dataframe

    backbone_df = backbone_df.sort_values(by=['atom'], ascending=True).reset_index(drop=True)
    return backbone_df


def add_edge_info(dataframe, atomlist, cutoff,GLY_dataframe):
    """
    :param dataframe:
    :param args:
    :param GLY_dataframe: Gly sidechain x,y,z
    :return: edge_tuple list, edge feature list
    """

    edge_tuple_list = []
    edge_feature_list = []
    cut_off_index = []
    cut_off_column = []

    # Add in edges for amino acids that distance of any two atoms from different proteins less than 5A.
    distmat = compute_distmat(dataframe, atomlist)
    #interacting_atoms = (distmat <= args.cutoff)

    if atomlist != None: ############if compute distance only using atom in the atomlist
        for col in distmat.columns:
            target_index = distmat[distmat[col] <= cutoff].index.tolist()
            target_column = [int(col)] * len(target_index)
            cut_off_index += target_index
            cut_off_column += target_column

        resi1 = dataframe.loc[cut_off_index]["node_id"].values
        resi2 = dataframe.loc[cut_off_column]["node_id"].values
    else:   ############if compute distance only using atom in the atomlist
        interacting_atoms = np.where(distmat <= cutoff)   # the distance of any atom is less than 5A
        # interacting_atoms = get_interacting_atoms_(args.cutoff, distmat)
        resi1 = dataframe.loc[interacting_atoms[0]]["node_id"].values
        resi2 = dataframe.loc[interacting_atoms[1]]["node_id"].values

    interacting_resis = set(list(zip(resi1, resi2)))

    ###very slow
    for index, (i1, i2) in enumerate(interacting_resis):
        if i1 != i2 and (i1, i2) not in edge_tuple_list and (i2, i1) not in edge_tuple_list:

            dataframe1 = get_new_atom_dataframe(dataframe, i1, GLY_dataframe)
            dataframe2 = get_new_atom_dataframe(dataframe, i2, GLY_dataframe)  ###atom order:C, CA, N, O, SC

            # Computes the pairwise euclidean distances between every atom.
            pairse_distmat = cdist(dataframe1[["x", "y", "z"]], dataframe2[["x", "y", "z"]], metric="euclidean")
            edge_tuple_list.append((i1, i2))
            edge_tuple_list.append((i2, i1))
            edge_feature_list.append(pairse_distmat.reshape(-1).tolist())
            edge_feature_list.append(pairse_distmat.T.reshape(-1).tolist())

    return edge_tuple_list,edge_feature_list
    # t0 = time.clock()
    # a = dataframe.groupby(["node_id"])
    # print(time.clock() - t0)
    #
    # t0 = time.clock()
    # a.get_group("A100ASN").reset_index()
    # print(time.clock() - t0)
    #
    # for index in range(0, len(a)):
    #     dataframe1 = a[index]
    #     dataframe2 = a[index+1]
    #     pairse_distmat = cdist(dataframe1[["x", "y", "z"]], dataframe2[["x", "y", "z"]], metric="euclidean")


def compute_distmat_between_different_proteins(dataframe1, dataframe2, atom_list):
    """
    Computes the pairwise euclidean distances between every atom.

    Design choice: passed in a DataFrame to enable easier testing on
    dummy data.
    """

    #1_compute the distance of any two atoms from dataframe1 and dataframe2.
    if atom_list!=None:
        dataframe1 = dataframe1[dataframe1['atom'].isin(atom_list)]
        dataframe2 = dataframe2[dataframe2['atom'].isin(atom_list)]

    pairse_distmat = cdist(dataframe1[["x", "y", "z"]], dataframe2[["x", "y", "z"]], metric="euclidean")
    pairse_distmat = pd.DataFrame(pairse_distmat)
    pairse_distmat.index = dataframe1.index
    pairse_distmat.columns = dataframe2.index

    return pairse_distmat


def get_interface_edge(dataframe1, dataframe2, atomlist,cutoff,GLY_dataframe):
    '''
    :param dataframe1: receptor dataframe
    :param dataframe2: ligand dataframe
    :param args:
    :return: edge tuple and edge feature list
    '''
    edge_tuple_list = []
    edge_feature_list = []
    cut_off_index = []
    cut_off_column = []

    pairse_distmat = compute_distmat_between_different_proteins(dataframe1, dataframe2, atomlist)

    #2_obtain residue nodeid by atom index
    if atomlist != None:
        ############if compute distance only using atom in the atomlist
        for col in pairse_distmat.columns:
            target_index = pairse_distmat[pairse_distmat[col] <= cutoff].index.tolist()
            target_column = [int(col)] * len(target_index)
            cut_off_index += target_index
            cut_off_column += target_column
            del target_index, target_column

        resi1 = dataframe1.loc[cut_off_index]["node_id"].values
        resi2 = dataframe2.loc[cut_off_column]["node_id"].values
    else:
        ############if compute distance using all atom
        interacting_atoms = np.where(pairse_distmat <= cutoff)  # the distance of any atom is less than 5A
        # interacting_atoms = get_interacting_atoms_(args.cutoff, distmat)
        resi1 = dataframe1.loc[interacting_atoms[0]]["node_id"].values
        resi2 = dataframe2.loc[interacting_atoms[1]]["node_id"].values

    del pairse_distmat
    interacting_resis = set(list(zip(resi1, resi2)))   ###because we need (i1,i2) and (i2,i1) both

    for index, (i1, i2) in enumerate(interacting_resis):
        if i1 != i2 and (i1, i2) not in edge_tuple_list and (i2, i1) not in edge_tuple_list:
            d1 = get_new_atom_dataframe(dataframe1, i1,GLY_dataframe)
            d2 = get_new_atom_dataframe(dataframe2, i2,GLY_dataframe)  ###atom order:C, CA, N, O, SC

            # Computes the pairwise euclidean distances between every atom.
            pairse_distmat = cdist(d1[["x", "y", "z"]], d2[["x", "y", "z"]], metric="euclidean")
            edge_tuple_list.append((i1, i2))
            edge_tuple_list.append((i2, i1))
            edge_feature_list.append(pairse_distmat.reshape(-1).tolist())
            edge_feature_list.append(pairse_distmat.T.reshape(-1).tolist())

            del pairse_distmat

    return edge_tuple_list,edge_feature_list


def generate_graph_info(dataframe, atomlist, cutoff, GLY_dataframe):
    '''
    :param dataframe: pdb dataframe
    :param args:
    :return: node list, edge_tuple list, edge feature list
    '''
    node_list = add_node_info(dataframe)
    edge_tuple_list, edge_feature_list = add_edge_info(dataframe, atomlist, cutoff, GLY_dataframe)

    return node_list, edge_tuple_list, edge_feature_list