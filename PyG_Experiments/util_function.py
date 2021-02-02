import pandas as pd
from scipy.spatial.distance import euclidean, pdist, squareform, cdist
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import logging
import os,csv
from typing import List, Any
import torch
import networkx as nx
from scipy.stats import pearsonr,spearmanr
from sklearn import metrics
#from sort_pool_regression import SortPoolClassification

class minmaxScaler:
    """
    https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    """
    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None, mins: np.ndarray=None,maxs: np.ndarray=None):
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token
        self.min=mins
        self.max=maxs
        #minmax_scale= preprocessing.MinMaxScaler().fit(

    def fit(self, X) -> 'StandardScaler':
        """ a=np.array([1,2,3,None],dtype=np.float32)
            b=np.nanmean(a)
            print(f'a:{a}\tb:{b}')
        """
        if type(X) is list:
            X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)#X is the array(sample,num_tasks) use axis=0 get the mean for each task
        self.stds = np.nanstd(X, axis=0)
        self.mins=np.nanmin(X, axis=0)
        self.maxs=np.nanmax(X, axis=0)#when one task has no value, all is none then its mean is None
        self.means = np.where(np.isnan(self.means), None, self.means)#use the mean value(0) for missing nan
        return self

    def transform(self, X):
        if type(X) is list:
            X = np.array(X).astype(float)
        transformed_with_nan = (X - self.mins) / (self.maxs - self.mins)
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)#make the miss=0, miss is the min
        #print(f'minmaxScaler scaling the data now')
        return transformed_with_none

    def inverse_transform(self, X):
        if type(X) is list:
            X = np.array(X).astype(float)
        transformed_with_nan = X * (self.maxs - self.mins) + self.mins
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
        return transformed_with_none


class StandardScaler:
    """A StandardScaler normalizes a dataset.

    When fit on a dataset, the StandardScaler learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the StandardScaler subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):
        """
        Initialize StandardScaler, optionally with means and standard deviations precomputed.

        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: The token to use in place of nans.
        """
        self.means = means#consider when in multitasks need means is the array each task has its mean
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0th axis.

        :param X: A list of lists of floats.
        :return: The fitted StandardScaler.
        """
        if type(X) is list:
            X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)#X shold be list or np array
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)#use the mean value(0) for missing nan
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)#when stds=0 why let =1?

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats.
        :return: The transformed data.
        """
        if type(X) is list:
            X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data.
        """
        if type(X) is list:
            X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

def save_checkpoint(path,
                    model,
                    scaler=None,
                    #features_scaler: StandardScaler = None,
                    args=None):
    """
    Saves a model checkpoint.

    :param model: A QSARmodel.
    :param scaler: A StandardScaler fitted on the data.
    :param features_scaler: A StandardScaler fitted on the features.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'data_scaler': scaler if scaler is not None else None,
        #'features_scaler': { todo when use additional features
        #    'means': features_scaler.means,
        #    'stds': features_scaler.stds
        #} if features_scaler is not None else None
    }
    torch.save(state, path)


def load_checkpoint(path: str,
                    model,
                    current_args= None,
                    logger: logging.Logger = None):
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded model.
    """
    debug = logger.debug if logger is not None else print

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, data_scaler,loaded_state_dict = state['args'],state['data_scaler'], state['state_dict']

    if current_args is not None:
        args = current_args

    # args.cuda = cuda if cuda is not None else args.cuda

    # Build model
    #num_node_features, num_layers, hidden, dense_hidden, k = args
    #model = Classifier(args)
    #model = SortPoolClassification(num_node_features, num_layers, hidden, dense_hidden, k)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():

        if param_name not in model_state_dict:
            debug(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            debug(f'Pretrained parameter "{param_name}" '
                  f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                  f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)


    # if args.mode == 'gpu':
    #     debug('Moving model to cuda')
    #     model = model.cuda()

    return model, args, data_scaler

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def reset(self):
        self.counter = 0
        self.early_stop = False

    def __call__(self,
                 val_metric,
                 path: str,
                 model,
                 mode,
                 scaler: StandardScaler = None,
                 args=None
                 ):

        score = -val_metric
        if mode == 'max':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint_process(val_metric, mode, path, model, scaler, args)
            elif score > self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint_process(val_metric, mode, path, model, scaler, args)
                self.counter = 0
        else:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint_process(val_metric, mode, path, model, scaler, args)
            elif score < self.best_score - self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint_process(val_metric, mode, path, model, scaler, args)
                self.counter = 0


    def save_checkpoint_process(self,val_loss, mode, path, model, scaler, args):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if mode == 'min':
                print(f'monitor decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(f'monitor increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_checkpoint(path, model, scaler, args)
        self.val_loss_min = val_loss


def get_task_names(path: str, cols_to_read) -> List[str]:
    """
    Gets the task names from a data CSV file.

    :param path: Path to a CSV file.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :return: A list of task names.
    """
    task_names = []
    for i in range(1, len(cols_to_read)):
        col = cols_to_read[i]
        task_names.append(get_header(path)[col])

    return task_names


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header

def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)

def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger


def form_atom_dataframe(decoy_path):
    """
    :param decoy_path:
    :return:
    receptor dataframe, ligand dataframe: which includes the info for receptor and ligand
    """
    r_atomic_data = []
    l_atomic_data = []
    with open(decoy_path,'r') as file:
        line = file.readline()               # call readline()
        while line[0:4]!='ATOM':
            line=file.readline()

        count=1
        is_the_same_pdb=True
        pre_serial_number = 0

        while line:
            dat_in = line[0:80].split()
            data = dict()
            if(dat_in[0]=='ATOM'):
                data["record_name"] = line[0:6].strip(" ")
                data["serial_number"] = int(line[6:11].strip(" "))
                data["atom"] = line[12:15].strip(" ")
                data["resi_name"] = line[17:20]
                data["chain_id"] = line[21]
                data["resi_num"] = int(line[23:26])
                data["x"] = float(line[30:37])
                data["y"] = float(line[38:45])
                data["z"] = float(line[46:53])


                if(pre_serial_number > data["serial_number"]):
                    is_the_same_pdb = False

                if (is_the_same_pdb):
                    r_atomic_data.append(data)
                    r_atomic_df = pd.DataFrame(r_atomic_data)
                    r_atomic_df["node_id"] = (
                            str(1)
                            + r_atomic_df["chain_id"]
                            + r_atomic_df["resi_num"].map(str)
                            + r_atomic_df["resi_name"]
                    )
                else:
                    l_atomic_data.append(data)
                    l_atomic_df = pd.DataFrame(l_atomic_data)
                    l_atomic_df["node_id"] = (
                            str(2)
                            + l_atomic_df["chain_id"]
                            + l_atomic_df["resi_num"].map(str)
                            + l_atomic_df["resi_name"]
                    )

                pre_serial_number = int(line[6:11].strip(" "))
                count=count+1
            line = file.readline()
    print('in total, we have %d atoms in receptor, %d atoms in ligand'%(len(r_atomic_df),len(l_atomic_df)))
    return r_atomic_df,l_atomic_df


def get_complex(graph1, graph2,cut_off):
    graph3 = nx.union(graph1, graph2)

    pairse_distmat = cdist(graph1.dataframe[["x", "y", "z"]], graph2.dataframe[["x", "y", "z"]], metric="euclidean")
    # Computes the pairwise euclidean distances between every atom.
    interacting_atoms = np.where(pairse_distmat <= cut_off)

    resi1 = graph1.dataframe.loc[interacting_atoms[0]]["node_id"].values
    resi2 = graph2.dataframe.loc[interacting_atoms[1]]["node_id"].values

    interacting_resis = set(list(zip(resi1, resi2)))
    kind = ["interaction"]
    for i1, i2 in interacting_resis:
        if i1 != i2:
            if graph3.has_edge(i1, i2):
                for k in kind:  # if there is edge between i1 and i2, add a new attribute "kind"
                    graph3.edges[i1, i2]["kind"].add(k)
            else:
                graph3.add_edge(i1, i2, kind=set(kind))  # if there is not edge between i1 and i2, add a new edge with attribute "kind"

        u=i1
        v=i2
        assert graph3.has_edge(u, v), "Edge not present in graph."
        # Defensive programming checks end.

        # Encode one-of-K for bond type.
        bond_set = graph3.edges[u, v]["kind"]

        # Encode one-of-K for bond type.
        bond_lb = LabelBinarizer()
        BOND_TYPES = [
            "hydrophobic",
            "disulfide",
            "hbond",
            "ionic",
            "aromatic",
            "aromatic_sulphur",
            "cation_pi",
            "backbone",
            "delaunay",
            "interaction"
        ]
        bond_lb.fit(BOND_TYPES)

        bonds = np.zeros(len(BOND_TYPES))
        if len(bond_set) > 0:
            bond_array = bond_lb.transform([i for i in bond_set])

            for b in bond_array:
                bonds = bonds + b

        bond_features = bonds
        graph3.edges[u, v]["features"] = np.concatenate((bond_features,))


    return graph3

    # graph1_node_list = list(graph1.nodes)
    # graph2_node_list = list(graph2.nodes)
    # dataframe1 = graph1.dataframe
    # dataframe2 = graph2.dataframe
    # cut_off = 10
    # cut_off = cut_off**2
    #
    # for i in range(len(graph1_node_list)):
    #     for j in range(len(graph2_node_list)):
    #         d1 = dataframe1[dataframe1['node_id'] == graph1_node_list[i]].reset_index()
    #         d2 = dataframe2[dataframe2['node_id'] == graph2_node_list[i]].reset_index()
    #         min_distance = 1000000
    #         residue1_len = len(d1)
    #         residue2_len = len(d2)
    #
    #         for m in range(residue1_len):
    #             atom1=d1.loc[m]
    #             for n in range(residue2_len):
    #                 atom2 = d2.loc[n]
    #                 distance = (atom1["x"]-atom2["x"])**2 + (atom1["y"]-atom2["y"])**2 + (atom1["z"]-atom2["z"])**2
    #                 if distance <= min_distance:
    #                     min_distance = distance
    #
    #         if min_distance <= cut_off:
    #             graph3.add_edge(graph1_node_list[i], graph2_node_list[i], kind={"interaction"})

def compute_PCC(valid_preds, valid_targets, complex_names):

    dict = {'pred':valid_preds, 'target':valid_targets , 'complex_names': complex_names}
    data = pd.DataFrame(dict, columns= dict.keys())
    #get all complex names according to the column
    all_complex_names = data['complex_names'].unique()
    target_sub_corr = []
    for i in range(len(all_complex_names)):
        current_complex = all_complex_names[i]
        current_complex_results = data.loc[data['complex_names'] == current_complex]
        if len(current_complex_results) > 1:
            corr, _ = pearsonr(current_complex_results['pred'],
                               current_complex_results['target'])
        else:
            continue
        target_sub_corr.append(corr)
        #print("target name: " + current_complex + "; Correlation: " + str(target_sub_corr[i]))

    average_corr = sum(target_sub_corr)/len(all_complex_names)
    del dict, data, all_complex_names, target_sub_corr
    return average_corr

def compute_SPCC(valid_preds, valid_targets, complex_names):

    dict = {'pred':valid_preds, 'target':valid_targets , 'complex_names': complex_names}
    data = pd.DataFrame(dict, columns= dict.keys())
    #get all complex names according to the column
    all_complex_names = data['complex_names'].unique()
    target_sub_corr = []
    for i in range(len(all_complex_names)):
        current_complex = all_complex_names[i]
        current_complex_results = data.loc[data['complex_names'] == current_complex]
        if len(current_complex_results) > 1:
            corr, _ = spearmanr(current_complex_results['pred'],
                               current_complex_results['target'])
        else:
            continue
        target_sub_corr.append(corr)
        #print("target name: " + current_complex + "; Correlation: " + str(target_sub_corr[i]))

    average_corr = sum(target_sub_corr)/len(all_complex_names)
    del dict, data, all_complex_names, target_sub_corr
    return average_corr

def compute_top_EF(valid_preds, valid_targets, complex_names, top_num_list, save_folder):
    dict = {'pred': valid_preds, 'target': valid_targets, 'complex_names': complex_names}
    source_data = pd.DataFrame(dict, columns=dict.keys())

    all_complex_names = source_data['complex_names'].unique()

    # df_results = pd.DataFrame(columns=['complex_name'] + ['EF_max'] + ratio_list)
    top_list = []
    for complex_name in all_complex_names:

        df = source_data.loc[source_data['complex_names'] == complex_name].sort_values(by=['pred', 'target'], ascending=False).reset_index(drop=True)
        EF_max = (df.shape[0]) / (df["target"].sum(axis=0))
        if EF_max >= 1200:
            continue
        top_EF_dic = {}
        top_EF_dic['complex_names'] = complex_name
        top_EF_dic['EF_max'] = EF_max
        for EF_top in top_num_list:
            top_dataframe = df.iloc[0:EF_top]
            top_accuracy = (top_dataframe["target"].sum(axis=0)) / (top_dataframe.shape[0])
            top_EF = top_accuracy * EF_max
            top_EF_dic[EF_top] = top_EF
        top_list.append(top_EF_dic)
        del top_EF_dic

    top_dataframe = pd.DataFrame(top_list, columns= ['complex_names'] + ['EF_max'] + top_num_list)

    # df_all = pd.concat([df_all, top_dataframe], axis=0, sort=False).reset_index(drop=True)

    EF_percent_dic = {}
    EF_percent_dic['complex_names'] = 'average'
    EF_percent_dic['EF_max'] = np.mean(top_dataframe['EF_max'])
    for EF_top in top_num_list:
        EF_percent_dic[EF_top] = np.mean(top_dataframe[EF_top])

    df_all = top_dataframe.append(EF_percent_dic, ignore_index=True)

    # EF_percent_dataframe = pd.DataFrame([EF_percent_dic], columns=['name'] + ['EF_max'] + EF_list)

    # df_all = pd.concat([EF_percent_dataframe, df_all], axis=0, sort=False).reset_index(drop=True)
    # path = os.path.join(fir_dir, 'enrichment_result', '20200330-2', 'enrichment_factor-1.csv')
    # makedirs(path, isfile=True)
    df_all.to_csv(os.path.join(save_folder, 'top_EF.csv'), index=False)

    for key, values in EF_percent_dic.items():
        print(str(key) + ':' + str(values))

def compute_ratio_EF(valid_preds, valid_targets, complex_names, max_count, save_folder, save_file_name):
    if len(set(valid_preds)) == 1:
        return 0
    dict = {'pred': valid_preds, 'target': valid_targets, 'complex_names': complex_names}
    source_data = pd.DataFrame(dict, columns=dict.keys())

    all_complex_names = source_data['complex_names'].unique()

    top_list = []
    for complex_name in all_complex_names:
        df = source_data.loc[source_data['complex_names'] == complex_name].sort_values(by=['pred', 'target'], ascending=False).reset_index(drop=True)

        max_positive = df["target"].sum(axis=0)
        EF_max = df.shape[0] / max_positive
        top_EF_dic = {}
        top_EF_dic['complex_names'] = complex_name
        top_EF_dic['EF_max'] = EF_max

        df = df.loc[df['pred'] > 0]

        if len(df) < max_count:
            #append target=predict=0
            for idx in range(max_count - len(df), max_count):
                df.loc[idx] = [0,0,complex_name]

        if max_positive < max_count:
            for k in range(1, max_positive+ 1):
                top_dataframe = df.iloc[0:k]
                top_accuracy = (top_dataframe["target"].sum(axis=0)) / (top_dataframe.shape[0])
                top_EF = top_accuracy * EF_max
                top_EF_dic[k] = top_EF
            for i in range(max_positive + 2, max_count+ 1):
                top_EF_dic[i] = top_EF_dic[k]
        else:
            for k in range(1, max_count+ 1):
                top_dataframe = df.iloc[0:k]
                top_accuracy = (top_dataframe["target"].sum(axis=0)) / (top_dataframe.shape[0])
                top_EF = top_accuracy * EF_max
                top_EF_dic[k] = top_EF

        top_list.append(top_EF_dic)
        del top_EF_dic

    count_list = [i for i in range(1,max_count + 1)]
    top_dataframe = pd.DataFrame(top_list, columns=['complex_names'] + ['EF_max'] + count_list)

    #df_all = pd.concat([df_all, top_dataframe], axis=0, sort=False).reset_index(drop=True)

    EF_percent_dic = {}
    EF_percent_dic['complex_names'] = 'average'
    EF_percent_dic['EF_max'] = np.mean(top_dataframe['EF_max'])
    for i in range(1, max_count +1):
        EF_percent_dic[i] = np.mean(top_dataframe[i])

    df_all = top_dataframe.append(EF_percent_dic, ignore_index= True)

    #EF_percent_dataframe = pd.DataFrame([EF_percent_dic], columns=['name'] + ['EF_max'] + EF_list)

    #df_all = pd.concat([EF_percent_dataframe, df_all], axis=0, sort=False).reset_index(drop=True)
    #path = os.path.join(fir_dir, 'enrichment_result', '20200330-2', 'enrichment_factor-1.csv')
    #makedirs(path, isfile=True)
    df_all.to_csv(os.path.join(save_folder, save_file_name), index=False)

    # for key, values in EF_percent_dic.items():
    #     print(str(key) + ':' + str(values))

def compute_EF(valid_preds, valid_targets, complex_names, accumulative_num : int):
    if len(set(valid_preds)) == 1:
        return 0.0

    dict = {'pred': valid_preds, 'target': valid_targets, 'complex_names': complex_names}
    source_data = pd.DataFrame(dict, columns=dict.keys())

    all_complex_names = source_data['complex_names'].unique()

    ef_list = []

    for complex_name in all_complex_names:
        df = source_data.loc[source_data['complex_names'] == complex_name].sort_values(by=['pred', 'target'], ascending=False).reset_index(drop=True)
        max_positive = df["target"].sum(axis=0)


        df = df.loc[df['pred'] > 0]

        if len(df) < accumulative_num:
            # append target=predict=0
            for idx in range(accumulative_num - len(df), accumulative_num):
                df.loc[idx] = [0, 0, complex_name]

        if max_positive < accumulative_num:
            for k in range(1, max_positive + 1):
                top_dataframe = df.iloc[0:k]
                top_accuracy = (top_dataframe["target"].sum(axis=0)) / (top_dataframe.shape[0]+ 0.00001)
                top_EF = top_accuracy * EF_max
                ef_list.append(top_EF)
            for i in range(max_positive + 2, accumulative_num + 1):
                ef_list.append(ef_list[k])
        else:
            for k in range(1, accumulative_num + 1):
                EF_max = (df.shape[0]) / (df["target"].sum(axis=0) + 0.00001)
                # k = math.ceil(0.001 * df.shape[0])
                top_dataframe = df.iloc[0:k]
                top_accuracy = (top_dataframe["target"].sum(axis=0)) / (top_dataframe.shape[0])
                top_EF = top_accuracy * EF_max
                ef_list.append(top_EF)

    ef_mean = np.mean(ef_list)

    del ef_list

    # ef_mean /= accumulative_num
    del dict, source_data, all_complex_names
    return ef_mean

def compute_success_rate(valid_preds, valid_targets, complex_names, accumulative_num : int):
    if len(set(valid_preds)) == 1:
        return 0.0, {}
    dict = {'pred': valid_preds, 'target': valid_targets, 'complex_names': complex_names}
    source_data = pd.DataFrame(dict, columns=dict.keys())
    all_complex_names = source_data['complex_names'].unique()
    success_rate_at_k = {}

    for k in range(1, accumulative_num + 1):
        success_rate_k = 0.0

        for complex_name in all_complex_names:

            df = source_data.loc[source_data['complex_names'] == complex_name].sort_values(by=['pred', 'target'], ascending=False).reset_index(drop=True)
            if len(set(df['pred'].tolist())) == 1:
                continue
            max_positive = df["target"].sum(axis=0)
            df = df.loc[df['pred'] > 0]

            if len(df) < accumulative_num:
                # append target=predict=0
                for idx in range(accumulative_num - len(df), accumulative_num):
                    df.loc[idx] = [0, 0, complex_name]

            if max_positive < k:
                top_dataframe = df.iloc[0: max_positive]
                top_count = top_dataframe["target"].sum(axis=0)
                if top_count != 0:
                    success_rate_k += 1
            else:
                top_dataframe = df.iloc[0 : k]
                top_count = top_dataframe["target"].sum(axis=0)
                if top_count!=0:
                    success_rate_k +=1

        success_rate_k = success_rate_k / len(all_complex_names)
        success_rate_at_k.update({k:success_rate_k})

    success_rate_mean = sum(list(success_rate_at_k.values()))/ accumulative_num
    del dict, source_data, all_complex_names
    return success_rate_mean, success_rate_at_k



def real_rannk_hit():
    raise NotImplemented

def auc_per_complex(all_pred, all_gt, all_complex_names, pos_label):
    unique_complex_list = set(all_complex_names)
    average_auc = 0.0
    arr_all_pred = np.array(all_pred, np.float)
    arr_all_gt = np.array(all_gt, np.float)
    arr_all_complex_names = np.array(all_complex_names, np.str)
    for current_complex in unique_complex_list:
        mask = (arr_all_complex_names == current_complex)
        fpr, tpr, _ = metrics.roc_curve(arr_all_gt[mask], arr_all_pred[mask], pos_label=pos_label)
        average_auc += metrics.auc(fpr, tpr)
        del mask
    average_auc /= len(unique_complex_list)
    del arr_all_pred, arr_all_gt, arr_all_complex_names
    return average_auc

def rpc_per_complex(all_pred, all_gt, all_complex_names,pos_label):
    unique_complex_list = set(all_complex_names)
    average_rpc = 0.0
    arr_all_pred = np.array(all_pred, np.float)
    arr_all_gt = np.array(all_gt, np.float)
    arr_all_complex_names = np.array(all_complex_names, np.str)
    for current_complex in unique_complex_list:
        mask = (arr_all_complex_names == current_complex)
        precision, recall, _ = metrics.precision_recall_curve(arr_all_gt[mask], arr_all_pred[mask], pos_label=pos_label)
        average_rpc += metrics.auc(recall, precision)
        del mask
    average_rpc /= len(unique_complex_list)
    del arr_all_pred, arr_all_gt, arr_all_complex_names
    return average_rpc


def save_node_significance(node_significance, node_belong_decoy_num, interface_pos, decoy_name , save_file_name):
    dict ={ 'significance': node_significance.reshape((-1)).tolist(), 'decoy_name' : node_belong_decoy_num.reshape((-1)).tolist(), 'at_interface': interface_pos.reshape((-1)).tolist()}
    df_node_significance = pd.DataFrame(dict, columns=['significance','decoy_name','at_interface'])
    #recover the decoy names
    df_node_significance['decoy_name'] = df_node_significance['decoy_name'].apply(lambda x: decoy_name[x][0])
    df_node_significance.to_csv(save_file_name,index=False, mode='a')
    del df_node_significance, node_belong_decoy_num, interface_pos, decoy_name

def read_txt(path):
    """
    read path and save the content into list
    """

    list = []
    with open(path, 'r') as f:
        for line in f:
            list.append(line.strip())

    return list