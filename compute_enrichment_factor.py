import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from torch_geometric.data import DataLoader
from util_function import compute_ratio_EF, compute_top_EF,load_checkpoint
from sort_pool_regression import GATSortPool,test_classification,test_regression
from import_data import PPDocking, arguments

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

def plot_results(complex_name,top_accuracy_list,path):

    makedirs(path)
    fig,ax = plt.subplots()

    ax.plot(top_accuracy_list, '.-')
    ax.set_ylim([0, 1.0])
    ax.grid(True)
    ax.set_xlabel('Top ranked considered')
    ax.set_ylabel('Hit Rate')
    ax.set_title(complex_name)

    plt.savefig(os.path.join(path, "Top ranked considered.pdf"))


def calculate_EF(fold_num, args, checkpoint_folder, ratio_list, top_list, mean_std_dict = None, save_prediction_folder = None):
    all_complex_names = []
    all_scores = []
    all_targets = []
    mean = 0
    std = 1
    # load normalized mean and std
    if mean_std_dict is not None:
        df_mean_std = pd.DataFrame(mean_std_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #declare model

    for n in range(fold_num):

        ##load test data
        test_dataset = PPDocking(root= r'', fold= n, hop= args.hop, mode= args.mode, split= 'test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=6)
        #read checkpoint
        model, _, _ = load_checkpoint(
            os.path.join(checkpoint_folder,'{:d}fold_{:s}_model.pt'.format(n, args.mode)),
            GATSortPool(args).to(device))

        #predict
        if args.mode == 'classification':
            
            _, _, _,_, fold_complex_names, fold_decoy_names, fold_scores, fold_targets = \
            test_classification(model, device, test_loader)

        else:
            mean = df_mean_std.loc[df_mean_std['fold'] == n]['mean']
            std = df_mean_std.loc[df_mean_std['fold'] == n]['std']
            _, _, _,_, fold_scores, fold_complex_names, fold_decoy_names, fold_targets = \
            test_regression(model, device, test_loader, mean= mean, std= std)

        dict = {'decoy_no': fold_decoy_names, 'label':fold_targets, 'prediction':fold_scores}
        fold_results = pd.DataFrame(dict)
        if save_prediction_folder is not None and not os.path.exists(save_prediction_folder):
            os.makedirs(save_prediction_folder)

        fold_results.to_csv(os.path.join(save_prediction_folder, '{:d}fold_predict_results.csv'.format(n)), index=False)


        all_complex_names.extend(fold_complex_names)
        all_scores.extend(fold_scores)
        all_targets.extend(fold_targets)
        del test_dataset, test_loader, fold_complex_names, fold_scores, fold_targets, model#,fold_results,dict

    compute_ratio_EF(all_scores, all_targets, all_complex_names, ratio_list, checkpoint_folder)
    compute_top_EF(all_scores, all_targets, all_complex_names, top_list, checkpoint_folder)


if __name__ == '__main__':
    args = arguments(num_node_features=26, num_layers=2, hidden=32,
                     dense_hidden=512, k=100, interface_only=True, hop=-1, head= 4, mode= 'classification', batch_size=1000, fold_num= 5)
    checkpoint_folder= r'/home/fei/Desktop/docking/trained_model/best_model_reimplementation_valloss/'
    ratio_list = [0.0001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    top_list = [20, 50, 100, 200, 500]
    #if args.mode == 'regression':
        #read mean and std


    calculate_EF(5, args, checkpoint_folder, ratio_list, top_list,save_prediction_folder= checkpoint_folder)



