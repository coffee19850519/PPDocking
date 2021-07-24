import pandas as pd
import os, torch, json
import numpy as np
from pdb2sql import interface

from import_data import arguments, PPDocking
from torch_geometric.data import InMemoryDataset
from sort_pool_regression import train_classification,test_regression,test_classification, train_regression, GATSortPool
from util_function import load_checkpoint
from torch_geometric.data import DataLoader

def read_meta_learner_scores(GCN_folder, GAT_folder, num_folds):

    for i in range(num_folds):
        GCN_result_path = os.path.join(GCN_folder, f'fold_{i}', '20200330-2', 'predsLogModel.csv')
        GCN_results = pd.read_csv(GCN_result_path)


        GAT_result_path = os.path.join(GAT_folder, f'{i}fold_predict_results.csv')
        GAT_results = pd.read_csv(GAT_result_path)

        pred_mean = np.array([GCN_results['prediction'], GAT_results['prediction']]).mean(axis= 0)

        GCN_results['ensemble_prediction'] = pred_mean

        GCN_results.to_csv(GCN_result_path, index= False)


class DockingBulkData(InMemoryDataset):
    def __init__(self,
                 root,
                 fold,
                 hop,
                 mode,
                 ensemble_idx,
                 split='train',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        assert split in ['train', 'val', 'test']

        super(DockingBulkData, self).__init__(root, transform, pre_transform, pre_filter)

        if split == 'train':
            self.data, self.slices = torch.load(self.root + r'/{:d}/{:d}_train_data.pt'.format(ensemble_idx, fold))
        elif split == 'val':
            self.data, self.slices = torch.load(self.root + r'/{:d}/{:d}_val_data.pt'.format(ensemble_idx,  fold))
        elif split == 'test':
            self.data, self.slices = torch.load(self.root + r'/{:d}/{:d}_test_data.pt'.format(ensemble_idx, fold))

def train_meta_learner(args, data_folder, model_save_folder,meta_idx):
    nfold_results = []
    train_summary = []
    save_folder = os.path.join(model_save_folder, str(meta_idx))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    for n in range(0, args.fold_num):
        print(str(n) + ' fold starts ... ...')

        train_dataset = DockingBulkData(root=data_folder, split='train', fold=n, hop=args.hop, mode=args.mode, ensemble_idx= meta_idx)
        val_dataset = DockingBulkData(root=data_folder, split='val', fold=n, hop=args.hop, mode=args.mode, ensemble_idx= meta_idx)

        model, summary = train_classification(train_dataset, val_dataset, args, fold=n,
                                              model_save_folder=save_folder)
        train_summary.extend(summary)
        del train_dataset, val_dataset, model, summary

    # save training details into csv file
    df_train_summary = pd.DataFrame(train_summary)
    df_train_summary.to_csv(os.path.join(save_folder, 'train_summary.csv'), index= False)

    del nfold_results, args, df_train_summary, train_summary



def predict_meta_learner(args, test_loader, current_data_fold, fold_num, model_save_folder):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GATSortPool(args).to(device)
    currrent_test_loss = current_test_auc = current_test_rpc = current_test_ef1 = 0
    #load all fold model to do prediction
    for fold_idx in range(0, fold_num):
        model, _, _ = load_checkpoint(os.path.join(model_save_folder,
                        '{:d}fold_{:s}_model.pt'.format(fold_idx, args.mode)), net)
        #do prediction on corresponding test data
        test_loss, test_auc, test_rpc, test_ef1, all_complex_names, all_decoy_names, all_scores, all_targets = \
            test_classification(model, device, test_loader)

        # save to file at save_folder
        df_predict = pd.DataFrame({'decoy_no': all_decoy_names, 'complex_name': all_complex_names, 'label': all_targets,
                                   'prediction': all_scores})
        df_predict.to_csv(os.path.join(model_save_folder, r'{:d}model_{:d}data_prediction.csv'.format(fold_idx, current_data_fold)))
        if fold_idx == current_data_fold:
           currrent_test_loss = test_loss
           current_test_auc = test_auc
           current_test_rpc = test_rpc
           current_test_ef1 = test_ef1

        del test_loss, test_auc, test_rpc, test_ef1, all_complex_names, all_decoy_names, all_scores, all_targets,model,df_predict

    del net

    return {'fold': current_data_fold, 'loss':currrent_test_loss, 'auc': current_test_auc, 'rpc': current_test_rpc, 'ef1': current_test_ef1}



def bulk_train_meta_learners(start_idx, end_idx, args, data_folder, model_save_folder):

    for i in range(start_idx, end_idx):
        print('start training No. {:d} meta-learner...'.format(i))
        train_meta_learner(args, data_folder, model_save_folder, i)

def bulk_predict_by_mata_learners(start_idx, end_idx, args, data_folder, model_save_folder):

    all_fold_test_performance = []

    for fold_idx in range(0, args.fold_num):

        # read 3600-test data
        test_dataset = PPDocking(data_folder, fold_idx, args.hop, args.mode, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=8)


        #load checkpoints from meta-learners and do prediction
        for i in range(start_idx, end_idx):
            save_folder = os.path.join(model_save_folder, str(i))
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            test_performance_dict = \
            predict_meta_learner(args, test_loader, fold_idx, args.fold_num, save_folder)
            all_fold_test_performance.append({'meta_learner_idx':i, 'test_performance': test_performance_dict})
            del test_performance_dict, save_folder

        del test_dataset, test_loader

    #re-organize test performance
    df_test_performance = pd.DataFrame(all_fold_test_performance)
    for i in range(start_idx, end_idx):
        save_folder = os.path.join(model_save_folder, str(i))
        current_learner_performance = df_test_performance[df_test_performance['meta_learner_idx'] == i]['test_performance']
        df_currrent_learner_test_performance = pd.DataFrame(current_learner_performance.tolist())
        df_currrent_learner_test_performance.to_csv(os.path.join(save_folder, 'all_fold_test_performance.csv'), index= False)
        del df_currrent_learner_test_performance

    del all_fold_test_performance, df_test_performance
# # create stacked model input dataset as outputs from the ensemble
# def stacked_dataset(members, inputX):
# 	stackX = None
# 	for model in members:
# 		# make prediction
# 		yhat = model.predict(inputX, verbose=0)
# 		# stack predictions into [rows, members, probabilities]
# 		if stackX is None:
# 			stackX = yhat
# 		else:
# 			stackX = dstack((stackX, yhat))
# 	# flatten predictions to [rows, members x probabilities]
# 	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
# 	return stackX

# fit a model based on the outputs from the ensemble members
# def fit_stacked_model(members, inputX, inputy):
# 	# create dataset using ensemble
# 	stackedX = stacked_dataset(members, inputX)
# 	# fit standalone model
# 	model = LogisticRegression()
# 	model.fit(stackedX, inputy)
#     yhat = model.predict(stackedX)
# 	return model

# make a prediction with the stacked model
# def stacked_prediction(members, model, inputX):
# 	# create dataset using ensemble
# 	stackedX = stacked_dataset(members, inputX)
# 	# make a prediction
#
# 	return yhat

def extract_test_results_into_csv(start_idx, end_idx, fold_num, model_save_folder):
    for fold_idx in range(0, fold_num):

        df_fold_all_test_results_from_meta_learner = pd.DataFrame(columns=['decoy_no', 'complex_name', 'label'])
        col_list = []
        for meta_idx in range(start_idx, end_idx):
            test_results_save_folder = os.path.join(model_save_folder, str(meta_idx))
            df_current_fold_results = pd.read_csv(os.path.join(test_results_save_folder, r'{:d}model_{:d}data_prediction.csv'.format(fold_idx, fold_idx)))

            if len(df_fold_all_test_results_from_meta_learner) == 0:
                df_fold_all_test_results_from_meta_learner['decoy_no'] = df_current_fold_results['decoy_no']
                df_fold_all_test_results_from_meta_learner['complex_name'] = df_current_fold_results['complex_name']
                df_fold_all_test_results_from_meta_learner['label'] = df_current_fold_results['label']

            df_fold_all_test_results_from_meta_learner[str(meta_idx)] = df_current_fold_results['prediction']
            col_list.append(str(meta_idx))
            del df_current_fold_results
        # compute some statics of all predictions from all meta-learners

        df_predictions_from_meta_learners = df_fold_all_test_results_from_meta_learner[col_list]
        df_fold_all_test_results_from_meta_learner['mean'] = df_predictions_from_meta_learners.mean(axis = 1)
        df_fold_all_test_results_from_meta_learner['median'] = df_predictions_from_meta_learners.median(axis=1)
        df_fold_all_test_results_from_meta_learner['max'] = df_predictions_from_meta_learners.max(axis=1)
        df_fold_all_test_results_from_meta_learner['Qn90'] = df_predictions_from_meta_learners.quantile(q = 0.9, axis=1)
        df_fold_all_test_results_from_meta_learner['Qn95'] = df_predictions_from_meta_learners.quantile(q = 0.95, axis=1)
        df_fold_all_test_results_from_meta_learner['Qn80'] = df_predictions_from_meta_learners.quantile(q = 0.80, axis=1)
        df_fold_all_test_results_from_meta_learner['Qn85'] = df_predictions_from_meta_learners.quantile(q=0.85, axis=1)

        # save as a csv file at model_save_folder
        df_fold_all_test_results_from_meta_learner.to_csv(os.path.join(model_save_folder, '{:d}fold_all_learner_results.csv'.format(fold_idx)), index= False)
        del col_list, df_predictions_from_meta_learners, df_fold_all_test_results_from_meta_learner


if __name__ == '__main__':

    args = arguments(num_node_features=26, num_layers=2, hidden=32,
                     dense_hidden=256, k=100, interface_only=True, hop=-1, head=2, batch_size=256, fold_num=5,
                     mode='classification')

    data_folder = r'/home/fei/Desktop/docking/data/ensemble_data/best_GAT_subsets/'

    model_folder = r'/home/fei/Desktop/docking/trained_model/best_GAT_ensemble_parameters/different_hop_100k_2head/hop2/'

    bulk_train_meta_learners(start_idx=0, end_idx= 20, args= args,  data_folder= data_folder, model_save_folder= model_folder)


    # args = arguments(num_node_features=26, num_layers=2, hidden=32,
    #                  dense_hidden=256, k=100, interface_only=True, hop=-1, head=2, batch_size=256, fold_num=5,
    #                  mode='classification')

    bulk_predict_by_mata_learners(start_idx=0, end_idx= 20, args= args,  data_folder= data_folder, model_save_folder= model_folder)


    extract_test_results_into_csv(start_idx=0, end_idx=20, fold_num=5, model_save_folder=model_folder)