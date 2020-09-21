import pandas as pd
import os, torch, json
import numpy as np

from PyG_Experiments.import_data import arguments, PPDocking
from torch_geometric.data import InMemoryDataset
# from PyG_Experiments.sort_pool_regression import train_classification,test_regression,test_classification, train_regression, GATSortPool
from PyG_Experiments.util_function import load_checkpoint,rpc_per_complex
from torch_geometric.data import DataLoader
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score,f1_score



def read_meta_learner_scores(GCN_folder, GAT_folder, num_folds):

    for i in range(num_folds):
        GCN_result_path = os.path.join(GCN_folder, f'fold_{i}', '20200330-2', 'predsLogModel.csv')
        GCN_results = pd.read_csv(GCN_result_path)


        GAT_result_path = os.path.join(GAT_folder, f'{i}fold_predict_results.csv')
        GAT_results = pd.read_csv(GAT_result_path)

        pred_mean = np.array([GCN_results['prediction'], GAT_results['prediction']]).mean(axis= 0)

        GCN_results['ensemble_prediction'] = pred_mean

        GCN_results.to_csv(GCN_result_path)


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
    df_train_summary.to_csv(os.path.join(save_folder, 'train_summary.csv'), index = False)

    del nfold_results, args, df_train_summary, train_summary



def predict_meta_learner(args, test_loader, current_data_fold, fold_num, model_save_folder):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GATSortPool(args).to(device)
    currrent_test_loss = current_test_auc = current_test_rpc = current_test_ef = 0
    #load all fold model to do prediction
    for fold_idx in range(0, fold_num):
        model, _, _ = load_checkpoint(os.path.join(model_save_folder,
                        '{:d}fold_{:s}_model.pt'.format(fold_idx, args.mode)), net)
        #do prediction on corresponding test data
        test_loss, test_auc, test_rpc, test_ef, all_complex_names, all_decoy_names, all_scores, all_targets = \
            test_classification(model, device, test_loader)

        # save to file at save_folder
        df_predict = pd.DataFrame({'decoy_no': all_decoy_names, 'complex_name': all_complex_names, 'label': all_targets,
                                   'prediction': all_scores})
        df_predict.to_csv(os.path.join(model_save_folder, r'{:d}_model_{:d}_data_prediction.csv'.format(fold_idx, current_data_fold)), index = False)
        if fold_idx == current_data_fold:
           currrent_test_loss = test_loss
           current_test_auc = test_auc
           current_test_rpc = test_rpc
           current_test_ef = test_ef

        del test_loss, test_auc, test_rpc, test_ef, all_complex_names, all_decoy_names, all_scores, all_targets,model,df_predict

    del net

    return {'fold': current_data_fold, 'loss':currrent_test_loss, 'auc': current_test_auc, 'rpc': current_test_rpc, 'ef': current_test_ef}



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
                                 num_workers=6)


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
        df_currrent_learner_test_performance.to_csv(os.path.join(save_folder, 'all_fold_test_performance.csv'), index = False)
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

    df_fold_all_test_results_from_meta_learner = pd.DataFrame(columns=['decoy_no', 'complex_name', 'label'])
    col_list = []
    for meta_idx in range(start_idx, end_idx + 1):
        test_results_save_folder = os.path.join(model_save_folder,
             "200809_dataset(classification)train{:d}_val{:d}_testall_vauc_savecheckpoint_GCNE_3layers_max_JK_AllGlobalPooling".format(meta_idx,meta_idx))
        df_current_fold_results = pd.read_csv(os.path.join(test_results_save_folder, r'data_prediction.csv'))

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
    df_fold_all_test_results_from_meta_learner['std'] = df_predictions_from_meta_learners.std(axis=1)
    df_fold_all_test_results_from_meta_learner['mad'] = df_predictions_from_meta_learners.mad(axis=1)
    df_fold_all_test_results_from_meta_learner['Q1'] = df_predictions_from_meta_learners.quantile(q = 0.25, axis=1)
    df_fold_all_test_results_from_meta_learner['Q3'] = df_predictions_from_meta_learners.quantile(q = 0.75, axis=1)
    df_fold_all_test_results_from_meta_learner['max'] = df_predictions_from_meta_learners.max(axis=1)
    df_fold_all_test_results_from_meta_learner['min'] = df_predictions_from_meta_learners.min(axis=1)


    # save as a csv file at model_save_folder
    df_fold_all_test_results_from_meta_learner.to_csv(os.path.join(model_save_folder, 'all_learner_results.csv'), index = False)
    del col_list, df_predictions_from_meta_learners
    return df_fold_all_test_results_from_meta_learner

def generate_statics_from_meta_learners(fold_num, model_save_folder):
    for fold_idx in range(0, fold_num):
        df_current_fold_prediction = pd.read_csv(os.path.join(model_save_folder, r'{:d}fold_merge_learner_results.csv'.format(fold_idx)))

        #get all column names of df_current_fold_prediction and then exclude the two
        col_list = list(df_current_fold_prediction)
        col_list.remove('decoy_no')
        col_list.remove('label')
        col_list.remove('complex_name')
        #declare a new dataframe to save the statics as ggregation prediction results
        df_ensemble_results = pd.DataFrame(columns = ['decoy_no', 'label'])
        df_ensemble_results['decoy_no'] = df_current_fold_prediction['decoy_no']
        df_ensemble_results['label']  = df_current_fold_prediction['label']
        df_ensemble_results['complex_name'] = df_current_fold_prediction['complex_name']
        df_meta_learner_predictions = df_current_fold_prediction[col_list]

        df_ensemble_results['max'] = df_meta_learner_predictions.max(axis=1)
        df_ensemble_results['median'] = df_meta_learner_predictions.median(axis=1)
        df_ensemble_results['mean'] = df_meta_learner_predictions.mean(axis=1)
        #df_ensemble_results['Qn90'] = df_meta_learner_predictions.quantile(q= 0.9, axis= 1)
        # df_ensemble_results['Qn95'] = df_meta_learner_predictions.quantile(q= 0.95, axis= 1)
        # df_ensemble_results['Qn85'] = df_meta_learner_predictions.quantile(q= 0.85, axis= 1)
        # df_ensemble_results['Qn80'] = df_meta_learner_predictions.quantile(q= 0.8, axis= 1)
        # df_ensemble_results['Qn75'] = df_meta_learner_predictions.quantile(q=0.75, axis=1)
        # df_ensemble_results['Qn70'] = df_meta_learner_predictions.quantile(q=0.7, axis=1)
        # df_ensemble_results['Qn60'] = df_meta_learner_predictions.quantile(q=0.6, axis=1)

        #save to another csv file as aggregation prediction results
        df_ensemble_results.to_csv(os.path.join(model_save_folder, r'{:d}fold_ensemble_results.csv'.format(fold_idx)), index = False)
        del col_list, df_meta_learner_predictions, df_current_fold_prediction,df_ensemble_results

#def calculate_all_learner_EF1():


def merge_results_from_new_meta_learners(selected_list_from_original_results, original_result_folder, new_start_idx, new_end_idx, new_result_folder, fold_num):
    # model_list = []
    # for model_idx in range(new_start_idx, new_end_idx):
    #     model_list.append('new_' + str(model_idx))
    #do merge
    # for fold_idx in range(0, fold_num):
        #current_fold_all_model_results = pd.DataFrame(columns=['decoy_no'])
    original_current_fold_predictions = pd.read_csv(
        os.path.join(original_result_folder, r'all_learner_results.csv'))
    #original_current_fold_predictions = original_current_fold_predictions.sort_values(by='decoy_no').reset_index(drop=True)

    # for model_idx in range(new_start_idx, new_end_idx):
    #     new_current_fold_predictions = pd.read_csv(
    #         os.path.join(new_result_folder, r'fold_{:d}'.format(fold_idx), r'{:d}model'.format(model_idx),
    #                      'predsLogModel.csv'))
    #     new_current_fold_predictions = new_current_fold_predictions.sort_values(by='decoy_no').reset_index(drop=True)
    #
    #     original_current_fold_predictions['new_' + str(model_idx)] = new_current_fold_predictions['prediction']
    #     del new_current_fold_predictions

    #save merge results into
    #selected_learner_predictions = original_current_fold_predictions[['decoy_no', 'label', 'complex_name'] + selected_list_from_original_results +  model_list]
    selected_learner_predictions = original_current_fold_predictions[
        ['decoy_no', 'label', 'complex_name'] + selected_list_from_original_results]
    selected_learner_predictions.to_csv(os.path.join(original_result_folder,r'merge_learner_results.csv'), index = False)
    del original_current_fold_predictions

    return selected_learner_predictions



if __name__ == '__main__':

    # args = arguments(num_node_features=26, num_layers=2, hidden=32,
    #                  dense_hidden=256, k=100, interface_only=True, hop=-1, head=2, batch_size=512, fold_num=5,
    #                  mode='classification')
    #
    # data_folder = r'/home/fei/Desktop/docking/data/ensemble_data/'
    #
    # model_folder = r'/home/fei/Desktop/docking/trained_model/ensemble_models/'
    #
    # bulk_train_meta_learners(start_idx=2, end_idx= 41, args= args,  data_folder= data_folder, model_save_folder= model_folder)
    #
    #
    # args = arguments(num_node_features=26, num_layers=2, hidden=32,
    #                  dense_hidden=256, k=100, interface_only=True, hop=-1, head=2, batch_size=256, fold_num=5,
    #                  mode='classification')
    # bulk_predict_by_mata_learners(start_idx=2, end_idx= 41, args= args,  data_folder= data_folder, model_save_folder= model_folder)

    # model_folder = r'/run/media/fei/Project/docking/ENSEMBLE_MODEL/'
    # extract_test_results_into_csv(start_idx=0, end_idx= 2, fold_num = 5, model_save_folder= model_folder)
    # original_result_folder = r'/run/media/fei/Project/docking/ENSEMBLE_MODEL/'
    # new_result_folder = r'/run/media/fei/Project/docking/GCN_ensemble/'

    # merge_results_from_new_meta_learners(['66', '126', '3', '162','163','170','2','141','64','181','25', '91', '164', '171', '9', '123', '94', '17', '51', '153'], original_result_folder = original_result_folder, new_start_idx = 0,
    #                                      new_end_idx = 12, new_result_folder = new_result_folder, fold_num= 5)
    # generate_statics_from_meta_learners(fold_num= 5, model_save_folder= original_result_folder)

    # model_folder = r'/run/media/fei/Project/docking/ENSEMBLE_MODEL/best_GAT_ensemble/'
    # #generate_statics_from_meta_learners(start_idx=0, end_idx=200, fold_num= 5, model_save_folder= model_folder)
    # generate_statics_from_meta_learners(fold_num= 5, model_save_folder= model_folder)


    model_save_folder = r'/home/fei/Research/saved_checkpoints/200809_dataset(classification)train1-11_val1-11_testall_vauc_savecheckpoint_GCNE_3layers_max_JK_AllGlobalPooling/'
    start_idx = 1
    end_idx = 11
    fold_num = 5
    pos_label = 1
    indicators_list = [str(i) for i in range(start_idx, end_idx +1)]
    indicators1_list = ['mean', 'median', 'std','mad','Q1', 'Q3','max','min']

    selected_learner_predictions = extract_test_results_into_csv(start_idx, end_idx, fold_num, model_save_folder)

    # original_result_folder = r'/home/fei/Research/saved_checkpoints/200809_dataset(classification)train1-9_val1-9_testall_vauc_savecheckpoint_GCNE_3layers_max_JK_AllGlobalPooling/'
    # new_result_folder = r'/home/fei/Research/saved_checkpoints/200809_dataset(classification)train1-9_val1-9_testall_vauc_savecheckpoint_GCNE_3layers_max_JK_AllGlobalPooling/'

    # selected_learner_predictions = merge_results_from_new_meta_learners(['mean', 'median', 'std', 'mad', 'Q1', 'Q3'], original_result_folder = original_result_folder, new_start_idx = 0,
    #                                      new_end_idx = 12, new_result_folder = new_result_folder, fold_num= 5)

    # selected_learner_predictions = pd.read_csv(os.path.join(model_save_folder, 'all_learner_results.csv'))
    # auc_max = 0
    # auc_max_indicators = ""
    all_targets = list(selected_learner_predictions["label"])
    all_complex_names = list(selected_learner_predictions["complex_name"])
    theshold_list =[i*0.01 for i in range(60,101)]

    all_scores_label_list = []
    for select_result in indicators1_list:
        # print(select_result)
        all_scores = list(selected_learner_predictions[select_result])

        precision, recall, _ = metrics.precision_recall_curve(np.array(all_targets, np.float),np.array(all_scores, np.float), pos_label=pos_label)
        auc_precision_recall = metrics.auc(recall, precision)
        print(select_result + ":" + str(auc_precision_recall))


        for theshold in theshold_list:
            all_scores_label = []
            for i in range(len(all_scores)):
                if all_scores[i] >= theshold:
                    all_scores_label.append(1)
                else:
                    all_scores_label.append(0)

            recall_score_result = recall_score(np.array(all_targets, np.float), np.array(all_scores_label, np.float), pos_label=pos_label)
            precision_score_result = precision_score(np.array(all_targets, np.float), np.array(all_scores_label, np.float), pos_label=pos_label)
            f1_score_result = f1_score(np.array(all_targets, np.float),
                                                     np.array(all_scores_label, np.float), pos_label=pos_label)
            if recall_score_result >= 0.8:
                all_scores_label_list.append(all_scores_label)
                print("theshold: {:f}".format(theshold))
                print("recall score: {:f}; precision_score: {:f}; f1_score: {:f}".format(recall_score_result,
                                                                                         precision_score_result,
                                                                                         f1_score_result))
            #     break
            # else:
            #     continue
    # final_label_list = []
    # for i in range(0,len(all_targets)):
    #     negtive_number = 0
    #     for j in range(0, len(all_scores_label_list)):
    #         negtive_number += all_scores_label_list[j][i]
    #
    #     if negtive_number >=(end_idx + 1)/2:
    #         final_label = 1
    #     else:
    #         final_label = 0
    #
    #     final_label_list.append(final_label)
    #
    # recall_score_result = recall_score(np.array(all_targets, np.float), np.array(final_label_list, np.float),
    #                                    pos_label=pos_label)
    # precision_score_result = precision_score(np.array(all_targets, np.float), np.array(final_label_list, np.float),
    #                                          pos_label=pos_label)
    # f1_score_result = f1_score(np.array(all_targets, np.float),
    #                            np.array(final_label_list, np.float), pos_label=pos_label)
    #
    # print("recall score: {:f}; precision_score: {:f}; f1_score: {:f}".format(recall_score_result,
    #                                                                          precision_score_result,
    #                                                                          f1_score_result))



        # if auc_precision_recall > auc_max:
        #     auc_max_indicators = select_result
        #     auc_max = auc_precision_recall

        # unique_complex_list = set(all_complex_names)
        # average_rpc = 0.0
        # auc_prc_dic = {}
        # arr_all_pred = np.array(all_scores, np.float)
        # arr_all_gt = np.array(all_targets, np.float)
        # arr_all_complex_names = np.array(all_complex_names, np.str)
        # for current_complex in unique_complex_list:
        #     mask = (arr_all_complex_names == current_complex)
        #     precision, recall, _ = metrics.precision_recall_curve(arr_all_gt[mask], arr_all_pred[mask], pos_label=pos_label)
        #     auc_rpc = metrics.auc(recall, precision)
        #     average_rpc += auc_rpc
        #     auc_prc_dic[current_complex] = auc_rpc
        #     del mask
        # # for name in auc_prc_dic.keys():
        # #     print(name + ":" + str(auc_prc_dic[name]))
        # average_rpc /= len(unique_complex_list)
        # print(select_result + " average prc:" + str(average_rpc))
        # del arr_all_pred, arr_all_gt, arr_all_complex_names

    # auc_max_indicators = "median"
    # all_scores = list(selected_learner_predictions[auc_max_indicators])
    # precision, recall, _ = metrics.precision_recall_curve(np.array(all_targets, np.float),np.array(all_scores, np.float), pos_label=pos_label)
    # auc_max = metrics.auc(recall, precision)
    #
    # plt.figure()
    # lw = 2
    # plt.plot(recall, precision, color = 'darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % auc_max)
    # plt.plot([1,0], [0,1], color = 'navy', lw = lw, linestyle = '--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("recall")
    # plt.ylabel("precision")
    # plt.title("Receiver operating characteristic")
    # plt.legend(loc = "lower right")
    # plt.show()

    del selected_learner_predictions



