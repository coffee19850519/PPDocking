import os
import torch, glob, json
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import DataLoader, DataListLoader
from sklearn import metrics
from tqdm import tqdm
from PyG_Experiments.compute_enrichment_factor import calculate_average_EF_from_meta_learners, \
    calculate_successful_rate, calculate_average_hit_count
from PyG_Experiments.GCNE_net import GCEonv_with_global_pool
from PyG_Experiments.self_attention_net import SFEConv_with_global_pool
from torch.nn import Linear,Conv1d, MaxPool1d
from torch_geometric.nn import data_parallel, global_mean_pool, GATConv, SAGPooling,GraphConv, TopKPooling, GCNConv ,JumpingKnowledge, global_max_pool, GlobalAttention,global_sort_pool,BatchNorm
from torch_geometric.utils import subgraph, to_undirected
from PyG_Experiments.losses import group_smooth_l1, group_L1, group_mse, rank_net_loss, rank_loss, weighted_l1_loss, list_top1_CE, list_net_loss, STlist_net_loss, rankMSE_loss_function
import PyG_Experiments.import_data

from point_cloud.test import DockPointNet

from point_cloud.data_convertor import PPDocking_point_cloud

from PyG_Experiments.util_function import EarlyStopping, compute_PCC, compute_EF, compute_SPCC, compute_success_rate,load_checkpoint
from PyG_Experiments.GRATConv import GRATConv, GlobalAttentionPooling
from PyG_Experiments.CGConv_net import CGConvSortPool, CGConvSortPool_by_occurrence, CGConv_by_occurrence_global_pool
from PyG_Experiments.sort_pool_by_occurrence import GATSortPool_by_occurrence,GATSortPool_by_occurrence_global_pool
from PyG_Experiments.CGConv_net import CGConv_by_occurrence_global_pool
# from PyG_Experiments.import_data import PPDocking
import torch_geometric.transforms as T


def train_classification(args, model, fold, model_save_folder, train_patch_size, train_subset_number, val_subset_number,earlystop,learning_rate):#,  mean, std):

    # transforms = T.Compose([
    #     T.RandomTranslate(0.01),
    #     T.RandomRotate(15, axis=0),
    #     T.RandomRotate(15, axis=1),
    #     T.RandomRotate(15, axis=2)
    # ])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.savecheckpoints_mode,
                                                           factor=0.8, patience=3,
                                                           min_lr=0.0001)

    # val_dataset = PyG_Experiments.import_data.PPDocking(root=args.root,
    #                                                     subset=val_subset_number,
    #                                                     fold=fold,
    #                                                     patch=0,
    #                                                     mode=args.mode,
    #                                                     split='val',
    #                                                     load_folder=args.val_classification_patch_path)

    val_dataset = PPDocking_point_cloud(root= args.root, fold= fold,
                                        split= 'val')

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_dataset = PPDocking_point_cloud(root= args.root, fold= fold,
                                        split= 'train')

    # train_dataset = PyG_Experiments.import_data.PPDocking(root=args.root,
    #                                                       subset=train_subset_number,
    #                                                       fold=fold,
    #                                                       patch=0,
    #                                                       mode=args.mode,
    #                                                       split='train',
    #                                                       load_folder=args.train_classification_patch_path
    #                                                       )
    sample_length = 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # sample_length += len(train_loader.dataset)

    for epoch in range(1, args.epochs):
        total_loss = 0

        summary = []
        model.train()

        for train_patch_number in range(train_patch_size):
            print('doing training: {:d} epoch {:d} patch'.format(epoch, train_patch_number))

            for step, data in enumerate(tqdm(train_loader)):
                data = data.to(args.device)
                optimizer.zero_grad()
                loss, y = model(data)

                # if data.y.size()[0] != out.size()[0]:
                #     print("Target size is not the same with input size")
                # loss = sigmoid_focal_loss(out, data.y, reduction= 'mean')
                # loss = F.binary_cross_entropy_with_logits(out, y, pos_weight=torch.tensor([args.pos_weights], device=args.device))

                # loss = F.binary_cross_entropy_with_logits(out, data.y)
                # out = F.log_softmax(out, dim=1)
                # loss = F.nll_loss(out, data.y.long())
                loss.backward()
                # if step%10 == 0:
                #     print('step: {:d} loss:{:.3f}'.format(step, float(loss)))
                total_loss += float(loss) * y.size(0)
                sample_length += y.size(0)
                #data.num_graphs
                optimizer.step()
                del data


        average_loss = total_loss / sample_length
        #, all_scores, all_targets, val_ef, val_success_rate
        val_loss, all_complex_names, all_decoy_names, all_scores, all_targets, auc, auc_precision_recall = val_classification(args, model, val_loader)
        # val_loss, val_auc, val_rpc, val_ef, all_complex_names, _, all_scores, all_targets = test_classification(args, model, fold, split = "val", max_subset_number = args.val_max_subset_number,patch_size = val_patch_size)

        # if args.savecheckpoints_monitor == "val_loss":
        #     monitor = val_loss
        # elif args.savecheckpoints_monitor == "val_rpc":
        #     monitor = val_rpc
        # elif args.savecheckpoints_monitor == "val_rpc+sr":
        #     monitor = val_rpc + val_success_rate
        # elif args.savecheckpoints_monitor == "val_sr":
        #     monitor = val_success_rate
        # elif args.savecheckpoints_monitor == "val_ef":
        #     monitor = val_ef
        # elif args.savecheckpoints_monitor == "val_sr+ef":
        #     monitor = val_ef * 0.5 + val_success_rate
        monitor = val_loss
        earlystop(monitor, os.path.join(model_save_folder, '{:d}_fold_{:s}_model.pt'.format(fold, args.mode)),  model, mode=args.savecheckpoints_mode, args=args)
        scheduler.step(monitor)

        lr = scheduler.optimizer.param_groups[0]['lr']

        # if best_val_error is None or val_error <= best_val_error:
        #     test_error = test(val_loader)
        #     best_val_error = val_error

        #, val_ef: {:4f}, val_success_rate: {:4f}, val_ef,val_success_rate, val_ef, val_success_rate, 'val_ef10':  val_ef, "val_success_rate":val_success_rate

        print('Epoch: {:03d}, LR: {:4f}, Loss: {:.4f},  val_loss: {:.4f}, val_auc:{:.4f}, val_prc:{:.4f}'.format(epoch, lr, average_loss, val_loss, auc,auc_precision_recall))
        summary.append({'fold':fold, 'epoch':epoch, 'lr':lr, 'train_loss':loss.item(), 'val_loss': val_loss, 'val_auc': auc, 'val_prc':auc_precision_recall})
        del  val_loss, all_complex_names, all_scores, all_targets
        # del val_loss, val_auc, val_rpc, all_complex_names, all_scores, all_targets
        if earlystop.early_stop:
            print("Early stopping")
            break

        # save training details into csv file
        df_train_summary = pd.DataFrame(summary)
        df_train_summary.to_csv(os.path.join(model_save_folder, r'train_summary.csv'), index=False, mode='a')
        del df_train_summary

    del val_dataset, val_loader, train_loader, train_dataset
    return model, lr


@torch.no_grad()
def val_classification(args, model, test_loader):
    all_scores = []
    all_targets = []
    all_val_loss = 0
    all_complex_names = []
    all_decoy_names = []
    model.eval()
    coupling_num = 0
    for data in tqdm(test_loader):
        all_complex_names.extend(np.array(data.complex_name).reshape(-1).tolist())
        all_decoy_names.extend(np.array(data.decoy_name).reshape(-1).tolist())
        data = data.to(args.device)
        with torch.no_grad():
            # out, node_significance, node_belong_decoy_num, interface_pos, decoy_name = model(data, data.num_graphs)
            val_loss, out, y = model(data)
            # del node_significance, node_belong_decoy_num, interface_pos, decoy_name
            #val_loss = sigmoid_focal_loss(out, data.y, reduction='mean')
            # val_loss = F.binary_cross_entropy_with_logits(out, y, pos_weight= torch.tensor([args.pos_weights],device = args.device))
            # out = torch.sigmoid(out)
            # val_loss = F.binary_cross_entropy(out, y)
            all_val_loss += float(val_loss) * y.size(0)
        all_scores.append(out)
        all_targets.append(y)
        coupling_num += y.size(0)
        del data

    all_scores = torch.cat(all_scores).cpu().numpy().reshape(-1).tolist()
    # all_scores = np.where(all_scores >= 0.5, 1, -1).tolist()

    all_targets = torch.cat(all_targets).cpu().numpy().reshape(-1).tolist()

    # val_acc = metrics.accuracy_score(all_targets, all_scores)

    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(all_targets, all_scores, pos_label=1)
    auc_precision_recall = metrics.auc(recall, precision)
    val_loss = all_val_loss / coupling_num
    # ef = compute_EF(all_scores, all_targets, all_complex_names)
    # success_rate = compute_success_rate(all_scores, all_targets, all_complex_names)
    del all_val_loss
    # return val_loss, auc, auc_precision_recall,all_complex_names, all_decoy_names, all_scores, all_targets, val_acc#, ef, success_rate
    return val_loss, all_complex_names, all_decoy_names, all_scores, all_targets, auc, auc_precision_recall

def test_classification(args, model, fold, split,max_subset_number,patch_size):
    all_scores = []
    all_targets = []
    all_test_loss = 0
    all_complex_names = []
    all_decoy_names = []
    sample_length = 0

    if split =='val':
        load_folder = args.val_classification_patch_path
    elif split=='test':
        load_folder = args.test_classification_patch_path

    # split = 'train'
    model.eval()
    for test_patch_number in range(patch_size):
        print('doing {:s}: {:d} fold {:d} patch'.format(split, fold, test_patch_number))
        # test_dataset = PyG_Experiments.import_data.PPDocking(root=args.root,
        #                                                       subset=max_subset_number,
        #                                                       fold=fold,
        #                                                       patch=test_patch_number,
        #                                                       mode='classification',
        #                                                       split=split,
        #                                                       load_folder =load_folder
        #                                                      )
        test_dataset = PPDocking_point_cloud(args.root, fold= fold, split = split)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        # sample_length += len(test_loader.dataset)

        for data in tqdm(test_loader):
            all_complex_names.extend(np.array(data.complex_name).reshape(-1).tolist())
            all_decoy_names.extend(np.array(data.decoy_name).reshape(-1).tolist())
            data = data.to(args.device)
            with torch.no_grad():
                # out, node_significance, node_belong_decoy_num, interface_pos, decoy_name = model(data, data.num_graphs)
                #
                # del node_significance, node_belong_decoy_num, interface_pos, decoy_name
                test_loss, out,y = model(data)
                #val_loss = sigmoid_focal_loss(out, data.y, reduction='mean')
                # test_loss = F.binary_cross_entropy_with_logits(out, y, pos_weight=torch.tensor([args.pos_weights], device=args.device))
                # out = torch.sigmoid(out)
                # test_loss = F.binary_cross_entropy(out, y)
                all_test_loss += float(test_loss) * y.size(0)

            sample_length += y.size(0)
            all_scores.append(out)
            all_targets.append(y)
            del data

        del test_loader, test_dataset


    all_scores = torch.cat(all_scores).cpu().numpy().reshape(-1).tolist()
    all_targets = torch.cat(all_targets).cpu().numpy().reshape(-1).tolist()
    # test_acc = metrics.accuracy_score(all_targets, all_scores)

    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(all_targets, all_scores, pos_label=1)
    auc_precision_recall = metrics.auc(recall, precision)
    test_loss = all_test_loss / sample_length
    # ef = compute_EF(all_scores, all_targets, all_complex_names)
    # success_rate = compute_success_rate(all_scores, all_targets, all_complex_names)
    # del all_test_loss
    # return val_loss, auc, auc_precision_recall, all_complex_names, all_decoy_names, all_scores, all_targets, test_acc#, ef, success_rate
    return test_loss, all_complex_names, all_decoy_names, all_scores, all_targets, auc, auc_precision_recall

# def train_regression(train_dataset, val_dataset, args, fold,  mean, std):
def train_regression(args, model, fold, model_save_folder, train_patch_size, val_patch_size, train_subset_number,
                             val_subset_number, earlystop, learning_rate):  # ,  mean, std):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.savecheckpoints_mode,
                                                           factor=0.8, patience=3,
                                                           min_lr=0.0001)
    sample_length = 0
    val_dataset = PyG_Experiments.import_data.PPDocking(root=args.root,
                                                        subset=val_subset_number,
                                                        fold=fold,
                                                        patch=0,
                                                        mode=args.mode,
                                                        split='val',
                                                        load_folder=args.val_classification_patch_path)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last= False)
    train_dataset = PyG_Experiments.import_data.PPDocking(root=args.root,
                                                          subset=train_subset_number,
                                                          fold=fold,
                                                          patch=0,
                                                          mode=args.mode,
                                                          split='train',
                                                          load_folder=args.train_classification_patch_path
                                                          )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last= False)
    sample_length += len(train_loader.dataset)
    model.train()
    for epoch in range(1, args.epochs):
        total_loss = []
        summary = []
        for train_patch_number in range(train_patch_size):
            print('doing training: {:d} epoch {:d} patch'.format(epoch, train_patch_number))

            for step, data in enumerate(train_loader):

                data = data.to(args.device)
                if args.regression_label_type == "DockQ":
                    regression_label = data.DockQ
                elif args.regression_label_type == 'IOU':
                    regression_label = data.IOU
                elif args.regression_label_type == 'normalized_DockQ':
                    regression_label = data.normalized_DockQ

                optimizer.zero_grad()

                out, node_significance, node_belong_decoy_num, interface_pos, decoy_name = model(data, data.num_graphs)
                # save_node_significance(node_significance, node_belong_decoy_num, interface_pos, decoy_name,
                #                        r'/home/fei/Desktop/gif/node.csv')

                del node_significance, node_belong_decoy_num, interface_pos, decoy_name

                # if args.regression_label_type == "DockQ":
                #     regression_label = torch.cat([data.DockQ for data in data_list]).to(out.device)
                    # regression_label =  data.DockQ
                if regression_label.size()[0] != out.size()[0]:
                    print("Target size is not the same with input size")

                # complex_list = []
                # for data in data_list:
                #     complex_list.extend(data.complex_name)
                # loss = F.binary_cross_entropy_with_logits(out, regression_label)
                #out = torch.sigmoid(out)
                # loss = rank_loss(out.view(-1), regression_label, data.complex_name, 'listNet', combined_loss_name= 'l1_loss', epoch= epoch)
                # loss = group_L1(out.view(-1), regression_label, data.complex_name, out.device)

                # loss = weighted_l1_loss(out.view(-1), regression_label, (data.CAPRI +1)* 3.0)  + 5 *list_top1_CE(out.view(-1), regression_label)
                # loss = F.smooth_l1_loss(out.view(-1), regression_label)
                # loss = group_mse(out.view(-1), regression_label, data.complex_name, out.device)
                loss = F.mse_loss(out.view(-1), regression_label)# + F.kl_div(F.log_softmax(out.view(-1, 1), 0),
                            #    F.softmax(regression_label.view(-1, 1), 0),reduction='sum')
                # loss = weighted_l1_loss(out.view(-1), regression_label, (data.CAPRI +1)* 3.0)
                # loss = STlist_net_loss(out.view(-1), regression_label)
                # loss = list_net_loss(out.view(-1), regression_label)
                # loss = rankMSE_loss_function(out.view(-1), regression_label)

                # print(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)

                optimizer.step()
                total_loss.append(loss.item() * data.num_graphs)

                # if step % 10 == 0:
                    # val_loss, val_auc, val_rpc, val_ef, val_success_rate, all_complex_names, _, all_scores, all_targets = val_classification(args, model, val_loader)
                    # val_loss, val_auc, val_rpc, val_ef, all_complex_names, _, all_scores, all_targets = test_classification(args, model, fold, split = "val", max_subset_number = args.val_max_subset_number,patch_size = val_patch_size)

        average_loss = np.sum(np.array(total_loss, np.float)) / sample_length
        val_loss, val_mae, val_pcc, val_spcc, val_ef, val_sr, _, _, _, _, _ = val_regression(args, model,
                                                                                             val_loader)

        if args.savecheckpoints_monitor == "val_loss":
            monitor = val_loss
        elif args.savecheckpoints_monitor == "val_mae":
            monitor = val_mae
        elif args.savecheckpoints_monitor == "val_pcc":
            monitor = val_pcc
        elif args.savecheckpoints_monitor == "val_sr+ef":
            monitor = val_ef * 0.5 + val_sr
        elif args.savecheckpoints_monitor == "val_sr":
            monitor = val_sr
        elif args.savecheckpoints_monitor == "val_spcc":
            monitor = val_spcc

        earlystop(monitor,
                  os.path.join(model_save_folder, '{:d}_fold_{:s}_model.pt'.format(fold, args.mode)), model,
                  mode=args.savecheckpoints_mode, args=args)
        scheduler.step(monitor)

        lr = scheduler.optimizer.param_groups[0]['lr']

        print('Epoch: {:03d}, step: {:d}, LR: {:4f}, Loss: {:.4f}, val_loss: {:.4f}, val_mae: {:.4f}, val_pcc: {:.4f}, val_spcc: {:.4f}, val_ef: {:.4f}, val_sr: {:.4f}'.format(
                epoch, step, lr, average_loss, val_loss.item(), val_mae.item(), val_pcc, val_spcc, val_ef, val_sr))
        summary.append(
            {'fold': fold, 'epoch': epoch, 'step': step , 'lr': lr, 'train_loss': average_loss, 'val_loss': val_loss.item(),
             'val_mae': val_mae.item(), 'val_pcc': val_pcc, 'val_spcc': val_spcc, 'val_ef': val_ef, 'val_sr': val_sr})


            #del train_loader, train_dataset



        # save training details into csv file
        df_train_summary = pd.DataFrame(summary)
        if epoch == 1 and fold == 0:
            df_train_summary.to_csv(os.path.join(model_save_folder, r'train_summary.csv'), index=False, mode='w')
        else:
            df_train_summary.to_csv(os.path.join(model_save_folder, r'train_summary.csv'), header= False, index=False, mode='a')
        del df_train_summary

        if earlystop.early_stop:
            print("Early stopping")
            break


    del val_dataset, val_loader, train_loader, train_dataset
    return model, lr


@torch.no_grad()
def val_regression(args, model, test_loader):

    sample_length = 0


    all_labels = []
    all_scores = []
    all_complex_names = []
    all_decoy_names = []
    all_categories = []

    model.eval()

    sample_length += len(test_loader.dataset)

    for data in test_loader:
        all_complex_names.extend(np.array(data.complex_name).reshape(-1).tolist())
        all_decoy_names.extend(np.array(data.decoy_name).reshape(-1).tolist())

        data = data.to(args.device)
        if args.regression_label_type == "DockQ":
            regression_label = data.DockQ
        elif args.regression_label_type == 'IOU':
            regression_label = data.IOU
        elif args.regression_label_type == 'normalized_DockQ':
            regression_label = data.normalized_DockQ

        with torch.no_grad():
            out, node_significance, node_belong_decoy_num, interface_pos, decoy_name = model(data, data.num_graphs)

            del node_significance, node_belong_decoy_num, interface_pos, decoy_name
            # val_loss = sigmoid_focal_loss(out, data.y, reduction='mean')

            #out = torch.sigmoid(out)
            # val_loss = F.l1_loss(out.view(-1), regression_label)
            # val_loss = rank_loss(out.view(-1), regression_label, data.complex_name, 'listNet', 'l1_loss')
            # val_loss = group_L1(out.view(-1), regression_label, data.complex_name, out.device)
            # vout += out - torch.mean(out)
            # vy += data.y - torch.mean(data.y)
        # error += (out.view(-1) - regression_label).abs().sum().item()  # MAE

        all_scores.extend(list(out.view(-1).cpu().numpy().reshape((-1))))
        all_labels.extend(list(regression_label.cpu().numpy().reshape((-1))))
        # all_val_loss.append(val_loss.item())
        all_categories.extend(list(data.y.cpu().numpy().reshape((-1))))
        # all_decoy_names.extend(np.array(data.decoy_name).reshape((-1)).tolist())

    # val_loss = rank_loss(torch.tensor(all_scores,dtype= torch.float, device= 'cpu'),
    #                      torch.tensor(all_labels, dtype= torch.float, device= 'cpu'),
    #                      all_complex_names, 'listNet', 'l1_loss').item()
    # val_loss = F.l1_loss(torch.tensor(all_scores,dtype= torch.float, device= 'cpu'),
    #                      torch.tensor(all_labels, dtype= torch.float, device= 'cpu'))
    # val_loss = F.smooth_l1_loss(torch.tensor(all_scores,dtype= torch.float, device= 'cpu'),
    #                      torch.tensor(all_labels, dtype= torch.float, device= 'cpu'))
    val_loss = F.mse_loss(torch.tensor(all_scores,dtype= torch.float, device= 'cpu'),
                         torch.tensor(all_labels, dtype= torch.float, device= 'cpu')) #+ F.kl_div(F.log_softmax(torch.tensor(all_scores, dtype=torch.float, device='cpu').view(-1, 1), 0),
             # F.softmax(torch.tensor(all_labels, dtype=torch.float, device='cpu').view(-1, 1), 0), reduction='sum')
    #val_loss = weighted_l1_loss(torch.tensor(all_scores,dtype= torch.float, device= 'cpu'),
      #                  torch.tensor(all_labels, dtype= torch.float, device= 'cpu'), data.CAPRI * 3.0)
    # val_loss = group_mse(torch.tensor(all_scores, dtype=torch.float, device='cpu'),
    #                             torch.tensor(all_labels, dtype= torch.float, device= 'cpu'),
    #                             all_complex_names, device= 'cpu')
   # val_loss =0
    val_mae = group_L1(torch.tensor(all_scores,dtype= torch.float, device= 'cpu'),torch.tensor(all_labels, dtype= torch.float, device= 'cpu'), all_complex_names, device= 'cpu')
    # val_loss = val_mae

    # torch.sum(vout * vy) / (torch.sqrt(torch.sum(vout ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    average_pcc = compute_PCC(all_scores, all_labels, all_complex_names)
    average_spcc = compute_SPCC(all_scores, all_labels, all_complex_names)
    average_ef = compute_EF(all_scores, all_categories, all_complex_names)
    average_sr = compute_success_rate(all_scores, all_categories, all_complex_names)

    return val_loss, val_mae, average_pcc,average_spcc, average_ef, average_sr, all_complex_names, all_decoy_names, all_scores,all_labels, all_categories



@torch.no_grad()
def test_regression(args, model, fold, split,max_subset_number,patch_size):

    sample_length = 0
    error = 0

    all_labels = []
    all_scores = []
    all_complex_names = []
    all_decoy_names = []
    all_categories = []

    if split == 'train':
        load_folder = args.train_classification_patch_path
    elif split =='val':
        load_folder = args.val_classification_patch_path
    else:
        load_folder = args.test_classification_patch_path

    model.eval()
    for test_patch_number in range(patch_size):
        print('doing {:s}: {:d} fold {:d} patch'.format(split, fold, test_patch_number))
        test_dataset = PyG_Experiments.import_data.PPDocking(root=args.root,
                                                             subset=max_subset_number,
                                                             fold=fold,
                                                             patch=test_patch_number,
                                                             mode=args.mode,
                                                             split=split,
                                                             load_folder=load_folder
                                                             )

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        sample_length += len(test_loader.dataset)

        for data in test_loader:
            all_complex_names.extend(np.array(data.complex_name).reshape(-1).tolist())
            all_decoy_names.extend(np.array(data.decoy_name).reshape(-1).tolist())

            data = data.to(args.device)
            if args.regression_label_type == "DockQ":
                regression_label = data.DockQ
            elif args.regression_label_type == 'IOU':
                regression_label = data.IOU
            elif args.regression_label_type == 'normalized_DockQ':
                regression_label = data.normalized_DockQ

            with torch.no_grad():
                out, node_significance, node_belong_decoy_num, interface_pos, decoy_name = model(data, data.num_graphs)

                del node_significance, node_belong_decoy_num, interface_pos, decoy_name
                # val_loss = sigmoid_focal_loss(out, data.y, reduction='mean')
                #out = torch.sigmoid(out)
                # val_loss = F.l1_loss(out.view(-1), regression_label)
                # val_loss = rank_loss(out.view(-1), regression_label, data.complex_name, 'listNet', 'l1_loss')
                # vout += out - torch.mean(out)
                # vy += data.y - torch.mean(data.y)
            #error += (out.view(-1) - regression_label).abs().sum().item()  # MAE
            all_scores.extend(list(out.view(-1).cpu().numpy().reshape((-1))))
            all_labels.extend(list(regression_label.cpu().numpy().reshape((-1))))
            # all_val_loss.append(val_loss.item())
            all_categories.extend(list(data.y.cpu().numpy().reshape((-1))))
            # all_decoy_names.extend(np.array(data.decoy_name).reshape((-1)).tolist())

    val_loss = F.mse_loss(out.view(-1), regression_label).item()
    # val_loss = F.kl_div(F.log_softmax(out, 0), regression_label).item()
    # val_loss = F.l1_loss(out.view(-1), regression_label).item()
    # torch.sum(vout * vy) / (torch.sqrt(torch.sum(vout ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    average_pcc = compute_PCC(all_scores, all_labels, all_complex_names)
    average_ef = compute_EF(all_scores, all_categories, all_complex_names)
    average_spcc = compute_SPCC(all_scores, all_labels, all_complex_names)
    average_sr = compute_success_rate(all_scores, all_categories, all_complex_names)
    return val_loss, average_pcc, average_spcc, average_ef, average_sr, all_complex_names, all_decoy_names, all_scores,all_labels, all_categories

class arguments:
    def __init__(self, num_node_features, num_edge_attr,regression_label_type,
                 num_layers, hidden, dense_hidden, k, interface_only, hop, head, mode, concat,
                 batch_size, fold_num, pos_weights, learning_rate, device, savecheckpoints_mode,
                 savecheckpoints_monitor, earlystop_patience, epochs,
                 train_classification_patch_path, val_classification_patch_path, test_classification_patch_path,
                 test_max_subset_number, boostrapping_train_start_number,boostrapping_train_max_number,
                 root,model_save_name,checkpoints_folder,pretrained_model_folder,pretrained_model_mode
                 ):
        self.num_node_features = num_node_features
        self.num_layers = num_layers
        self.num_edge_attr = num_edge_attr
        self.regression_label_type = regression_label_type
        self.hidden = hidden
        self.dense_hidden = dense_hidden
        self.k = k
        self.interface_only = interface_only
        self.hop = hop
        self.head = head
        self.mode = mode
        self.batch_size = batch_size
        self.fold_num = fold_num
        self.concat = concat
        self.pos_weights = pos_weights
        self.learning_rate = learning_rate
        self.device = device
        self.savecheckpoints_mode = savecheckpoints_mode
        self.savecheckpoints_monitor = savecheckpoints_monitor
        self.earlystop_patience = earlystop_patience
        self.epochs = epochs
        self.train_classification_patch_path = train_classification_patch_path
        self.val_classification_patch_path = val_classification_patch_path
        self.test_classification_patch_path = test_classification_patch_path
        self.boostrapping_train_start_number = boostrapping_train_start_number
        self.boostrapping_train_max_number = boostrapping_train_max_number
        self.test_max_subset_number = test_max_subset_number
        self.root = root
        self.model_save_name=model_save_name
        self.checkpoints_folder=checkpoints_folder
        self.pretrained_model_folder=pretrained_model_folder
        self.pretrained_model_mode = pretrained_model_mode


if __name__ == '__main__':
    args = arguments(
        # 1.node feature and edge feature
        num_node_features=36,
        num_edge_attr=25,
        regression_label_type="DockQ",

        # 2.model setting
        num_layers=2,
        hidden=32,
        dense_hidden=512,
        k=100,
        interface_only=True,
        hop=-1,
        head=4,
        mode='regression',
        concat=False,

        # 3.training setting
        fold_num=1,
        batch_size=300,
        pos_weights=3.0,
        # learning_rate=0.0005,
        learning_rate=[0.01, 0.01, 0.01, 0.01, 0.01],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        savecheckpoints_mode='max',
        savecheckpoints_monitor="val_sr+ef",
        earlystop_patience=20,
        epochs=201,

        # 4.dataset setting
        train_classification_patch_path="/run/media/fei/Windows/hanye/regression_patch/",
        val_classification_patch_path="/run/media/fei/Windows/hanye/all_small_regression_patch/",
        test_classification_patch_path="/run/media/fei/Windows/hanye/all_small_regression_patch/",

        boostrapping_train_start_number=1,
        boostrapping_train_max_number=1,
        test_max_subset_number=100,

        # 5.others
        root=r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/',
        model_save_name="200909_regression_valmse_savecheckpoint_GCN_2layers_no_JK_AllGlobalPooling_boostrapping_noSigmoid",
        checkpoints_folder=r'/home/fei/Research/saved_checkpoints/',
        pretrained_model_folder="/home/fei/Research/saved_checkpoints/200906_regression_valOriginalL1_savecheckpoint_selfAttention_3layers_no_JK_AllGlobalPooling_weightedL1Loss_new_valtest_noSigmoid/",
        pretrained_model_mode="classification"
    )

    label_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/'
    model_save_folder = os.path.join(args.checkpoints_folder, args.model_save_name)

    nfold_results = []
    # train_summary = []
    all_mean_std = []

    all_decoy_names = []
    all_complex_names = []
    all_targets = []
    all_scores = []
    all_labels = []
    all_acc = []
    for n in range(0, 1):#args.fold_num):

        print(str(n) + ' fold starts ... ...')
        # learning_rate = args.learning_rate
        learning_rate = args.learning_rate[n]

        training_model = GCEonv_with_global_pool(args)
        # training_model = SFEConv_with_global_pool(args)
        # training_model = GATEConv_with_global_pool(args)
        #training_model = DataParallel(training_model)

        training_model = training_model.to(args.device)

        ###load trained weights
        training_model, _, _ = load_checkpoint(
            os.path.join(args.checkpoints_folder, args.model_save_name,'{:d}_fold_{:s}_model.pt'.format(n, 'regression')), training_model)



        test_temp = glob.glob(os.path.join(args.test_classification_patch_path,
                                           '{:d}_fold_*_patch_test_data.pt'.format(n)))
        test_patch_size = len(test_temp)


        del test_temp

        # split = "test"
        if args.mode == 'classification':
            # test_loss, test_auc, test_rpc, test_ef, all_complex_names,_, all_scores, all_targets = test_classification(model, device, test_loader)
            test_acc, all_complex_names, all_decoy_names, all_scores, all_targets = test_classification(
                args=args, model=training_model, fold=n, split="val", max_subset_number=args.test_max_subset_number,
                patch_size=test_patch_size)

            all_complex_names.extend(complex_names)
            all_decoy_names.extend(decoy_names)
            all_scores.extend(scores)
            all_targets.extend(targets)
            all_acc.extend(test_acc)
            print('{:d} fold test_acc: {:.4f}'.format(n, test_acc))

            nfold_results.append({'fold': n, 'acc': test_acc})
            del test_loss, complex_names, decoy_names, scores, targets
        elif args.mode == 'regression':
            test_loss, test_pcc, test_spcc, test_ef,test_sr, complex_names, decoy_names, scores, targets, categories= test_regression(
                args=args, model=training_model, fold=n, split="val", max_subset_number=args.test_max_subset_number,
                patch_size=test_patch_size)

            all_complex_names.extend(complex_names)
            all_decoy_names.extend(decoy_names)
            all_scores.extend(scores)
            all_targets.extend(categories)
            all_labels.extend(targets)

            print('{:d} fold test_loss: {:.4f}, test_pcc: {:.4f}, test_spcc:{:.4f}, test_ef: {:.4f}, test_sr: {:.4f}'.format(n, test_loss, test_pcc, test_spcc, test_ef, test_sr))

            nfold_results.append({'fold': n,'test_loss': test_loss, 'pcc': test_pcc, 'spcc': test_spcc, 'ef': test_ef, 'sr': test_sr})
            del test_pcc, test_ef
        del training_model




    if args.mode == 'classification':
        # # save to file at save_folder
        df_predict = pd.DataFrame({'complex_name': all_complex_names, 'decoy_no': all_decoy_names, 'label': all_targets,
                                   'prediction': all_scores})
        df_predict.to_csv(os.path.join(model_save_folder, r'data_prediction.csv'), index=False)

        del df_predict, all_complex_names, all_decoy_names, all_targets, all_scores

        df_nfold_results = pd.DataFrame(nfold_results)
        mean_acc = np.mean(df_nfold_results['acc'])
        # mean_rpc = np.mean(df_nfold_results['rpc'])
        # mean_ef = np.mean(df_nfold_results['ef'])
        print('{:d} fold test mean_acc: {:.4f}'.format(n + 1, mean_acc))

        nfold_results.append({'mean_auc': mean_acc})

        with open(os.path.join(model_save_folder,
                               '{:d}fold_hop{:d}_{:s}_results.json'.format(n + 1, args.hop, args.mode)),
                  'w') as json_fp:
            json.dump(nfold_results, json_fp)

        del nfold_results, df_nfold_results, args

    else:
        # # save to file at save_folder
        df_predict = pd.DataFrame({'complex_name': all_complex_names, 'decoy_no': all_decoy_names, 'label': all_targets,
                                   'prediction': all_scores, 'regression_label': all_labels})
        df_predict.to_csv(os.path.join(model_save_folder, r'data_prediction.csv'), index=False)

        del df_predict, all_complex_names, all_decoy_names, all_targets, all_scores, all_labels

        df_nfold_results = pd.DataFrame(nfold_results)
        mean_pcc = np.mean(df_nfold_results['pcc'])
        #mean_mae = np.mean(df_nfold_results['mae'])
        mean_ef = np.mean(df_nfold_results['ef'])

        print('{:d} fold test mean_pcc: {:.4f},  test mean_ef: {:.4f}'.format(n + 1, mean_pcc, mean_ef))
        nfold_results.append({'mean_pcc': mean_pcc, 'mean_ef': mean_ef})

        # save
        with open(os.path.join(model_save_folder,
                               '{:d}fold_hop{:d}_{:s}_results.json'.format(n + 1, args.hop, args.mode)),
                  'w') as json_fp:
            json.dump(nfold_results, json_fp)
        del nfold_results, df_nfold_results, args

    max_count = 1000
    calculate_average_EF_from_meta_learners(model_save_folder, max_count)
    calculate_successful_rate(model_save_folder, max_count)
    calculate_average_hit_count(model_save_folder, max_count)