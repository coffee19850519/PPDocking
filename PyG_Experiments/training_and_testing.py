import torch
import torch.nn.functional as F
from torch.nn import Linear,Conv1d, MaxPool1d
from torch_geometric.nn import SAGEConv, global_mean_pool, GATConv, SAGPooling,GraphConv, TopKPooling, GCNConv ,JumpingKnowledge, global_max_pool, GlobalAttention,global_sort_pool,BatchNorm
from torch_geometric.data import DataLoader
from torch_geometric.utils import subgraph, to_undirected
import os
import numpy as np
from sklearn import metrics
from PyG_Experiments.util_function import EarlyStopping, compute_PCC, compute_EF, save_node_significance
from PyG_Experiments.GRATConv import GRATConv, GlobalAttentionPooling
from PyG_Experiments.CGConv_net import CGConvSortPool, CGConvSortPool_by_occurrence, CGConv_by_occurrence_global_pool
from PyG_Experiments.sort_pool_by_occurrence import GATSortPool_by_occurrence,GATSortPool_by_occurrence_global_pool
from PyG_Experiments.CGConv_net import CGConv_by_occurrence_global_pool

# from PyG_Experiments.import_data import PPDocking
import PyG_Experiments.import_data
from tqdm import tqdm


def train_classification(args, model, fold, model_save_folder,train_patch_size, val_patch_size):#,  mean, std):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.savecheckpoints_mode,
                                                           factor=0.8, patience=3,
                                                           min_lr=0.0001)

    earlystop = EarlyStopping(patience=args.earlystop_patience, verbose=True)#, delta= 0.00001)
    summary = []
    # val_dataset = PyG_Experiments.import_data.PPDocking(root=root,
    #                                                     subset=subset_number,
    #                                                     fold=fold,
    #                                                     patch=0,
    #                                                     mode='classification',
    #                                                     split='val')
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    for epoch in range(1, args.epochs):
        total_loss = 0
        sample_length = 0
        model.train()

        for train_patch_number in range(train_patch_size):
            print('doing training: {:d} epoch {:d} patch'.format(epoch, train_patch_number))
            train_dataset = PyG_Experiments.import_data.PPDocking(root=args.root,
                                      subset=args.train_subset_number,
                                      fold=fold,
                                      patch=train_patch_number,
                                      mode=args.mode,
                                      split='train',
                                      load_folder = args.train_classification_patch_path
                                      )

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            sample_length += len(train_loader.dataset)

            for i, data in enumerate(train_loader):

                data = data.to(args.device)
                optimizer.zero_grad()
                out, node_significance, node_belong_decoy_num, interface_pos, decoy_name = model(data, data.num_graphs)
                # save_node_significance(node_significance, node_belong_decoy_num, interface_pos, decoy_name,
                #                        r'/home/fei/Desktop/gif/node.csv')

                del node_significance, node_belong_decoy_num, interface_pos, decoy_name

                if data.y.size()[0] != out.size()[0]:
                    print("Target size is not the same with input size")

                # loss = sigmoid_focal_loss(out, data.y, reduction= 'mean')
                loss = F.binary_cross_entropy_with_logits(out, data.y, pos_weight=torch.tensor(args.pos_weights, device=args.device))
                # loss = F.binary_cross_entropy_with_logits(out, data.y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * data.num_graphs

            del train_loader, train_dataset


        average_loss = total_loss / sample_length

        # val_loss, val_auc, val_rpc, val_ef, all_complex_names, _, all_scores, all_targets = val_classification(model, device, val_patch_size, val_max_subset_number, root, args, fold, args.pos_weights)
        val_loss, val_auc, val_rpc, val_ef, all_complex_names, _, all_scores, all_targets = test_classification(args, model, fold, split = "val", max_subset_number = args.val_max_subset_number,patch_size = val_patch_size)

        if args.savecheckpoints_monitor == "val_loss":
            monitor = val_loss
        elif args.savecheckpoints_monitor == "val_rpc":
            monitor = val_rpc

        earlystop(monitor, os.path.join(model_save_folder, '{:d}_fold_classification_model.pt'.format(fold)),  model, mode=args.savecheckpoints_mode, args= args)
        scheduler.step(monitor)

        lr = scheduler.optimizer.param_groups[0]['lr']

        # if best_val_error is None or val_error <= best_val_error:
        #     test_error = test(val_loader)
        #     best_val_error = val_error

        print('Epoch: {:03d}, LR: {:4f}, Loss: {:.4f}, val_loss: {:.4f}, val_auc: {:.4f}, val_rpc: {:.4f}, val_ef1: {:4f}'.format(epoch, lr, average_loss, val_loss, val_auc, val_rpc, val_ef1))
        summary.append({'fold':fold, 'epoch':epoch, 'lr':lr, 'train_loss':loss.item(), 'val_loss': val_loss, 'val_auc': val_auc, 'val_rpc': val_rpc, 'val_ef1':  val_ef1})
        del val_loss, val_auc, val_rpc, val_ef1 , all_complex_names, all_scores, all_targets
        if earlystop.early_stop:
            print("Early stopping")
            break
    # del val_dataset, val_loader
    return model, summary


@torch.no_grad()
def val_classification(model, device, test_loader, pos_weights):
    all_scores = []
    all_targets = []
    all_val_loss = []
    all_complex_names = []
    all_decoy_names = []
    model.eval()
    for data in tqdm(test_loader):
        all_complex_names.extend(np.array(data.complex_name).reshape(-1).tolist())
        all_decoy_names.extend(np.array(data.decoy_name).reshape(-1).tolist())
        data = data.to(device)
        with torch.no_grad():
            out, node_significance, node_belong_decoy_num, interface_pos, decoy_name = model(data, data.num_graphs)

            del node_significance, node_belong_decoy_num, interface_pos, decoy_name
            #val_loss = sigmoid_focal_loss(out, data.y, reduction='mean')
            val_loss = F.binary_cross_entropy_with_logits(out, data.y, pos_weight= torch.tensor(pos_weights,device = device))
            out = torch.sigmoid(out)
        all_scores.append(out)
        all_targets.append(data.y)
        all_val_loss.append(val_loss)
    all_scores = torch.cat(all_scores).cpu().numpy().reshape(-1).tolist()
    all_targets = torch.cat(all_targets).cpu().numpy().reshape(-1).tolist()
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(all_targets, all_scores, pos_label=1)
    auc_precision_recall = metrics.auc(recall, precision)
    val_loss = torch.mean(torch.stack(all_val_loss)).item()
    ef1 = compute_EF(all_scores, all_targets, all_complex_names)
    del all_val_loss
    return val_loss, auc, auc_precision_recall, ef1, all_complex_names, all_decoy_names, all_scores, all_targets


def test_classification(args, model, fold, split,max_subset_number,patch_size):
    all_scores = []
    all_targets = []
    #all_val_loss = []
    all_complex_names = []
    all_decoy_names = []
    sample_length = 0

    if split =='val':
        load_folder = args.val_classification_patch_path
    elif split=='test':
        load_folder = args.test_classification_patch_path

    model.eval()
    for test_patch_number in range(patch_size):
        print('doing {:s}: {:d} fold {:d} patch'.format(split, fold, test_patch_number))
        test_dataset = PyG_Experiments.import_data.PPDocking(root=args.root,
                                                              subset=max_subset_number,
                                                              fold=fold,
                                                              patch=test_patch_number,
                                                              mode='classification',
                                                              split=split,
                                                              load_folder =load_folder
                                                             )

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        sample_length += len(test_loader.dataset)

        for data in tqdm(test_loader):
            all_complex_names.extend(np.array(data.complex_name).reshape(-1).tolist())
            all_decoy_names.extend(np.array(data.decoy_name).reshape(-1).tolist())
            data = data.to(args.device)
            with torch.no_grad():
                out, node_significance, node_belong_decoy_num, interface_pos, decoy_name = model(data, data.num_graphs)

                del node_significance, node_belong_decoy_num, interface_pos, decoy_name
                #val_loss = sigmoid_focal_loss(out, data.y, reduction='mean')
                #val_loss = F.binary_cross_entropy_with_logits(out, data.y, pos_weight=torch.tensor([args.pos_weights], device=args.device))
                out = torch.sigmoid(out)
            all_scores.append(out)
            all_targets.append(data.y)
            #all_val_loss.append(val_loss)

        del test_loader, test_dataset


    all_scores = torch.cat(all_scores).cpu().numpy().reshape(-1).tolist()
    all_targets = torch.cat(all_targets).cpu().numpy().reshape(-1).tolist()
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(all_targets, all_scores, pos_label=1)
    auc_precision_recall = metrics.auc(recall, precision)
    #val_loss = torch.mean(torch.stack(all_val_loss)).item()
    ef = compute_EF(all_scores, all_targets, all_complex_names)
    #del all_val_loss
    return auc, auc_precision_recall, ef, all_complex_names, all_decoy_names, all_scores, all_targets


def train_regression(train_dataset, val_dataset, args, fold,  mean, std):
    #device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True,
                            num_workers=6)


    model = GATSortPool(args).to(device)
    #model = GATRegression(train_dataset.num_node_features, 64).to(device)
    #model = TopK(train_dataset, 5, 50, ratio= 0.7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.8, patience=5,
                                                           min_lr=0.0001)

    earlystop = EarlyStopping(patience=30, verbose=True, delta= 0.00001)
    summary = []
    for epoch in range(1, 301):
        total_loss = 0
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out, node_significance, node_belong_decoy_num, interface_pos,decoy_name = model(data, data.num_graphs)

            del node_significance, node_belong_decoy_num, interface_pos,decoy_name
            # loss = F.binary_cross_entropy_with_logits(out, data.y)
            loss = F.smooth_l1_loss(out, data.dockq)
            # loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        average_loss = total_loss / len(train_loader.dataset)

        val_loss, val_mae, val_pcc, val_ef1, _,_,_,_ = test_regression(model, device, val_loader, mean, std)
        earlystop(val_pcc, '{:d}fold_regression_model.pt'.format(fold), model, mode= 'max' ,args=args)
        scheduler.step(val_pcc)

        lr = scheduler.optimizer.param_groups[0]['lr']

        # if best_val_error is None or val_error <= best_val_error:
        #     test_error = test(val_loader)
        #     best_val_error = val_error

        print('Epoch: {:03d}, LR: {:4f}, Loss: {:.4f}, val_loss: {:.4f}, val_mae: {:.4f}, val_pcc: {:.4f}, val_ef1: {:.4f}'.format(epoch, lr,average_loss,
                             val_loss, val_mae, val_pcc, val_ef1))
        summary.append(
            {'fold': fold, 'epoch': epoch, 'lr': lr, 'train_loss': loss, 'val_loss': val_loss, 'val_mae': val_mae,
             'val_pcc': val_pcc, 'val_ef1': val_ef1})
        if earlystop.early_stop:
            print("Early stopping")
            break

    return model, summary

@torch.no_grad()
def test_regression(model, device, test_loader, mean = 0, std = 1):
    all_labels = []
    all_scores = []
    all_val_loss = []
    all_complex_names = []
    all_decoy_names = []
    all_categories = []
    error = 0

    model.eval()
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out, node_significance, node_belong_decoy_num, interface_pos, decoy_name = model(data, data.num_graphs)
            del node_significance, node_belong_decoy_num, interface_pos, decoy_name
            val_loss = F.smooth_l1_loss(out, data.dockq)
        # vout += out - torch.mean(out)
        # vy += data.y - torch.mean(data.y)
        error += (out * std - data.dockq * std).abs().sum().item()  # MAE
        all_scores.extend((out * std + mean).cpu().numpy().reshape((-1)).tolist())
        all_labels.extend(data.dockq.cpu().numpy().reshape((-1)).tolist())
        all_val_loss.append(val_loss)
        all_complex_names.extend(np.array(data.complex_name).reshape((-1)).tolist())
        all_categories.extend(data.category.cpu().numpy().reshape((-1)).tolist())
        #all_decoy_names.extend(np.array(data.decoy_name).reshape((-1)).tolist())
    val_loss = torch.mean(torch.stack(all_val_loss)).item()
    #torch.sum(vout * vy) / (torch.sqrt(torch.sum(vout ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    average_pcc = compute_PCC(all_scores,all_labels, all_complex_names, mean, std)
    average_ef = compute_EF(all_scores, all_categories.cpu(), all_complex_names)
    del all_labels
    return val_loss, error / len(test_loader.dataset), average_pcc, average_ef1, all_scores, all_complex_names, all_decoy_names, all_categories
# # Normalize targets to mean = 0 and std = 1.
# mean = dataset.data.y.mean(dim=0, keepdim=True)
# std = dataset.data.y.std(dim=0, keepdim=True)
# dataset.data.y = (dataset.data.y - mean) / std
# mean, std = mean[:, 0].item(), std[:, 0].item()

# Split datasets.