import torch
import torch.nn.functional as F
from torch.nn import Linear,Conv1d, MaxPool1d
from torch_geometric.nn import SAGEConv, global_sort_pool,global_mean_pool, GATConv, SAGPooling,GraphConv, TopKPooling, GCNConv ,JumpingKnowledge, global_max_pool, GlobalAttention
from torch_geometric.data import DataLoader
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from util_function import EarlyStopping, compute_PCC, compute_EF1

class GATRegression(torch.nn.Module):
    def __init__(self, num_node_features,hidden):
        super(GATRegression, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden, heads=4, concat=True, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(
            hidden * 4, hidden, heads=1, concat=True, dropout=0.6)

        self.lin1 = Linear(hidden, int(hidden / 2))
        self.lin2 = Linear(int(hidden / 2), 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.elu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin2(x)

class TopK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, dense_hidden, ratio=0.8):
        super(TopK, self).__init__()
        self.num_layers = num_layers
        self.hidden = hidden
        self.conv1 = GraphConv(dataset.num_features, hidden, aggr='mean')
        self.attn1 = GlobalAttention(Linear(hidden, 1))
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.attns = torch.nn.ModuleList()
        self.convs.extend([
            #GCNConv(dataset.num_features, hidden, improved= False)
            GraphConv(hidden, hidden, aggr='mean')
            for i in range(num_layers - 1)
        ])

        self.attns.extend([
            GlobalAttention(Linear(hidden, 1))
            for i in range(num_layers - 1)
        ])

        # self.pools.extend(
        #     [TopKPooling(hidden, ratio) for i in range((num_layers) // 2)])

        self.pools.extend(
            [ SAGPooling(hidden, ratio)
            #TopKPooling(hidden, ratio)
             for i in range(num_layers - 1)])

        self.jump = JumpingKnowledge(mode='cat')

        #self.conv1d = Conv1d(,100,)
        # self.lin1 = Linear(num_layers * hidden, hidden)
        # self.lin2 = Linear(hidden, 1)

        self.latent_dim = hidden * (num_layers - 1) + 1
        self.conv1d_1 = Conv1d(1, 16, self.latent_dim, self.latent_dim)
        #self.maxpool1d = MaxPool1d(2, 2)
        self.conv1d_2 = Conv1d(16, 32, 5, 1)

        self.dense_dim = (int((self.latent_dim - 2) / 2 + 1) - 5 + 1) * 32

        self.lin0 = Linear(self.dense_dim, dense_hidden)
        self.lin1 = Linear(dense_hidden, int(dense_hidden * 0.5))
        self.lin2 = Linear(int(dense_hidden * 0.5), 1)


    def reset_poolparameters(self):
        self.conv1.reset_parameters()
        self.attn1.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for attn in self.attns:
            attn.reset_parameters()
        self.conv1d_1.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, graph_num):
        x, edge_index, batch, interface_tag = data.x, data.edge_index, data.batch, data.pos_batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [self.attn1(x, interface_tag)]

        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            # if i % 2 == 0 and i < len(self.convs) - 1:
            #     pool = self.pools[i // 2]
            #     x, edge_index, _, batch, _, _ = pool(x, edge_index,
            #                                          batch=batch)
            #if i < len(self.convs) - 1:

            pool = self.pools[i]
            attn = self.attns[i]
            x, edge_index, _, interface_tag, _, _ = pool(x, edge_index, batch= interface_tag)
            xs += [attn(x, interface_tag)]

        x = self.jump(xs)
        # reshape sort-pooled embedding
        x = x.view((-1, 1, self.num_layers * self.hidden ))
        x = F.relu(self.conv1d_1(x))
        #x = self.maxpool1d(x)
        x = F.relu(self.conv1d_2(x))

        # reshape 1d conv embedding to dense
        x = x.view((graph_num, -1))
        x = F.relu(self.lin0(x))
        # x=  F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin2(x)


    def __repr__(self):
        return self.__class__.__name__

class SortPoolRegression(torch.nn.Module):
    def __init__(self, arguments):
        super(SortPoolRegression, self).__init__()
        num_node_features, num_layers, hidden, dense_hidden, k, interface_only, head = arguments.num_node_features, \
               arguments.num_layers, arguments.hidden, arguments.dense_hidden, arguments.k, arguments.interface_only, arguments.head
        self.k = k
        self.interface_only = interface_only
        self.conv1 = GATConv(num_node_features, hidden, heads=head, concat=True,dropout= 0.2)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GATConv(hidden, hidden, heads=head, concat=False,dropout= 0.2))
        self.convn = GATConv(head * hidden, hidden = 1, dropout= 0.7)
        self.jump = JumpingKnowledge(mode='cat')

        self.latent_dim = head * hidden * (num_layers - 1) + 1
        self.conv1d_1 = Conv1d(1, 16, self.latent_dim, self.latent_dim)
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv1d_2 = Conv1d(16, 32, 5, 1)

        self.dense_dim = (int((k - 2) / 2 + 1) - 5 + 1) * 32
        self.lin0 = Linear(self.dense_dim, dense_hidden)
        self.lin1 = Linear(dense_hidden, int(dense_hidden * 0.5))
        self.lin2 = Linear(int(dense_hidden * 0.5), 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.convn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d_1.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, graph_num):
        x, edge_index, batch, interface_pos = data.x, data.edge_index, data.batch, data.interface_pos
        # batch = torch.mul(batch, interface_tags)

        x = F.relu(self.conv1(x, edge_index))
        xs = [x]

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]

        x = F.relu(self.convn(x, edge_index))
        xs += [x]
        x = self.jump(xs)
        if self.interface_only:
            x = global_sort_pool(x[interface_pos], batch[interface_pos], self.k)
        else:
            x = global_sort_pool(x, batch, self.k)

        # reshape sort-pooled embedding
        x = x.view((-1, 1, self.k * self.latent_dim))
        x = F.relu(self.conv1d_1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv1d_2(x))

        # reshape 1d conv embedding to dense
        x = x.view((graph_num, -1))
        x = F.relu(self.lin0(x))
        # x=  F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin2(x)

    def __repr__(self):
        return self.__class__.__name__

def sigmoid_focal_loss(
    inputs,
    targets,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss



class GATSortPool(torch.nn.Module):
    def __init__(self, arguments):
        super(GATSortPool, self).__init__()
        num_node_features, num_layers, hidden, dense_hidden, k, interface_only, head = arguments.num_node_features, \
             arguments.num_layers, arguments.hidden, arguments.dense_hidden, arguments.k, arguments.interface_only, arguments.head


        self.k = k
        self.interface_only = interface_only
        #self.conv1 = GCNConv(num_node_features, hidden)
        self.conv1 = GATConv(num_node_features, hidden, heads= head, concat= True, dropout= 0.3)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            #self.convs.append(GCNConv(hidden, hidden))
            self.convs.append(GATConv(head * hidden, hidden, heads= head, concat= True, dropout= 0.3))
        #self.convn = GCNConv(hidden, 1)
        self.convn = GATConv(head * hidden, 1, dropout= 0.3)

        self.jump = JumpingKnowledge(mode='cat')


        #self.latent_dim = hidden * (num_layers - 1) + 1
        self.latent_dim = head * hidden * (num_layers - 1) + 1
        self.conv1d_1 = Conv1d(1, 16, self.latent_dim, self.latent_dim)
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv1d_2 = Conv1d(16, 32, 5, 1)


        self.dense_dim = (int((k - 2) / 2 + 1) - 5 + 1) * 32
        self.lin0 = Linear(self.dense_dim, dense_hidden)
        self.lin1 = Linear(dense_hidden, int(dense_hidden * 0.5) )
        self.lin2 = Linear(int(dense_hidden * 0.5), 1)

    def reset_parameters(self):
        self.parameters = self.conv1.reset_parameters()
        self.convn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d_1.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, graph_num):
        x, edge_index, batch, interface_pos = data.x, data.edge_index, data.batch, data.interface_pos
        #print(x.shape)

        x = torch.tanh(self.conv1(x, edge_index))
        xs = [x]

        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            xs += [x]

        x = torch.tanh(self.convn(x, edge_index))
        xs += [x]
        x = self.jump(xs)
        if self.interface_only:
            x = global_sort_pool(x[interface_pos], batch[interface_pos], self.k)
        else:
            x = global_sort_pool(x, batch, self.k)

        #reshape sort-pooled embedding
        x = x.view((-1, 1, self.k * self.latent_dim))
        x = torch.relu(self.conv1d_1(x))
        x = self.maxpool1d(x)
        x = torch.relu(self.conv1d_2(x))

        #reshape 1d conv embedding to dense
        x = x.view((graph_num , -1))
        x = torch.relu(self.lin0(x))
        #x=  F.dropout(x, p=0.5, training=self.training)
        x = torch.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        del  xs
        return self.lin2(x)

    def __repr__(self):
        return self.__class__.__name__

def train_classification(train_dataset, val_dataset, args, fold, model_save_folder):#,  mean, std):
    #device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)


    #model = SortPoolRegression(train_dataset.num_node_features,2,32).to(device)
    #model = GATRegression(train_dataset.num_node_features, 64).to(device)
    #model = TopK(train_dataset, 3, 32, dense_hidden= 256 , ratio= 0.8).to(device)


    model = GATSortPool(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.8, patience=3,
                                                           min_lr=0.00001)
    earlystop = EarlyStopping(patience= 30, verbose= True)#, delta= 0.00001)
    summary = []
    for epoch in range(1, 201):
        total_loss = 0
        model.train()
        for i, data in enumerate(train_loader):

            data = data.to(device)
            optimizer.zero_grad()
            out = model(data, data.num_graphs)
            #loss = sigmoid_focal_loss(out, data.y, reduction= 'mean')
            loss = F.binary_cross_entropy_with_logits(out, data.y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        average_loss = total_loss / len(train_loader.dataset)

        val_loss, val_auc, val_rpc, val_ef1, all_complex_names,_, all_scores, all_targets = test_classification(model, device, val_loader)
        earlystop(val_loss, os.path.join(model_save_folder, '{:d}fold_classification_model.pt'.format(fold)),  model, mode='min', args= args)
        scheduler.step(val_loss)

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

    return model, summary


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
                                                           factor=0.8, patience=3,
                                                           min_lr=0.00001)

    earlystop = EarlyStopping(patience=30, verbose=True, delta= 0.00001)
    summary = []
    for epoch in range(1, 301):
        total_loss = 0
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data, data.num_graphs)
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
            out = model(data, data.num_graphs)
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
    average_ef1 = compute_EF1(all_scores, all_categories, all_complex_names)
    del all_labels
    return val_loss, error / len(test_loader.dataset), average_pcc, average_ef1, all_scores, all_complex_names, all_decoy_names, all_categories

@torch.no_grad()
def test_classification(model, device, test_loader):
    all_scores = []
    all_targets = []
    all_val_loss = []
    all_complex_names = []
    all_decoy_names = []
    model.eval()
    for data in test_loader:
        all_complex_names.extend(np.array(data.complex_name).reshape(-1).tolist())
        all_decoy_names.extend(np.array(data.decoy_name).reshape(-1).tolist())
        data = data.to(device)
        with torch.no_grad():
            out = model(data, data.num_graphs)
            #val_loss = sigmoid_focal_loss(out, data.y, reduction='mean')
            val_loss = F.binary_cross_entropy_with_logits(out, data.y, reduction='mean')
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
    ef1 = compute_EF1(all_scores, all_targets, all_complex_names)
    del  all_val_loss
    return val_loss, auc, auc_precision_recall, ef1, all_complex_names, all_decoy_names, all_scores, all_targets




# # Normalize targets to mean = 0 and std = 1.
# mean = dataset.data.y.mean(dim=0, keepdim=True)
# std = dataset.data.y.std(dim=0, keepdim=True)
# dataset.data.y = (dataset.data.y - mean) / std
# mean, std = mean[:, 0].item(), std[:, 0].item()

# Split datasets.



