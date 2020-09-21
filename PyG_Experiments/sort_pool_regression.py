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

class GATRegression(torch.nn.Module):
    def __init__(self, num_node_features,hidden):
        super(GATRegression, self).__init__()
        self.conv1 = \
            GATConv(num_node_features, hidden, heads=4, concat=True, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = \
            GATConv(
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
        x, edge_index, batch, interface_pos, decoy_name = data.x, data.edge_index, data.batch, data.interface_pos, data.decoy_name
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
        del xs
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


class GATGlobalAtttention(torch.nn.Module):
    def __init__(self, arguments):
        super(GATGlobalAtttention, self).__init__()
        num_node_features, num_layers, hidden, dense_hidden, k, interface_only, head  = arguments.num_node_features, \
             arguments.num_layers, arguments.hidden, arguments.dense_hidden, arguments.k, arguments.interface_only, arguments.head
        self.k = k
        self.interface_only = interface_only



        self.conv1 = GCNConv(num_node_features, hidden)
        # self.conv1 = GATConv(num_node_features, hidden, heads= head, concat= True, dropout= 0.3)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
            # self.convs.append(GATConv(head * hidden, hidden, heads= head, concat= True, dropout= 0.3))
        self.convn = GCNConv(hidden, 1)
        # self.convn = GATConv(head * hidden, 1, dropout= 0.3)
        self.jump = JumpingKnowledge(mode='cat')

        self.latent_dim = hidden * (num_layers - 1) + 1
        # self.latent_dim = head * hidden * (num_layers - 1) + 1
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
        x, edge_index, batch, interface_pos, decoy_name = data.x, data.edge_index, data.batch, data.interface_pos, data.decoy_name
        # batch = torch.mul(batch, interface_tags)

        x = torch.tanh(self.conv1(x, edge_index))
        xs = [x]

        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            xs += [x]

        x = torch.tanh(self.convn(x, edge_index))

        # save the x and corresponding decoy_name
        node_significance = x.cpu().detach().numpy()
        node_belong_decoy_num = batch.cpu().detach().numpy()

        xs += [x]
        x = self.jump(xs)

        if self.interface_only:
            # print('before pooling /n')
            # print(x[interface_pos])
            subgraph()

            x = global_sort_pool(x[interface_pos], batch[interface_pos], self.k)
            # print('after pooling /n')
            # print(x.view((-1, self.k, 33)))
        else:
            x = global_sort_pool(x, batch, self.k)

        # reshape sort-pooled embedding
        x = x.view((-1, 1, self.k * self.latent_dim))
        x = torch.relu(self.conv1d_1(x))
        x = self.maxpool1d(x)
        x = torch.relu(self.conv1d_2(x))

        # reshape 1d conv embedding to dense
        x = x.view((graph_num, -1))
        x = torch.relu(self.lin0(x))
        # x=  F.dropout(x, p=0.5, training=self.training)
        x = torch.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin2(x), node_significance, node_belong_decoy_num, interface_pos.cpu().detach().numpy(), decoy_name

    def __repr__(self):
        return self.__class__.__name__


class GRATNet(torch.nn.Module):
    def __init__(self, arguments):
        super(GRATNet, self).__init__()
        num_node_features, num_layers, hidden, dense_hidden, k, interface_only, head  = arguments.num_node_features, \
             arguments.num_layers, arguments.hidden, arguments.dense_hidden, arguments.k, arguments.interface_only, arguments.head
        self.k = k
        self.interface_only = interface_only
        self.conv1 = GRATConv(num_node_features, hidden, head, concat= True)
        self.glob_att1 = GlobalAttention(Linear(hidden, 1))

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GRATConv(hidden, hidden, head= head, concat= True))
            self.convs.append(GlobalAttention(Linear(hidden, 1)))
            #self.convs.append(GCNConv(hidden, hidden))
            #self.convs.append(GATConv(head * hidden, hidden, heads= head, concat= True, dropout= 0.3))
        #self.convn = GCNConv(hidden, 1)
        #self.convn = GATConv(head * hidden, 1, dropout= 0.3)
        self.convn = GRATConv(hidden, 1, head= 1, concat= False)
        self.glob_attn = GlobalAttention(Linear(hidden, 1))

        self.jump = JumpingKnowledge(mode='cat')

        #self.latent_dim = hidden * (num_layers - 1) + 1
        # self.latent_dim = head * hidden * (num_layers - 1) + 1
        # self.conv1d_1 = Conv1d(1, 16, self.latent_dim, self.latent_dim)
        # self.maxpool1d = MaxPool1d(2, 2)
        # self.conv1d_2 = Conv1d(16, 32, 5, 1)


        # self.dense_dim = (int((k - 2) / 2 + 1) - 5 + 1) * 32
        #self.lin0 = Linear(self.dense_dim, dense_hidden)
        dense_hidden = hidden * num_layers
        self.lin1 = Linear(dense_hidden, int(dense_hidden * 0.5) )
        self.lin2 = Linear(int(dense_hidden * 0.5), 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.convn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        # self.conv1d_1.reset_parameters()
        # self.conv1d_2.reset_parameters()
        # self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, graph_num):
        x, edge_index, batch, interface_pos, decoy_name = data.x, data.edge_index, data.batch, data.interface_pos, data.decoy_name
        #batch = torch.mul(batch, interface_tags)

        #conver graph to undirected
        edge_index = to_undirected(edge_index, len(x))

        #x = torch.tanh(self.conv1(x, edge_index,edge_attr= edge_attr))
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.glob_att1(x, batch)
        xs = [global_mean_pool(x[interface_pos], batch[interface_pos])]

        for i, conv in enumerate(self.convs):
            if i%2 == 0:
                x = torch.tanh(conv(x, edge_index))
            else:
                conv()
                xs += [x]

        x = torch.tanh(self.convn(x, edge_index))


        #save the x and corresponding decoy_name
        node_significance = x.cpu().detach().numpy()
        node_belong_decoy_num = batch.cpu().detach().numpy()

        xs += [x]
        x = self.jump(xs)

        if self.interface_only:
            #print('before pooling /n')
            #print(x[interface_pos])
            x = global_sort_pool(x[interface_pos], batch[interface_pos], self.k)
            #print('after pooling /n')
            #print(x.view((-1, self.k, 33)))
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
        return self.lin2(x), node_significance, node_belong_decoy_num, interface_pos.cpu().detach().numpy(),decoy_name

    def __repr__(self):
        return self.__class__.__name__


class GATSortPool(torch.nn.Module):
    def __init__(self, arguments):
        super(GATSortPool, self).__init__()
        num_node_features, num_layers, hidden, dense_hidden, k, interface_only, head, concat  = arguments.num_node_features, \
             arguments.num_layers, arguments.hidden, arguments.dense_hidden, arguments.k, arguments.interface_only, arguments.head, arguments.concat
        self.k = k
        self.interface_only = interface_only
        #self.conv1 = GRATConv(num_node_features, hidden, head, concat= True, dropout= 0.2)
        self.conv1 = GCNConv(num_node_features, hidden)



        #self.conv1 = GATConv(num_node_features, hidden, heads= head, concat= concat, dropout= 0.3)
        #self.norm1 = BatchNorm(hidden)
        self.convs = torch.nn.ModuleList()
        #self.norms = torch.nn.ModuleList()
        for i in range(num_layers-2):
            # self.convs.append(GRATConv(hidden, hidden, head= head, concat= True, dropout= 0.2))
            self.convs.append(GCNConv(hidden, hidden))
            # if not concat:
            #     self.convs.append(GATConv(hidden, hidden, heads= head, concat= concat, dropout= 0.3))
            #     #self.norms.append(BatchNorm(hidden))
            # else:
            #     self.convs.append(GATConv(head*hidden, head*hidden, heads=head, concat=concat, dropout=0.3))
                #self.norms.append(BatchNorm(head*hidden))
        self.convn = GCNConv(hidden, 1)
        # if not concat:
        #     self.convn = GATConv(hidden, 1, dropout= 0.3)
        #     #self.normn = BatchNorm(1)
        # else:
        #     self.convn = GATConv(head * hidden, 1, dropout=0.3)
        #     #self.normn = BatchNorm(1)

        # self.convn = GRATConv(head * hidden, 1, head= 1, concat= False, dropout= 0.2)
        #self.convn = GlobalAttentionPooling(Linear(hidden * (num_layers - 1), 1))


        self.jump = JumpingKnowledge(mode='cat')


        if not concat:
            self.latent_dim = hidden * (num_layers - 1) + 1
        else:
            self.latent_dim = head * hidden * (num_layers - 1) + 1
        self.conv1d_1 = Conv1d(1, 18, self.latent_dim, self.latent_dim)
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv1d_2 = Conv1d(18, 36, 5, 1)
        self.dense_dim = (int((k - 2) / 2 + 1) - 5 + 1) * 36

        self.lin0 = Linear(self.dense_dim, dense_hidden)


        #dense_hidden = hidden
        self.lin1 = Linear(dense_hidden, int(dense_hidden * 0.5) )

        # dense_nns = []
        # dense_nns.append(Linear(dense_hidden, int(dense_hidden * 0.5) ))
        # dense_nns.append(torch.nn.LeakyReLU())
        # dense_nns.appe.nd(torch.nn.Dropout(p = 0.2))
        # gate_nns = []
        # gate_nns.append(Linear(dense_hidden, 1))
        # gate_nns.append(torch.nn.LeakyReLU())
        # gate_nns.append(torch.nn.Dropout(p = 0.2))
        #
        # self.att = GlobalAttention(torch.nn.Sequential(*gate_nns), torch.nn.Sequential(*dense_nns))

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
        self.att.reset_parameters()

    def forward(self, data, graph_num):
        x, edge_index, edge_attr, batch, interface_pos, decoy_name = data.x, data.edge_index, data.edge_attr, data.batch, data.interface_pos, data.decoy_name
        #batch = torch.mul(batch, interface_tags)
        #print(data.complex_name)
        #conver graph to undirected
        #if data.
        # edge_index = to_undirected(edge_index, len(x))

        #x = torch.tanh(self.conv1(x, edge_index,edge_attr= edge_attr))
        #x = torch.tanh(self.norm1(self.conv1(x, edge_index)))        ###graph convolution 1
        x = torch.tanh(self.conv1(x, edge_index))
        xs = [x]

        # for conv,norm in zip(self.convs, self.norms):
        #     x = torch.tanh(norm(conv(x, edge_index)))
        #     xs += [x]

        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            xs += [x]

        # get contact subgraph and do last neighboring aggregation
        #contact_edge_index, _ = subgraph(interface_pos,edge_index)

        #x = torch.tanh(self.convn(x, contact_edge_index))
        #x = self.convn(self.jump(xs), batch)

        #save the x and corresponding decoy_name
        # node_significance = x.cpu().detach().numpy()
        # node_belong_decoy_num = batch.cpu().detach().numpy()
        node_significance = None
        node_belong_decoy_num = None
        x = torch.tanh(self.convn(x, edge_index))
        #x = torch.tanh(self.normn(self.convn(x, edge_index)))
        xs += [x]
        x = self.jump(xs)    #contact

        #get contact subgraph
        #subgraph()

        x = global_sort_pool(x[interface_pos], batch[interface_pos], self.k)    #sort acordding the last value
        #print(x.size())
        #x = self.att(x[interface_pos], batch[interface_pos])

        #reshape sort-pooled embedding
        x = x.view((-1, 1, self.k * self.latent_dim))
        x = torch.relu(self.conv1d_1(x))  #[1600,16,100]
        x = self.maxpool1d(x)
        x = torch.relu(self.conv1d_2(x))

        #reshape 1d conv embedding to dense
        x = x.view((graph_num , -1))
        #x = x.view((-1, self.dense_dim))
        x = torch.relu(self.lin0(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = torch.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        del xs, data #,contact_edge_index
        return self.lin2(x), node_significance, node_belong_decoy_num, interface_pos.cpu().detach().numpy(),decoy_name

    def __repr__(self):
        return self.__class__.__name__






