import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Conv2d,Conv1d, Linear as Lin, ReLU, Tanh, BatchNorm1d as BN, LayerNorm, Embedding,CosineEmbeddingLoss
from torch_geometric.datasets import  MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool,global_add_pool, global_mean_pool, radius, fps, DynamicEdgeConv,PointConv, GINEConv
from torch import Tensor
from torch_sparse import SparseTensor, set_diag
from PyG_Experiments.GCNE_net import gcn_norm
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj
from torch_geometric.nn.inits import reset, glorot, zeros

from point_cloud.data_convertor import PPDocking_point_cloud
from tqdm import tqdm
from torch_cluster import knn
import numpy as np

class DockPointNet_Module(MessagePassing):
    r""" Combining the pointNet++ set abstraction and Dynamic Edge Convolution Layers
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            `:obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.* defined by :class:`torch.nn.Sequential`.
        k (int): Max number of nearest neighbors.
        aggr (string): The aggregation operator to use (:obj:`"add"`,
            :obj:`"mean"`, :obj:`"max"`). (default: :obj:`"max"`)
        num_workers (int): Number of workers to use for k-NN computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn: Callable, k: int, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super(DockPointNet_Module, self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        self.nn = nn
        self.max_k = k
        self.num_workers = num_workers
        self.reset_parameters()
        # todo: try within and without sampling
        # fps sampling ratio parameter
        # self.ratio = 0.5
        # radius for ball sampling
        self.r = 0.1

    def reset_parameters(self):
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            pos: Tensor = None,
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:

        # if isinstance(x, Tensor):
        #     x: PairTensor = (x, x)
        # assert x[0].dim() == 2, \
        #     'Static graphs not supported in `DockPointNet_Module`.'
        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)
        assert pos[0].dim() == 2, \
            'Static graphs not supported in `DockPointNet_Module`.'

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])
        #
        # if pos is not None:
        #     idx = fps(pos, b, ratio=self.ratio)
        #     row, col = radius(pos, pos[idx], self.r, b, b[idx], max_num_neighbors=self.max_k)
        #     edge_index = torch.stack([col, row], dim=0)
        # else:

        # x[0] and x[1] are copied tensors of same input points
        # b[0] and b[1] denote the batches of every point in x[0] and x[1]
        # idx = fps(x, b, ratio=self.ratio)
        # row, col = radius(x, x[idx], self.r, b, b[idx], max_num_neighbors=self.max_k)

        edge_index = knn(pos[0], pos[1], self.max_k, b[0], b[1])

        # row, col = radius(x[0], x[1], self.r, b[0], b[1], max_num_neighbors=self.max_k)
        # edge_index = torch.stack([col, row], dim=0)


        del pos
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:

        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.max_k)

# returns sequence of linear layers, relu activation, and batch normalization of specified depth and width
def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), LayerNorm(channels[i]), Dropout(0.5))
        for i in range(1, len(channels))
    ])

class SAModule(torch.nn.Module):
    def __init__(self,  feature_nn,  global_nn , r, aggr):
        super(SAModule, self).__init__()
        # self.ratio = ratio
        # self.r = r
        self.k = r
        self.conv = PointConv(feature_nn, global_nn, aggr= aggr)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, pos, batch):
        # idx = fps(pos, batch, ratio=self.ratio)
        # row, col = radius(pos, pos[idx], self.k, batch, batch[idx],
        #                   max_num_neighbors=20)
        row, col = radius(pos, pos, self.k, batch, batch,
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)

        # edge_index = knn(pos, pos, self.k, batch, batch)
        x = self.conv(x, (pos, pos), edge_index)
        # x = self.conv(x, (pos, pos[idx]), edge_index)
        # pos, batch = pos[idx], batch[idx]
        return x, pos, batch


# class GlobalSAModule(torch.nn.Module):
#     def __init__(self, nn):
#         super(GlobalSAModule, self).__init__()
#         self.nn = nn
#
#     def forward(self, x, pos, batch):
#         x = self.nn(torch.cat([x, pos], dim=1))
#         x = global_max_pool(x, batch)
#         pos = pos.new_zeros((x.size(0), 3))
#         batch = torch.arange(x.size(0), device=batch.device)
#         return x, pos, batch

# class PointConv(MessagePassing):
#
#     def __init__(self, local_nn: Optional[Callable] = None,
#                  # pos_nn: Optional[Callable] = None,
#                  global_nn: Optional[Callable] = None,
#                  add_self_loops: bool = True, **kwargs):
#         super(PointConv, self).__init__(aggr='max', **kwargs)
#
#         self.local_nn = local_nn
#         self.global_nn = global_nn
#         # self.pos_nn = pos_nn
#         self.add_self_loops = add_self_loops
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         reset(self.local_nn)
#         # reset(self.pos_nn)
#         reset(self.global_nn)
#
#     def forward(self, x: Union[OptTensor, PairOptTensor],
#                 pos: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
#         """"""
#         if not isinstance(x, tuple):
#             x: PairOptTensor = (x, None)
#
#         if isinstance(pos, Tensor):
#             pos: PairTensor = (pos, pos)
#
#         if self.add_self_loops:
#             if isinstance(edge_index, Tensor):
#                 edge_index, _ = remove_self_loops(edge_index)
#                 edge_index, _ = add_self_loops(edge_index,
#                                                num_nodes=pos[1].size(0))
#             elif isinstance(edge_index, SparseTensor):
#                 edge_index = set_diag(edge_index)
#
#         # propagate_type: (x: PairOptTensor, pos: PairTensor)
#         out = self.propagate(edge_index, x=x, pos=pos, size=None)
#
#         if self.global_nn is not None:
#             out = self.global_nn(out)
#
#         return out
#
#     def message(self, x_i: Optional[Tensor], x_j: Optional[Tensor], pos_i: Tensor,
#                 pos_j: Tensor) -> Tensor:
#         msg = pos_j - pos_i
#         #original PointConv
#         if x_j is not None:
#             msg = torch.cat([x_j, msg], dim=1)
#         if self.local_nn is not None:
#             msg = self.local_nn(msg)
#         # modified PointConv
#         # if x_j is not None:
#         #     msg = self.pos_nn(msg)
#         # if self.local_nn is not None:
#         #     x_j = self.local_nn(torch.cat([x_j, (x_j - x_i)], dim =1))
#         # if msg.size(1) == x_j.size(1):
#         #     msg = msg + x_j
#         # else:
#         #     msg = torch.cat([x_j, msg], dim=1)
#         return msg
#
#     def __repr__(self):
#         return '{}(local_nn={}, global_nn={})'.format(self.__class__.__name__,
#                                                       self.local_nn,
#                                                       self.global_nn)

# re-indexes the residue_num globally
# can identify which res_num belongs to which protein/batch based on the batch variable
# TODO: if data is not sequential per protein: per protein sort, save sorted indeces, run below, then revert sort
def global_index_residue(residue_list_batch, decoy_list, device):
    '''

    Args:
        residue_list_batch (list): has batch number of residue indexes for atoms in a protein's interface
        decoy_list (list): has batch number of protein source pdb filenames

    Returns:
        residue_index_batch (Tensor): index all atoms by protein.residue globally
        residue_to_decoy_batch (Tensor) : index of which residues belong to which proteins
    '''

    # this will store source_protein_file.local_residue for every atom
    full_residue_number_list = []
    # this will store the protein index for each residue: ex.) different residues from same protein have same value
    residue_to_decoy_batch = []
    # compose the full residue number for all atoms in batch
    # this operation could be done with batch object alternitively
    for decoy_idx, decoy in enumerate(decoy_list):
        for residue_number in residue_list_batch[decoy_idx]:
            full_residue_number_list.append(decoy + '.' + residue_number)
        # set the same index for all residues in a same decoy
        residue_to_decoy_batch.extend(np.repeat(decoy_idx, len(set(residue_list_batch[decoy_idx]))).tolist())

    # initialize the variables needed in loop
    current_res_index = 0
    current_residue_number = full_residue_number_list[0]
    residue_index_batch = []

    # index atoms by protein.residue globally
    for full_residue_number in full_residue_number_list:
        if current_residue_number != full_residue_number:
            current_res_index += 1
            current_residue_number = full_residue_number

        residue_index_batch.append(current_res_index)

    del full_residue_number_list, current_res_index, current_residue_number

    return torch.tensor(residue_index_batch, dtype= torch.long, device= device),\
           torch.tensor(residue_to_decoy_batch, dtype= torch.long, device= device)


def organize_by_correspondence(x, residue_list, decoy_list,  correspondence):
    # notice: here the x, residue_list and correspondence are all on batch mode
    '''

    Args:
        x: residue embeddings aggregated from atom-level embeddings
        residue_list: has batch number of residue labels for atoms in a protein's interface
        decoy_list: has batch number of protein source pdb filenames
        correspondence: ground_truth residue connection labels for all interfaces

    Returns:
        x_s (Tensor): x source points in interfaces
        x_t (Tensor): x target points in interfaces
        all_ground_truth_list (Tensor): labels for edges if they exist in groundtruth or not
    '''

    all_source_residue_index = []
    all_target_residue_index = []
    all_ground_truth_list = []
    full_residue_list = []
    source_residue_index = []
    target_residue_index = []
    # loop through all protein interfaces by index
    # store all interface connections in all_source_residue_index and all_target_residue_index w/ protein_source_filename.local_residue_label labels
    # store labels for if connections exist in true protein pairing or not
    for sample_idx in range(len(decoy_list)):
        # get current complex edge labels
        current_correspondence = correspondence[sample_idx]

        # get list of residues for the current interface by protein_source_filename.local_residue_label
        # cannot use set() to remove duplicated ones, may change their orders
        for residue_num in {}.fromkeys(residue_list[sample_idx]).keys():
            full_residue_list.append(decoy_list[sample_idx] + '.' + residue_num)


        # loop through all connections in current interface by index
        for res_idx in range(len(current_correspondence)):
            # save edge information from correspondence with protein complex source label
            all_source_residue_index.append(decoy_list[sample_idx] + '.' +  current_correspondence[res_idx][0])
            all_target_residue_index.append(decoy_list[sample_idx] + '.' + current_correspondence[res_idx][1])

            # if source and target are bonded, then append 1 as label else append -1
            if current_correspondence[res_idx][2]:
                all_ground_truth_list.append(1)
            else:
                all_ground_truth_list.append(-1)

    # convert protein_source_filename.local_residue_label into index labels from list of residues across all interfaces
    for idx in range(len(all_source_residue_index)):
        source_residue_index.append(full_residue_list.index(all_source_residue_index[idx]))
        target_residue_index.append(full_residue_list.index(all_target_residue_index[idx]))

    # get x values for given source and target indices
    x_s = x.index_select(dim=0, index=torch.tensor(source_residue_index, dtype= torch.long, device= x.device))
    x_t = x.index_select(dim=0, index=torch.tensor(target_residue_index, dtype=torch.long, device=x.device))

    del all_source_residue_index, all_target_residue_index, full_residue_list, source_residue_index, target_residue_index

    return x_s, x_t, torch.tensor(all_ground_truth_list, dtype= torch.float,device= x.device)




class GCNEConv(MessagePassing):

    '''
    create a message based on edge and source point attributes and then scale that message by node degree
    '''

    def __init__(self, in_channels: int, out_channels: int, edge_channels: int,
                 improved: bool = False,  **kwargs):

        super(GCNEConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.edge_channels = edge_channels
        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        glorot(self.weight)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr:Tensor,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(edge_index, Tensor):
            # input: edge_weight is None
            # output: edge_weight scaling factor for message aggregation of each edge, based on node degree
            edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight, x.size(self.node_dim),
            self.improved, True, dtype=x.dtype)

        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                self.improved, True, dtype=x.dtype)

        # add self-loop edge attr
        self_loop_attr = torch.zeros(x.size(0), self.edge_channels)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, edge_attr= edge_attr,
                             size=None)

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: Tensor, edge_attr: Tensor) -> Tensor:

        x_j = torch.matmul(torch.cat([x_j, edge_attr], dim=1) , self.weight)
        return edge_weight.view(-1, 1) * x_j + self.bias



    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class DockPointNet(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, out_channels):
        super().__init__()

        # point-cloud layers for atomic non-bond embeddings
        # self.tnet = MLP([3, 64, 64, 3])
        self.conv1 = SAModule(MLP([3 + in_channels, 64]), None, 0.2, aggr= 'add')

        # GINE conv layers for atomic bond embeddings
        self.gcn1 = GCNEConv(in_channels + edge_dim, 64, edge_dim)

        # embed atomic represenations
        self.atom_embed_layers = MLP([64, 64])

        # point_cloud layers for residue level embeddings
        self.res_embed_layers = MLP([64, 64])
        self.res_conv1 = SAModule(MLP([3 + 64, 128]), MLP([128, 128]), 0.2, aggr= 'add')
        # self.res_conv2 = SAModule(MLP([3 + 64, 128]), 8.5, 0.8)

        self.lin1 = Lin(256, 128)
        # self.lin2 = Lin(128, 64)
        # self.lin2 = Lin(128, out_channels)
        self.loss = CosineEmbeddingLoss(reduction='none')
        self.reset_parameters()
        # self.mlp = Seq(
        #     MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
        #     Lin(256, out_channels))

    def reset_parameters(self):
        # self.embed_layer.reset_parameters()
        # self.embed_norm.reset_parameters()
        self.gcn1.reset_parameters()
        self.conv1.reset_parameters()
        self.res_conv1.reset_parameters()
        self.lin1.reset_parameters()
        reset(self.res_embed_layers)
        reset(self.atom_embed_layers)

    def forward(self, data):
        '''

        Args:
            data:   x: has atom attributes
                    pos: has atom position
                    edge_index: has atomic bond connectivity
                    edge_attr: has atomic bond edge info
                    residue_index_list: has batch number of residue labels for atoms in a protein's interface
                    decoy_name/decoy_list: has batch number of protein source pdb filenames
                    complex_name/target_list: has batch number of protein names
                    correspondence: ground_truth residue connection labels for protein-protein interactions


        Returns:

        '''

        x, residue_list, pos, decoy_list,target_list, correspondence,edge_index, edge_attr, batch = \
            data.x.float(), data.residue_index_list, data.pos.float(), data.decoy_name, data.complex_name, data.correspondence,\
            data.edge_index, data.edge_attr, data.batch

        # get atomic point embeddings w/ PointNet++Conv
        x1 = self.conv1(x, pos, batch)
        x1, pos1, batch1 = x1

        # get atomic point embeddings based on bond edges
        x2 = self.gcn1(x, edge_index, edge_attr)

        # embed summed points to 64d vector space w/ MLP to get final atomic representations
        atom_x = self.atom_embed_layers(x1 + x2)

        # get index all atoms by protein.residue globally and index of which residues belong to which proteins
        residue_index_batch, residue_to_decoy_batch = global_index_residue(residue_list, decoy_list, x.device)

        # aggregate point features by residue
        res_x = global_add_pool(atom_x , residue_index_batch)

        # get positions for each residue
        res_pos = global_mean_pool(pos, residue_index_batch)

        # embed residue represenations in 64d vector space
        res_x = self.res_embed_layers(res_x)

        # get residue point embeddings w/ PointNet++Conv
        x3 = self.res_conv1(res_x, res_pos, residue_to_decoy_batch)
        x3, pos3, batch3 = x3

        # organize source and target points for connections between residues in interfaces
        x_s, x_t, y = organize_by_correspondence(x3, residue_list, decoy_list, correspondence)

        # get pos_labels by 1 if pos 0 if neg, neg_labels by 1 if neg, 0 if pos
        pos_labels = torch.where(y != 1, torch.zeros_like(y), y)
        neg_labels = 1 - pos_labels
        # calculate class weights according to ground truth distribution
        # pos_weights = torch.sum(1 - y, dim=0) / torch.sum(y, dim=0)
        pos_weights = torch.sum(neg_labels, dim=0) / torch.sum(pos_labels, dim=0)
        # loss = F.binary_cross_entropy_with_logits(out, y.view((-1, 1)), pos_weights)
        unweighted_loss = self.loss(x_s, x_t, y)
        #pos_labels will locate the positive correspondence pairs

        loss = torch.mean((pos_labels * pos_weights + neg_labels * 1) * unweighted_loss)
        del pos_weights
        if self.training:
            return loss, y
        else:
            return loss, torch.cosine_similarity(x_s, x_t) , y




def train(loader, model, optimizer):
    model.train()
    total_loss = 0
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # loss = F.binary_cross_entropy_with_logits(out, data.y, pos_weight=torch.tensor([4], device=data.y.device))
        out = F.log_softmax(out, dim=1)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        # break
    return total_loss / len(loader.dataset)


def test(loader, model):
    model.eval()

    correct = 0
    for data in tqdm(loader):
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)




if __name__ == '__main__':
    path = r'/run/media/fei/Windows/hanye/point_cloud_format/'
    pdb_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/3_pdb/'
    data_split_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/11_dataset_splitting/resplit_result/'
    label_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/'
    interface_folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/14_interface_info/'

#     transforms =  T.Compose([
#     T.RandomTranslate(0.01),
#     T.RandomRotate(15, axis=0),
#     T.RandomRotate(15, axis=1),
#     T.RandomRotate(15, axis=2)
# ])

    train_dataset =  MNISTSuperpixels(r'/run/media/fei/Windows/hanye/MNIST/')
    val_dataset =  MNISTSuperpixels(r'/run/media/fei/Windows/hanye/MNIST/', train= False)
    # train_dataset = PPDocking_point_cloud(path, 0, 'train', pdb_folder, data_split_folder, label_folder,
    #                                   interface_folder, transform= transforms)
    #
    # val_dataset = PPDocking_point_cloud(path, 0, 'val', pdb_folder, data_split_folder, label_folder,
    #                                   interface_folder)

    train_loader = DataLoader(
        train_dataset, batch_size=200, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=200, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DockPointNet((train_dataset.num_features+2),  train_dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(1, 201):
        loss = train(train_loader, model, optimizer)
        val_acc = test(val_loader, model)
        print('Epoch {:03d}, Loss: {:.4f} Val_acc: {:.4f}'.format(epoch, loss, val_acc))
        scheduler.step()
        # break
