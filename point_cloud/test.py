import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Conv2d,Conv1d, Linear as Lin, ReLU, Tanh, BatchNorm1d as BN, LayerNorm, Embedding,CosineEmbeddingLoss
from torch_geometric.datasets import  MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool,global_add_pool, global_mean_pool, radius, fps, DynamicEdgeConv,PointConv
from torch_geometric.nn.conv.ppf_conv import get_angle
from torch import Tensor
from torch_sparse import SparseTensor, set_diag
from PyG_Experiments.GCNE_net import gcn_norm
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj
from torch_geometric.nn.inits import reset, glorot, zeros
from point_cloud.focal_loss import BinaryFocalLoss
from point_cloud.data_convertor import PPDocking_point_cloud
from collections import Counter
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
        Seq(Lin(channels[i - 1], channels[i]), Tanh(), Dropout(0.5)) # LayerNorm(channels[i]),
        for i in range(1, len(channels))
    ])

class PPFConv(MessagePassing):
    def __init__(self, local_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 add_self_loops: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'max')
        super(PPFConv, self).__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)

    def normalized_point_pair_features(self, pos_i: Tensor, pos_j: Tensor, normal_i: Tensor,
                            normal_j: Tensor, radius : float) -> Tensor:
        pseudo = pos_j - pos_i
        angle1 = get_angle(normal_i, pseudo)
        angle2 = get_angle(normal_j, pseudo)
        angle3 = get_angle(normal_i, normal_j)
        return torch.stack([
            pseudo.norm(p=2, dim=1) / radius,
            torch.sin(angle1),
            torch.cos(angle1),
            torch.sin(angle2),
            torch.cos(angle2),
            torch.sin(angle3),
            torch.cos(angle3)
        ], dim=1)

    def forward(self, x: Union[OptTensor, PairOptTensor],
                pos: Union[Tensor, PairTensor],
                normal: Union[Tensor, PairTensor],
                edge_index: Adj,
                radius: float) -> Tensor:  # yapf: disable
        """"""
        if not isinstance(x, tuple):
            x: PairOptTensor = (x, None)

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if isinstance(normal, Tensor):
            normal: PairTensor = (normal, normal)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                               num_nodes=pos[1].size(0))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairOptTensor, pos: PairTensor, normal: PairTensor)  # noqa
        out = self.propagate(edge_index, x=x, pos=pos, normal=normal,
                             size=None, radius= radius)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out

    def message(self, x_j: OptTensor, pos_i: Tensor, pos_j: Tensor,
                normal_i: Tensor, normal_j: Tensor, radius_i : float) -> Tensor:
        msg = self.normalized_point_pair_features(pos_i, pos_j, normal_i, normal_j, radius_i)
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def __repr__(self):
        return '{}(local_nn={}, global_nn={})'.format(self.__class__.__name__,
                                                      self.local_nn,
                                                      self.global_nn)



class SAModule(torch.nn.Module):
    def __init__(self,  feature_nn,  global_nn , r):
        super(SAModule, self).__init__()
        # self.ratio = ratio
        # self.r = r
        self.k = r
        self.conv = PPFConv(feature_nn, global_nn)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, pos, normal, batch):
        # idx = fps(pos, batch, ratio=self.ratio)
        # row, col = radius(pos, pos[idx], self.k, batch, batch[idx],
        #                   max_num_neighbors=20)
        row, col = radius(pos, pos, self.k, batch, batch,
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)

        # edge_index = knn(pos, pos, self.k, batch, batch)
        x = self.conv(x, (pos, pos), normal, edge_index, float(self.k))
        # x = self.conv(x, (pos, pos[idx]), edge_index)
        # pos, batch = pos[idx], batch[idx]
        return x, pos, batch



class Res_Module(torch.nn.Module):
    def __init__(self,  feature_nn,  global_nn , k):
        super(Res_Module, self).__init__()
        self.k = k
        self.conv = PointConv(feature_nn, global_nn)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, pos, batch):
        edge_index = knn(pos, pos, self.k, batch, batch)

        x = self.conv(x, (pos, pos), edge_index)

        return x, pos, batch


# re-indexes the residue_num globally
# can identify which res_num belongs to which protein/batch based on the batch variable
# TODO: if data is not sequential per protein: per protein sort, save sorted indeces, run below, then revert sort
def global_index_residue(residue_list_batch, decoy_list, device):

    full_residue_number_list = [i for item in residue_list_batch for i in item]
    residue_index_batch= []
    residue_to_decoy_batch = []
    residue_name_list = []

    #compose the full residue number for all atoms in batch
    for decoy_idx, decoy in enumerate(decoy_list):
        # for residue_number in residue_list_batch[decoy_idx]:
        #     full_residue_number_list.append(decoy + '.' + residue_number)
        residue_name_list.extend({}.fromkeys(residue_list_batch[decoy_idx]).keys())
        #set the same index for all residues in a same decoy
        residue_to_decoy_batch.extend(np.repeat(decoy_idx, len(set(residue_list_batch[decoy_idx]))).tolist())

    #initialize the variables needed in statment
    current_res_index = 0
    current_residue_number = full_residue_number_list[0]

    for full_residue_number in full_residue_number_list:
        if current_residue_number != full_residue_number:
            current_res_index += 1
            current_residue_number = full_residue_number

        residue_index_batch.append(current_res_index)

    del current_res_index, current_residue_number

    return torch.tensor(residue_index_batch, dtype= torch.long, device= device),\
           torch.tensor(residue_to_decoy_batch, dtype= torch.long, device= device), \
           residue_name_list


def organize_by_correspondence(x, residue_list, decoy_list, chain, correspondence):
    # notice: here the x, residue_list and correspondence are all on batch mode
    # The x should be the residue embeddings aggregated from atom-level embeddings

    all_ground_truth_list = []
    # decoy_chain_list = [i for item in chain for i in item]
    source_residue_index = []
    target_residue_index = []
    # residue_to_chain_batch = []

    for sample_idx in range(len(decoy_list)):

        current_correspondence = correspondence[sample_idx]
        #get uniform residue number and keep their orders for each decoy
        # cannot use set() to remove duplicated ones, may change their orders
        # for residue_num in {}.fromkeys(residue_list[sample_idx]).keys():
        #     full_residue_list.append(decoy_list[sample_idx] + '.' + residue_num)
        #     decoy_chain_list.append(decoy_list[sample_idx] + '.' + residue_num.split('.')[0])

        for res_idx in range(len(current_correspondence)):
            source_residue_index.append(residue_list.index(current_correspondence[res_idx][0]))
            target_residue_index.append(residue_list.index(current_correspondence[res_idx][1]))
            if current_correspondence[res_idx][2]:
                all_ground_truth_list.append(1)
            else:
                all_ground_truth_list.append(0)

    # according to full_residue_number_list, extract decoy-chain level batch index
    # initialize the variables needed in statement
    # current_chain_index = 0
    # current_chain = decoy_chain_list[0]

    # for chain_id in decoy_chain_list:
    #     if current_chain != chain_id:
    #         current_chain_index += 1
    #         current_chain = chain_id
    #
    #     residue_to_chain_batch.append(current_chain_index)


    x_s = x.index_select(dim=0, index=torch.tensor(source_residue_index, dtype= torch.long, device= x.device))
    x_t = x.index_select(dim=0, index=torch.tensor(target_residue_index, dtype=torch.long, device=x.device))

    del source_residue_index, target_residue_index#, decoy_chain_list

    return x_s, x_t, torch.tensor(all_ground_truth_list, dtype= torch.float,device= x.device)




class GCNEConv(MessagePassing):

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
    def __init__(self, in_channels, edge_dim):
        super().__init__()
        # point-cloud layers for atomic non-bond embeddings
        # self.tnet = MLP([3, 64, 64, 3])

        # self.conv1 = SAModule(MLP([4 + in_channels, 64]), MLP([64, 128]), 3)
        self.conv2 = SAModule(MLP([7 + in_channels, 7 + in_channels, 128]), None, 5)
        self.conv3 = SAModule(MLP([7 + in_channels, 7 + in_channels, 128]), None, 8.5)
        self.conv4 = SAModule(MLP([7 + in_channels, 7 + in_channels, 128]), None, 10)

        #GINE conv layers for atomic bond embeddings
        # self.gcn1 = GCNEConv(in_channels + edge_dim, 64, edge_dim)
        # self.gcn_norm = LayerNorm(64)

        self.atom_embed_layers = MLP([384, 256])

        # point_cloud layers for residue level embeddings
        # self.residue_feature_embed_layers =  MLP([26,256])
        self.res_embed_layers = MLP([256, 512])
        # self.residue_concat_embed_layers = MLP([512,1024])
        # self.res_conv1 = Res_Module(MLP([3 + 256, 512]), MLP([512, 512]), 20)
        # self.res_conv2 = SAModule(MLP([3 + 64, 128]), 8.5, 0.8)


        self.lin1 = Lin(512, 1)
        # self.lin2 = Lin(128, 1)
        # self.lin2 = Lin(128, out_channels)
        # self.loss = CosineEmbeddingLoss()

        self.loss = BinaryFocalLoss(alpha=4)

        self.reset_parameters()
        # self.mlp = Seq(
        #     MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
        #     Lin(256, out_channels))

    def reset_parameters(self):
        # self.embed_layer.reset_parameters()
        # self.embed_norm.reset_parameters()
        # self.gcn1.reset_parameters()
        # self.gcn_norm.reset_parameters()
        # self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        # self.res_conv1.reset_parameters()
        self.lin1.reset_parameters()
        reset(self.res_embed_layers)
        reset(self.atom_embed_layers)
        # reset(self.residue_concat_embed_layers)
        # reset(self.residue_feature_embed_layers)

    def forward(self, data):
        x, residue_list, pos, normal, decoy_list,target_list, chain, correspondence,edge_index, edge_attr, residue_features, batch = \
            data.x.float(), data.residue_index_list, data.pos.float(), data.norm.float(),data.decoy_name, data.complex_name, data.chain, data.correspondence,\
            data.edge_index, data.edge_attr, data.residue_features.float() , data.batch

        # x0 = self.tnet(pos)
        # read_out_x = []
        # x1 = self.conv1(torch.cat([x, pos], dim=-1), batch)
        # x =  self.embed_norm(torch.relu(self.embed_layer(torch.argmax(x,dim =1))))
        # x = self.conv1(x, pos, batch)
        # x = self.conv2(x,  pos, batch)
        # x3 = self.conv3(*x2)
        # x1 = self.conv1(x, pos, normal, batch)
        x2 = self.conv2(x, pos, normal, batch)
        x3 = self.conv3(x, pos, normal, batch)
        x4 = self.conv4(x, pos, normal, batch)
        # x1, pos1, batch1 = x1
        x2, pos2, batch2 = x2
        x3, pos3, batch3 = x3
        x4, pos4, batch4 = x4
        # x2 = F.dropout(self.gcn_norm(torch.relu(self.gcn1(x, edge_index, edge_attr))),training= self.training)


        atom_x = self.atom_embed_layers(torch.cat((x2, x3, x4), dim= 1))

        atom_to_residue_index_batch, residue_to_decoy_batch, residue_list_batch = global_index_residue(residue_list, decoy_list, x.device)


        res_x = global_max_pool(atom_x , atom_to_residue_index_batch)
        # res_pos = global_mean_pool(pos, atom_to_residue_index_batch)
        # res_normals =
        # residue_feature_embeddings = self.residue_feature_embed_layers(residue_features)
        res_x = self.res_embed_layers(res_x)
        # res_final_embeddings = self.residue_concat_embed_layers(res_x,)
        # x3 = self.res_conv1(res_x, res_pos, residue_to_decoy_batch)
        # x4 = self.res_conv2(res_x, res_pos, residue_to_decoy_batch)
        # x3, pos3, batch3 = x3
        # x4, pos4, batch2 = x4


        x_s, x_t, y = organize_by_correspondence(res_x, residue_list_batch, decoy_list, chain, correspondence)

        out = self.lin1(x_s - x_t)
        # x = F.dropout(x, p=0.5, training=self.training)
        # out = self.lin2(x)
        # x_s, x_t, y = organize_by_correspondence(self.lin2(x), residue_list, decoy_list, correspondence)
        # x = torch.relu(self.lin2(x))
        # x = F.dropout(x, p=0.2, training=self.training)

        # pos_labels = torch.where(y != 1, torch.zeros_like(y), y)
        # neg_labels = 1 - pos_labels

        # calculate class weights according to ground truth distribution
        # pos_weights = torch.sum(1 - y, dim=0) / torch.sum(y, dim=0)
        # pos_weights = torch.sum(neg_labels, dim=0) / torch.sum(pos_labels, dim=0)
        # loss = F.binary_cross_entropy_with_logits(out, y.view((-1, 1)), pos_weights)
        # unweighted_loss = self.loss(x_s, x_t, y)
        #pos_labels will locate the positive correspondence pairs

        # loss = torch.mean((pos_labels * pos_weights + neg_labels * 1) * unweighted_loss)
        # loss = self.loss(x_s, x_t, y)
        loss = self.loss(out.squeeze(), y)
        # del pos_weights

        if self.training:
            return loss, y
        else:
            out = torch.sigmoid(out)
            return loss, out, y, fnat(out, 0.4, correspondence)



def fnat(out, threshold, correspondence):
    fnat = []
    current_start_index = 0
    for current_correspondence in correspondence:
        current_coupling_num = len(current_correspondence)
        current_fnat = torch.sum(out[current_start_index: current_start_index + current_coupling_num].greater_equal(threshold).float())
        fnat.append(float(current_fnat))
        current_start_index += current_coupling_num
    return fnat


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