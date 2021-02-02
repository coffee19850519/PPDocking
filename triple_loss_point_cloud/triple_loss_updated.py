import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Conv2d, Conv1d, Linear as Lin, ReLU, LeakyReLU, BatchNorm1d as BN, \
    LayerNorm, Embedding, CosineEmbeddingLoss, TripletMarginLoss
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool, radius, fps, DynamicEdgeConv, \
    PointConv
from torch_geometric.nn.conv.ppf_conv import get_angle
from torch import Tensor
from torch_sparse import SparseTensor, set_diag
from PyG_Experiments.GCNE_net import gcn_norm
from torch_geometric.utils import remove_self_loops, add_self_loops, to_dense_batch
from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj
from torch_geometric.nn.inits import reset, glorot, zeros
import torch.autograd as grad
from point_cloud.focal_loss import BinaryFocalLoss

from tqdm import tqdm
from torch_cluster import knn
import numpy as np

torch.autograd.set_detect_anomaly(True)


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
                                       normal_j: Tensor, radius: float) -> Tensor:
        pseudo = pos_j - pos_i
        angle1 = get_angle(normal_i+1e-8, pseudo+1e-8)
        angle2 = get_angle(normal_j+1e-8, pseudo+1e-8)
        angle3 = get_angle(normal_i+1e-8, normal_j+1e-8)
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
                local_edge_attr,
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
        # out = self.propagate(edge_index, x=x, pos=pos, normal=normal, edge_attr=local_edge_attr, size=None)
        out = self.propagate(edge_index, x=x, pos=pos, normal=normal, edge_attr=local_edge_attr, size=None,
                             radius=radius)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out

    def message(self, x_j: OptTensor, pos_i: Tensor, pos_j: Tensor,
                normal_i: Tensor, normal_j: Tensor, edge_attr: Tensor, radius_i: float) -> Tensor:
        msg = self.normalized_point_pair_features(pos_i, pos_j, normal_i, normal_j, radius_i)
        if edge_attr is not None:
            msg = torch.cat([msg, edge_attr], dim=1)
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def __repr__(self):
        return '{}(local_nn={}, global_nn={})'.format(self.__class__.__name__,
                                                      self.local_nn,
                                                      self.global_nn)


# returns sequence of linear layers, relu activation, and batch normalization of specified depth and width
def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), LayerNorm(channels[i]), Dropout(0.5))
        for i in range(1, len(channels))
    ])


class SAModule(torch.nn.Module):
    def __init__(self, feature_nn, global_nn, r, conv_type):
        super(SAModule, self).__init__()
        # self.ratio = ratio
        # self.r = r
        self.k = r
        if conv_type == 'ppf':
            self.conv = PPFConv(feature_nn, global_nn)
        if conv_type == 'point_conv':
            self.conv = PointConv(feature_nn, global_nn)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, pos, batch, edge_attr, edge_index, normal=None):
        # idx = fps(pos, batch, ratio=self.ratio)
        # row, col = radius(pos, pos[idx], self.k, batch, batch[idx],
        #                   max_num_neighbors=20)
        row, col = radius(pos, pos, self.k, batch, batch, max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        '''
        targets, sources = radius(pos, pos, self.k, batch, batch, max_num_neighbors=64)
        local_edge_index = torch.stack([sources, targets], dim=0)
        transposed_local_edge_index = torch.transpose(local_edge_index, 0, 1)
        transposed_edge_index = torch.transpose(edge_index, 0, 1)

        # TODO could probably still be optimized more
        # this also changes device to cpu can change to .float() if still want gpu
        transposed_edge_index = transposed_edge_index.type(torch.FloatTensor)
        transposed_local_edge_index = transposed_local_edge_index.type(torch.FloatTensor)

        # radius runs faster on cpu
        matched_local_edge, matched_global_edge = radius(transposed_edge_index, transposed_local_edge_index, 1e-8,
                                                         max_num_neighbors=1)
        matched_local_edge = matched_local_edge.to(edge_attr.device)
        matched_global_edge = matched_global_edge.to(edge_attr.device)

        edge_attr_indices = torch.Tensor([edge_attr.shape[0] - 1]).to(edge_attr.device)
        edge_attr_indices = edge_attr_indices.repeat(transposed_local_edge_index.shape[0]).type(torch.int64)
        edge_attr_indices[matched_local_edge] = matched_global_edge
        local_edge_attr = torch.index_select(edge_attr, 0, edge_attr_indices)

        # edge_index = knn(pos, pos, self.k, batch, batch)
        # x = self.conv(x, (pos, pos), normal, local_edge_index, local_edge_attr)


        if normal is not None:
            x = self.conv(x, (pos, pos), normal, local_edge_index, local_edge_attr, self.k)
        else:
            x = self.conv(x, (pos, pos), local_edge_index, local_edge_attr, self.k)
        '''
        if normal is not None:
            x = self.conv(x, (pos, pos), normal, edge_index, edge_attr, self.k)
        else:
            self.conv(x, (pos, pos), None, edge_index, edge_attr, self.k)
        return x, pos, batch


class RESModule(torch.nn.Module):
    def __init__(self, feature_nn, global_nn, k):
        super(RESModule, self).__init__()

        self.k = k
        self.conv = PointConv(feature_nn, global_nn)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, pos, batch):
        # row, col = radius(pos, pos, self.k, batch, batch,
        #                   max_num_neighbors=64)
        # edge_index = torch.stack([col, row], dim=0)

        edge_index = knn(pos, pos, self.k, batch, batch)

        x = self.conv(x, (pos, pos), edge_index)

        return x, pos, batch


# re-indexes the residue_num globally
# can identify which res_num belongs to which protein/batch based on the batch variable
def global_index_residue(residue_list_batch_A, residue_list_batch_B, decoy_list, device):
    residue_index_batch_native = []
    residue_index_batch_decoy = []
    residue_in_native = []
    residue_in_decoy = []

    full_residue_number_list_native = []
    full_residue_number_list_decoy = []

    # compose the full residue number for all atoms in batch
    for decoy_idx, decoy in enumerate(decoy_list):
        for residue_number in residue_list_batch_A[decoy_idx]:
            full_residue_number_list_native.append(residue_number)
        for residue_number in residue_list_batch_B[decoy_idx]:
            full_residue_number_list_decoy.append(residue_number)

        # set the same index for all residues in a same decoy
        residue_in_native.extend(np.repeat(decoy_idx, len(set(residue_list_batch_A[decoy_idx]))).tolist())
        residue_in_decoy.extend(np.repeat(decoy_idx, len(set(residue_list_batch_B[decoy_idx]))).tolist())

    # initialize the variables needed in statment
    current_res_index = 0
    current_residue_number = full_residue_number_list_native[0]
    for full_residue_number in full_residue_number_list_native:
        if current_residue_number != full_residue_number:
            current_res_index += 1
            current_residue_number = full_residue_number

        residue_index_batch_native.append(current_res_index)

    current_res_index = 0
    current_residue_number = full_residue_number_list_decoy[0]
    for full_residue_number in full_residue_number_list_decoy:
        if current_residue_number != full_residue_number:
            current_res_index += 1
            current_residue_number = full_residue_number

        residue_index_batch_decoy.append(current_res_index)

    # del full_residue_number_list_A,full_residue_number_list_B, current_res_index, current_residue_number

    return torch.tensor(residue_index_batch_native, dtype=torch.long, device=device), \
           torch.tensor(residue_in_native, dtype=torch.long, device=device), \
           torch.tensor(residue_index_batch_decoy, dtype=torch.long, device=device), \
           torch.tensor(residue_in_decoy, dtype=torch.long, device=device)


def organize_by_correspondence(x_A, x_B, residue_list_A, residue_list_B, decoy_list, correspondence):
    # notice: here the x, residue_list and correspondence are all on batch mode
    # The x should be the residue embeddings aggregated from atom-level embeddings

    index_pad_native = 0
    index_pad_complex = 0
    anchor_indices_A = []
    anchor_indices_B = []
    pos_indices_A = []
    pos_indices_B = []
    neg_indices_A = []
    neg_indices_B = []
    for sample_idx in range(len(decoy_list)):
        current_correspondence = correspondence[sample_idx]
        current_native_residues = residue_list_A[sample_idx]
        current_generated_complex_residues = residue_list_B[sample_idx]
        native_residue_set = list(dict.fromkeys(current_native_residues))
        generated_complex_residue_set = list(dict.fromkeys(current_generated_complex_residues))

        # stripped_residue_list_A = []
        # for temp_residue in native_residue_set:
        #     temp_residue = temp_residue.split(".")
        #     temp_residue = temp_residue[-2] + "." + temp_residue[-1]
        #     stripped_residue_list_A.append(temp_residue)
        #
        # stripped_residue_list_B = []
        # for temp_residue in generated_complex_residue_set:
        #     temp_residue = temp_residue.split(".")
        #     temp_residue = temp_residue[-2] + "." + temp_residue[-1]
        #     stripped_residue_list_B.append(temp_residue)

        for native_A_edge, native_B_edge, pos_A_edge, pos_B_edge, neg_A_edge, neg_B_edge in current_correspondence:
            native_A_index = native_residue_set.index(native_A_edge) + index_pad_native
            native_B_index = native_residue_set.index(native_B_edge) + index_pad_native

            pos_A_index = generated_complex_residue_set.index(pos_A_edge) + index_pad_complex
            pos_B_index = generated_complex_residue_set.index(pos_B_edge) + index_pad_complex

            neg_A_index = generated_complex_residue_set.index(neg_A_edge) + index_pad_complex
            neg_B_index = generated_complex_residue_set.index(neg_B_edge) + index_pad_complex

            anchor_indices_A.append(native_A_index)
            anchor_indices_B.append(native_B_index)

            pos_indices_A.append(pos_A_index)
            pos_indices_B.append(pos_B_index)

            neg_indices_A.append(neg_A_index)
            neg_indices_B.append(neg_B_index)

        index_pad_native += len(native_residue_set)
        index_pad_complex += len(generated_complex_residue_set)
        del native_residue_set, generated_complex_residue_set

    anchor_A_embeddings = x_A.index_select(dim=0,
                                           index=torch.tensor(anchor_indices_A, dtype=torch.long, device=x_A.device))
    anchor_B_embeddings = x_A.index_select(dim=0,
                                           index=torch.tensor(anchor_indices_B, dtype=torch.long, device=x_A.device))

    pos_A_embeddings = x_B.index_select(dim=0, index=torch.tensor(pos_indices_A, dtype=torch.long, device=x_A.device))
    pos_B_embeddings = x_B.index_select(dim=0, index=torch.tensor(pos_indices_B, dtype=torch.long, device=x_A.device))

    neg_A_embeddings = x_B.index_select(dim=0, index=torch.tensor(neg_indices_A, dtype=torch.long, device=x_A.device))
    neg_B_embeddings = x_B.index_select(dim=0, index=torch.tensor(neg_indices_B, dtype=torch.long, device=x_A.device))

    anchor_edges = torch.cat((anchor_A_embeddings, anchor_B_embeddings), dim=1)
    positive_edges = torch.cat((pos_A_embeddings, pos_B_embeddings), dim=1)
    negative_edges = torch.cat((neg_A_embeddings, neg_B_embeddings), dim=1)

    return anchor_edges, positive_edges, negative_edges


#
#
# class GCNEConv(MessagePassing):
#
#     def __init__(self, in_channels: int, out_channels: int, edge_channels: int,
#                  improved: bool = False,  **kwargs):
#
#         super(GCNEConv, self).__init__(aggr='add', **kwargs)
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.improved = improved
#         self.edge_channels = edge_channels
#         self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
#         self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         zeros(self.bias)
#         glorot(self.weight)
#
#     def forward(self, x: Tensor, edge_index: Adj,
#                 edge_attr:Tensor,
#                 edge_weight: OptTensor = None) -> Tensor:
#
#         if isinstance(edge_index, Tensor):
#             edge_index, edge_weight = gcn_norm(  # yapf: disable
#             edge_index, edge_weight, x.size(self.node_dim),
#             self.improved, True, dtype=x.dtype)
#
#         elif isinstance(edge_index, SparseTensor):
#             edge_index = gcn_norm(  # yapf: disable
#                 edge_index, edge_weight, x.size(self.node_dim),
#                 self.improved, True, dtype=x.dtype)
#
#         # add self-loop edge attr
#         self_loop_attr = torch.zeros(x.size(0), self.edge_channels)
#         self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
#         edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
#
#         # propagate_type: (x: Tensor, edge_weight: OptTensor)
#         out = self.propagate(edge_index, x=x, edge_weight=edge_weight, edge_attr= edge_attr,
#                              size=None)
#
#         return out
#
#     def message(self, x_i: Tensor, x_j: Tensor, edge_weight: Tensor, edge_attr: Tensor) -> Tensor:
#
#         x_j = torch.matmul(torch.cat([x_j, edge_attr], dim=1) , self.weight)
#         return edge_weight.view(-1, 1) * x_j + self.bias
#
#
#
#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
#                                    self.out_channels)

# class TripletLoss(torch.nn.Module):
#     """
#     Triplet loss
#     Takes embeddings of an anchor sample, a positive sample and a negative sample
#     """
#
#     def __init__(self, margin):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, anchor, positive, negative, size_average=True):
#         distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
#         distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
#         losses = F.relu(distance_positive - distance_negative + self.margin)
#         return losses.mean() if size_average else losses.sum()

class tnet(torch.nn.Module):
    def __init__(self, in_features, K):
        super().__init__()

        self.K = K

        self.conv1 = torch.nn.Conv1d(in_features,64,1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.fc3 = torch.nn.Linear(256, K * K)

        self.identity = grad.Variable(torch.eye(self.K).float().view(-1).cuda())


    def forward(self, data, batch):

        batch_size = torch.max(batch)+1
        data, bools = to_dense_batch(data, batch)
        data = data.view(batch_size, 3, -1)

        tmp1 = self.conv1(data)
        temp2 = self.bn1(tmp1)
        x = F.relu(temp2)
        # x = F.relu(self.bn1(self.conv1(data)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x,2).values

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x+=self.identity

        x = x.view(data.shape[0],self.K,self.K)
        data = data.view(data.shape[0], data.shape[2], data.shape[1])

        return x, data, bools

def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.

        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
    """

    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(embeddings, torch.t(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diagonal(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = torch.unsqueeze(square_norm, 0) - 2.0 * dot_product + torch.unsqueeze(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.clamp(distances, min=0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
       Args:
           labels: tf.int32 `Tensor` with shape [batch_size]
       Returns:
           mask: tf.bool `Tensor` with shape [batch_size, batch_size]
       """

    # Check that i and j are distinct
    indices_equal = torch.eye(labels.shape[0]).bool()
    indices_not_equal = torch.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

    # Combine the two masks
    mask = torch.logical_and(indices_not_equal, labels_equal)

    return mask.float().cuda()

def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

    mask = torch.logical_not(labels_equal)

    return mask.float().cuda()

def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """

    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive  = _get_anchor_positive_triplet_mask(labels)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist  = torch.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist,_ = torch.max(anchor_positive_dist, dim=1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist,_ = torch.max(pairwise_dist, dim=1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist,_ = torch.min(anchor_negative_dist, dim=1, keepdims=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    temp_loss = hardest_positive_dist - hardest_negative_dist + margin
    triplet_loss = torch.clamp(temp_loss, min=0.0)

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    del hardest_positive_dist, mask_anchor_negative, max_anchor_negative_dist, anchor_negative_dist, hardest_negative_dist

    return triplet_loss


class DockPointNet(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, out_channels):
        super().__init__()

        self.t1 = tnet(3,3)

        # point-cloud layers for atomic non-bond embeddings
        # self.tnet = MLP([3, 64, 64, 3])
        # self.conv1 = SAModule(MLP([4 + in_channels, 64, 128]), None, 3, 'ppf')
        # self.conv2 = SAModule(MLP([4 + in_channels, 4 + in_channels,  128]), None, 5, 'ppf')
        # self.conv3 = SAModule(MLP([4 + in_channels, 4 + in_channels,  128]), None, 8.5, 'ppf')
        # self.conv4 = SAModule(MLP([4 + in_channels, 4 + in_channels,  128]), None, 10, 'ppf')
        self.conv2 = SAModule(MLP([7 + in_channels, 7 + in_channels, 128]), None, 5, 'ppf')
        self.conv3 = SAModule(MLP([7 + in_channels, 7 + in_channels, 128]), None, 8.5, 'ppf')
        self.conv4 = SAModule(MLP([7 + in_channels, 7 + in_channels, 128]), None, 10, 'ppf')
        # GCNE conv layers for atomic bond embeddings
        # self.gcn1 = GCNEConv(in_channels + edge_dim, 128, edge_dim)
        # self.gcn2 = GCNEConv(128 + edge_dim, 128, edge_dim)
        # self.gcn_norm = LayerNorm(128)

        self.atom_embed_layers = MLP([384, 512])

        # point_cloud layers for residue level embeddings
        self.res_embed_layers = MLP([512, 512])
        # self.res_conv1 = RESModule(MLP([3 + 512, 512]), MLP([512, 1024]), 5)

        # self.res_conv1 = SAModule(MLP([3 + 256, 512]), MLP([512, 1024]), 8.5, 'point_conv')
        # self.res_conv2 = SAModule(MLP([3 + 64, 128]), 8.5, 0.8)

        # self.lin1 = Lin(1024, out_channels)
        # self.lin2 = Lin(128, 1)
        # self.loss = CosineEmbeddingLoss(margin= 0, reduction='none')
        # self.loss = CosineEmbeddingLoss()
        # self.loss = BinaryFocalLoss(alpha=4)
        self.loss = TripletMarginLoss()
        self.loss2 = torch.nn.MSELoss()
        self.reset_parameters()
        # self.mlp = Seq(
        #     MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
        #     Lin(256, out_channels))

    def reset_parameters(self):
        # self.embed_layer.reset_parameters()
        # self.embed_norm.reset_parameters()

        # self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        # self.gcn1.reset_parameters()
        # self.gcn2.reset_parameters()
        # self.gcn_norm.reset_parameters()
        # self.res_conv1.reset_parameters()
        # self.lin1.reset_parameters()
        reset(self.res_embed_layers)
        reset(self.atom_embed_layers)

    def forward(self, data):
        x_A, residue_num_A, pos_A, normal_A, edge_index_A, edge_attr_A, batch_A, \
        x_B, residue_num_B, pos_B, normal_B, edge_index_B, edge_attr_B, batch_B, decoy_list, target_list, correspondence, fnat_gt = \
            data.x_A.float(), data.residue_num_list_A, data.pos_A.float(), data.norm_A.float(), \
            data.edge_index_A, data.edge_attr_A, data.x_A_batch, \
            data.x_B.float(), data.residue_num_list_B, data.pos_B.float(), data.norm_B.float(), \
            data.edge_index_B, data.edge_attr_B, data.x_B_batch, data.decoy_name, data.complex_name, data.correspondence, data.fnat

        batched_norm_A, bools2 = to_dense_batch(normal_A, batch_A)
        t_matrix_A, temp_pos_A, bools1 = self.t1(pos_A, batch_A)
        temp_pos_A = torch.bmm(temp_pos_A, t_matrix_A)
        batched_norm_A = torch.bmm(batched_norm_A,t_matrix_A)
        pos_A = temp_pos_A[bools1]
        normal_A = batched_norm_A[bools2]

        batched_norm_B, bools3 = to_dense_batch(normal_B, batch_B)
        t_matrix_B, temp_pos_B, bools4 = self.t1(pos_B, batch_B)
        temp_pos_B = torch.bmm(temp_pos_B, t_matrix_B)
        batched_norm_B = torch.bmm(batched_norm_B, t_matrix_B)
        pos_B = temp_pos_B[bools4]
        normal_B = batched_norm_B[bools3]


        # x_A2 = self.conv2(x_A, pos_A, batch_A, None, None, None)
        # x_B2 = self.conv2(x_B, pos_B, batch_B, None, None, None)
        # x_A3 = self.conv3(x_A, pos_A, batch_A, None, None, None)
        # x_B3 = self.conv3(x_B, pos_B, batch_B, None, None, None)
        # x_A4 = self.conv4(x_A, pos_A, batch_A, None, None, None)
        # x_B4 = self.conv4(x_B, pos_B, batch_B, None, None, None)

        x_A2 = self.conv2(x_A, pos_A, batch_A, None, None, normal_A)
        x_B2 = self.conv2(x_B, pos_B, batch_B, None, None, normal_B)
        x_A3 = self.conv3(x_A, pos_A, batch_A, None, None, normal_A)
        x_B3 = self.conv3(x_B, pos_B, batch_B, None, None, normal_B)
        x_A4 = self.conv4(x_A, pos_A, batch_A, None, None, normal_A)
        x_B4 = self.conv4(x_B, pos_B, batch_B, None, None, normal_B)

        atom_x_A = self.atom_embed_layers(torch.cat((x_A2[0], x_A3[0], x_A4[0]), dim=1))
        atom_x_B = self.atom_embed_layers(torch.cat((x_B2[0], x_B3[0], x_B4[0]), dim=1))

        residue_index_batch_A, residue_to_decoy_batch_A, residue_index_batch_B, residue_to_decoy_batch_B = \
            global_index_residue(residue_num_A, residue_num_B, decoy_list, x_A.device)

        res_x_A = global_max_pool(atom_x_A, residue_index_batch_A)
        res_x_B = global_max_pool(atom_x_B, residue_index_batch_B)

        res_x_A = self.res_embed_layers(res_x_A)
        res_x_B = self.res_embed_layers(res_x_B)

        anchor_edges, positive_edges, negative_edges = organize_by_correspondence(res_x_A, res_x_B, residue_num_A,
                                                                                  residue_num_B, decoy_list,
                                                                                  correspondence)

        tmp_label_pos = torch.ones(positive_edges.shape[0])
        tmp_label_neg = torch.zeros(negative_edges.shape[0])
        triplet_labels = torch.cat((tmp_label_pos,tmp_label_neg),0)
        triplet_embeddings = torch.cat((positive_edges,negative_edges),0)
        triplet_loss = batch_hard_triplet_loss(triplet_labels, triplet_embeddings, margin=2, squared=False)

        # loss = self.loss(anchor_edges, positive_edges, negative_edges)
        # identity = grad.Variable(torch.eye(3).cuda(),requires_grad=False)
        # reg_loss_A = self.loss2(torch.bmm(t_matrix_A,t_matrix_A.permute(0,2,1)),identity.expand(t_matrix_A.shape[0],-1,-1))
        # reg_loss_B = self.loss2(torch.bmm(t_matrix_B, t_matrix_B.permute(0, 2, 1)),identity.expand(t_matrix_B.shape[0], -1, -1))
        # total_loss = loss + reg_loss_A + reg_loss_B

        # if self.training:
        del anchor_edges, positive_edges, negative_edges


        return triplet_loss
        # else:
        # out = torch.sigmoid(out)
        ## out = 1 - out # keep the larger value
        # return loss, out, y, fnat(out, 0.4, correspondence)


def fnat(out, threshold, correspondence):
    fnat = []
    current_start_index = 0
    for current_correspondence in correspondence:
        current_coupling_num = len(current_correspondence)
        current_fnat = torch.sum(
            out[current_start_index: current_start_index + current_coupling_num].greater_equal(threshold).float())
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

    train_dataset = MNISTSuperpixels(r'/run/media/fei/Windows/hanye/MNIST/')
    val_dataset = MNISTSuperpixels(r'/run/media/fei/Windows/hanye/MNIST/', train=False)
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
    model = DockPointNet((train_dataset.num_features + 2), train_dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(1, 201):
        loss = train(train_loader, model, optimizer)
        val_acc = test(val_loader, model)
        print('Epoch {:03d}, Loss: {:.4f} Val_acc: {:.4f}'.format(epoch, loss, val_acc))
        scheduler.step()
        # break