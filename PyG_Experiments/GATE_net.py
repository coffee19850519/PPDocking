from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import JumpingKnowledge, global_max_pool, global_add_pool, global_mean_pool
from PyG_Experiments.GlobalSelfAttention import GlobalSelfAttention,GlobalHardAttention

class GATEConv_with_global_pool(torch.nn.Module):
    def __init__(self, arguments):
        super(GATEConv_with_global_pool, self).__init__()
        num_node_features, num_edge_attr, num_layers, hidden, dense_hidden,  k, interface_only, head, concat = \
            arguments.num_node_features, arguments.num_edge_attr, \
            arguments.num_layers, arguments.hidden, arguments.dense_hidden,  arguments.k,\
            arguments.interface_only, arguments.head, arguments.concat
        self.k = k
        self.interface_only = interface_only

        self.conv1 = GATEConv(num_node_features, hidden,num_edge_attr= num_edge_attr, heads= head, concat= concat,dropout= 0.3)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATEConv(hidden, hidden, num_edge_attr= num_edge_attr, heads= head, concat= concat,dropout= 0.3))

        self.gat_jump = JumpingKnowledge(mode='max')
        self.pool_jump = JumpingKnowledge(mode='cat')

        self.glob_self_att = GlobalSelfAttention()
        #self.glob_hard_att = GlobalHardAttention()

        #if use 4 typed of global pooling operators
        self.lin1 = Linear(hidden * 4, int(hidden * 2))
        self.lin2 = Linear(int(hidden * 2), 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.att.reset_parameters()


    def forward(self, data, graph_num):
        x, edge_index, edge_attr, batch, interface_pos, decoy_name = \
            data.x, data.edge_index, \
            torch.reciprocal(data.edge_attr), \
            data.batch, data.interface_pos, data.decoy_name


        x = torch.tanh(self.conv1(x, edge_index, edge_attr))
        xs = [x]

        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index, edge_attr))
            xs += [x]

        #x = torch.tanh(self.convn(x, edge_index))
        #todo: normalize interface_count
        #xs += [torch.softmax(interface_count)]

        x = self.gat_jump(xs)
        del xs

        #attempt several global pooling strategies
        xs = [
              global_add_pool(x[interface_pos], batch[interface_pos]),

              global_mean_pool(x[interface_pos], batch[interface_pos]),
              # try adding attention scores to global pooling to replace global_
              self.glob_self_att(x[interface_pos], batch[interface_pos]),
              #self.glob_hard_att(x[interface_pos],interface_count[interface_pos], batch[interface_pos],),
              global_max_pool(x[interface_pos], batch[interface_pos])
              # weighted sum all interface nodes
              # scatter_add(softmax(interface_count[interface_pos], batch[interface_pos]).view(-1,1)
              #             * x[interface_pos], batch[interface_pos], dim=0)
              ]

        x = self.pool_jump(xs)


        #reshape 1d conv embedding to dense
        x = x.view((graph_num , -1))
        x = torch.tanh(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        del xs, data
        node_significance = None
        node_belong_decoy_num = None
        return self.lin2(x), node_significance, node_belong_decoy_num, interface_pos.cpu().detach().numpy(),decoy_name

    def __repr__(self):
        return self.__class__.__name__


class GATEConv(MessagePassing):

    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, num_edge_attr: int,heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        super(GATEConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.num_edge_attr = num_edge_attr

        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(num_edge_attr, heads * out_channels)

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: Tensor,
                size: Size = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = alpha_r = (x_l * self.att_l).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                num_nodes = size[1] if size is not None else num_nodes
                num_nodes = x_r.size(0) if x_r is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(num_nodes, self.num_edge_attr)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_encoder(edge_attr)

        out = self.propagate(edge_index, x=(x_l, x_r),edge_embeddings =edge_embeddings,
                             alpha=(alpha_l, alpha_r), size=size)
        del edge_embeddings
        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                edge_embeddings:Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        x_j += edge_embeddings.view(-1, self.heads, self.out_channels)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)



# class GATEConv(MessagePassing):
#
#     def __init__(self, in_channels, out_channels, num_edge_attr, heads=1, concat=True,
#                  negative_slope=0.2, dropout=0, bias=True, **kwargs):
#         super(GATEConv, self).__init__(aggr='add', **kwargs)
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.concat = concat
#         self.negative_slope = negative_slope
#         self.dropout = dropout
#         self.num_edge_attr = num_edge_attr
#         ### Mapping 0/1 edge features to embedding
#         self.edge_encoder = torch.nn.Linear(num_edge_attr, heads * out_channels)
#
#         self.__alpha__ = None
#
#         self.lin = Linear(in_channels, heads * out_channels, bias=False)
#
#         self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
#         self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
#
#         if bias and concat:
#             self.bias = Parameter(torch.Tensor(heads * out_channels))
#         elif bias and not concat:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         glorot(self.lin.weight)
#         glorot(self.att_i)
#         glorot(self.att_j)
#         zeros(self.bias)
#
#     def forward(self, x, edge_index, edge_attr, return_attention_weights=False):
#         """"""
#
#         if torch.is_tensor(x):
#             x = self.lin(x)
#             x = (x, x)
#         else:
#             x = (self.lin(x[0]), self.lin(x[1]))
#
#         edge_index, _ = remove_self_loops(edge_index)
#
#         # add self loops in the edge space
#         edge_index,_ = add_remaining_self_loops(edge_index, num_nodes=x[0].size(0))
#
#         # add features corresponding to self-loop edges.
#         self_loop_attr = torch.zeros(x[0].size(0), self.num_edge_attr)
#         self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
#         edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
#
#         edge_embeddings = self.edge_encoder(edge_attr)
#
#         out = self.propagate(edge_index, x=x,edge_embedding = edge_embeddings,
#                              return_attention_weights=return_attention_weights)
#         del edge_embeddings
#         if self.concat:
#             out = out.view(-1, self.heads * self.out_channels)
#         else:
#             out = out.mean(dim=1)
#
#         if self.bias is not None:
#             out = out + self.bias
#
#         if return_attention_weights:
#             alpha, self.__alpha__ = self.__alpha__, None
#             return out, (edge_index, alpha)
#         else:
#             return out
#
#     def message(self, x_i, x_j, edge_index_i, size_i, edge_embedding,
#                 return_attention_weights):
#         # Compute attention coefficients.
#         x_i = x_i.view(-1, self.heads, self.out_channels)
#         x_j = x_j.view(-1, self.heads, self.out_channels)
#         #x_j += edge_embedding.view(-1, self.heads, self.out_channels)
#         alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)
#         alpha = F.leaky_relu(alpha, self.negative_slope)
#         alpha = softmax(alpha, edge_index_i, size_i)
#
#         if return_attention_weights:
#             self.__alpha__ = alpha
#
#         # Sample attention coefficients stochastically.
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#
#         return x_j * alpha.view(-1, self.heads, 1)
#
#     def __repr__(self):
#         return '{}({}, {}, heads={})'.format(self.__class__.__name__,
#                                              self.in_channels,
#                                              self.out_channels, self.heads)