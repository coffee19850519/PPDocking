
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter,LayerNorm,BatchNorm1d,InstanceNorm1d
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul_
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import global_mean_pool,JumpingKnowledge, global_max_pool, global_add_pool,GCNConv,GlobalAttention


# from torch_geometric.nn.norm import batch_norm
from torch.nn import Linear
from PyG_Experiments.GlobalSelfAttention import GlobalSelfAttention,GlobalHardAttention


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul_(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul_(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNEConv(MessagePassing):


    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int, edge_channels: int,
                 concat: bool = True, improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        super(GCNEConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.edge_channels = edge_channels
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.concat = concat
        # if not add_self_loops:
        #     self.skip = torch.nn.Parameter(torch.ones(1))
        # Mapping edge features to embedding
        if concat:
            self.edge_encoder = torch.nn.Linear(edge_channels, 32, bias=False)
            self.edge_norm = LayerNorm(32, elementwise_affine=True)
        else:
            self.edge_encoder = torch.nn.Linear(edge_channels, out_channels, bias= False)
            self.edge_norm = LayerNorm(out_channels, elementwise_affine=True)
        self.node_encoder = torch.nn.Linear(in_channels, out_channels, bias= False)

        self.output_norm = LayerNorm(out_channels,elementwise_affine=True)

        # if concat:
        #     self.weight_j = Parameter(torch.Tensor(out_channels + 32, out_channels))


        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        # if self.concat:
        #     glorot(self.weight_j)
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.edge_norm.reset_parameters()
        self.output_norm.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr:Tensor,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache



        # add features corresponding to self-loop edges.
        if self.add_self_loops:
            self_loop_attr = torch.zeros(x.size(0), self.edge_channels)
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        node_embeddings = self.node_encoder(x)
        edge_embeddings = self.edge_encoder(edge_attr)
        edge_embeddings = self.edge_norm(edge_embeddings)

        if self.concat:
            # if self.add_self_loops:
            #     x = torch.matmul(x, self.weight_j)
            out = self.propagate(edge_index, x=node_embeddings, edge_weight=edge_weight, edge_attr=edge_embeddings,
                                 size=None)
        else:

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=node_embeddings, edge_weight=edge_weight, edge_attr= edge_embeddings,
                                 size=None)

        if self.bias is not None:
            out = out + self.bias

        return self.output_norm(out)


    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: Tensor,edge_attr: Tensor) -> Tensor:
        if self.concat:
            # x_i = torch.matmul(torch.cat([x_i, edge_attr], dim=1), self.weight)
            x_j = torch.matmul(torch.cat([x_j, edge_attr], dim=1) , self.weight_j)
            return edge_weight.view(-1, 1) * x_j
        else:
            return edge_weight.view(-1, 1) * (x_j + edge_attr)
        # return edge_weight.view(-1, 1) * x_j

    # def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
    #     if self.add_self_loops:
    #         return aggr_out
    #     else:
    #         alpha = torch.sigmoid(self.skip)
    #         return (1 - alpha) * x + aggr_out * alpha

    # def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
    #     return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



class GCEonv_with_global_pool(torch.nn.Module):
    def __init__(self, arguments):
        super(GCEonv_with_global_pool, self).__init__()
        num_node_features, num_edge_attr, num_layers, hidden, dense_hidden,  k, interface_only, head, concat = \
            arguments.num_node_features, arguments.num_edge_attr, \
            arguments.num_layers, arguments.hidden, arguments.dense_hidden,  arguments.k,\
            arguments.interface_only, arguments.head, arguments.concat
        self.k = k
        self.interface_only = interface_only

        self.conv1 = GCNEConv(num_node_features, hidden, num_edge_attr, concat= concat, add_self_loops= True)
        # self.conv1 = GCNConv(num_node_features, hidden, add_self_loops=True)
        # self.norm1 = LayerNorm(hidden, elementwise_affine= True)
        self.convs = torch.nn.ModuleList()
        # self.norms = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            # self.convs.append(GCNConv(hidden, hidden, add_self_loops=True))
            self.convs.append(GCNEConv(hidden, hidden, num_edge_attr, concat= concat, add_self_loops= True))
            # self.norms.append(LayerNorm(hidden, elementwise_affine= True))

        # self.jump_node_embeddings = JumpingKnowledge(mode='cat')
        # self.jump_graph_embeddings = JumpingKnowledge(mode='cat')
        self.glob_self_att = GlobalSelfAttention()
        # self.glob_hard_att = GlobalHardAttention()
        # self.glob_pool = GlobalAttention(Linear(hidden, 1, bias= False))
        #if use 4 types of global pooling operators
        self.lin1 = Linear(hidden * 1, int(hidden * 0.5))
        # self.norm = LayerNorm(int(hidden * 1), elementwise_affine= True)
        self.lin2 = Linear(int(hidden * 0.5), 1)
        #self.norm2 = LayerNorm(1)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        # self.norm1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        # for n in self.norms:
        #     n.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        # self.norm.reset_parameters()



    def forward(self, data, graph_num):
        x, edge_index, edge_attr, batch, interface_pos, decoy_name = \
            data.x, data.edge_index, \
            data.edge_attr, \
            data.batch, data.interface_pos, data.decoy_name

        # x = torch.tanh(self.conv1(x, edge_index))
        x = torch.tanh(self.conv1(x, edge_index, edge_attr))
        # x = F.dropout(x, training=self.training)
        # xs = [x]

        for conv_idx in range(0, len(self.convs)):
            # x = torch.tanh(self.convs[conv_idx](x, edge_index))
            x = torch.tanh(self.convs[conv_idx](x, edge_index, edge_attr))
            # x = F.dropout(x, training= self.training)
            # xs += [x]

        #x = torch.tanh(self.convn(x, edge_index))
        #todo: normalize interface_count
        #xs += [torch.softmax(interface_count)]

        # x = self.jump_node_embeddings(xs)    #contact
        # del xs

        #attempt several global pooling strategies
        #xs = [
              #global_add_pool(x[interface_pos], batch[interface_pos]),

              # global_mean_pool(x[interface_pos], batch[interface_pos]),

              # try adding attention scores to global pooling to replace global_
              # self.glob_pool(x[interface_pos], batch[interface_pos]),
              # self.glob_self_att(x[interface_pos], batch[interface_pos]),
              # self.glob_hard_att(x[interface_pos],interface_count[interface_pos], batch[interface_pos]),
              # global_max_pool(x[interface_pos], batch[interface_pos])
              # weighted sum all interface nodes
              # scatter_add(softmax(interface_count[interface_pos], batch[interface_pos]).view(-1,1)
              #             * x[interface_pos], batch[interface_pos], dim=0)
             # ]

        # x = self.jump_graph_embeddings(xs)
        x = self.glob_self_att(x[interface_pos], batch[interface_pos])

        #reshape 1d conv embedding to dense
        x = x.view((graph_num , -1))
        # x = torch.tanh(self.norm(self.lin1(x)))
        x = torch.tanh(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        del  data
        node_significance = None
        node_belong_decoy_num = None
        return torch.sigmoid(self.lin2(x)), node_significance, node_belong_decoy_num, interface_pos.cpu().detach().numpy(),decoy_name

    def __repr__(self):
        return self.__class__.__name__