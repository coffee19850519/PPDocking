import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros,reset


class GRATConv(MessagePassing):
    r"""
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, head,
                 #edge_weight = True,
                 concat=True,
                 bias=True,
                 dropout = 0,
                 **kwargs):
        super(GRATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = head
        self.concat = concat
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels,
                                             self.heads * out_channels))

        #self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        # if edge_weight:
        #     self.edge_weight = Parameter(torch.Tensor(self.heads, self.heads))
        # else:
        #     self.register_parameter('edge_weight', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.linear = torch.nn.Linear(self.heads * out_channels, out_channels, bias)
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.linear = None
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        #glorot(self.edge_weight)
        zeros(self.bias)

    def add_self_loops(self, edge_index, edge_attr = None):

        num_nodes = edge_index.max().item() + 1

        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)

        if edge_attr is not None:

            loop_attr = torch.ones((num_nodes, self.heads ), device= edge_index.device)
            edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        edge_index = torch.cat([edge_index, loop_index], dim=1)
        del loop_index
        return edge_index, edge_attr


    def forward(self, x, edge_index, size=None, edge_attr=None, return_attention_weights=False):

        # add self loops for all nodes
        # remove_self_loops is included before, so there are no duplicates
        if size is None and torch.is_tensor(x):
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr= edge_attr)
            edge_index, edge_attr = self.add_self_loops(edge_index,edge_attr)

        # get node latent embeddings
        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            # for Pair Tensor
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        out = self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr, return_attention_weights=return_attention_weights)

        if return_attention_weights:
            alpha, self.alpha = self.alpha, None
            return out, alpha
        else:
            return out

    # x_i is target, x_j is source (i.e. messages coming to x_i from x_j)
    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr, return_attention_weights):
        # Compute self attention coefficients.
        assert x_i is not None
        #assert edge_attr is not None

        # Key: source/neighbor node embeddings
        # reshape tensor for # of heads
        x_j = x_j.view(-1, self.heads, self.out_channels)

        # Query: center/target node embeddings
        # reshape tensor for # of heads
        x_i = x_i.view(-1, self.heads, self.out_channels)

        # dim 0 is batch, so transpose on dim 1 head and dim 2 features
        # contiguous makes x_j_t an in memory tensor containing the data from x_j
        x_j_t = x_j.transpose(1, 2).contiguous()

        # batch matrix multiply
        # outputs (b,head,1) (i.e. raw how much to listen to each head)
        attention_scores = torch.bmm(x_i, x_j_t).sum(dim=-1)

        # scale attention scores by dimensionality of embedding channels
        attention_scores.mul_(1.0 / torch.sqrt(torch.tensor(self.out_channels, dtype=torch.float, device= x_i.device)))

        # how much to listen to each head 0 to 1 scale
        alpha = softmax(attention_scores, edge_index_i, size_i)

        if return_attention_weights:
            self.alpha = alpha

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # edge_attr is defined by distance between points
        # edge embedding
        # apply distance embeddings to neighbors
        if edge_attr is not None:
            # scale edge attr
            # 1e-16 is smoothing constant
            edge_attr = 1 / (edge_attr.view(-1, self.heads, 1) + 1e-16)
            edge_attr = torch.matmul(edge_attr, self.edge_weight)
            x_j = x_j * alpha.view(-1, self.heads, 1) * edge_attr
            del alpha, attention_scores, x_j_t
            return x_j, attention_scores
        else:
            # since we're assigning attention values by entire heads, chunks of the feature vector are weighted equally instead of feature-wise
            x_j = x_j * alpha.view(-1, self.heads, 1)
            del alpha, attention_scores, x_j_t
            return x_j

    # gets the output of aggregation procedure (defined at initialization)
    # aggr_out is (b,heads,features)
    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
            aggr_out = self.linear(aggr_out)
        else:
            aggr_out = aggr_out.mean(dim=1)

        # TODO what is purpose of this bias: change this to only work for yes bias, no concat
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)


class GlobalAttentionPooling(torch.nn.Module):
    r"""
    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
    """
    def __init__(self, gate_nn):
        super(GlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)


    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)

        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, size)
        #out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)

'''
class DotProductAttention(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim):
        super().__init__()

        self.scale = 1.0 / np.sqrt(query_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, mask, query, keys, values):
        # query: [B,Q] (hidden state, decoder output, etc.)
        # keys: [T,B,K] (encoder outputs)
        # values: [T,B,V] (encoder outputs)
        # assume Q == K

        # compute energy
        query = query.unsqueeze(1)  # [B,Q] -> [B,1,Q]
        keys = keys.permute(1, 2, 0)  # [T,B,K] -> [B,K,T]
        energy = torch.bmm(query, keys)  # [B,1,Q]*[B,K,T] = [B,1,T]
        energy = self.softmax(energy.mul_(self.scale))

        # apply mask, renormalize
        energy = energy * mask
        energy.div(energy.sum(2, keepdim=True))

        # weight values
        values = values.transpose(0, 1)  # [T,B,V] -> [B,T,V]
        combo = torch.bmm(energy, values).squeeze(1)  # [B,1,T]*[B,T,V] -> [B,V]

        return (combo, energy)'''
