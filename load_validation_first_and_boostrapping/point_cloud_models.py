import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, Parameter, ReLU, Conv1d, Dropout, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
from torch import Tensor
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops, softmax
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_cluster import knn
from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj
from torch_geometric.nn.inits import reset, glorot, zeros

class SAModule(torch.nn.Module):
    # ratio is sampling ratio
    # r is radius for ball
    # nn is the point embedding network
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r

        # from looking at the source code for PointConv, there is not affine transformation matrix multiplication, just a point embedding
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

# not a great name for this module
# just final pointNet clustering for aggregated set of points
# no sampling/grouping, taking all points
class GlobalSAModule(torch.nn.Module):
    # nn is point embedding network
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

# returns sequence of linear layers, relu activation, and batch normalization of specified depth and width
def MLP(channels, batch_norm=True):

    # the reason they're handling this with a linear layer, is because they break up the input tensor into position and batch seperately for radius function in MessagePassing module
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

    # return Seq(*[
    #     Seq(Conv1d(channels[i - 1], channels[i], 1, stride=1), ReLU(), BN(channels[i]))
    #     for i in range(1, len(channels))
    # ])

# defines pointNet++ architecture
# remember to put in training mode 
class Point_Net2(torch.nn.Module):
    def __init__(self):
        super(Point_Net2, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 10)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        # fully connected classifier
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)

# defines dgcnn architecture
# instantiation example Net(num_classes, knn_num)
class DynamicGraphCNN(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)

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
        # radius for ball sampling
        self.r = 0.2

    def reset_parameters(self):
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        assert x[0].dim() == 2, \
            'Static graphs not supported in `DockPointNet_Module`.'

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = batch
        elif isinstance(batch, tuple):
            assert batch is not None
            b = batch[0]

        # x[0] and x[1] are copied tensors of same input points
        # point position data x and batch information b need to be seperated as input to radius function
        # why this could not be instead handled inside of the function, I do not know
        row, col = radius(x[0], x[1], self.r, b, b, max_num_neighbors=self.max_k)
        edge_index = torch.stack([col, row], dim=0)

        # propagate_type: (x: PairTensor)
        # x.shape = (N,3)
        print('edge index')
        print(edge_index.shape)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:

        print('xi')
        print(x_i.shape)
        print('xj')
        print(x_j.shape)
        # TODO do reshape before and after
        out = self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
        return out

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.max_k)

class Att_DockPointNet_Module(MessagePassing):
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
    def __init__(self, in_channels, out_channels, head, k: int, aggr: str = 'max', dropout = 0, bias=True, **kwargs):
        super(Att_DockPointNet_Module, self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        self.max_k = k
        self.heads = head
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        # radius for ball sampling
        self.r = 0.2

        # used to generate node embeddings prior to attention
        # generates large tensor that will be split into given # of heads
        self.weight = Parameter(torch.Tensor(in_channels, self.heads * out_channels))
        self.linear = torch.nn.Linear(self.heads * out_channels, out_channels, bias)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)

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

    def forward(self, x: Union[Tensor, PairTensor], batch: Union[OptTensor, Optional[PairTensor]] = None, edge_attr=None, return_attention_weights=False) -> Tensor:
        
        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = batch
        elif isinstance(batch, tuple):
            assert batch is not None
            b = batch[0]

        # x[0] and x[1] are copied tensors of same input points
        # point position data x and batch information b need to be seperated as input to radius function
        row, col = radius(x, x, self.r, b, b, max_num_neighbors=self.max_k)
        edge_index = torch.stack([col, row], dim=0)

        # add self loops for all nodes
        # remove_self_loops is included before, so there are no duplicates
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr= edge_attr)
        edge_index, edge_attr = self.add_self_loops(edge_index,edge_attr)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, return_attention_weights=return_attention_weights)

        if return_attention_weights:
            alpha, self.alpha = self.alpha, None
            return out, alpha
        else:
            return out

    # x.shape = (N,3)
    # x_i is target, x_j is source (i.e. messages coming to x_i from x_j)
    def message(self, edge_index_i, x_i: Tensor, x_j: Tensor, edge_attr, return_attention_weights) -> Tensor:

        # get node latent embeddings
        x_i = torch.matmul(x_i, self.weight)
        x_j = torch.matmul(x_j, self.weight)

        # Query: center/target node embeddings
        # reshape tensor for # of heads
        x_i = x_i.view(-1, self.heads, self.out_channels)

        # Key: source/neighbor node embeddings
        # reshape tensor for # of heads
        x_j = x_j.view(-1, self.heads, self.out_channels)

        # dim 0 is batch, so transpose on dim 1 head and dim 2 features
        # contiguous makes x_j_t an in memory tensor containing the data from x_j
        x_j_t = x_j.transpose(1, 2).contiguous()

        # batch matrix multiply
        # outputs (b,head,1) (i.e. raw how much to listen to each head)
        attention_scores = torch.bmm(x_i, x_j_t).sum(dim=-1)

        # scale attention scores by dimensionality of embedding channels
        attention_scores.mul_(1.0 / torch.sqrt(torch.tensor(self.out_channels, dtype=torch.float, device= x_i.device)))

        # how much to listen to each head 0 to 1 scale
        alpha = softmax(attention_scores, edge_index_i)

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

            # reshape for the library's automatic aggregation
            x_j = x_j.view(-1, self.heads * self.out_channels)

            del alpha, attention_scores, x_j_t
            return x_j
    
    # gets the output of aggregation procedure (defined at initialization)
    # aggr_out is (b,heads,features)
    def update(self, aggr_out):
        
        aggr_out = self.linear(aggr_out)

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)


class DockPointNet(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max'):
        super().__init__() 


        # self.conv1 = DockPointNet_Module(MLP([2 * 3, 64, 64, 64]), k, aggr)
        # self.conv2 = DockPointNet_Module(MLP([2 * 64, 128]), k, aggr)
        # self.lin1 = MLP([128 + 64, 1024])


        self.conv1 = Att_DockPointNet_Module(3,27,9,k=k)
        self.conv2 = Att_DockPointNet_Module(27,64,9,k=k)
        self.lin1 = MLP([64+27, 128])
        self.mlp = Seq(
            MLP([128, 128]), Dropout(0.5), MLP([128, 128]), Dropout(0.5),
            Lin(128, out_channels))

    def forward(self, data):
        '''
            data is expected to be a pytorch_geometric Batch object Batch(batch=[2048], pos=[2048, 3], y=[8])  
        '''

        pos, batch = data.pos, data.batch

        # print('pos')
        # print(pos.shape)

        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)

        # # x0 = self.tnet(pos)
        # # pos and batch parts are passed as seperate args for the radius function docking module
        # x1 = self.conv1(pos, batch)
        # x2 = self.conv2(x1, batch)
        # out = self.lin1(torch.cat([x1, x2], dim=1))
        # out = global_max_pool(out, batch)
        # out = self.mlp(out)
        # return F.log_softmax(out, dim=1)



