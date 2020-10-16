import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool, radius, fps
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj
from torch_geometric.nn.inits import reset

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
        # fps sampling ratio parameter
        self.ratio = 0.5
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
        # b[0] and b[1] denote the batches of every point in x[0] and x[1]
        idx = fps(x[0], b, ratio=self.ratio)
        row, col = radius(x[0], x[0][idx], self.r, b, b[idx], max_num_neighbors=self.max_k)
        edge_index = torch.stack([col, row], dim=0)

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.max_k)

# returns sequence of linear layers, relu activation, and batch normalization of specified depth and width
def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')
pre_transform, transform = T.NormalizeScale(), T.SamplePoints(256)
train_dataset = ModelNet(path, '10', True, transform, pre_transform)
test_dataset = ModelNet(path, '10', False, transform, pre_transform)
train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=8, shuffle=False)


class DockPointNet(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max'):
        super().__init__()

        # self.tnet = MLP([3, 64, 64, 3])
        self.conv1 = DockPointNet_Module(MLP([2 * 3, 64, 64, 64]), k, aggr)
        self.conv2 = DockPointNet_Module(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        pos, batch = data.pos, data.batch

        # x0 = self.tnet(pos)
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.num_classes, k=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        # break
    return total_loss / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, 201):
    loss = train()
    test_acc = test(test_loader)
    print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))
    scheduler.step()
    # break
