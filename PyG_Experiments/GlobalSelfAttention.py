import torch
from torch_scatter import scatter_add,scatter_max, scatter_std
from torch_geometric.utils import softmax, to_dense_batch
from torch.nn.functional import dropout

# from torch_geometric.nn.inits import reset
# def restricted_softmax(src, dim: int = -1, margin: float = 0.):
#     src_max = torch.clamp(src.max(dim=dim, keepdim=True)[0], min=0.)
#     out = (src - src_max).exp()
#     out = out / (out.sum(dim=dim, keepdim=True) + (margin - src_max).exp())
#     return out

class GlobalSelfAttention(torch.nn.Module):

    def __init__(self):
        super(GlobalSelfAttention, self).__init__()
        # self.gamma = torch.nn.Parameter(torch.zeros(1))
        # self.out_channels = out_channels
        # self.linear_out = torch.nn.Linear(out_channels * 2, out_channels, bias=False)
        # self.linear_out.reset_parameters()

    def forward(self, x, batch, size=None):
        """"""
        size = batch[-1].item() + 1 if size is None else size
        batch_x, mask = to_dense_batch(x, batch)
        batch_size, num_nodes, _  = batch_x.size()
        #mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)


        attention_scores = torch.matmul(batch_x, batch_x.transpose(-2, -1).contiguous())
        attention_scores = attention_scores / torch.tensor(batch_x.size(-1), dtype= torch.float, device= x.device)
        scores = attention_scores.sum(dim= 1)[mask]

        scores = softmax(scores, batch, num_nodes=size)
        scores = dropout(scores, p= 0.1, training= self.training)
        #inspired from the implementation of torchnlp's attention
        #https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html

        # concat -> (batch_size * output_len, 2*dimensions)
        # combined = torch.cat((alpha.view(-1,1) * x, x), dim=-1)
        # combined = combined.view(batch_size , 2 * self.out_channels)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        # output = self.linear_out(combined).view(-1, self.out_channels)
        # alpha = torch.sigmoid(self.gamma)
        # output = alpha * scores.view(-1,1) * x + (1 - alpha) * x
        # print(alpha)
        out = scatter_add(scores.view(-1,1) * x, batch, dim=0, dim_size=size)
        del attention_scores, x, scores, batch_x, mask #,combined
        return out


class GlobalHardAttention(torch.nn.Module):
    def __init__(self):
        super(GlobalHardAttention, self).__init__()
        #self.gamma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x,interface_count, batch, size=None):
        size = batch[-1].item() + 1 if size is None else size
        attention_scores = softmax(interface_count, batch).view(-1, 1)
        out = scatter_add(attention_scores * x + x, batch, dim=0, dim_size= size)
        del attention_scores
        return out