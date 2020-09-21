import torch
import torch.nn.functional as F
from torch.nn import Linear,Conv1d, MaxPool1d
from torch_geometric.nn import SAGEConv, global_mean_pool, GATConv, SAGPooling, GCNConv ,JumpingKnowledge, global_max_pool, GlobalAttention, global_sort_pool, global_add_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter_max
from torch_scatter.utils import broadcast

class GATSortPool_by_occurrence(torch.nn.Module):
    def __init__(self, arguments):
        super(GATSortPool_by_occurrence, self).__init__()
        num_node_features, num_layers, hidden, dense_hidden, k, interface_only, head, concat  = arguments.num_node_features, \
             arguments.num_layers, arguments.hidden, arguments.dense_hidden, arguments.k, arguments.interface_only, arguments.head, arguments.concat
        self.k = k
        self.interface_only = interface_only
        # self.conv1 = GRATConv(num_node_features, hidden, head, concat= True, dropout= 0.2)
        self.conv1 = GCNConv(num_node_features, hidden)

        # self.conv1 = GATConv(num_node_features, hidden, heads= head, concat= concat, dropout= 0.3)


        self.convs = torch.nn.ModuleList()

        for i in range(num_layers-1):
            # self.convs.append(GRATConv(hidden, hidden, head= head, concat= True, dropout= 0.2))
            self.convs.append(GCNConv(hidden, hidden))
            # self.convs.append(GATConv(hidden, hidden, heads= head, concat= concat, dropout= 0.3))
        #self.convn = GCNConv(hidden, 1)
        #self.convn = GATConv(head * hidden, 1, dropout= 0.3)

        # self.convn = GRATConv(head * hidden, 1, head= 1, concat= False, dropout= 0.2)
        #self.convn = GlobalAttentionPooling(Linear(hidden * (num_layers - 1), 1))


        self.jump = JumpingKnowledge(mode='cat')


        if concat:
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
        #self.convn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

        self.conv1d_1.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.att.reset_parameters()

    def forward(self, data, graph_num):
        x, edge_index, edge_attr, batch, interface_pos, decoy_name, interface_count = \
            data.x, data.edge_index, data.edge_attr, data.batch, data.interface_pos, data.decoy_name, data.interface_count.type(torch.float) / 5400
        #batch = torch.mul(batch, interface_tags)
        #print(data.complex_name)
        #conver graph to undirected
        #if data.
        # edge_index = to_undirected(edge_index, len(x))

        #x = torch.tanh(self.conv1(x, edge_index,edge_attr= edge_attr))
        x = torch.tanh(self.conv1(x, edge_index))        ###graph convolution 1

        xs = [x]

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

        #x = torch.tanh(self.convn(x, edge_index))

        #get nodes at interface
        xs += [interface_count.view(-1, 1)]
        x = self.jump(xs)    #contact

        x = global_sort_pool(x[interface_pos], batch[interface_pos], self.k)    #sort acordding the last value

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



class GATSortPool_by_occurrence_global_pool(torch.nn.Module):
    def __init__(self, arguments):
        super(GATSortPool_by_occurrence_global_pool, self).__init__()
        num_node_features, num_layers, hidden, dense_hidden, k, interface_only, head, concat  = arguments.num_node_features, \
             arguments.num_layers, arguments.hidden, arguments.dense_hidden, arguments.k, arguments.interface_only, arguments.head, arguments.concat
        self.k = k
        self.interface_only = interface_only
        #self.conv1 = GRATConv(num_node_features, hidden, head, concat= True, dropout= 0.2)
        #self.conv1 = GCNConv(num_node_features, hidden)

        self.conv1 = GATConv(num_node_features, hidden, heads= head, concat= concat, dropout= 0.3)

        self.convs = torch.nn.ModuleList()

        for i in range(num_layers-1):
            # self.convs.append(GRATConv(hidden, hidden, head= head, concat= True, dropout= 0.2))
            # self.convs.append(GCNConv(hidden, hidden))
            self.convs.append(GATConv(hidden, hidden, heads= head, concat= concat, dropout= 0.3))
        #self.convn = GCNConv(hidden, 1)
        #self.convn = GATConv(head * hidden, 1, dropout= 0.3)

        # self.convn = GRATConv(head * hidden, 1, head= 1, concat= False, dropout= 0.2)
        #self.convn = GlobalAttentionPooling(Linear(hidden * (num_layers - 1), 1))


        self.jump = JumpingKnowledge(mode='cat')


        #if use 3 typed of global pooling operators
        # self.lin0 = Linear(hidden * 4, int(hidden * 2))

        self.lin1 = Linear(int(hidden * 3 * 2), int(hidden * 3))

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

        self.lin2 = Linear(int(hidden * 3), 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        #self.convn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()


    def forward(self, data, graph_num):
        x, edge_index, edge_attr, batch, interface_pos, decoy_name, interface_count = \
            data.x, data.edge_index, data.edge_attr, data.batch, data.interface_pos, data.decoy_name, data.interface_count
        #batch = torch.mul(batch, interface_tags)
        #print(data.complex_name)
        #conver graph to undirected
        #if data.
        # edge_index = to_undirected(edge_index, len(x))

        #x = torch.tanh(self.conv1(x, edge_index,edge_attr= edge_attr))
        x = torch.tanh(self.conv1(x, edge_index))        ###graph convolution 1

        xs = [x]

        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            xs += [x]

        #x = torch.tanh(self.convn(x, edge_index))
        #todo: normalize interface_count
        #xs += [torch.softmax(interface_count)]

        x = self.jump(xs)    #contact
        del xs

        #attempt several global pooling strategies



        xs = [
              global_add_pool(x[interface_pos], batch[interface_pos]),
              global_mean_pool(x[interface_pos], batch[interface_pos]),
              global_max_pool(x[interface_pos], batch[interface_pos]),
              # weighted sum all interface nodes
              # scatter_add(softmax(interface_count[interface_pos], batch[interface_pos])* x[interface_pos], batch[interface_pos], dim=0)
              ]

        x = self.jump(xs)

        #x = self.att(x[interface_pos], batch[interface_pos])

        #reshape 1d conv embedding to dense
        x = x.view((graph_num , -1))
        #x = x.view((-1, self.dense_dim))
        # x = torch.relu(self.lin0(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = torch.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        del xs, data #,contact_edge_index
        node_significance = None
        node_belong_decoy_num = None
        return self.lin2(x), node_significance, node_belong_decoy_num, interface_pos.cpu().detach().numpy(),decoy_name

    def __repr__(self):
        return self.__class__.__name__