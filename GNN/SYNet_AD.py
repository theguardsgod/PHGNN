import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from .dynunet_block import get_conv_layer, UnetResBlock
from monai.networks.layers.utils import get_norm_layer
from ..utils import feature_normalized, create_graph_from_embedding, sample_mask, preprocess_adj, preprocess_features

def process_adj_SF(A):
    adj = create_graph_from_embedding(A, name='knn', n=30)

    adj, edge = preprocess_adj(adj)
    adj = adj.todense()
    adj = torch.from_numpy(adj).to(torch.float32)
    return adj

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, input_feature, adjacency):

        support = torch.mm(input_feature, self.weight)
        output = torch.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output
class SYNet_AD(nn.Module):
    def __init__(self, input_dim, output_dim, spatial_dims=3, in_channels=3, dims=[32, 64, 128, 256], dropout=0.2):
        super(SYNet_AD, self).__init__()
        nd_1 = 128
        nd_2 = 64
        nd_3 = 32
        n_4 = 2

        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(5, 5, 5), stride=(5, 5, 5),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )

        self.gcn1_1 = GraphConvolution(input_dim, nd_1)  # AD-NC
        self.gcn1_2 = GraphConvolution(nd_1, nd_2)

        self.gcn2_1 = GraphConvolution(93, nd_3)
        self.gcn2_2 = GraphConvolution(nd_3, output_dim)

        self.linear1 = torch.nn.Linear(93, n_4)  # AD-NC
        self.linear2 = torch.nn.Linear(n_4, output_dim)
        self.linear3 = torch.nn.Linear(output_dim, output_dim)


        self.W = nn.Parameter(torch.ones(input_dim, 93))  # nxd

    def forward(self, adjacency1, feature, adjacency2):
        # adjacency1:S  adjacency2:A feature:X
        lam1 = 1e-5
        lam2 = 1e-3
        # y_hat = F.relu(self.gcn1_1(feature.t(), adjacency1))
        # y_hat = F.dropout(y_hat, 0.2, training=self.training)
        # y_hat = self.gcn1_2(y_hat, adjacency1)
        # s_hat = process_adj_SF((y_hat.cpu().detach().numpy())).to(device)

        x_hat = F.relu(self.gcn2_1(feature, adjacency2))
        x_hat = F.dropout(x_hat, 0.2, training=self.training)
        x_hat = self.gcn2_2(x_hat, adjacency2)
        a_hat = process_adj_SF(x_hat.cpu().detach().numpy())

        # s_hat += lam1 * torch.eye(s_hat.size(0)).to(y_hat.device)
        # a_hat += lam1 * torch.eye(a_hat.size(0)).to(y_hat.device)
        #
        # # (A'+I+A)XS'(S'+I+S)
        # s_hat += lam2 * adjacency1.to(y_hat.device)  # dxd
        # a_hat += lam2 * adjacency2.to(y_hat.device)  # nxn


        X_s = torch.mm(a_hat, feature)
        new_feature = torch.mm(a_hat, X_s)
        h = self.linear1(new_feature)
        h = F.relu(self.linear2(h))
        h = self.linear3(h)

        logits = h

        return logits, X_s