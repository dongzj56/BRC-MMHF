import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.data import data as D
from torch.nn import Linear
import torch_scatter
from math import sqrt
from .Layers import EncoderLayer, Decoder, Predictor
from .Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from .Layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from .embed import DataEmbedding, CustomEmbedding,DataEmbedding_new

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        configs.device = torch.device("cuda")
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in

        # Hypergraph design code
        self.individual = configs.individual
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Tran = nn.Linear(self.pred_len, self.pred_len)
        self.enc_embedding=DataEmbedding(configs.enc_in, configs.d_model, configs.dropout)
        self.mask, self.all_size, self.adj= get_mask(configs.seq_len, configs.window_size, configs.inner_size, configs.khop, configs.device)
        self.Ms_length = sum(self.all_size)
        self.conv_layers = eval(configs.CSCM)(configs.enc_in, configs.window_size, configs.d_bottleneck)
        self.conv1 = HypergraphConv(configs)
        # self.conv2 = TMP(128, 512)
        self.convtra = nn.Linear(self.channels, self.channels)
        self.convtra.weight = nn.Parameter(
            (1 / (self.channels)) * torch.ones([self.channels, self.channels]))
        self.out_tran = nn.Linear(self.Ms_length, self.pred_len)

        self.predictor=Predictor(self.channels,self.pred_len*self.channels)
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.indexes=refer_points(self.all_size, configs.window_size, configs.device)

    def forward(self, x,x_mark_enc):
        # instance_normal
        mean_enc=x.mean(1,keepdim=True).detach()
        std_enc=torch.sqrt(torch.var(x,dim=1,keepdim=True,unbiased=False)+1e-5).detach()
        x = x - mean_enc
        x=x / std_enc
        _, _, N = x.shape
        # Hypergraph code
        # seq_enc = self.enc_embedding(x, x_mark_enc)
        mask=self.mask
        # mask = self.mask.repeat(len(x), 1, 1)
        mask = torch.tensor(mask).to(x.device)
        seq_enc = self.conv_layers(x).to(x.device)
        seq_enc = torch.tensor(seq_enc, dtype=torch.float).to(x.device)

        adj = self.adj
        hw = None
        x_out = self.conv1(seq_enc, mask, adj, hw)
        # Fusion module
        if self.individual:
            # Implement the last layer of pyramid
            indexes = self.indexes.repeat(x.size(0), 1, 1, x.size(2)).to(x.device)
            indexes = indexes.view(x_out.size(0), -1, x_out.size(2))
            all_enc = torch.gather(x_out, 1, indexes)
            seq_enc = all_enc.view(x.size(0), self.all_size[0], -1).permute(0, 2, 1)
            out = x[:, -1, :]
            out_tra = self.predictor(out).view(x.size(0), self.pred_len, -1).permute(0, 2, 1)
            x = self.Linear(x.permute(0,2,1))
        else:
            x = self.Linear(x.permute(0, 2, 1))
            out_tra = self.out_tran(x_out.permute(0, 2, 1))

        x = x + out_tra
        x=self.Linear_Tran(x).permute(0,2,1)

        x = x * std_enc+ mean_enc
        return x # [Batch, Output length, Channel]


class TMP(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention=True,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=False):
        super(TMP, self).__init__(aggr='add')
        self.soft=nn.Softmax(dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        # self.fc = nn.Linear(self.out_channels * heads, self.out_channels)
        self.W=nn.Linear(1,512)
        self.dropout1 = nn.Dropout(0.1)
        # self.dropout_rate=0.1
        ## Attention calculation between hyperedges

        # Information aggregation attention from hyperedge to node
        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    # Initialize weight and bias parameters
    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)
    
    # Calculate degree of hypergraph, then calculate normalized weight of hypergraph based on degree
    def __forward__(self,
                    x,
                    hyperedge_index,adj,
                    hyperedge_weight=None,
                    alpha=None):
        aa = hyperedge_index[0]
        dd=x.size(0)

        if hyperedge_weight is None:
            # D is the degree of the node
            D = degree(hyperedge_index[0], x.size(0), x.dtype)
        else:
            D_1 = torch_scatter.scatter_add(
                hyperedge_weight[hyperedge_index[1, 0:13263]],
                hyperedge_index[0, 0:13263],
                dim=0,
                dim_size=x.size(0))
            D_2 = torch_scatter.scatter_add(
                hyperedge_weight[hyperedge_index[1, 13264:112859]],
                hyperedge_index[0, 13264:112859],
                dim=0,
                dim_size=x.size(0))
            D = torch.cat((D_1, D_2), dim=0)
            # ---------------------------------------------------------
        D = 1.0 / D
        D[D == float("inf")] = 0


        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B_1 = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        B_2 = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        B=B_1
        # ---------------------------------------------------------

        B[B == float("inf")] = 0
        if hyperedge_weight is not None:
            # B = B * hyperedge_weight; two next line is added by myself
            B = B * hyperedge_weight.t()
            B = B.t()

        # Call propagate method to execute message passing, passing node features and normalized weights
        # Propagate executes message aggregation and node feature update operations
        # Propagate runs twice because it needs to execute source to target and target to source
        # Output result is hypergraph convolution result
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        return out


    # Message calculates the message received by each node in message passing
    # Multiply input node features by hyperedge normalized weights
    # And reorganize results according to number of heads and output channels
    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j####
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        return out
    
    # Forward is a wrapper for __forward__ method, passing input node features and hypergraph, returning hypergraph convolution result
    def forward(self, x, hyperedge_index, adj, hyperedge_weight=None):
        r"""
        Args:
            x (Tensor): Node feature matrix :math:`\mathbf{X}`
            hyper_edge_index (LongTensor): Hyperedge indices from
                :math:`\mathbf{H}`.
            hyperedge_weight (Tensor, optional): Sparse hyperedge weights from
                :math:`\mathbf{W}`. (default: :obj:`None`)
        """
        adj1=adj
        x = torch.matmul(x, self.weight)
        alpha = None
        B,L,D=x.shape
        scale=1./sqrt(D)

        adj = torch.tensor(adj).to(x.device)
        for i in range(x.size(0)):
            x_new = x[i, :, :]
            hyperedge_index_new = hyperedge_index[i, :, :]
            if self.use_attention:
                x_new1 = x_new
                x_new = x_new.view(x_new.size(0), self.heads, -1) 
                x_i= x_new[hyperedge_index_new[0]]

                edges=hyperedge_index_new[1]
                
                # Non-repetitive aggregation
                # node-edge
                unique_edges, inverse_indices = torch.unique(edges,return_inverse=True)
                aggregated_values = torch.stack([torch.sum(x_i[edges == edge]) for edge in unique_edges])
                aggregated_values=aggregated_values.unsqueeze(0)
                
                # Attention calculation between hyperedges
                attention_weights = (torch.softmax(aggregated_values * adj, dim=-1))

                attended_value = torch.einsum("hh,hb->hb", attention_weights, aggregated_values.T)
                attended_value=attended_value.to(torch.float32)
                attended_value = self.dropout1(self.W(attended_value))
                restored_edges = attended_value[inverse_indices]
                x_j=restored_edges.view(hyperedge_index_new.size(1), self.heads, -1)

                ### edge-node
                alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
                alpha = F.leaky_relu(alpha, self.negative_slope)
                alpha = softmax(alpha, hyperedge_index_new[0], x_new.size(0))

                alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            out_batch = self.__forward__(x_new, hyperedge_index_new, adj, hyperedge_weight, alpha)
            out_batch = out_batch.view(-1, 1, self.out_channels)

            if i==0:
                out=out_batch
            else:
                out=torch.cat((out,out_batch),1)
        out=out.transpose(0,1)


        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class Predictor(nn.Module):

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data):
        out = self.linear(data)
        out = out
        return out


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, self.W(key).transpose(-2, -1))  # Calculate attention scores

        if mask is not None:
            scores.masked_fill_(mask == 0, -float('inf'))  # Set invalid positions to negative infinity

        attention_weights = torch.softmax(scores, dim=-1)  # Calculate attention weights
        attended_value = torch.matmul(attention_weights, value)  # Weighted sum of value based on weights

        return attended_value, attention_weights

class HypergraphConv(MessagePassing):
    def __init__(self, configs, use_attention=True, heads=1, concat=True, negative_slope=0.2, dropout=0.1, bias=False):
        super(HypergraphConv, self).__init__(aggr='add')
        self.soft=nn.Softmax(dim=0)
        self.in_channels = configs.enc_in
        self.d_model = configs.d_model
        self.out_channels = configs.dec_in
        self.use_attention = use_attention
        self.W = nn.Linear(1, configs.enc_in)
        self.dropout1 = nn.Dropout(0.1)
        # self.fc = nn.Linear(self.out_channels * heads, self.out_channels)
        ## hyperedge_attention
        self.W_query=nn.Linear(configs.enc_in,configs.enc_in)
        self.W_key=nn.Linear(configs.enc_in,configs.enc_in)
        self.W_value = nn.Linear(configs.enc_in,configs.enc_in)
        self.C=500


        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(torch.Tensor(self.d_model, configs.dec_in))
            # self.att1 = Parameter(torch.Tensor(1, heads, int(out_channels / heads)))
            self.att = Parameter(torch.Tensor(1, heads, 2 * int(configs.dec_in / heads))) # Check if input/output dimensions correspond if error occurs
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(self.d_model, configs.dec_in))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * configs.dec_in))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(configs.dec_in))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    # Initialize weight and bias parameters
    def reset_parameters(self):
        glorot(self.weight)
