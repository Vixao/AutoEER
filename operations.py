import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from scipy.sparse.linalg import eigs
import scipy.sparse as sp
import numpy as np
from torch.nn.modules.module import Module
import torch
import torch.nn as nn
from torch.nn import Parameter
from enum import IntEnum
import h5py
import torch
import torch.nn as nn
from torch.nn import Parameter
from enum import IntEnum
from utils import exp_mask_for_high_rank

OPS = {
  'none' : lambda C : Zero(),
  'skip_connect' : lambda C : Identity(),
  'cnn' : lambda C : CNN(C, C, (1, 3), 1),
  'lstm': lambda C : LSTM(C, C),
  'scnn' : lambda C : SCNN(C, C, (3, 3), 1),
  'dgcn': lambda C, cheb, nodevec1, nodevec2, alpha: Dgcn(2, cheb, C, C, nodevec1, nodevec2, alpha),
  'lgg':lambda C : LGGNet(C, C),
  'channelWiseAttention':lambda C : ChannelWiseAttention(C),
  'multi_dimensional_attention':lambda C : Multi_Dimensional_Attention(C),
  'temporal_att': lambda C: TransformerLayer(C),
  'spatial_att': lambda C: SpatialTransformerLayer(C),
}

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class LSTM_batchfirst(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_i = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(torch.Tensor(hidden_size))

        self.W_f = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_f = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(torch.Tensor(hidden_size))

        self.W_g = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_g = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(torch.Tensor(hidden_size))

        self.W_o = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_o = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(torch.Tensor(hidden_size))


        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def _init_states(self, x):
        h_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=x.dtype).to(x.device)
        c_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=x.dtype).to(x.device)
        return h_t, c_t

    def forward(self, x, init_states=None):
        batch_size, seq_size, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = self._init_states(x)
        else:
            h_t, c_t = init_states

        for t in range(seq_size):
            x_t = x[:, t, :]

            i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
            g_t = torch.tanh(x_t @ self.W_g + h_t @ self.U_g + self.b_g)
            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t)
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)

class linear(nn.Module):
    """
    Linear for 2d feature map
    """
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))  # bias=True

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        return self.mlp(x)

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class SCNN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, padding, affine=False):
      super(SCNN, self).__init__()
      self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, 64, kernel_size=kernel_size, stride=1, padding=1, bias=False),
        nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=1, bias=False),
        nn.Conv2d(128, 256, kernel_size=kernel_size, stride=1, padding=1, bias=False),
        nn.Conv2d(256, C_in, kernel_size=kernel_size, stride=1, padding=0, bias=False),
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=1, bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
      )
    def forward(self, x):
      return self.op(x)

class CNN(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride=1, dilation=1):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.filter_conv = CausalConv2d(c_in, c_out, kernel_size, stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(c_out, affine=False)

    def forward(self, x):

        x = self.relu(x)
        output = (self.filter_conv(x))
        output = self.bn(output)

        return output

class CausalConv2d(nn.Conv2d):
    """
    单向padding
    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self._padding = (kernel_size[-1] - 1) * dilation
        super(CausalConv2d, self).__init__(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=(0, self._padding),
                                           dilation=dilation,
                                           groups=groups,
                                           bias=bias)

    def forward(self, input):
        result = super(CausalConv2d, self).forward(input)
        if self._padding != 0:
            return result[:, :, :, :-self._padding]
        return result

class LSTM(nn.Module):
    def __init__(self, c_in, c_out):
        super(LSTM, self).__init__()
        self.lstm = LSTM_batchfirst(c_in*4, c_in*4)

    def forward(self, x):
        """
        :param x: [batch_size, channel f_in, band N, T]
        :return:
        """
        b, C, N, T = x.shape
        x = x.permute(0, 3, 1, 2)  # [b, T, f_in, band]
        x = x.reshape(b, T, -1)  # [b, T, 4*f_in (band * channel) ) ]
        output, state = self.lstm(x)
        output = output.reshape(b, T, C, N)
        output = output.permute(0, 2, 3, 1)
        return output

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self):
    super(Zero, self).__init__()

  def forward(self, x):
    return x.mul(0.)


class Dgcn(nn.Module):
    """
    K-order chebyshev graph convolution layer
    """

    def __init__(self, K, cheb_polynomials, c_in, c_out, nodevec1, nodevec2, alpha):
        super(Dgcn, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.c_in = c_in
        self.c_out = c_out
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.alpha = alpha
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(c_out, affine=False)
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Theta = nn.ParameterList(  # weight matrices
            [nn.Parameter(torch.FloatTensor(4, 4).to(self.DEVICE)) for _ in range(K)])
        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K):
            self.theta_k = self.Theta[k]
            nn.init.xavier_uniform_(self.theta_k)
    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return: [batch_size, f_out, N, T]
        """
        x = self.relu(x) 

        batch_size, num_nodes, c_in, timesteps = x.shape
        adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=1)

        outputs = []
        for step in range(timesteps):
            graph_signal = x[:, :, :, step]  
            output = torch.zeros(
                batch_size, num_nodes, 4).to(self.DEVICE)  

            for k in range(self.K):
                alpha, beta = F.softmax(self.alpha[k] , dim=0)
                T_k = alpha * self.cheb_polynomials[k] + beta * adp
                self.theta_k = self.Theta[k]  
                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                output = output + rhs.matmul(self.theta_k)  

            outputs.append(output.unsqueeze(-1))
        outputs = F.relu(torch.cat(outputs, dim=-1)) # Concatenate the output of each time step
        outputs = self.bn(outputs)  

        return outputs

class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=False):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

class linear(nn.Module):
    """
    Linear for 2d feature map
    """
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))  # bias=True

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        return self.mlp(x)
        
def scaled_Laplacian(W):
    """
    compute ilde{L}
    :param W: adj_mx
    :return: scaled laplacian matrix
    """
    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real  # k largest real part of eigenvalues

    return (2 * L) / lambda_max - np.identity(W.shape[0])

def cheb_polynomial(L_tilde, K):
    """
    compute a list of chebyshev polynomials from T_0 to T{K-1}
    :param L_tilde: scaled laplacian matrix
    :param K: the maximum order of chebyshev polynomials
    :return: list(np.ndarray), length: K, from T_0 to T_{K-1}
    """
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i-1] - cheb_polynomials[i-2])

    return cheb_polynomials

def asym_adj(adj):
    '''
    The purpose of this function is to normalize the adjacency matrix so that it can be used in a diffusion convolution layer. 
    This allows for more accurate calculations when performing convolutions on graphs.
    '''
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight)-self.bias
        output = F.relu(torch.matmul(adj, output))
        return output


class LGGNet(nn.Module):
    def __init__(self, c_in, c_out):
        # input_size: EEG frequency x channel x datapoint
        super(LGGNet, self).__init__()
        #idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format('gen'), 'r')['data']))
        #channels = sum(idx_local_graph)
        channels = c_in
        #self.idx = idx_local_graph
        self.channel = channels
        self.brain_area = self.channel
        self.c_in= c_in
        self.c_out= 4
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # diag(W) to assign a weight to each local areas
        size = (1, self.channel , 4)
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)
        # aggregate function
        #self.aggregate = Aggregator(self.idx)

        # trainable adj weight for global network
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)
        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(self.brain_area)
        self.bn_ = nn.BatchNorm1d(self.brain_area)
        # learn the global network of networks
        self.GCN = GraphConvolution(size[-1], 4)

    def forward(self, x):
        batch_size, num_nodes, c_in, timesteps = x.shape
        outputs = []
        for step in range(timesteps):
            out = x[:, :, :, step]
            out = torch.reshape(out, (out.size(0), out.size(1), -1))
            out = self.local_filter_fun(out, self.local_filter_weight)
            adj = self.get_adj(out)
            out = self.bn(out)
            out = self.GCN(out, adj)
            out = self.bn_(out)
            outputs.append(out.unsqueeze(-1))
        outputs = torch.cat(outputs, dim=-1) # Concatenate the output of each time step
        return outputs

    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_adj(self, x, self_loop=True):
        # x: b, node, feature
        adj = self.self_similarity(x)   # b, n, n
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(self.DEVICE)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def self_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s

class ChannelWiseAttention(nn.Module):
    def __init__(self, C, weight_decay=0.00004):
        super(ChannelWiseAttention, self).__init__()
        self.C = C
        self.weight_decay = weight_decay
        self.weight = nn.Parameter(torch.empty(C, C), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(C), requires_grad=True)
        nn.init.orthogonal_(self.weight)

    def forward(self, feature_map):
        B, C, H, W = feature_map.shape
        # L2 regularization of weight and bias
        l2_reg = torch.norm(self.weight) + torch.norm(self.bias)
        l2_reg = l2_reg * self.weight_decay
        
        # Averaging the feature map along the spatial dimension
        mean_feature_map = torch.mean(feature_map, dim=[2, 3], keepdim=True)#(B,C,1,1)
        
        # Linear transformation for each channel
        channel_wise_attention_fm = torch.matmul(mean_feature_map.view(-1, self.C), self.weight) + self.bias#(B,C)
        
        # Apply sigmoid activation function to get the attention weights
        channel_wise_attention_fm = torch.sigmoid(channel_wise_attention_fm)
        
        # Extending attention weights to spatial dimensions
        attention = channel_wise_attention_fm.view(-1,self.C , 1).repeat(1 , 1 , H * W).view(-1,self.C, H, W)
        
        # Multiplying attention with the original feature map
        attended_fm = attention * feature_map
        
        return attended_fm

class Multi_Dimensional_Attention(nn.Module): 
    def __init__(self, C): 
        super(Multi_Dimensional_Attention, self).__init__()
        ivec = C * 4
        self.C = C
        self.linear = nn.Linear(ivec, ivec, bias=True) 
        self.bn = nn.BatchNorm1d(ivec)
        self.rep_mask = 64
        # Define the activation function 
        self.activation = F.elu 


    def forward(self, rep_tensor):
        #print("rep_tensor.shape",rep_tensor.shape)
        rep_tensor = rep_tensor.reshape(rep_tensor.shape[0],-1,rep_tensor.shape[3])
        #print("rep_tensor.shape",rep_tensor.shape)
        rep_tensor = rep_tensor.permute(0, 2, 1)
        bs, sl, fs = rep_tensor.shape
        #print("rep_tensor.shape",rep_tensor.shape)
        # Compute the attention map
        map1 = self.linear(rep_tensor)
        map1 = self.bn( map1.reshape (-1, fs) )
        map1 = self.activation(map1.reshape(bs, sl, fs))
        map2 = self.linear(map1)
        map2 = self.bn(map2.reshape (-1, fs))
        map2 = map2.reshape(bs, sl, fs)
   
        # Mask out the padded values
        #map2_masked = exp_mask_for_high_rank(map2, self.rep_mask)

        # Apply softmax to get attention weights
        soft = F.softmax(map2, dim=1)
        # Compute the weighted sum of rep_tensor
        attn_output = soft * rep_tensor
        #print("attn_output.shape",attn_output.shape)
        output = attn_output.permute(0, 2, 1)
        output = output.reshape(output.shape[0], self.C, 4, 4)
        #print("output.sahpe",output.shape)

        return output

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(TransformerLayer, self).__init__()
        # Initialization parameters
        # d_model: dimensions of inputs and outputs
        # d_ff: dimension of the middle layer
        # dropout: probability of dropout
        # n_heads: number of attention heads
        # activation: type of activation function
        # output_attention: whether to return attention weights
        d_model = d_model * 4
        self.attention = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.d_model = d_model

    def forward(self, x, attn_mask=None):
        b, C, N, T = x.shape
        x = x.permute(0, 3, 1, 2)  # [b, T, f_in, band]
        x = x.reshape(b, T, -1)  # [b, T, 4*f_in (band * channel) ) ]
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        # A dropout operation is performed on the new sequence and added to the original sequence to realize the residual join
        x = x + self.dropout(new_x)
        
        y = x = self.norm1(x) 
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, T, C, N)
        output = output.permute(0, 2, 3, 1)

        return output

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape 
        _, S, _ = keys.shape 
        H = self.n_heads 

        # queries = queries.transpose(-1, 1)
        # keys = keys.transpose(-1, 1)
        # values = values.transpose(-1, 1)

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        
        if self.mix:
            out = out.transpose(2, 1).contiguous() 
        
        out = out.view(B, L, -1) 

        return self.out_projection(out), attn 
    
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()

        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape 
        _, S, _, D = values.shape 
        scale = self.scale or 1. / math.sqrt(E) 

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) 
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L) 

            scores.masked_fill_(attn_mask.mask, -np.inf) 

        A = self.dropout(torch.softmax(scale * scores, dim=-1)) 
        V = torch.einsum("bhls,bshd->blhd", A, values) 

        if self.output_attention:
            return (V.contiguous(), A) 
        else:
            return (V.contiguous(), None) 
        
class TriangularCausalMask():
    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(DEVICE)

    @property
    def mask(self):
        return self._mask

class SpatialTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(SpatialTransformerLayer, self).__init__()
        self.attention = SpatialAttentionLayer(
            SpatialFullAttention(attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.d_model = d_model

    def forward(self, x):
        b, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1)  # [64, 4, 4, 32] b, T, band, C
        x = x.reshape(-1, N, C)  # [64*4, 4, 32]
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, -1, N, C)
        output = output.permute(0, 3, 2, 1)

        return output
    
class SpatialAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(SpatialAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        # shape=[b*T, N, C]
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class SpatialFullAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(SpatialFullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
