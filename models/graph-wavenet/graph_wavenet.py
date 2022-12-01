# This is an implementation of Graph WaveNet in the following paper:
# [Graph WaveNet for Deep Spatial - Temporal Graph Modeling, IJCAI 2019](https: // arxiv.org / abs / 1906.00121).

import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn

from share import eps


class Linear(nn.Block):
    '''An implementation of linear layer.

    Args:
        units (int): dimensionality of the output space.

    Input:
        **data**: data should have shape (x1, x2, ..., xn, in_units).

    Input:
        **out**: out will have shape (x1, x2, ..., xn, units).
    '''

    def __init__(self, units):
        super(Linear, self).__init__()
        self.blk = nn.Dense(units, flatten=False)

    def forward(self, x):
        return self.blk(x)


class TemporalConvNet(nn.Block):
    '''An implementation of temporal convolution network (TCN).

    Args:
        channels (int): the number of ouput channels.
        layers (int): the number of layers of TCN.
        kernel_size (int): convolution kernel size.

    Input:
        **data**: 3D input tensor with shape (batch_size, in_channels, width) when layout is NCW.

    Output:
        **out**: 3D output tensor with shape (batch_size, channels, out_width) when layout is NCW. out_width is calculated as::

        out_width_{i+1} = floor((width_i-dilation*(kernel_size-1)-1)/strides)+1
    '''

    def __init__(self, channels, layers, kernel_size):
        super(TemporalConvNet, self).__init__()
        dilation = 1
        self.blk = nn.Sequential()
        with self.blk.name_scope():
            for i in range(layers):
                self.blk.add(nn.Conv1D(channels=channels, kernel_size=kernel_size,
                             dilation=dilation, strides=kernel_size * dilation))
                dilation = dilation * 2

    def forward(self, x):
        x = self.blk(x)
        return x


class GatedTCN(nn.Block):
    '''An implementation of gated TCN.

    Args:
        channels (int): the number of ouput channels.
        layers (int): the number of layers of TCN.
        kernel_size (int): convolution kernel size.

    Input:
        **data**: 3D input tensor with shape (batch_size, in_channels, width) when layout is NCW.

    Output:
        **out**: 3D output tensor with shape (batch_size, channels, out_width) when layout is NCW. out_width is calculated as::

        out_width_{i+1} = floor((width_i-dilation*(kernel_size-1)-1)/strides)+1
    '''

    def __init__(self, channels, layers, kernel_size):
        super(GatedTCN, self).__init__()
        self.tcn_a = TemporalConvNet(channels, layers, kernel_size)
        self.tcn_b = TemporalConvNet(channels, layers, kernel_size)

    def forward(self, x):
        a = self.tcn_a(x)
        b = self.tcn_b(x)
        out = nd.tanh(a) * nd.sigmoid(b)
        return out


class GCN(nn.Block):
    '''An implementation of graph convolution network (GCN).

    Args:
        A (ndarray): adjacency matrix of the graph.
        units (in): dimensionality of the output space.
        layers (int): number of layers in GCN.
        dim_embed (int): dimensionality of the embed space.

    Input:
        **data**: 3D input tensor with shape (batch_size, num_nodes, in_units) when layout is NCW.

    Output:
        **out**: 3D output tensor with shape (batch_size, num_nodes, units) when layout is NCW.
    '''

    def __init__(self, A, units, layers, dim_embed):
        super(GCN, self).__init__()
        self.layers = layers
        self.units = units
        self.dim_embed = dim_embed
        N = A.shape[0]
        A_tilde = A + nd.eye(N, ctx=A.ctx)
        D_tilde = 1 / (nd.sum(A_tilde, axis=1) + eps)
        self.P = A_tilde / D_tilde
        with self.name_scope():
            self.W0 = self.params.get('W0', allow_deferred_init=True, init=mx.init.Uniform())
            self.W0_adp = self.params.get('W0_adp', allow_deferred_init=True, init=mx.init.Uniform())
            self.Ws = self.params.get('Ws', allow_deferred_init=True, init=mx.init.Uniform())
            self.Ws_adp = self.params.get('Ws_adp', allow_deferred_init=True, init=mx.init.Uniform())
            self.E1 = self.params.get('E1', allow_deferred_init=True, init=mx.init.Uniform())
            self.E2 = self.params.get('E2', allow_deferred_init=True, init=mx.init.Uniform())

    def forward(self, x):
        batch_size, num_nodes, in_units = x.shape
        self.W0.shape = (in_units, self.units)
        self.W0_adp.shape = (in_units, self.units)
        self.Ws.shape = (self.layers-1, self.units, self.units)
        self.Ws_adp.shape = (self.layers-1, self.units, self.units)
        self.E1.shape = (num_nodes, self.dim_embed)
        self.E2.shape = (num_nodes, self.dim_embed)
        for param in [self.W0, self.W0_adp, self.Ws, self.Ws_adp, self.E1, self.E2]:
            param._finish_deferred_init()
        P = self.P.as_in_context(x.ctx)
        A_tilde_adp = nd.dot(self.E1.data(x.ctx), nd.transpose(self.E2.data(x.ctx)))  # (N,N)
        A_tilde_adp = nd.softmax(nd.relu(A_tilde_adp), axis=1)
        P_adp = A_tilde_adp
        x = nd.transpose(x, axes=(1, 0, 2))  # (num_nodes, batch_size, in_featurs)
        x = nd.dot(nd.dot(P, x), self.W0.data(x.ctx)) + nd.dot(nd.dot(P_adp, x), self.W0_adp.data(x.ctx))
        for i in range(1, self.layers):
            P = P * self.P.as_in_context(x.ctx)
            P_adp = P_adp * A_tilde_adp
            x = nd.dot(nd.dot(P, x), self.Ws.data(x.ctx)[i-1]) + nd.dot(nd.dot(P_adp, x), self.Ws_adp.data(x.ctx)[i-1])
        x = nd.transpose(x, axes=(1, 0, 2))  # (batch_size, num_nodes, in_featurs)
        return x


class ResidualBlock(nn.Block):
    '''An implementation of the residual block as shown in figure 3.

    Args:
        channels (int): the number of ouput channels.
        tcn_layers (int): the number of layers of TCN.
        A (ndarray): adjacency matrix of the graph.
        units (in): dimensionality of the output space.
        gcn_layers (int): number of layers in GCN.
        dim_embed (int): dimensionality of the embed space.

    Input:
        **data**: 3D input tensor with shape (batch_size, in_channels, in_units) when layout is NCW.

    Output:
        **out**: 3D output tensor with shape (batch_size, channels, units) when layout is NCW.
    '''

    def __init__(self, channels, tcn_layers, kernel_size, A, units, gcn_layers, dim_embed):
        super(ResidualBlock, self).__init__()
        self.gated_tcn = GatedTCN(channels, tcn_layers, kernel_size)
        self.gcn = GCN(A, units, gcn_layers, dim_embed)
        self.linear = Linear(units)

    def forward(self, x):
        out = self.gated_tcn(x)
        out = self.gcn(out)
        out = out + self.linear(x)
        return out


class GraphWaveNet(nn.Block):
    '''An implementation of graph wavenet.

    Args:
        N (int): the number of nodes in graph (number of output channels).
        A (ndarray): adjacency matrix of the graph.
        Tp (int): The number of future time steps (number of features in the ouput).
        tcn_layers (int): the number of layers of TCN.
        kernel_size (int): convolution kernel size.
        gcn_layers (int, optional): number of layers in GCN.
        dim_embed (int, optional): dimensionality of the embed space.
        rb_layers (int, optional): the number of parallel residual block layers.

    Input:
        **data**: 4D input tensor with shape (batch_size, N, F, T).

    Output:
        **out**: 3D output tensor with shape (batch_size, N, Tp).
    '''

    def __init__(self, N, A, Tp, seq_len=300, tcn_layers=3, kernel_size=2, gcn_layers=3, dim_embed=32, rb_layers=5):
        super(GraphWaveNet, self).__init__()
        self.rb_layers = rb_layers
        self.rbs = []
        for i in range(rb_layers):
            self.rbs.append(ResidualBlock(N, tcn_layers, kernel_size, A, Tp, gcn_layers, dim_embed))
            self.register_child(self.rbs[-1])
        self.linear0 = Linear(seq_len * rb_layers)
        self.linear1 = Linear(Tp)
        self.linear2 = Linear(Tp)

    def forward(self, x):
        batch_size, N, F, T = x.shape
        x = nd.reshape(x, shape=(batch_size, N, -1))
        x = self.linear0(x)  # (batch_size, N, seq_len * rb_layers)
        x = nd.reshape(x, shape=(batch_size, N, -1, self.rb_layers))  # (batch_size, N, seq_len, rb_layers)
        x_list = [self.rbs[i](x[...,i]) for i in range(self.rb_layers)]
        x = nd.add_n(*x_list)
        x = nd.relu(x)
        x = self.linear1(x)
        x = nd.relu(x)
        x = self.linear2(x)
        return x
