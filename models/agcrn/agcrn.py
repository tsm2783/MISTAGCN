# This is an implementation of AGCRN in the following paper:
# [Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting, NeurIPS 2020](...).

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

from share import merge_list_mx_ndarray


class AGCRN_Layer(nn.Block):
    r'''
    This is implement of adaptive-graph-convolutional-recurrent-network (AGCRN),\
    according to formula (7) in the ariginal paper, with parameters\
        num_nodes: number of nodes in graph.\
        dim_hidden: dimension of output (the hidden state) at each node, h_t.shape[-1], also h_{t-1}.shape[-1].\
        dim_embed: dimension of the embedded feature vector at each node, E.shape[-1].\
    '''
    def __init__(self, num_nodes, dim_embed, dim_hidden, **kwargs):
        super(AGCRN_Layer, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.dim_hidden = dim_hidden
        self.linear = nn.Dense(dim_hidden, flatten=False)
        with self.name_scope():
            self.E = self.params.get('E', shape=(num_nodes, dim_embed), allow_deferred_init=True)
            self.Wz = self.params.get('Wz', shape=(2*dim_hidden, dim_embed, num_nodes, dim_hidden), allow_deferred_init=True)
            self.bz = self.params.get('bz', shape=(dim_embed, dim_hidden), allow_deferred_init=True)
            self.Wr = self.params.get('Wr', shape=(2*dim_hidden, dim_embed, num_nodes, dim_hidden), allow_deferred_init=True)
            self.br = self.params.get('br', shape=(dim_embed, dim_hidden), allow_deferred_init=True)
            self.Wh = self.params.get('Wh', shape=(2*dim_hidden, dim_embed, num_nodes, dim_hidden), allow_deferred_init=True)
            self.bh = self.params.get('bh', shape=(dim_embed, dim_hidden), allow_deferred_init=True)

    def forward(self, x_state):
        #x: B, num_nodes, dim_in
        #state (h_{t-1}): B, num_nodes, dim_hidden
        #out: x_state, concatation of x and (next) state, along the last dimension.
        if len(x_state.shape) == 2:
            x_state = nd.expand_dims(x_state, axis=0)

        ctx = x_state.ctx
        num = x_state.shape[0]

        E, Wz, bz, Wr, br, Wh, bh = self.E.data(ctx), self.Wz.data(ctx), self.bz.data(ctx), self.Wr.data(ctx), self.br.data(ctx), self.Wh.data(ctx), self.bh.data(ctx)
        A_tild = nd.softmax(nd.relu(nd.dot(E, nd.transpose(E))), axis=-1)  #num_nodes, num_nodes

        out = []
        for i in range(num):
            x_state_temp = x_state[i]
            x = x_state_temp[:, :-self.dim_hidden]
            x = self.linear(x)
            state = x_state_temp[:, -self.dim_hidden:]
            x_state_temp = nd.concat(x, state, dim=-1)

            z = nd.dot(A_tild, x_state_temp)
            z = nd.dot(nd.transpose(z), E)
            z = nd.dot(nd.reshape(z, (-1,)), nd.reshape(Wz, (-1, self.num_nodes, self.dim_hidden)))
            z = nd.sigmoid(z + nd.dot(E, bz))

            r = nd.dot(A_tild, x_state_temp)
            r = nd.dot(nd.transpose(r), E)
            r = nd.dot(nd.reshape(r, (-1,)), nd.reshape(Wr, (-1, self.num_nodes, self.dim_hidden)))
            r = nd.sigmoid(r + nd.dot(E, br))

            x_state_temp = nd.concat(x, r*state, dim=-1)
            h = nd.dot(A_tild, x_state_temp)
            h = nd.dot(nd.transpose(h), E)
            h = nd.dot(nd.reshape(h, (-1,)), nd.reshape(Wh, (-1, self.num_nodes, self.dim_hidden)))
            h = nd.sigmoid(h + nd.dot(E, bh))

            state = z * state + (1-z) * h
            out.append(nd.concat(x, state, dim=-1))
        out = merge_list_mx_ndarray(out)

        return out


class AGCRN(nn.Block):
    r'''
    This is implement of adaptive-graph-convolutional-recurrent-network (AGCRN),\
    according to formula (7) in the ariginal paper.\
    '''
    def __init__(self, num_nodes, dim_embed, dim_hidden, **kwargs):
        super(AGCRN, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.dim_hidden = dim_hidden

        self.net = nn.Sequential()
        with self.name_scope():
            self.net.add(AGCRN_Layer(num_nodes, dim_embed, dim_hidden))
            self.net.add(AGCRN_Layer(num_nodes, dim_embed, dim_hidden))
            self.net.add(AGCRN_Layer(num_nodes, dim_embed, dim_hidden))
            self.net.add(nn.LeakyReLU(0.01))

    def forward(self, x):
        #x: num, num_nodes, dim_input (num, N, T)
        #out: num, num, num_nodes, dim_hidden (num, N, Tp)
        if len(x.shape) == 2:
            x = nd.expand_dims(x, axis=0)
        ctx = x.ctx
        num = x.shape[0]
        state = nd.random.uniform(shape=(num, self.num_nodes, self.dim_hidden), ctx=ctx)
        x_state = nd.concat(x, state, dim=-1)
        out = self.net(x_state)
        out = out[..., -self.dim_hidden:]
        return out
