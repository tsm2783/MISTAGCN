# This is an implementation of STGCN in the following paper:
# [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting, (IJCAI-18)](...).

import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn

from share import K, eps
from share import get_max_eigenvalue, merge_list_mx_ndarray


class TemporalGatedConv(mx.gluon.Block):
    r'''
    This is temporal gated convolution layer in spatio temporal graph convolutional networks (STGCN), where\
        N is the number of nodes,\
        C is the number of input channels,\
        T is the number of input time steps,\
        C_out is the number of output channels,\
        T_out is the number of ouput time steps.
    '''
    def __init__(self, T,  C_out, T_out, **kwargs):
        super(TemporalGatedConv, self).__init__(**kwargs)
        self.C_out = C_out
        # (batch_size, in_channels, width) -> (batch_size, channels, out_width)
        # out_width = width-kernel_size+1
        self.conv = nn.Conv2D(channels=2*C_out, kernel_size=(1, T-T_out+1))
        self.linear1 = nn.Dense(T_out, flatten=False)
        self.linear2 = nn.Dense(T_out, activation='sigmoid', flatten=False)
        self.conv1 = nn.Conv2D(channels=C_out, kernel_size=(1, T-T_out+1))

    def forward(self, x):
        # x (num, N, C, T)
        # out (num, N, C_out, T_out)
        x = nd.transpose(x, axes=(0,2,1,3)) #num, C, N, T
        x1 = self.conv(x)
        x11 = self.linear1(x1[:, :self.C_out, :, :])  #glu: lin1(x) * sigma(lin2(x))
        x12 = self.linear2(x1[:, -self.C_out:, :, :])
        x = x11 * x12 + self.conv1(x)
        out = nd.transpose(x, axes=(0,2,1,3))
        return out


# class SpaticalGraphConv_1st(mx.gluon.Block):
#     r'''
#     This is spatial graph conv block in spatio temporal graph convolutional networks (STGCN), where\
#         A is the weighted ajacency matrix of the graph,\
#         N is the number of nodes,\
#         C is the number of input channels,\
#         T is the number of input time steps,\
#         C_out is the number of output channels,\
#         T_out is the number of ouput time steps,\
#         Theta contains the parameters for GCN with Chebyshev ploynomials.
#     '''
#     def __init__(self, A, N, C, T, C_out, T_out, **kwargs):
#         super(STConvBlock, self).__init__(**kwargs)
#         self.N = N
#         self.C = C
#         self.C_out = C_out
#         self.T_out = T_out
#         # spatial graph conv
#         self.cheb_p = self.gen_cheb_p(A)
#         with self.name_scope():
#             self.Theta = self.params.get('Theta', shape=(K, T, T_out), init=mx.init.Xavier())

#     def forward(self, x):
#         # x (N, C, T)
#         # out (N, C_out, T_out)
#         ctx = x.ctx
#         Theta = self.Theta.data(ctx)
#         out = []
#         for co in range(self.C_out):
#             y = nd.zeros(shape=(self.N, self.T_out))
#             for c in range(self.C):
#                 for k in range(K):
#                     y = y + nd.dot(nd.dot(self.cheb_p, x[:,c,:]), Theta[k])
#             out.append(y)
#         out = merge_list_mx_ndarray(out)
#         out = nd.transpose(out, axes=(1,0,2))
#         return out

#     def gen_cheb_p(self, A):
#         # first order approximation of Chebyshev Polynomial
#         N = self.N
#         ctx = A.ctx
#         A_tilde = A + nd.eye(N, ctx=ctx)
#         D_tilde = nd.sum(A_tilde, axis=1)
#         D_tilde_12 = 1 / (nd.sqrt(D_tilde) + eps)
#         cheb_p = D_tilde_12 * A_tilde * D_tilde_12
#         return cheb_p


class SpaticalGraphConv(mx.gluon.Block):
    r'''
    This is spatial graph conv block in spatio temporal graph convolutional networks (STGCN), where\
        A is the weighted ajacency matrix of the graph,\
        N is the number of nodes,\
        C is the number of input channels,\
        T is the number of input time steps,\
        C_out is the number of output channels,\
        T_out is the number of ouput time steps,\
        Theta contains the parameters for GCN with Chebyshev ploynomials.
    '''
    def __init__(self, A, T, C_out, T_out, **kwargs):
        super(SpaticalGraphConv, self).__init__(**kwargs)
        self.C_out = C_out
        self.T_out = T_out
        # spatial graph conv
        self.cheb_p = self.gen_cheb_p(A)
        with self.name_scope():
            self.Theta = self.params.get('Theta', shape=(K, T, T_out), init=mx.init.Xavier())

    def forward(self, x):
        # x (num, N, C, T)
        # out (num, N, C_out, T_out)
        ctx = x.ctx
        num, N, C, T = x.shape
        C_out, T_out = self.C_out, self.T_out
        cheb_p = self.cheb_p.as_in_context(ctx)
        Theta = self.Theta.data(ctx)
        out = []
        x = nd.transpose(x, axes=(1,0,2,3))  #N, num, C, T
        for co in range(C_out):
            y = nd.zeros(shape=(N, num, T_out), ctx=ctx)
            for c in range(C):
                for k in range(K):
                    y = y + nd.dot(nd.dot(cheb_p[k], x[:,:,c,:]), Theta[k])
            out.append(y)
        out = merge_list_mx_ndarray(out) #C_out, N, num, T_out
        out = nd.transpose(out, axes=(2,1,0,3)) #num, N, C_out, T_out
        return out

    def gen_cheb_p(self, A):
        '''
        Generate Chebshev Poloynomial from matrix A.
        '''
        N = len(A)
        D = nd.sum(A, axis=1)
        D_12 = 1 / (nd.sqrt(D) + eps)
        L = nd.eye(N, ctx=A.ctx) - (D_12 * A) * D_12  #element wise multiply
        lambda_max = get_max_eigenvalue(L)
        L_tilde = 2 / lambda_max * L - nd.eye(N, ctx=A.ctx)
        cheb_p = [nd.eye(N, ctx=A.ctx), L_tilde]
        for k in range(2, K):
            cheb_p.append(2 * L_tilde * cheb_p[-1] - cheb_p[-2])
        cheb_p = merge_list_mx_ndarray(cheb_p)  #K, N, N
        return cheb_p


class STConvBlock(mx.gluon.Block):
    r'''
    This is spatial-temporal block in spatio temporal graph convolutional networks (STGCN), where\
        A is the weighted ajacency matrix of the graph,\
        N is the number of nodes,\
        C is the number of input channels,\
        T is the number of input time steps,\
        C_out is the number of output channels,\
        T_out is the number of ouput time steps,\
    '''
    def __init__(self, A, T, T_out, **kwargs):
        super(STConvBlock, self).__init__(**kwargs)
        self.temp_conv1 = TemporalGatedConv(T, 16, int(.75*T + .25*T_out)) #T>T_out
        self.spatial_graph_conv = SpaticalGraphConv(A, int(.75*T + .25*T_out), 8, int(.25*T + .75*T_out))
        self.temp_conv2 = TemporalGatedConv(int(.25*T + .75*T_out), 16, T_out) #T>T_out

    def forward(self, x):
        # x (num, N, C, T)
        # out (num, N, C_out, T_out)
        x = self.temp_conv1(x)
        x = self.spatial_graph_conv(x)
        x = nd.relu(x)
        x = self.temp_conv2(x)
        out = nd.L2Normalization(x, eps)  #introduces to prevent overfitting
        return out

class STGCN(mx.gluon.Block):
    r'''
    This is spatial-temporal block in spatio temporal graph convolutional networks (STGCN), where\
        A is the weighted ajacency matrix of the graph,\
        N is the number of nodes,\
        T is the number of input time steps,\
        T_out is the number of ouput time steps.
    '''
    def __init__(self, A, T, T_out, **kwargs):
        super(STGCN, self).__init__(**kwargs)
        assert T>T_out, 'The number of prediction time steps should be no greater than number of history time steps!'
        self.st_block1 = STConvBlock(A, T, int(.5*T + .5*T_out))
        self.st_block2 = STConvBlock(A, int(.5*T + .5*T_out), T_out)
        self.linear = nn.Dense(T_out, flatten=False)

    def forward(self, x):
        # x (num, N, T)
        # out (num, N, Tp)
        x = nd.expand_dims(x, axis=2)  #num, N, 1, T
        num, N, C, T = x.shape
        x = self.st_block1(x)
        x = self.st_block2(x)
        out = self.linear(nd.reshape(x, shape=(num, N, -1)))
        return out