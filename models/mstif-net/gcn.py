import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd
from mxnet.gluon.parameter import Parameter
from mxnet.ndarray.ndarray import eye
from mxnet.numpy.multiarray import zeros_like
import mxnet.numpy as mnp

from share import *

class GCN(mx.gluon.Block):
    r'''
    This is a graph convolution network block, where\
        ajacency matrix adj can be ADG, AIG or ACG (ndarray),\
        K is the order of Chebyshev polynomial (int),\
        n_filter1 and n_filter2 are the number of features of GCN layer1 and layer2, respectively,\
        n_filter is the number of features of the last 'linear+relu' layer (int).
    '''
    def __init__(self, adj, n_filter1, n_filter2, n_filter, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.adj = adj
        self.n_filter1 = n_filter1
        self.n_filter2 = n_filter2
        # Chebyshev polynomials and parameters
        A = adj
        N = len(A)
        D = nd.sum(A, axis=1)
        eps = 0.0001
        D_12 = 1 / (D + eps)
        self.L = eye(N, ctx=A.ctx) - (D_12 * A) * D_12  #element wise multiply
        # parameter and layers to be trained
        self.Theta1 = self.params.get('Theta1', allow_deferred_init=True)
        self.Theta2 = self.params.get('Theta2', allow_deferred_init=True)
        self.linear_relu = nn.Dense(n_filter, activation='relu')

    def forward(self, x):
        r'''
        x is the input (ndarray, n x N x T),\
        out is the output of this block, the latent variables (ndarray, n x n_filter).
        '''
        #number of samples, number of nodes, time slices
        batch_size, N, T = x.shape
        # defer the shape of params
        self.Theta1.shape = (T, self.n_filter1)
        self.Theta2.shape = (self.n_filter1, self.n_filter2)
        for param in [self.Theta1, self.Theta2]:
            param._finish_deferred_init()
        # process
        L = self.L.as_in_context(x.ctx)
        out = nd.sigmoid(nd.dot(nd.dot(L, nd.transpose(x, axes=(1,0,2))), self.Theta1.data(ctx=x.ctx)))   #(N,n,n_filter1)
        out = nd.sigmoid(nd.dot(nd.dot(L, out), self.Theta2.data(ctx=out.ctx)))   #(N,n,n_filter2)
        out = nd.transpose(out, axes=(1,0,2))  #(n,N,n_filter2)
        out = self.linear_relu(out)
        # return
        return out