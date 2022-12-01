import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn

from share import Tr, Td, Tw, Tp, K, eps
from share import merge_list_mx_ndarray, dot, get_max_eigenvalue


class ST_block(mx.gluon.Block):
    r'''
    This is spatial-temporal block in multi information spatial temporal attention graph convolution network (MISTAGCN), where\
        As is an array of ajacency matrices (ndarray, As[i] denotes the i'th adjacency matrix),\
        N is the number of nodes in the graph structure (int),\
        F is the number of features recorded on each node (int),\
        T is the number of time slices in recording a sample X_t (int),\
        F1 is the expected number of features in the output (int),\
        T1 is the expected number of time steps in the output (int),\
        W_1, W_2, W_3, bs are weight parameters of spatial attention mechanism (nd.ndarray),\
        U_1, U_2, U_3, be are weight parameters of temporal attention mechanism (nd.ndarray),\
        Theta contains the parameters for GCN with Chebyshev ploynomial (nd.ndarray).
    '''
    def __init__(self, A, N, F, T, F1, T1, **kwargs):
        super(ST_block, self).__init__(**kwargs)
        self.N = N
        self.F = F
        self.T = T
        self.F1 = F1
        self.T1 = T1
        # Chebyshev polynomials and parameters
        self.cheb_p = self.gen_cheb_p(A)
        # ResNet, the other path to target, (N, F, T) -> (N, F1, T1)
        self.conv = nn.Conv2D(channels=N, kernel_size=(F-F1+1, T-T1+1), activation='relu')
        # parameters to train
        with self.name_scope():
            self.W1 = self.params.get('W1', shape=(T, ), init=mx.init.Uniform())
            self.W2 = self.params.get('W2', shape=(F, ), init=mx.init.Uniform())
            self.W3 = self.params.get('W3', shape=(F, ), init=mx.init.Uniform())
            self.W4 = self.params.get('W4', shape=(T, ), init=mx.init.Uniform())
            self.U1 = self.params.get('U1', shape=(F, ), init=mx.init.Uniform())
            self.U2 = self.params.get('U2', shape=(N, ), init=mx.init.Uniform())
            self.U3 = self.params.get('U3', shape=(N, ), init=mx.init.Uniform())
            self.U4 = self.params.get('U4', shape=(F, ), init=mx.init.Uniform())
            self.Theta = self.params.get('Theta', shape=(K, F, F1), init=mx.init.Uniform())
            self.W_conv = self.params.get('W_conv', shape=(T-T1+1,), init=mx.init.Uniform())
            self.b_conv = self.params.get('b_conv', shape=(N, F1), init=mx.init.Zero())

    def forward(self, x):
        # Input x is of shape (num, N, F, T),\
        # output out is of shape (num, N, F1, T1).
        x1 = self.st_att_gcn(x)
        x1 = self.temporal_leaky_conv(x1)
        out = x1 + self.conv(x)
        return out

    def gen_cheb_p(self, A):
        '''
        Generate Chebshev poloynomidal from matrix A.
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

    def st_att_gcn (self, x):
        r'''
        Combine spatial-temporal attention with gcn, where\
            x: (num, N, F, T)
            out: (num, N, F1, T)
        '''
        ctx = x.ctx
        num, N, F, T = x.shape
        cheb_p = self.cheb_p.as_in_context(ctx)
        W1, W2, W3, W4 = self.W1.data(ctx), self.W2.data(ctx), self.W3.data(ctx), self.W4.data(ctx)
        U1, U2, U3, U4 = self.U1.data(ctx), self.U2.data(ctx), self.U3.data(ctx), self.U4.data(ctx)
        Theta = self.Theta.data(ctx)
        # spatial attention
        xs = nd.transpose(x, axes=(1,0,2,3))
        s = nd.dot(nd.dot(xs, W1), W2)
        s = nd.dot(s, nd.dot(W3, nd.dot(W4, nd.transpose(xs)))) #NxN
        s = nd.softmax(s, axis=1)
        # temporal attention
        xe = nd.transpose(x, axes=(3,0,1,2)) #T,num, N, F
        e = nd.dot(nd.dot(xe, U1), U2)
        e = nd.dot(e, nd.dot(U3, nd.dot(U4, nd.transpose(xe)))) #TxT
        e = nd.softmax(e, axis=0)
        # apply temporal attension
        x = nd.dot(x, e)  #num, N, F, T
        # apply spatial attension GCN
        x = nd.transpose(x, axes=(1,0,3,2)) #N, num, T, F
        out = nd.zeros(shape=(N, num, T, self.F1), ctx=ctx)
        for k in range(K):
            xk = nd.dot((cheb_p[k] * s), x)
            xk = nd.dot(xk, Theta[k])
            out = out + xk
        out = nd.transpose(out, axes=(1,0,3,2)) #num, N, F1, T
        return out

    def temporal_leaky_conv(self, x):
        '''
        This is a 1d convelution function, where
            x: shape(num, N, F, T)
            out: shape(num, N, F, T1)
        '''
        ctx = x.ctx
        T, T1 = self.T, self.T1
        W_conv, b_conv = self.W_conv.data(ctx), self.b_conv.data(ctx)
        out = []
        for j in range(T1):
            y = nd.dot(x[...,j:j+T-T1+1], W_conv)
            y = nd.broadcast_add(y, b_conv, axis=0)
            y = nd.relu(y)
            out.append(y)
        out = merge_list_mx_ndarray(out)
        out = nd.transpose(out, axes=(1,2,3,0))
        return out


class MISTAGCN_1(mx.gluon.Block):
    r'''
    This is the first part in multi information spatial temporal attention graph convolution network (MISTAGCN), where\
        A1 and A2 are ajacency matrices for two different graph structures (nd.ndarray),\
        N is the number of nodes in the graph structure (int),\
        F is the number of features recorded on each node (int).
    '''
    def __init__(self, A1, A2, N, F, **kwargs):
        super(MISTAGCN_1, self).__init__(**kwargs)
        # construct 6 blocks to treat 6 input cases <AIG, Xr>, <AIG, Xd>, <AIG, Xw>, <ACG, Xr>, <ACG, Xd>, <ACG, Xw>
        # because parameters to train differ from input cases
        # with self.name_scope():
        #     self.blk_aig_r = ST_block(ACG, N, F, Tr, 1, Tp)
        #     self.blk_aig_d = ST_block(ACG, N, F, Td*Tp, 1, Tp)
        #     self.blk_aig_w = ST_block(ACG, N, F, Tw*Tp, 1, Tp)
        #     self.blk_acg_r = ST_block(ADG, N, F, Tr, 1, Tp)
        #     self.blk_acg_d = ST_block(ADG, N, F, Td*Tp, 1, Tp)
        #     self.blk_acg_w = ST_block(ADG, N, F, Tw*Tp, 1, Tp)
        self.submodules = [
            ST_block(A1, N, F, Tr, 1, Tp),
            ST_block(A1, N, F, Td*Tp, 1, Tp),
            ST_block(A1, N, F, Tw*Tp, 1, Tp),
            ST_block(A2, N, F, Tr, 1, Tp),
            ST_block(A2, N, F, Td*Tp, 1, Tp),
            ST_block(A2, N, F, Tw*Tp, 1, Tp),
        ]
        for sm in self.submodules:
            self.register_child(sm)


    def forward(self, xr, xd, xw):
        # input xr, xd, xw are of shapes (num, N, F, Tr), (num, N, F, Td*Tp), (num, N, F, Tw*Tp), respectively, (num <= batch_size).
        # output y is of shape (num, N, F(6), Tp).
        # collect outputs
        # out = self.blk_aig_r(xr)
        # out=nd.concat(out, self.blk_aig_d(xd), dim=-2)
        # out=nd.concat(out, self.blk_aig_w(xw), dim=-2)
        # out=nd.concat(out, self.blk_acg_r(xr), dim=-2)
        # out=nd.concat(out, self.blk_acg_d(xd), dim=-2)
        # out=nd.concat(out, self.blk_acg_w(xw), dim=-2)
        x_list = [xr, xd, xw, xr, xd, xw]
        submodule_outputs = [self.submodules[idx](x_list[idx]) for idx in range(len(x_list))]
        # out = nd.concatenate(submodule_outputs, axis=-2)
        out = submodule_outputs[0]
        for i in range(1,len(submodule_outputs)):
            out = nd.concat(out, submodule_outputs[i], dim=-2)
        return out


class MISTAGCN_2(mx.gluon.Block):
    r'''
    This is the second part in multi information spatial temporal attention graph convolution network (MISTAGCN), where\
        A is the ajacency matrix of graph (nd.ndarray),\
        N is the number of nodes in the graph structure (int).
    '''
    def __init__(self, A, N, **kwargs):
        super(MISTAGCN_2, self).__init__(**kwargs)
        self.blk = nn.Sequential()
        with self.name_scope():
            self.blk.add(ST_block(A, N, 6, Tp, 6, Tp))
            self.blk.add(ST_block(A, N, 6, Tp, 6, Tp))
            self.blk.add(ST_block(A, N, 6, Tp, 1, Tp))

    def forward(self, x):
        # Input x is of shape (num, N, F(6), Tp),
        # output out is of shape (num, N, 1, Tp).
        out = self.blk(x)
        return out


class MISTAGCN(mx.gluon.Block):
    r'''
    This is spatial-temporal block in multi information spatial temporal attention graph convolution network (MISTAGCN), where parameters\
        ACG is the ajacency matrix of correlation graph (nd.ndarray),\
        ADG is the ajacency matrix of distance graph (nd.ndarray),\
        AIG is the ajacency matrix of interaction graph (nd.ndarray),\
        N is the number of nodes in the graph structure (int),\
        F is the number of features recorded on each node (int).
    '''
    def __init__(self, ADG, ACG, AIG, N, F, **kwargs):
        super(MISTAGCN, self).__init__(**kwargs)
        with self.name_scope():
            self.blk1 = MISTAGCN_1(ACG, ADG, N, F)
            self.blk2 = MISTAGCN_2(AIG, N)

    def forward(self, xr, xd, xw):
        # input x is of shape (num, N, F, T), num equals batch_size for almost all cases, except for the last batch (num <= batch_size).
        # output y is of shape (num, N, Tp).
        out = self.blk1(xr, xd, xw)
        out = self.blk2(out)
        out = out[:,:,0,:]
        return out