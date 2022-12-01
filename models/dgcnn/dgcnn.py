import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
import numpy as np
import mxnet.numpy as mnp


class TDL(nn.Block):
    r'''
    This is an implementation of tensor decompossition layer (TDL) in dynamic graph convolutional neural networks (DGCNN), where\
        A is the adjacency matrix of distance graph ADG (nd.ndarray).\
        U1 nad U2 are projection factors.
    '''
    def __init__(self, **kwargs):
        super(TDL, self).__init__(**kwargs)
        # parameters for Laplacian Matrix Estimator
        with self.name_scope():
            self.U1 = self.params.get('U1', allow_deferred_init=True)
            self.U2 = self.params.get('U2', allow_deferred_init=True)

    def forward(self, x, A):
        if len(x.shape) == 3:
            x = nd.expand_dims(x.as_np_ndarray(), axis=0)
        batch_size, N, F, T = x.shape
        p, m, c = N, T, F
        x1 = nd.transpose(x, axes=(0,1,3,2))  #a batch of \mathcal{X}
        ctx = x.ctx

        # global Laplacian Ls
        D = nd.sum(A, axis=1)
        Ls = nd.diag(D) - A

        # parameter reinitial
        r1 = p // 5
        r2 = m // 3
        if r1 == 0: r1 = 1  #in case p or m so small
        if r2 == 0: r2 = 1
        self.U1.shape = (p, r1)
        self.U2.shape = (m, r2)
        for param in [self.U1, self.U2]:
            param._finish_deferred_init()

        out = []
        for chi in x1:  #chi == \mathcal{X}
            # Tensor Decomposition (TDL)
            G = self.prod2(self.prod1(chi, self.U1.data(ctx)), self.U2.data(ctx))
            chi_s = self.prod2(self.prod1(G, nd.transpose(self.U1.data(ctx))), nd.transpose(self.U2.data(ctx)))
            chi_e = chi - chi_s  #(p,m,c)
            # Unfolding-Normalization
            chi_s = nd.L2Normalization(chi_s)
            chi_e = nd.L2Normalization(chi_e)
            x_s = nd.reshape(chi_s, shape=(p, m*c))
            x_e = nd.reshape(chi_e, shape=(p, m*c))
            temp = nd.sum(nd.linalg.extractdiag(nd.dot(nd.dot(x_s.transpose(), Ls), x_s))) + 0.1 * nd.sum(nd.linalg.extractdiag(nd.dot(x_e.transpose(), x_e)))  #formla in paper not correct
            out.append(temp)
        # concatenate the elements of a list
        out1 = out[0]
        for i in range(1,len(out)):
            out1 = nd.concat(out1, out[i], dim=0)

        return out1

    def prod1(self, x, u):
        '''Implementation of Eq. 6, where
            x is of shape (p,m,c),
            u is of shape (p,r1),
            out is of shape (r1,m,c).'''
        out = nd.dot(nd.transpose(u, axes=(1,0)),x)
        return out

    def prod2(self, x, u):
        '''Implementation of Eq. 6, where
            x is of shape (p,m,c),
            u is of shape (m,r2),
            out is of shape (p,r2,c).'''
        out = nd.dot(nd.transpose(x, axes=(0,2,1)),u)
        out = nd.transpose(out, axes=(0,2,1))
        return out


class Temp_conv(nn.Block):
    r'''
    This is an implementation of 'Gated CNNs for Extracting Temporal Features', where parameter\
        units is the number of features of the output, i.e. c_out (int).
    '''
    def __init__(self, channels, units, **kwargs):
        super(Temp_conv, self).__init__(**kwargs)
        self.gcnn_t1 = nn.Sequential()
        self.gcnn_t1.add(nn.Conv1D(channels=channels, kernel_size=(1,)))
        self.gcnn_t1.add(nn.Dense(units=units, flatten=False))
        self.gcnn_t2 = nn.Sequential()
        self.gcnn_t2.add(nn.Conv1D(channels=channels, kernel_size=(1,)))
        self.gcnn_t2.add(nn.Dense(units=units, flatten=False))

    def forward(self, x):
        '''Input x is of shape (p,m,c_in), out is of shape (p,channels,units).'''
        out = self.gcnn_t1(x) * nd.sigmoid(self.gcnn_t2(x))
        return out


class Spatial_graph_conv(nn.Block):
    r'''
    This is an implementation of 'Graph CNNs for Extracting Spatial Features', where parameter\
        units is the number of features of the output, i.e. c_out (int),\
    '''
    def __init__(self, K, units, **kwargs):
        super(Spatial_graph_conv, self).__init__(**kwargs)
        self.K = K
        self.c_out = units
        with self.name_scope():
            self.Theta = self.params.get('Theta', allow_deferred_init=True)

    def forward(self, x, L):
        r'''
        Input x is of shape (m,p,c_in),\
        L is the normalized Laplacian matrix (nd.ndarray),\
        out is of shape (m,p,c_out).
        '''
        p, m, c_in = x.shape
        self.Theta.shape = (self.K, c_in, self.c_out)
        for param in [self.Theta]:
            param._finish_deferred_init()

        cheb_p = [nd.eye(p, ctx=x.ctx), L]
        for k in range(2, self.K):
            cheb_p.append(2 * L * cheb_p[-1] - cheb_p[-2])
        out = nd.zeros((p,m,self.c_out), ctx=x.ctx)
        for k in range(self.K):
            out = out + nd.dot(nd.dot(cheb_p[k], x), self.Theta.data(ctx=x.ctx)[k])
        return out


class DST_conv(nn.Block):
    r'''
    This is an implementation of 'Dynamic Spatial-Temporal Convolutional Block', where parameter\
        K is the order of Chebyshev polynomials (int),\
        channel_list is a list of channels for Temp_conv layers (list),\
        unit_list_temp is a list of units for Temp_conv layers (list),\
        unit_sp is the number of units for Spatial_graph_conv layer (int),\
    '''
    def __init__(self, K, channel_list, unit_list_temp, unit_sp, **kwargs):
        super(DST_conv, self).__init__(**kwargs)
        self.layer1 = Temp_conv(channel_list[0], unit_list_temp[0])
        self.layer2 = Spatial_graph_conv(K, unit_sp)
        self.layer3 = Temp_conv(channel_list[1], unit_list_temp[1])

    def forward(self, x, L):
        '''Input x is of shape (m,p,c_in), out is of shape (m,p,c_out).'''
        out = self.layer1(x)
        out = self.layer2(out, L)
        out = self.layer3(out)
        return out


class DGCNN(nn.Block):
    r'''
    This is an implementation of dynamic graph convolutional neural networks (DGCNN), where parameter\
        A is ADG (nd.ndarray),\
        K is the order of Chebyshev polinomial (int),\
        Tp is the target number of time steps to predict (int),\
        U1 and U2 are parameters pre-trained through model 'TDL' (nd.ndarray).
    '''
    def __init__(self, K, Tp, **kwargs):
        super(DGCNN, self).__init__(**kwargs)

        # parameters and layers for Laplacian Matrix Estimator
        self.conv1 = nn.Conv2D(channels=3, kernel_size=(1,1))
        self.conv2 = nn.Conv2D(channels=1, kernel_size=(1,1))
        # layers to be trained
        self.dst_conv1 = DST_conv(K, channel_list=[16, 16], unit_list_temp=[8, 8], unit_sp=8)
        self.dst_conv2 = DST_conv(K, channel_list=[16, 16], unit_list_temp=[8, 8], unit_sp=8)
        self.outlayer = nn.Dense(Tp, flatten=False)

    def forward(self, x, A, U1, U2):
        if len(x.shape) == 3:   # in case there is only on element in the last batch, len(Z.shape) == 3, extend the dimension to 4
            Z = nd.expand_dims(x.as_np_ndarray(), axis=0)
        batch_size, N, F, T = x.shape
        p, m, c = N, T, F
        x1 = nd.transpose(x, axes=(0,1,3,2))  #a batch of \mathcal{X}
        ctx = x.ctx

        # global Laplacian Ls
        D = nd.sum(A, axis=1)
        Ls = D - A

        out = []
        for chi in x1:  #chi == \mathcal{X}
            # Laplacian Matrix Estimator
            # Tensor Decomposition (TDL)
            G = self.prod2(self.prod1(chi, U1), U2)
            chi_s = self.prod2(self.prod1(G, nd.transpose(U1)), nd.transpose(U2))
            chi_e = chi - chi_s
            # Unfolding-Normalization
            chi_s = nd.L2Normalization(chi_s)
            chi_e = nd.L2Normalization(chi_e)
            x_s = nd.reshape(chi_s, shape=(p, m*c))
            x_e = nd.reshape(chi_e, shape=(p, m*c))
            x_stack = [nd.dot(x_s, x_e.transpose()), nd.dot(x_e, x_s.transpose()), nd.dot(x_e, x_e.transpose())]
            # convert to ndarray of higher dimension
            x_stack = [nd.expand_dims(_, axis=0) for _ in x_stack]
            temp = x_stack[0]
            temp = nd.concat(temp, x_stack[1], dim=0)
            temp = nd.concat(temp, x_stack[2], dim=0)
            x_stack = temp
            # 2-D Conv
            x_stack = nd.expand_dims(x_stack, axis=0)  #shape (1,3,p,p)
            z_e = self.conv1(x_stack)  #shape (1,3,p,p)
            z_e = self.conv2(x_stack)  #shape (1,1,p,p)
            z_e = z_e[0,0]  #shape (p,p)
            b = nd.sum(x_stack[0], axis=0) + z_e
            # Estimator, 'We compute the global Laplacian matrix based on the distances among sensors employed on each road segment'
            bls = nd.dot(b, Ls)  #matrix multiplication
            bls = nd.L2Normalization(bls)  #in order to force equation (18) to converge, bls should be mapped to norm(bls)=p<1, this is not mensioned in the paper
            bls = bls / 3
            temp = nd.eye(p, ctx=x.ctx)
            Le = nd.zeros_like(Ls)
            for i in range(6):
                temp = -1 * nd.dot(temp, bls)
                Le = Le + nd.dot(Ls, temp)
            # Normalization
            L = Ls + 0.5 * (Le + Le.transpose())
            D = nd.abs(nd.sum(L - nd.diag(nd.linalg_extractdiag(L)), axis=1))
            eps = 0.0001
            D_12 = 1 / (nd.sqrt(D) + eps)
            L = D_12 * L * D_12

            # the main process as shown in Fig. 1
            out1 = self.dst_conv1(chi, L)
            out1 = self.dst_conv2(out1, L)
            out1 = nd.reshape(out1, shape=(out1.shape[0], out1.shape[1]*out1.shape[2]))
            out1 = self.outlayer(out1)

            out.append(out1)

        # convert a list of arrays to an array of higher dimension
        out = [nd.expand_dims(_, axis=0) for _ in out]
        out1 = out[0]
        for i in range(1,len(out)):
            out1 = nd.concat(out1, out[i], dim=0)
        out =  out1

        return out

    def prod1(self, x, u):
        '''Implementation of Eq. 6, where
            x is of shape (p,m,c),
            u is of shape (p,r1),
            out is of shape (r1,m,c).'''
        out = nd.dot(nd.transpose(u, axes=(1,0)),x)
        return out

    def prod2(self, x, u):
        '''Implementation of Eq. 6, where
            x is of shape (p,m,c),
            u is of shape (m,r2),
            out is of shape (p,r2,c).'''
        out = nd.dot(nd.transpose(x, axes=(0,2,1)),u)
        out = nd.transpose(out, axes=(0,2,1))
        return out

    def normalize(self, x):
        '''Normalize a tensor x of shape (p,m,c) along axis 1.'''
        p,m,c = x.shape
        mean = nd.sum(x, axis=1) / m
        mean = nd.expand_dims(mean, axis=-1)  #(p,c,1)
        x = nd.transpose(x, axes=(0,2,1))  #(p,c,m)
        std = nd.broadcast_sub(x, mean)
        std = nd.sum(std**2, axis=-1) / m
        std = nd.sqrt(std)
        std = nd.expand_dims(std, axis=-1)  #(p,c,1)
        x1 = nd.broadcast_div(nd.broadcast_sub(x, mean), std + 0.0001)  #(p,c,m)
        return nd.transpose(x1, axes=(0,2,1))  #(p,m,c)