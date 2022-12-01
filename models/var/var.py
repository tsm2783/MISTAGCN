import imp
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, Parameter


class VAR(nn.Block):
    r'''
    This is a vector autoregression model, which maps graph signal X_{t-T+1,...,t} -> X_{t+1,...,t+Tp} -> Y_{1,...,Tp}. Parameter
        F: number of features at each node.
        T: number of time steps of historical data.
        Tp: number of time steps of future data.
    '''
    def __init__(self, N, F, T, Tp, **kwargs):
        super(VAR, self).__init__(**kwargs)
        self.Tp = Tp
        with self.name_scope():
            self.w = self.params.get('weight', shape=(T,F,F), init=mx.init.Xavier())
            self.b = self.params.get('bias', shape=(1,N,F), init=mx.init.Zero())
            self.linear = nn.Dense(1, flatten=False, activation='relu')

    def forward(self, x):
        #input: (num, N, F, T)
        #output: (num, N, Tp)
        num, N, F, T = x.shape  #for the last batch, num <= batch_size
        Tp = self.Tp
        ctx = x.ctx
        w, b = self.w.data(ctx), self.b.data(ctx)
        for i in range(Tp):
            y = nd.zeros(shape=(num, N, F), ctx=ctx)
            for j in range(T):
                y = y + nd.dot(x[...,-j-1], w[j])
            y = nd.broadcast_add(nd.transpose(y), nd.transpose(b))
            y = nd.transpose(y)
            y = nd.expand_dims(y, axis=-1)
            x = nd.concat(x, y, dim=-1)
        x = x[..., -Tp:]  #num, N, F, Tp
        x = nd.transpose(x, axes=(0,1,3,2)) #num, N, Tp, F
        out = self.linear(x) #num, N, Tp, 1
        out = out[:,:,:,0]
        return out