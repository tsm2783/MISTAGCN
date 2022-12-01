# This is an implementation of MISTIF-Net in the following paper:
# [Urban ride-hailing demand prediction with multiple spatio-temporal information fusion network, Transportation Research Part C (2020)](...).

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

from gcn import GCN
from vae import VAE
from seq2seq import Seq2Seq

class MSTIF_NET(mx.gluon.Block):
    r'''
    This is a class discribed in paper <Urban ride-hailing demand prediction with multiple spatio-temporal information fusion network>, where
        adj1, adj2, adj3 are three ajacency matrices, which denote the three graph structures (ndarray),\
        n_filter is the number of filters (out_dim) for output (latent variables) (int), \
        n_linear_out is the number of features of the out of the full connected layer (int),\
        hidden_size_encoder, n_layers_enconder, hidden_size_decoder, n_layers_deconder are parameters of the Seq2Seq layer,\
        channels, kernel_size are parameters of the Deconvolution layer.
    '''
    def __init__(self,
                adj1, adj2, adj3, n_filter1, n_filter2, n_filter,  #parameters for GCN
                n_fusion_out,  #number of features of ouput of the information fusion layer, i.e. FNN
                out_dim, n_hidden_size_encoder, n_layers_enconder, n_hidden_size_decoder, n_layers_deconder,  #parameters for Seq2Seq
                Tp,
                **kwargs):
        super(MSTIF_NET, self).__init__(**kwargs)
        self.n_seq2seq_out = out_dim
        self.Tp = Tp
        self.gcn1 = GCN(adj1, n_filter1, n_filter2, n_filter)
        self.gcn2 = GCN(adj1, n_filter1, n_filter2, n_filter)
        self.gcn3 = GCN(adj1, n_filter1, n_filter2, n_filter)
        self.linear = nn.Dense(n_fusion_out, activation='relu')
        self.seq2seq = Seq2Seq(out_dim, n_hidden_size_encoder, n_layers_enconder, n_hidden_size_decoder, n_layers_deconder)
        self.deconv_weight = self.params.get('Deconv_weight', allow_deferred_init=True)  #the deconvolution layer is a linear layer
        self.deconv_bias = self.params.get('Deconv_bias', allow_deferred_init=True)

    def forward(self, x, z):
        r'''
        Input x is an array of the traffic flows (flow_out and flow_in).
            output out is an array of traffic flows in the next Tp time slices.
        '''
        if len(x.shape) == 2:
            nd.expand_dims(x, axis=0)
        batch_size, N, T = x.shape
        # parameter initialization
        self.deconv_weight.shape = (self.n_seq2seq_out, N*self.Tp)
        self.deconv_bias.shape = (N*self.Tp, )
        for param in [self.deconv_weight, self.deconv_bias]:
            param._finish_deferred_init()
        # compute the latent variables
        latent = self.gcn1(x)   #(batch_size, n_filter)
        latent = nd.concat(latent, self.gcn2(x), dim=1)
        latent = nd.concat(latent, self.gcn3(x), dim=1)
        latent = nd.concat(latent, z, dim=1)
        input_seq2seq = self.linear(latent)
        # seq2seq forward
        out = self.seq2seq(input_seq2seq)
        # decovolution
        out = nd.LeakyReLU(nd.dot(out, self.deconv_weight.data(ctx=out.ctx)) + self.deconv_bias.data(ctx=out.ctx))
        out = nd.reshape(out, shape=(batch_size, N, self.Tp))
        return out
