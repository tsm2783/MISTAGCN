from mxnet.gluon import nn, rnn
from mxnet import ndarray as nd

from share import *

class Seq2Seq(nn.Block):
    r'''
    This is a seq2seq class, where
        out_dim is the number of features of the output (int),\
        hidden_size_encoder is the number of units in each GRU layer of the encoder (int),\
        n_layers_enconder is the number of layers of the GRU of the encoder (int),\
        hidden_size_decoder is the number of units in each GRU layer of the decoder (int),\
        n_layers_deconder is the number of layers of the GRU of the decoder (int).
    '''
    def __init__(self, out_dim, n_hidden_size_encoder, n_layers_enconder, n_hidden_size_decoder, n_layers_deconder, **kwargs):
        super(Seq2Seq, self).__init__(**kwargs)
        # encoder
        # self.embedding = nn.Embedding(input_dim, embed_dim)
        self.gru1 = rnn.GRU(n_hidden_size_encoder, n_layers_enconder)
        # decoder
        self.gru2 = rnn.GRU(n_hidden_size_decoder, n_layers_deconder)
        self.linear = nn.Dense(out_dim)

    def forward(self, x):
        if len(x.shape) == 1:
            x = nd.expand_dims(x, axis=0)
        batch_size, in_features = x.shape
        # encoder
        # out = self.embedding(x)   #(batch_size, in_features, embed_dim)
        # out = nd.Embedding(x, weight=nd.ones(shape=(100,20)), input_dim=100, output_dim=20)
        out = nd.expand_dims(x, axis=-1)  #expand to 3 dimension
        out = nd.transpose(out, axes=(1,0,2))  #(in_features, batch_size, embed_dim) or (sequence_length, batch_size, input_size) for gru
        out = self.gru1(out)  #(in_features, batch_size, n_hidden_size_encoder)
        # decoder
        out = self.gru2(out)  #(in_features, batch_size, n_hidden_size_decoder)
        out = nd.transpose(out, axes=(1,0,2))  #(batch_size, in_features, n_hidden_size_decoder)
        out = self.linear(out)  #(batch_size, out_dim)
        return out