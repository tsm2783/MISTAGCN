from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.parameter import Parameter

class VAE(nn.Block):
    r'''
    This is a class of convolutional variation autoencoders (VAE), where\
        n_encoder_filter_list is the list of number of units in the hidden layers, the encoder part (list),\
        n_decoder_filter_list is the list of number of units in the hidden layers, the decoder part (list),\
        n_latent is the numer of features of the latent variables (int),\
        n_out is the numer of features of the output (int).
    '''
    def __init__(self, n_encoder_filter_list, n_latent, n_decoder_filter_list, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = nn.Sequential()
        for n_filter in n_encoder_filter_list:
            self.encoder.add(nn.Dense(n_filter, activation='sigmoid'))
        self.encoder.add(nn.Dense(n_latent*2))  #mu, sigma
        self.decoder = nn.Sequential()
        for n_filter in n_decoder_filter_list:
            self.decoder.add(nn.Dense(n_filter, activation='sigmoid'))
        # parameters for a linear layer
        self.n_decoder_filter = n_decoder_filter_list[-1]
        self.W = self.params.get('W', allow_deferred_init=True)

    def forward(self, x):
        if len(x.shape) == 2:
            x = nd.expand_dims(x, axis=0)
        batch_size, N, T = x.shape
        self.W.shape = (self.n_decoder_filter, N*T)
        for param in [self.W]:
            param._finish_deferred_init()
        h = self.encoder(x)  #(batch_size, n_latent*2)
        mu, sigma = nd.split(h, num_outputs=2)
        eps = nd.random_normal(loc=0, scale=1, shape=mu.shape, ctx=mu.ctx)
        z = mu + 0.5 * sigma * eps  #(batch_size, n_latent)
        out = self.decoder(z)
        out = nd.dot(out, self.W.data(x.ctx))
        out = nd.reshape(out, shape=(batch_size, N, T))
        return (z, out)