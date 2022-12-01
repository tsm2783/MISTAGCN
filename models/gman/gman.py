import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
import math


def concat(data, dim=0):
    '''Concatenate a list of array along a special axis.

    Args:
        data (list of ndarray): the list of arrays to concatenate.
        dim (int, optional): the axis along which concatenation is done. Defaults to 0.

    Returns:
        ndarray: the concatenated array.
    '''
    out = data[0]
    for i in range(1,len(data)):
        out = nd.concat(out, data[i], dim=dim)
    return out


class FCs(nn.Block):
    """Full connected layers.

    Args:
        units (int or list of int): number of units of the output(s).
    """
    def __init__(self, units, **kwargs):
        super(FCs, self).__init__(**kwargs)
        if isinstance(units, int):
            units = [units]
        self.batch_norm = nn.BatchNorm()
        self.blk = nn.Sequential()
        with self.blk.name_scope():
            for unit in units:
                self.blk.add(nn.Dense(unit, activation='relu', flatten=False))

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.blk(x)
        return x


class STE(nn.Block):
    '''Spatio-temporal embedding layer.

    Args:
        T (int): number of time steps in one day,
        D (int): output dims.
    '''
    def __init__(self, T, D, **kwargs):
        super(STE, self).__init__(**kwargs)
        self.T = T
        self.se_embed = FCs([D, D])
        self.te_embed = FCs([D, D])

    def forward(self, se, te):
        # se (ndarray): spatial embedding (N, D),
        # te (ndarray): temporal embedding (batch_size, P+Q, 2).(dayofweek, timeofday),
        # out (ndarray): [batch_size, P + Q, N, D].
        # spatial embedding
        se = nd.expand_dims(nd.expand_dims(se, axis=0), axis=0)
        se = self.se_embed(se)
        # temporal embedding
        dayofweek = nd.one_hot(te[...,0], depth = 7)
        timeofday = nd.one_hot(te[...,1], depth = self.T)
        te = nd.concat(dayofweek, timeofday, dim = -1)
        te = nd.expand_dims(te, axis = 2)
        te = self.te_embed(te)
        out = se + te
        return out


class SpatialAttention(nn.Block):
    '''An implementation of the spatial attention mechanism.

    Args:
        K (int) : number of attention heads,
        d (int) : dimension of each attention head outputs.
    '''
    def __init__(self, K, d, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        D = K * d
        self.d = d
        self.K = K
        self.fc_q = FCs(D)
        self.fc_k = FCs(D)
        self.fc_v = FCs(D)
        self.fc = FCs([D, D])

    def forward(self, x, ste):
        # x (ndarray) : input sequence, with shape (batch_size, num_step, num_nodes, K*d)
        # ste (ndarray) : spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d)
        # out (ndarray) : spatial attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        ctx = x.ctx
        batch_size = len(x)
        K, d = self.K, self.d
        x = nd.concat(x, ste, dim=-1)
        query = self.fc_q(x)
        key = self.fc_k(x)
        value = self.fc_v(x)
        query = concat(nd.split(query, K, axis=-1), dim=0)  #[K * batch_size, num_step, N, d]
        key = concat(nd.split(key, K, axis=-1), dim=0)  #[K * batch_size, num_step, N, d]
        key = nd.swapaxes(key, -1, -2)
        value = concat(nd.split(value, K, axis=-1), dim=0)  #[K * batch_size, num_step, N, d]
        attention = mx.np.matmul(query.as_np_ndarray(), key.as_np_ndarray())  #[K * batch_size, num_step, N, N]
        attention = attention.as_nd_ndarray()
        attention = attention / math.sqrt(d)
        attention = nd.softmax(attention, axis=-1)
        x = mx.np.matmul(attention.as_np_ndarray(), value.as_np_ndarray())  #[batch_size, num_step, N, D]
        x = x.as_nd_ndarray()
        x = concat(nd.split(x, K, axis=0), dim=-1)
        x = self.fc(x)
        return x


class TemporalAttention(nn.Block):
    '''An implementation of the temporal attention mechanism.

    Args:
        K (int) : number of attention heads.
        d (int) : dimension of each attention head outputs.
    '''
    def __init__(self, K, d, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        D = K * d
        self.K = K
        self.d = d
        self.fc_q = FCs(D)
        self.fc_k = FCs(D)
        self.fc_v = FCs(D)
        self.fc = FCs([D, D])

    def forward(self, x, ste):
        # x (ndarray) : input sequence, with shape (batch_size, num_step, num_nodes, K*d)
        # ste (ndarray) : spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d)
        # return : temporal attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        ctx = x.ctx
        batch_size = len(x)
        K, d = self.K, self.d
        x = nd.concat(x, ste, dim=-1)
        query = self.fc_q(x)
        key = self.fc_k(x)
        value = self.fc_v(x)
        query = concat(nd.split(query, K, axis=-1), dim=0)  #[K * batch_size, num_step, N, d]
        key = concat(nd.split(key, K, axis=-1), dim=0)  #[K * batch_size, num_step, N, d]
        value = concat(nd.split(value, K, axis=-1), dim=0)  #[K * batch_size, num_step, N, d]
        query = nd.transpose(query, axes=(0,2,1,3))  #[K * batch_size, N, num_step, d]
        key = nd.transpose(key, axes=(0,2,3,1))  #[K * batch_size, N, d, num_step]
        value = nd.transpose(value, axes=(0,2,1,3))  #[K * batch_size, N, num_step, d]
        attention = mx.np.matmul(query.as_np_ndarray(), key.as_np_ndarray())  #[K * batch_size, N, num_step, num_step]
        attention = attention.as_nd_ndarray()
        attention = attention / math.sqrt(d)
        attention = nd.softmax(attention, axis=-1)
        x = mx.np.matmul(attention.as_np_ndarray(), value.as_np_ndarray())
        x = x.as_nd_ndarray()
        x = nd.transpose(x, axes=(0,2,1,3))  #[K * batch_size, num_step, N, d]
        x = concat(nd.split(x, K, axis=0), dim=-1)
        x = self.fc(x)
        return x


class GatedFusion(nn.Block):
    '''An implementation of the gated fusion mechanism.

    Args:
        D (int) : dimension of output.
    '''
    def __init__(self, D, **kwargs):
        super(GatedFusion, self).__init__(**kwargs)
        self.fc_xs = FCs(D)
        self.fc_xt = FCs(D)
        self.fc_h = FCs([D, D])

    def forward(self, hs, ht):
        # hs (ndarray) : spatial attention scores, with shape (batch_size, num_step, num_nodes, D),
        # ht (ndarray) : temporal attention scores, with shape (batch_size, num_step, num_nodes, D),
        # out (ndarray) : spatial-temporal attention scores, with shape (batch_size, num_step, num_nodes, D).
        xs = self.fc_xs(hs)
        xt = self.fc_xt(ht)
        z = nd.sigmoid(xs + xt)
        h = z * hs + (1-z) * ht
        h = self.fc_h(h)
        return h


class ST_Block(nn.Block):
    """
    An implementation of the spatial-temporal attention block.

    Args:
        K (int) : number of attention heads,
        d (int) : dimension of each attention head outputs.
    """
    def __init__(self, K, d, **kwargs):
        super(ST_Block, self).__init__(**kwargs)
        self.satt = SpatialAttention(K, d)
        self.tatt = TemporalAttention(K, d)
        self.gated_fusion = GatedFusion(K * d)

    def forward(self, x_ste):
        # x (ndarray) : input sequence, with shape (batch_size, num_step, num_nodes, K*d),
        # ste (ndarray) : spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d),
        # out (ndarray) : attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        x, ste = x_ste  #expand the tuple
        hs = self.satt(x, ste)
        ht = self.tatt(x, ste)
        h = self.gated_fusion(hs, ht)
        x = x + h
        out = (x, ste)
        return out


class TransformAttention(nn.Block):
    """
    An implementation of the tranform attention mechanism.

    Args:
        K (int) : number of attention heads,
        d (int) : dimension of each attention head outputs.
    """
    def __init__(self, K, d, **kwargs):
        super(TransformAttention, self).__init__(**kwargs)
        D = K * d
        self.K = K
        self.d = d
        self.fc_q = FCs(D)
        self.fc_k = FCs(D)
        self.fc_v = FCs(D)
        self.fc = FCs([D, D])

    def forward(self, x, ste_hist, ste_pred):
        # x (ndarray) : input sequence, with shape (batch_size, num_hist, num_nodes, K*d),
        # ste_hist (ndarray) : spatial-temporal embedding for history, with shape (batch_size, num_hist, num_nodes, K*d),
        # ste_pred (ndarray) : spatial-temporal embedding for prediction, with shape (batch_size, num_pred, num_nodes, K*d),
        # out (ndarray): output sequence for prediction, with shape (batch_size, num_pred, num_nodes, K*d).
        batch_size = x.shape[0]
        K, d = self.K, self.d
        query = self.fc_q(ste_pred)  #[batch_size, Q, N, K * d]
        key = self.fc_k(ste_hist)  #[batch_size, P, N, K * d]
        value = self.fc_v(x)  #[batch_size, P, N, K * d]
        query = concat(nd.split(query, K, axis=-1), dim=0)  #[K * batch_size, Q, N, d]
        key = concat(nd.split(key, K, axis=-1), dim=0)  #[K * batch_size, P, N, d]
        value = concat(nd.split(value, K, axis=-1), dim=0)  #[K * batch_size, P, N, d]
        query = nd.transpose(query, axes=(0,2,1,3))  #[K * batch_size, N, Q, d]
        key = nd.transpose(key, axes=(0,2,3,1))  #[K * batch_size, N, d, P]
        value = nd.transpose(value, axes=(0,2,1,3))  #[K * batch_size, N, P, d]
        attention = mx.np.matmul(query.as_np_ndarray(), key.as_np_ndarray())
        attention = attention.as_nd_ndarray()
        attention = attention / math.sqrt(d)
        attention = nd.softmax(attention, axis=-1)
        x = mx.np.matmul(attention.as_np_ndarray(), value.as_np_ndarray())  #[batch_size, Q, N, D]
        x = x.as_nd_ndarray()
        x = nd.transpose(x, axes=(0,2,1,3))
        x = concat(nd.split(x, K, axis=0), dim=-1)
        x = self.fc(x)
        return x


class GMAN(nn.Block):
    """An implementation of GMAN.

    Args:
        L (int) : number of STAtt blocks in the encoder/decoder,
        K (int) : number of attention heads,
        d (int) : dimension of each attention head outputs,
        P (int): number of history steps,
        Q (int): number of prediction steps,
        T (int): number of steps in a day.
    """
    def __init__(self, L, K, d, P, T, **kwargs):
        super(GMAN, self).__init__(**kwargs)
        D = K * d
        self.D = D
        self.P = P
        self.T = T
        self.st_embedding = STE(T, D)
        self.encoder = nn.Sequential()
        for _ in range(L):
            self.encoder.add(ST_Block(K, d))
        self.decoder = nn.Sequential()
        for _ in range(L):
            self.decoder.add(ST_Block(K, d))
        self.transform_attention = TransformAttention(K, d)
        self.fc_1 = FCs([D, D])
        self.fc_2 = FCs([D, 1])

    def forward(self, x, se, te):
        # x (ndarray): input sequence, with shape (batch_size, num_hist, num_nodes),
        # se (ndarray): spatial embedding, with shape (num_nodes, K * d) = (num of nodes, D),
        # te (ndarray): temporal attention, with shape (batch_size, num_his + num_pred, 2)  (time-of-day, day-of-week).
        # out (ndarray): ouput sequence for prediction, with shape (batch_size, num_pred, num of nodes).
        ctx = x.ctx
        se = se.as_in_context(ctx)
        x = nd.expand_dims(x, axis=-1)
        x = self.fc_1(x)
        ste = self.st_embedding(se, te)
        ste_hist = ste[:, :self.P]
        ste_pred = ste[:, self.P:]

        # encoder
        x_ste_hist = (x, ste_hist)
        x_ste_hist = self.encoder(x_ste_hist)
        x = x_ste_hist[0]
        # tranform
        x = self.transform_attention(x, ste_hist, ste_pred)
        # decoder
        x_ste_pred = (x, ste_pred)
        x_ste_pred = self.decoder(x_ste_pred)
        x = x_ste_pred[0]

        x = self.fc_2(x)
        x = nd.squeeze(x, axis = 3)
        return x