import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
import math


def nconv(x, A):
    '''A new type of convolution.

    Args:
        x (ndarray): The input.
        A (ndarray): The matrix to multiply.

    Returns:
        ndarray: The calculated result.
    '''
    x = x.as_np_ndarray()
    x = mx.np.einsum('ncvl,vw->ncwl', x, A)
    x = x.as_nd_ndarray()
    return x


def cat(xs, dim=0):
    '''Concatenate a list of arrays along specified dimension.

    Args:
        xs (list of ndarrays): The array list to concatenate.
        dim (int, optional): The dimension along which the concatenation is processed. Defaults to 0.

    Returns:
        ndarray: The concatenated data.
    '''
    out = xs[0]
    len_dim = xs[0].shape[dim]
    for i in range(1, len_dim):
        out = nd.concat(out, x[i], dim=dim)
    return out


class Linear(nn.Block):
    '''An implement of linear function.

    Args:
        c_in (int): The number of channel in the input.
        c_out (int): The number of channel in the output.
    '''

    def __init__(self, c_in, c_out, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.mlp = nn.Conv2D(c_out, kernel_size=(1, 1), c_in=c_in)

    def forward(self, x):
        x = self.mlp(x)
        return x


class GCN(nn.Block):
    '''An implement of graph convolution networks (GCN).

    Args:
        c_in (int): The number of channel in the input.
        c_out (int): The number of channel in the output.
        drop_out (double): Fraction of the input that gets dropped out during training time.
        support_len (int, optional): ...
        order (int, optional): The order of GCN.
    '''

    def __init__(self, c_in, c_out, dropout, support_len=3, order=2, **kwargs):
        super(GCN, self).__init__(**kwargs)
        c_in = (order*support_len+1) * c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = cat(out, dim=1)
        h = self.mlp(h)
        h = nd.Dropout(h, self.dropout)
        return h


class GWNET(nn.Block):
    def __init__(self, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2, **kwargs):
        super(GWNET, self).__init__(**kwargs)
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.Sequential()
        self.gate_convs = nn.Sequential()
        self.residual_convs = nn.Sequential()
        self.skip_convs = nn.Sequential()
        self.bn = nn.Sequential()
        self.gconv = nn.Sequential()

        self.start_conv = nn.Conv2D(residual_channels, kernel_size=(1,1), in_channels=in_dim)
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        with self.name_scope():
            if gcn_bool and addaptadj:
                if aptinit is None:
                    if supports is None:
                        self.supports = []
                    self.nodevec1 = self.params.get('nodevec1', shape=(num_nodes, 10), init=mx.init.Uniform())
                    self.nodevec2 = self.params.get('nodevec2', shape=(10, num_nodes), init=mx.init.Uniform())
                    self.supports_len += 1
                else:
                    if supports is None:
                        self.supports = []
                    m, p, n = mx.np.linalg.svd(aptinit.as_np_ndarray())
                    m, p, n = m.as_nd_ndarray(), p.as_nd_ndarray(), n.as_nd_ndarray()
                    initemb1 = nd.dot(m[:, :10], nd.diag(p[:10] ** 0.5))
                    initemb2 = nd.dot(nd.diag(p[:10] ** 0.5), n[:, :10].t())
                    self.nodevec1 = self.params.get('nodevec1', shape=initemb1.shape, init=initemb1)
                    self.nodevec2 = self.params.get('nodevec2', shape=initemb1.shape, init=initemb1)
                    self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(GCN(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = torch.softmax(torch.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = torch.relu(skip)
        x = torch.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
