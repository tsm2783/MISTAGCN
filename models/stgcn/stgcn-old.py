from mxnet import nd
from mxnet.gluon import nn

from share import K
from share import get_max_eigenvalue


class GCN(nn.Block):
    r'''
    This is an implement graph convolutional network, where\
        A is ajacency matrix of the graph.
    '''
    def __init__(self, A, in_feats, out_feats, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.in_feats = in_feats
        self.out_feats = out_feats

        A = A.cpu()
        A = nd.array(A.numpy())  #torch.Tensor > mxnet.ndarray
        N = len(A)
        D = nd.sum(A, axis=1)
        eps = 0.0001
        D_12 = 1 / (nd.sqrt(D) + eps)
        L = nd.eye(N, ctx=A.ctx) - (D_12 * A) * D_12  #element wise multiply
        lambda_max = get_max_eigenvalue(L)
        L_tilde = (2 / lambda_max) * L - nd.eye(N, ctx=A.ctx)
        L_tilde = L_tilde.as_nd_ndarray()
        cheb_p = [nd.eye(N, ctx=A.ctx), L_tilde]
        for k in range(2, K):
            cheb_p.append(2 * L_tilde * cheb_p[-1] - cheb_p[-2])
        # convert to torch tensor
        cheb_p = [torch.tensor(_.asnumpy()).to(device) for _ in cheb_p]
        cheb_p = [_.to(device) for _ in cheb_p]
        self.cheb_polynomial = cheb_p

        # W^{1} and W^{2} in the paper
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(N, out_feats))  # bias, optional

        # initialize the weight and bias
        self.reset_parameters()


    def reset_parameters(self):
        '''
        Reinitialize learnable parameters
        ** Glorot, X. & Bengio, Y. (2010)
        ** Critical, otherwise the loss will be NaN
        '''

        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


    def forward(self, x):
        '''
        formular:
            h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})
        Inputs:
            x:
                H^{l}, i.e. Node features with shape [num_nodes, batch_size, in_feats, channels]
        Returns:
            out:
                H^{l+1}, i.e. Node embeddings of the l+1 layer with the shape [num_nodes, batch_size, out_feats, channels]
        '''
        num_nodes, batch_size, in_feats, channels  = x.shape  #?
        K = len(self.cheb_polynomial)
        out = torch.zeros((num_nodes, batch_size, self.out_feats, channels), device=device)
        for i in range(batch_size):
            for j in range(in_feats):
                for k in range(K):
                    Tk = self.cheb_polynomial[k]
                    out += torch.relu(torch.matmul(torch.matmul(Tk, x[:,i,j,:]), self.weight) + self.bias)
        return out


class TemporalConvLayer(nn.Module):
    '''
    Section 3.3 in the paper
    Gated 1D temporal convolution layer (Conv2d used in the 1D way due to the input dim)
    ==> We only care about H_out if input:[N, C_in, H_in, W_in] and output:[N, C_out, H_out, W_out]
        where H corresponds to the timestep dimension

    Inputs:
        c_in: input channels
        c_out: output channels
        kernel: kernel size for timestep axis
        dia: spacing between kernel elements
        x: input with the shape [batch_size, c_in, timesteps, num_nodes]

    Return:
        gated_conv: output with the shape (we assume that dia = 1)
        [batch_size, c_out, timesteps-kernel_size[0]+1, num_nodes-kernel_size[1]+1]
        i.e. [batch, c_out, timestep-1, num_nodes] if kernel_size = (2, 1)

    '''
    def __init__(self, c_in, c_out, kernel = 2, dia = 1):
        super(TemporalConvLayer, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.conv = nn.Conv2d(c_in, 2*c_out, (kernel, 1), 1, dilation = dia, padding = (0,0)).to(device)
        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, x):
        conv_x = self.conv(x)
        P = conv_x[:, 0:self.c_out, :, :]
        Q = conv_x[:, -self.c_out:, :, :]
        gated_conv = P * self.sigmoid(Q)
        return gated_conv


class TemporalConvLayer_Residual(nn.Module):
    '''
    ** 'TemporalConvLayer' with the residual connection **

    Inputs:
        c_in: input channels
        c_out: output channels
        kernel: kernel size for timestep axis
        dia: spacing between kernel elements
        x: input with the shape [batch_size, c_in, timesteps, num_nodes]

    Return:
        gated_conv: output with the shape (we assume that dia = 1)
        [batch_size, c_out, timesteps-kernel_size[0]+1, num_nodes-kernel_size[1]+1]
        i.e. [batch, c_out, timestep-1, num_nodes] if kernel_size = (2, 1)

    '''
    def __init__(self, c_in, c_out, kernel = 2, dia = 1):
        super(TemporalConvLayer_Residual, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.conv = nn.Conv2d(c_in, 2*c_out, (kernel, 1), 1, dilation = dia, padding = (0,0))
        if self.c_in > self.c_out:
            self.conv_self = nn.Conv2d(c_in, c_out, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # get the last two dims of 'x'
        b, _, T, n = list(x.size())
        if self.c_in > self.c_out:
            # [batch, c_out, timesteps, num_nodes]
            x_self = self.conv_self(x)
        elif self.c_in < self.c_out:
            # [batch, c_out, timesteps, num_nodes]
            x_self = torch.cat([x, torch.zeros([b, self.c_out - self.c_in, T, n]).to(x)], dim=1)
        else:
            x_self = x
        conv_x = self.conv(x)
        # get the timesteps dim of 'conv(x)'
        _, _, T_new, _ = list(conv_x.size())
        # need 'x_self' has the same shape of 'P'
        x_self = x_self[:, :, -T_new:, :]
        P = conv_x[:, 0:self.c_out, :, :]
        Q = conv_x[:, -self.c_out:, :, :]
        # residual connection added
        gated_conv = (P + x_self) * self.sigmoid(Q)
        return gated_conv


class SpatialConvLayer(nn.Module):
    '''
    Section 3.2 in the paper
    Graph convolution layer (GCN used here as the spatial CNN)

    Inputs:
        c_in: input channels
        c_out: output channels
        x: input with the shape [batch_size, c_in, timesteps, num_nodes]

    Return:
        y: output with the shape [batch_size, c_out, timesteps, num_nodes]

    '''
    def __init__(self, A, K, c_in, c_out):
        super(SpatialConvLayer, self).__init__()
        self.gc = GCN(A, K, c_in, c_out).to(device)

    def forward(self, x):
        # [batch, c_in, ts, nodes] --> [nodes, c_in, ts, batch]
        x = x.transpose(0, 3)
        # [nodes, c_in, ts, batch] --> [nodes, batch, ts, c_in]
        x = x.transpose(1, 3)
        # output: [nodes, batch, ts, c_out]
        output = self.gc(x)
        # [nodes, batch, ts, c_out] --> [nodes, c_out, ts, batch]
        output = output.transpose(1, 3)
        # [nodes, c_out, ts, batch] --> [batch, c_out, ts, nodes]
        output = output.transpose(0, 3)
        # return with the shape: [batch, c_out, ts, nodes]
        return torch.relu(output)


class OutputLayer(nn.Module):
    '''
    several temporal conv layers with a fully-connected layer as the output layer

    Inputs:
        c: input channels, c_in = c_out = c
        T: same as the timesteps dimention in 'x'
        n: number of nodes
        x: input with the shape [batch_size, c_in, timesteps, num_nodes]

    Outputs:
        y: output with the shape [batch_size, 1, 1, num_nodes]

    '''

    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation = 1, padding = (0,0)).to(device)
        self.ln = nn.LayerNorm([n, c]).to(device)
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation = 1, padding = (0,0)).to(device)
        self.fc = nn.Conv2d(c, 1, 1).to(device)

    def forward(self, x):
        # maps multi-steps to one
        # [batch, c_in, ts, nodes] --> [batch, c_out_1, 1, nodes]
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # [batch, c_out_1, 1, nodes] --> [batch, c_out_2, 1, nodes]
        x_t2 = self.tconv2(x_ln)
        # maps multi-channels to one
        # [batch, c_out_2, 1, nodes] --> [batch, 1, 1, nodes]
        return self.fc(x_t2)


class OutputLayer_Simple(nn.Module):
    '''

    ** Simplified version of 'OutputLayer' **

    The second half of the section 3.4 in the paper
    Single temporal conv layers with a fully-connected layer as the output layer

    Inputs:
        c: input channels, c_in = c_out = c
        T: same as the timesteps dimention in 'x'
        n: number of nodes
        x: input with the shape [batch_size, c_in, timesteps, num_nodes]

    Outputs:
        y: output with the shape [batch_size, 1, 1, num_nodes]

    '''

    def __init__(self, c, T):
        super(OutputLayer_Simple, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation = 1, padding = (0,0)).to(device)
        self.fc = nn.Conv2d(c, 1, 1).to(device)

    def forward(self, x):
        # [batch, c_in, ts, nodes] --> [batch, c_out = c_in, 1, nodes]
        x_t1 = self.tconv1(x)
        # [batch, c_out = c_in, 1, nodes] --> [batch, 1, 1, nodes]
        return self.fc(x_t1)


class STGCN(nn.Module):
    '''
    STGCN network described in the paper (Figure 2)

    Inputs:
        c: channels, e.g. [1, 64, 16, 64, 64, 16, 64] where [1, 64] means c_in and c_out for the first temporal layer
        T: window length, e.g. 12
        Tp: prediction window length, e.g. 4
        n: num_nodes
        p: dropout after each 'sandwich', i.e. 'TSTN', block
        control_str: model strcture controller, e.g. 'TSTNTSTN'; T: Temporal Layer, S: Spatio Layer, N: Norm Layer
        x: input feature matrix with the shape [batch, 1, T, n]

    Return:
        y: output with the shape [batch, 1, Tp, n]

    '''

    def __init__(self, A, K, c, T, Tp, n, p, control_str):
        super(STGCN, self).__init__()
        self.Tp = Tp
        self.stg_one_step = STGCN_one_step(A, K, c, T, n, p, control_str).to(device)

    def forward(self, x):
        T = x.shape[-2]
        for i in range(T):
            x1 = x[:,:,-T:,:]
            y1 = self.stg_one_step(x1)
            x = torch.concat((x, y1), dim=-2)
        y = x[:,:,-T:,:]
        return y

class STGCN_one_step(nn.Module):
    '''
    STGCN network described in the paper (Figure 2)

    Inputs:
        c: channels, e.g. [1, 64, 16, 64, 64, 16, 64] where [1, 64] means c_in and c_out for the first temporal layer
        T: window length, e.g. 12
        n: num_nodes
        g: fixed DGLGraph
        p: dropout after each 'sandwich', i.e. 'TSTN', block
        control_str: model strcture controller, e.g. 'TSTNTSTN'; T: Temporal Layer, S: Spatio Layer, N: Norm Layer
        x: input feature matrix with the shape [batch, 1, T, n]

    Return:
        y: output with the shape [batch, 1, 1, n]

    '''

    def __init__(self, A, K, c, T, n, p, control_str):

        super(STGCN_one_step, self).__init__()

        self.control_str = control_str
        self.num_layers = len(control_str)
        self.num_nodes = n
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p)
        # Temporal conv kernel size set to 3
        self.Kt = 3
        # c_index controls the change of channels
        c_index = 0
        num_temporal_layers = 0

        # construct network based on 'control_str'
        for i in range(self.num_layers):

            layer_i = control_str[i]

            # Temporal Layer
            if layer_i == 'T':
                self.layers.append(TemporalConvLayer_Residual(c[c_index], c[c_index + 1], kernel = self.Kt).to(device))
                c_index += 1
                num_temporal_layers += 1

            # Spatio Layer
            if layer_i == 'S':
                self.layers.append(SpatialConvLayer(A, K, c[c_index], c[c_index + 1]).to(device))
                c_index += 1

            # Norm Layer
            if layer_i == 'N':
                # TODO: The meaning of this layernorm
                self.layers.append(nn.LayerNorm([n, c[c_index]]).to(device))

        # c[c_index] is the last element in 'c'
        # T - (self.Kt - 1) * num_temporal_layers returns the timesteps after previous temporal layer transformations cuz dialiation = 1
        self.output = OutputLayer(c[c_index], T - (self.Kt - 1) * num_temporal_layers, self.num_nodes).to(device)

    def forward(self, x):
        # Example:
        # batch=64, input_channel=1, window_length=12, num_nodes=207, temporal_kernel = 2
        #   input.shape: torch.Size([64, 1, 12, 207])
        #   T output.shape: torch.Size([64, 64, 11, 207])
        #   S output.shape: torch.Size([64, 16, 11, 207])
        #   T output.shape: torch.Size([64, 64, 10, 207])
        #   N output.shape: torch.Size([64, 64, 10, 207])
        #   T output.shape: torch.Size([64, 64, 9, 207])
        #   S output.shape: torch.Size([64, 16, 9, 207])
        #   T output.shape: torch.Size([64, 64, 8, 207])
        #   OutputLayer output.shape: torch.Size([64, 1, 1, 207])

        for i in range(self.num_layers):
            layer_i = self.control_str[i]
            if layer_i == 'N':
                # x.permute(0, 2, 3, 1) leads
                # [batch, channel, timesteps, nodes] to [batch, timesteps, nodes, channel]
                # self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) leads
                # [batch, timesteps, nodes, channel] to [batch, channel, timesteps, nodes]
                x = self.dropout(self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            else:
                # x.shape is [batch, channel, timesteps, nodes]
                x = self.layers[i](x)
        return self.output(x)  # [batch, 1, 1, nodes]



class STGCN_WAVE(nn.Module):
    '''
    Improved variation of the above STGCN network
        + Extra temporal conv, layernorm, and sophisticated output layer design
        + temporal conv with increasing dialations like in TCN

    Inputs:
        c: channels, e.g. [1, 16, 32, 64, 32, 128] where [1, 16] means c_in and c_out for the first temporal layer
        T: window length, e.g. 144, which should larger than the total dialations
        n: num_nodes
        g: fixed DGLGraph
        p: dropout
        control_str: model strcture controller, e.g. 'TNTSTNTSTN'; T: Temporal Layer, S: Spatio Layer, N: Norm Layer
        x: input feature matrix with the shape [batch, 1, T, n]

    Return:
        y: output with the shape [batch, 1, 1, n]

    Notice:
        ** Temporal layer changes c_in to c_out, but spatial layer doesn't change
           in this way where c_in = c_out = c
    '''

    def __init__(self, A, K, c, T, n, p, control_str):

        super(STGCN_WAVE, self).__init__()

        self.control_str = control_str
        self.num_layers = len(control_str)
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p).to(device)
        # c_index controls the change of channels
        c_index = 0
        # diapower controls the change of dilations in temporal CNNs where dilation = 2^diapower
        diapower = 0

        # construct network based on 'control_str'
        for i in range(self.num_layers):

            layer_i = control_str[i]

            # Temporal Layer
            if layer_i == 'T':
                # Notice: dialation = 2^diapower (e.g. 1, 2, 4, 8) so that
                # T_out = T_in - dialation * (kernel_size - 1) - 1 + 1
                # if padding = 0 and stride = 1
                self.layers.append(TemporalConvLayer_Residual(c[c_index], c[c_index + 1], dia = 2**diapower).to(device))
                diapower += 1
                c_index += 1

            # Spatio Layer
            if layer_i == 'S':
                self.layers.append(SpatialConvLayer(A, K, c[c_index], c[c_index]).to(device))

            # Norm Layer
            if layer_i == 'N':
                # TODO: The meaning of this layernorm
                self.layers.append(nn.LayerNorm([n, c[c_index]]).to(device))

        # c[c_index] is the last element in 'c'
        # T + 1 - 2**(diapower) returns the timesteps after previous temporal layer transformations
        # 'n' will be needed by LayerNorm inside of the OutputLayer
        self.output = OutputLayer(c[c_index], T + 1 - 2**(diapower), n).to(device)

    def forward(self, x):
        # Example:
        # batch=8, input_channel=1, window_length=144, num_nodes=207, temporal_kernel = 2
        #   x.shape: torch.Size([8, 1, 144, 207])
        #   T output.shape: torch.Size([8, 16, 143, 207])
        #   N output.shape: torch.Size([8, 16, 143, 207])
        #   T output.shape: torch.Size([8, 32, 141, 207])
        #   S output.shape: torch.Size([8, 32, 141, 207])
        #   T output.shape: torch.Size([8, 64, 137, 207])
        #   N output.shape: torch.Size([8, 64, 137, 207])
        #   T output.shape: torch.Size([8, 32, 129, 207])
        #   S output.shape: torch.Size([8, 32, 129, 207])
        #   T output.shape: torch.Size([8, 128, 113, 207])
        #   Outputlayer output.shape: torch.Size([8, 1, 1, 207])

        for i in range(self.num_layers):
            layer_i = self.control_str[i]
            if layer_i == 'N':
                # x.permute(0, 2, 3, 1) leads
                # [batch, channel, timesteps, nodes] to [batch, timesteps, nodes, channel]
                # self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) leads
                # [batch, timesteps, nodes, channel] to [batch, channel, timesteps, nodes]
                x = self.dropout(self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            else:
                # x.shape is [batch, channel, timesteps, nodes]
                x = self.layers[i](x)
        return self.output(x)  # [batch, 1, 1, nodes]