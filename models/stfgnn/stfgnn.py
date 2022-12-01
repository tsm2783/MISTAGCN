import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
import math
import numpy as np


class PositionEmbedding(nn.Block):
    """An implementation of STSGCL.

    Args:
        input_length (int): T
        num_vertices (int): N
        embedding_size (int): C, num_features
        temporal (bool): whether equip temporal embedding
        spatial (bool): whether equip spatial embedding
    """
    def __init__(self, input_length, num_vertices, embedding_size,
                 temporal=True, spatial=True, 
                 **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.temporal = temporal
        self.spatial = spatial
        # parameters to train
        with self.name_scope():
            self.temporal_emb = self.params.get('temporal_emb', shape=(1, input_length, 1, embedding_size), init=mx.init.Xavier())  #(1, T, 1, C)
            # self.temporal_emb = self.params.get('temporal_emb', shape=(input_length, 1, embedding_size), init=mx.init.Xavier())  #(T, 1, C)
            self.spatial_emb = self.params.get('spatial_emb', shape=(1, 1, num_vertices, embedding_size), init=mx.init.Xavier())  #(1, 1, N, C)
            # self.spatial_emb = self.params.get('spatial_emb', shape=(1, num_vertices, embedding_size), init=mx.init.Xavier())  #(1, N, C)

    def forward(self, x):
        # x (ndarray): input sequence, with shape (B, T, N, C) -- (batch_size, input_length, num_nodes, features).
        # out (ndarray): ouput sequence for prediction, with shape (B, T, N, C).
        ctx = x.ctx
        B, T, N, C = x.shape

        if self.temporal:
            out = nd.broadcast_add(x, self.temporal_emb.data(ctx))
        if self.spatial_emb:
            out = nd.broadcast_add(x, self.spatial_emb.data(ctx))

        # out = []
        # for i in range(B):
        #     xi = x[i]
        #     if self.temporal:
        #         xi = nd.transpose(xi, (0,2,1))
        #         emb = nd.transpose(self.temporal_emb.data(ctx), (0,2,1))
        #         xi = nd.broadcast_add(xi, emb)
        #     if self.spatial:
        #         xi = nd.transpose(xi, (1,2,0))
        #         emb = nd.transpose(self.spatial_emb.data(ctx), (1,2,0))
        #         xi = nd.broadcast_add(xi, emb)
        #     out.append(nd.expand_dims(xi, axis=0))
        # out = nd.concat(*out, dim=0)

        return out


class GCN_operation(nn.Block):
    """An implementation of STSGCL.

    Args:
        ADJ (ndarray): (4N, 4N).
        filters (int): C'.
        num_vertices (int): N.
        use_mask (bool): whether mask the adjacency matrix or not.
    """
    def __init__(self, ADJ, filters,  num_vertices, 
                use_mask=True, 
                **kwargs):
        super(GCN_operation, self).__init__(**kwargs)
        self.ADJ = ADJ
        self.use_mask = use_mask
        # mask_init_value = ADJ != 0
        # mask_init_value = nd.ones(shape=(4*num_vertices, 4*num_vertices))
        with self.name_scope():
            # self.mask = self.params.get('mask', shape=(4*num_vertices, 4*num_vertices), init=mask_init_value)
            self.mask = self.params.get('mask', shape=(4*num_vertices, 4*num_vertices), init=mx.init.One())
        self.full = nn.Dense(filters, activation='relu', flatten=False)

    def forward(self, x):
        # x (ndarray): input sequence, with shape (4N, B, C).
        # out (ndarray): ouput sequence for prediction, with shape (4N, B, C').
        ctx = x.ctx
        if self.use_mask:
            ADJ = self.ADJ.as_in_context(ctx) * self.mask.data(ctx)
        x = nd.dot(ADJ, x)
        x = self.full(x)
        return x


class STSGCM(nn.Block):
    """An implementation of STSGCM, multiple stacked gcn layers with cropping and max operation.

    Args:
        ADJ (ndarray): (4N, 4N).
        list_filters (list[int]): list of C'.
        num_vertices (int): N.
        use_mask (bool): whether mask the adjacency matrix or not.
    """
    def __init__(self, ADJ, list_filters, num_vertices, use_mask=True, **kwargs):
        super(STSGCM, self).__init__(**kwargs)
        self.ADJ = ADJ
        self.list_filters = list_filters
        self.num_vertices = num_vertices
        self.submodules = []
        for i in range(len(list_filters)):
            self.submodules.append(
                GCN_operation(ADJ, list_filters[i], num_vertices, use_mask)
            )
            self.register_child(self.submodules[-1])
            features = list_filters[i]

    def forward(self, x):
        # x (ndarray): input sequence, with shape (4N, B, C).
        # out (ndarray): ouput sequence for prediction, with shape (N, B, C').
        ctx = x.ctx
        need_concat = [self.submodules[idx](x) for idx in range(len(self.list_filters))]
        # shape of each element is (1, N, B, C')
        need_concat = [
           nd.expand_dims(
                nd.slice(
                    i,
                    begin=(self.num_vertices, None, None),
                    end=(2 * self.num_vertices, None, None)
                ), 0
            ) for i in need_concat
        ]
        # shape is (N, B, C')
        x = nd.max(nd.concat(*need_concat, dim=0), axis=0)
        return x


class SthgcnLayerIndividual(nn.Block):
    """An implementation of SthgcnLayerIndividual, multiple individual STSGCMs.

    Args:
        ADJ (ndarray): (4N, 4N).
        input_length (int): T.
        num_vertices (int): N.
        features (int): C.
        list_filters (list[int]): list of C'.
        use_mask (bool): whether mask the adjacency matrix or not.
        temporal (bool): whether equip temporal embedding.
        spatial (bool): whether equip spatial embedding.
    """
    def __init__(self, ADJ, input_length, num_vertices, features, list_filters,
                use_mask = True,
                temporal=True, spatial=True,
                **kwargs):
        super(SthgcnLayerIndividual, self).__init__(**kwargs)
        self.input_length = input_length
        self.num_vertices = num_vertices
        self.features = features
        self.position_embedding = PositionEmbedding(input_length, num_vertices, features, temporal, spatial)
        self.conv1 = nn.Conv2D(features, kernel_size=(1,1), strides=(1,1), dilation=(1,1)) #kernel_size=(1,2) and dilation=(1,3) -> T'=T-3 -> T should be large enough
        self.conv2 = nn.Conv2D(features, kernel_size=(1,1), strides=(1,1), dilation=(1,1))
        self.stsgcm = STSGCM(ADJ, list_filters, num_vertices, use_mask)
        self.linear = nn.Dense(list_filters[-1], activation='relu', flatten=False)  #added

    def forward(self, x):
        # x (ndarray): input sequence, with shape (B, T, N, C).
        # out (ndarray): ouput sequence for prediction, with shape (B, T, N, C').
        ctx = x.ctx
        B, T, N, C = x.shape
        x = self.position_embedding(x) #(B, T, N, C)
        temp = nd.transpose(x, (0,3,2,1))  #(B, C, N, T)
        left = nd.sigmoid(self.conv1(temp))
        right = nd.tanh(self.conv2(temp))
        data_time_axis = left * right
        data_res = nd.transpose(data_time_axis, (0, 3, 2, 1))  #(B, T, N, C)

        need_concat = []
        for i in range(self.input_length - 3):
            # shape is (B, 4, N, C)
            t = nd.slice(x, begin=(None, i, None, None), end=(None, i + 4, None, None))
            # shape is (B, 4N, C)
            t = nd.reshape(t, (-1, 4 * self.num_vertices, self.features))
            # shape is (4N, B, C)
            t = nd.transpose(t, (1, 0, 2))
            # shape is (N, B, C')
            t = self.stsgcm(t)
            # shape is (B, N, C')
            t = nd.swapaxes(t, 0, 1)
            # shape is (B, 1, N, C')
            need_concat.append(nd.expand_dims(t, axis=1))
        # added
        for i in range(self.input_length-3, self.input_length):
            t = nd.slice(x, begin=(None, i, None, None), end=(None, i + 1, None, None)) #(B, 1, N, C)
            t = self.linear(t) #(B, 1, N, C')
            need_concat.append(t)

        # shape is (B, T, N, C')
        out = nd.concat(*need_concat, dim=1)
        out = out + data_res
        return out


class OutputLayer(nn.Block):
    """An implementation of the output layer.

    Args:
        predict_length (int) : number of steps to be predicted, T'.
    """
    def __init__(self, 
                # num_vertices, input_length, features,
                # filters=128, predict_length=3, 
                predict_length=1, 
                **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        self.full = nn.Dense(1, activation='relu', flatten=False)
        self.full1 = nn.Dense(predict_length, activation='relu', flatten=False)

    def forward(self, x):
        # x (ndarray): input sequence, with shape (B, T, N, C) -- (batch_size, input_length, num_nodes, features)
        # out (ndarray): ouput sequence for prediction, with shape (B, T', N) -- (batch_size, predict_length, num_nodes).
        ctx = x.ctx
        B, T, N, C = x.shape

        # x = nd.swapaxes(x, 1, 2)  #(B, N, T, C)
        # x = nd.reshape(x, shape=(-1, N, T * C))  #(B, N, T * C)
        # x = self.full(x)  #(B, N, C')
        # x = self.full1(x)  #(B, N, T')
        # x = nd.swapaxes(x, 1, 2)  #(B, T', N)

        x = self.full(x) #(B, T, N, 1)
        x = x[..., 0]  #(B, T, N)
        x = nd.transpose(x, (0,2,1))  #(B, N, T)
        x = self.full1(x)  #(B, N, T')
        x = nd.transpose(x, (0,2,1)) #(B, T', N)

        return x


class STSGCL(nn.Block):
    """An implementation of STSGCL.

    Args:
        ADJ (ndarray) : (a combination of temporal and spatial) adjacency matrix, with shape (4N, 4N).
        input_length (int): T.
        num_vertices (int): N.
        features (int): C.
        list_filters (list): list of C'.
        use_mask (bool): whether mask the adjacency matrix or not.
        module_type (str): in {'sharing', 'individual'}.
        temporal (bool): whether equip temporal embedding.
        spatial (bool): whether equip spatial embedding.
    """
    def __init__(self, ADJ, input_length, num_vertices, features, list_filters,
                use_mask=True,
                module_type='individual', 
                temporal=True, spatial=True, 
                **kwargs):
        super(STSGCL, self).__init__(**kwargs)
        # assert module_type in {'sharing', 'individual'}
        self.module_type = module_type
        self.sthgcn_layer_individual = SthgcnLayerIndividual(ADJ, input_length, num_vertices, features, list_filters, use_mask, temporal, spatial)

    def forward(self, x):
        # x (ndarray): input sequence, with shape (B, T, N, C) -- (batch_size, input_length, num_nodes, features).
        # out (ndarray): ouput sequence for prediction, with shape (B, T, N, C') -- (batch_size, hidden_length, num_nodes, num_hidden_features).
        ctx = x.ctx
        if self.module_type == 'individual':
            x = self.sthgcn_layer_individual(x)
        return x


class STSGCN(nn.Block):
    """An implementation of STSGCM, multiple stacked gcn layers with cropping and max operation.

    Args:
        ADJ (ndarray): (a combination of temporal and spatial) adjacency matrix, with shape (4N, 4N).
        input_length (int): T.
        num_vertices (int): N.
        features (int): C.
        list_2d_filters (list[list[int]]): a list of lists of filters (number of hidden features).
        predict_length (int): number of steps to predict.
        use_mask (bool): whether mask the adjacency matrix or not.
        module_type (str): in {'sharing', 'individual'}.
        temporal (bool): whether equip temporal embedding.
        spatial (bool): whether equip spatial embedding.
    """
    def __init__(self, ADJ, input_length, num_vertices, features, 
                list_2d_filters, predict_length,
                use_mask=True,  # mask_init_value=None,
                module_type='individual',
                temporal=True, spatial=True, 
                # rho=1, 
                **kwargs):
        super(STSGCN, self).__init__(**kwargs)
        self.predict_length = predict_length
        # with self.name_scope():
        #     if use_mask:
        #         if mask_init_value is None:
        #             raise ValueError("mask init value is None!")
        #         mask = self.params.get('mask', shape=(4*num_vertices, 4*num_vertices), init=mask_init_value)
        #         self.ADJ = mask.data(ADJ.ctx) * ADJ
        #     else:
        #         self.ADJ = ADJ
        
        self.blk = nn.Sequential()
        for idx, list_filters in enumerate(list_2d_filters):
            self.blk.add(STSGCL(ADJ, input_length, num_vertices, features, list_filters, use_mask, module_type, temporal, spatial))
            # input_length -= 3
            features = list_filters[-1]

        # self.submodules = []
        # for i in range(predict_length):
        #     self.submodules.append(
        #         OutputLayer(num_vertices, input_length, features, filters=128, predict_length=1)
        #     )
        #     self.register_child(self.submodules[-1])

        # added
        self.outlayer = OutputLayer(predict_length=predict_length)

    def forward(self, x):
        # x (ndarray): input sequence, with shape (B, T, N, C) -- (batch_size, input_length, num_nodes, features)
        # out (ndarray): ouput sequence for prediction, with shape (B, T', N) -- (batch_size, predict_length, num_nodes).
        ctx = x.ctx
        x = self.blk(x)

        # # (B, 1, N)
        # need_concat = [self.submodules[idx](x) for idx in range(self.predict_length)]
        # # shape is (N, B, C')
        # out = nd.concat(*need_concat, dim=1)

        # added
        x = self.outlayer(x)
        
        return x


class STFGNN(nn.Block):
    """An implementation of STFGNN.

    Args:
        ADJ_s (ndarray) : adjacency matrix of spatial graph.
        ADJ_t (ndarray) : adjacency matrix of temporal graph.
        input_length (int): T.
        num_vertices (int): N.
        predict_length (int) : number of steps to predict.
        use_mask (bool): whether mask the adjacency matrix or not.
        module_type (str): in {'sharing', 'individual'}.
        temporal (bool): whether equip temporal embedding.
        spatial (bool): whether equip spatial embedding.
    """
    def __init__(self, ADJ_s, ADJ_t, 
                input_length, num_vertices, 
                predict_length=3, 
                use_mask=True,  # mask_init_value=None, 
                module_type='individual',
                temporal=True, spatial=True, 
                **kwargs):
        super(STFGNN, self).__init__(**kwargs)
        ADJ = self.construct_adj_fusion(ADJ_s, ADJ_t, 4)
        # mask_init_value = mx.init.Constant(value=(ADJ != 0).astype('float32').tolist())
        # mask_init_value = ADJ != 0
        # ADJ = ADJ * mask_init_value  #no effiect here
        list_2d_filters = [[64, 64, 64],
                            [64, 64, 64],
                            [64, 64, 64]]
        self.full = nn.Dense(64, activation='relu', flatten=False)
        self.stsgcn = STSGCN(ADJ, input_length, num_vertices, 64, list_2d_filters, predict_length, use_mask, module_type, temporal, spatial)

    def forward(self, x):
        # x (ndarray): input sequence, with shape (B, T, N, C) -- (batch_size, input_length, num_nodes, features).
        # out (ndarray): ouput sequence for prediction, with shape (B, T', N) -- (batch_size, num_pred, num_nodes).
        ctx = x.ctx
        x = self.full(x)
        x = self.stsgcn(x)
        return x

    def construct_adj_fusion(self, A, A_dtw, steps):
        '''
        construct a bigger adjacency matrix using the given matrix

        Parameters
        ----------
        A: nd.ndarray, adjacency matrix, shape is (N, N)

        A_dtw: nd.ndarray, adjacency matrix, shape is (N, N)

        steps: how many times of the does the new adj mx bigger than A

        Returns
        ----------
        new adjacency matrix: mixed matrix, shape is (N * steps, N * steps)

        ----------
        This is 4N_1 mode:

        [T, 1, 1, T
        1, S, 1, 1
        1, 1, S, 1
        T, 1, 1, T]

        '''

        N = len(A)
        adj = nd.zeros([N * steps] * 2) # "steps" = 4 !!!  #(4N, 4N)

        # construct the diagonal blocks
        for i in range(steps):
            if (i == 1) or (i == 2):
                adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A
            else:
                adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw
        # sub diagonal blocks, identity matrices
        for i in range(N):
            for k in range(steps - 1):
                adj[k * N + i, (k + 1) * N + i] = 1
                adj[(k + 1) * N + i, k * N + i] = 1
        # the block in the upper-right corner of the matrix and the one in the bottom-left corner
        adj[3 * N: 4 * N, 0:  N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]
        adj[0 : N, 3 * N: 4 * N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]
        # the sub sub diagonal blocks
        adj[2 * N: 3 * N, 0 : N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
        adj[0 : N, 2 * N: 3 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
        adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
        adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]

        for i in range(len(adj)):
            adj[i, i] = 1

        return adj