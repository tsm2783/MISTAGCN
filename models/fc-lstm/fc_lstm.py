import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn,rnn

class LSTM_m(nn.Block):
    r'''
    This is a LSTM extension model, which accepts the last Tr samples and yiels next Tp samples, (num, N, T) -> (num, N, Tp).
    '''
    def __init__(self, Tp, **kwargs):
        super(LSTM_m, self).__init__(**kwargs)
        self.Tp = Tp
        self.lstm = rnn.LSTM(hidden_size=Tp, num_layers=7) #default layout, 'TNC' = 'sequence length, batch size, feature dimensions'
        self.fc = nn.Dense(Tp, activation='relu', flatten=False)  #map elements of ndarray from [0,1] to [0,\inf]

    def forward(self, x):
        num, N, F, T = x.shape
        x = nd.reshape(x, shape=(num, N, -1))
        out = nd.transpose(x, axes=(1,0,2))   #(sequence_length, batch_size, input_size) <-> (N, num, T)
        out = self.lstm(out)  #(sequence_length, batch_size, num_hidden) <-> (N, num, Tp)
        out = self.fc(out) #(N, num, Tp)
        out = nd.transpose(out, axes=(1,0,2))  #(num, N, Tp)
        return out