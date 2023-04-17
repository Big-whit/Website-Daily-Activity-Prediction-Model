import torch
import torch.nn
import torch.nn.functional as F


class LSCNN(torch.nn.Module):
    def __init__(self, model_params):
        super(LSCNN, self).__init__()
        # device
        self.device = model_params['device']

        # loss function
        # BCE
        self.criterion_1 = torch.nn.BCELoss()
        self.criterion_1.to(self.device)
        # MSE
        self.criterion_2 = torch.nn.MSELoss()
        self.criterion_2.to(self.device)

        # batch_size
        self.batch_size = model_params['batch_size']
        # day
        self.day = model_params['day']
        # action user : one_hot_num
        self.a_feat_size = model_params['a_feat_size']
        self.u_feat_size = model_params['u_feat_size']
        # action user : feature_num
        self.a_field_size = model_params['a_field_size']
        self.u_field_size = model_params['u_field_size']

        # embedding layer
        self.embedding_size = model_params['embedding_size']
        self.a_embeddings = torch.nn.Embedding(self.a_feat_size, self.embedding_size)
        self.u_embeddings = torch.nn.Embedding(self.u_feat_size, self.embedding_size)

        # LSTM parameters
        self.lstm_1_input_size = self.embedding_size * self.a_field_size
        self.lstm_2_input_size = model_params['lstm_2_input_size']
        self.hidden_size = model_params['hidden_size']
        # LSTM
        self.lstm_1 = torch.nn.LSTM(input_size=self.lstm_1_input_size, hidden_size=self.hidden_size, batch_first=True)
        self.lstm_2 = torch.nn.LSTM(input_size=self.lstm_2_input_size, hidden_size=self.hidden_size, batch_first=True)

        # max pool
        self.lscnn_conv2_kernel = model_params['lscnn_conv2_kernel']
        self.lscnn_conv2_output_size = model_params['lscnn_conv2_output_size']
        self.lscnn_pool_kernel = model_params['lscnn_pool_kernel']

        #  conv2d
        self.conv2 = torch.nn.Conv2d(1, self.lscnn_conv2_output_size, self.lscnn_conv2_kernel, padding='same',
                                     stride=1, bias=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=self.lscnn_pool_kernel, stride=2)

        # linear parameters
        h2 = (self.day - self.lscnn_conv2_kernel + 3 - self.lscnn_pool_kernel) // 2 + 1
        w2 = (self.a_field_size - self.lscnn_conv2_kernel + 3 - self.lscnn_pool_kernel) // 2 + 1
        liner_1_input_size = self.lscnn_conv2_output_size * h2 * w2
        # linear
        self.fc_1 = torch.nn.Linear(liner_1_input_size, self.day * self.lstm_2_input_size)
        self.fc_out = torch.nn.Linear(self.hidden_size, 1, bias=False)

        # dropout
        self.dropout = torch.nn.Dropout(model_params['dropout_p'])

        # activation function
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.wd = 0.5

    def forward(self, ui, uv, ai, av, y=None, time=None, loss_func='MSE'):
        # batch_size: 32
        # ui、uv: [batch_size, u_field_size]
        # ai、av: [batch_size, day, a_field_size]
        # y: [batch_size, 1]

        # LSTM
        # a_emb: [batch_size, day, a_field_size * embedding_size]
        a_emb = self.a_embeddings(ai)
        a_emb = torch.multiply(a_emb, av.reshape(-1, self.day, self.a_field_size, 1))
        a_emb = a_emb.reshape(self.batch_size, self.day, -1)

        # output_1: [batch_size, day, hidden_size]
        output_1, (h_n, c_n) = self.lstm_1(a_emb)
        # y_deep_1: [batch_size, hidden_size]
        y_deep_1 = torch.sum(output_1, dim=1)
        # y_pred_1: [batch_size,1]
        y_pred_1 = self.sigmoid(self.fc_out(y_deep_1))

        # CNN + LSTM
        # av: [batch_size, 1, day, a_field_size]
        av = torch.unsqueeze(av, dim=1)
        # av_conv: [batch_size, lscnn_conv2_output_size, h1, w1]
        # h1: self.day - self.lscnn_conv2_kernel + 3
        # w1: self.a_field_size - self.lscnn_conv2_kernel + 3
        av_conv = self.conv2(av)
        av_conv = self.relu(av_conv)
        # av_maxpool: [batch_size, lscnn_conv2_output_size, h2, w2]
        # h2: (h1 - self.lscnn_pool_kernel) // 2 + 1
        # w2: (w1 - self.lscnn_pool_kernel) // 2 + 1
        av_maxpool = self.maxpool(av_conv)
        # av_fc: [batch_size, lscnn_conv2_output_size * h2 * w2]
        av_fc = av_maxpool.reshape(self.batch_size, -1)

        # linear
        # av_fc: [batch_size, day * lstm_2_input_size]
        av_fc = self.dropout(av_fc)
        av_fc = self.relu(self.fc_1(av_fc))

        # av_fc: [batch_size, day, lstm_2_input_size]
        av_fc = av_fc.reshape(self.batch_size, self.day, -1)
        # output_2: [batch_size, day, hidden_size]
        output_2, (h_n, c_n) = self.lstm_2(av_fc)
        # y_deep_2: [batch_size, hidden_size]
        y_deep_2 = torch.sum(output_2, dim=1)
        # y_pred_2: [batch_size, 1]
        y_pred_2 = self.sigmoid(self.fc_out(y_deep_2))

        # add sum
        y_pred = ((y_pred_1 + y_pred_2) * self.wd)

        # y_true_bool (Active or not): [batch_size, 1](int)
        esp = 1e-5
        y_true_bool = y.clone()
        y_true_bool[y >= esp] = 1.0
        y_true_bool[y < esp] = 0.0
        y_true_bool = y_true_bool.to(self.device)

        if y is not None:
            if loss_func == 'BCE':
                loss = self.criterion_1(y_pred, y_true_bool)
            else:
                loss = self.criterion_2(y_pred, y)
            return loss, y_pred
        else:
            return y_pred

    def batch_norm_layer(self, x):
        bn = self.batch_norm(x)
        return
