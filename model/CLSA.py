import torch
import torch.nn
import torch.nn.functional as F


class CLSA(torch.nn.Module):
    def __init__(self, model_params):
        super(CLSA, self).__init__()
        # device
        self.device = model_params['device']

        # loss function
        # BCE
        self.criterion_1 = torch.nn.BCELoss()
        self.criterion_1.to(self.device)
        # MSE
        self.criterion_2 = torch.nn.MSELoss()
        self.criterion_2.to(self.device)

        # batch size
        self.batch_size = model_params['batch_size']

        # day
        self.day = model_params['day']
        # action user : one_hot_num
        self.a_feat_size = model_params['a_feat_size']
        self.u_feat_size = model_params['u_feat_size']
        # action user : feature_num
        self.a_field_size = model_params['a_field_size']
        self.u_field_size = model_params['u_field_size']

        # conv2 parameters
        self.clas_conv2_kernel = model_params['clsa_conv2_kernel']
        self.clsa_conv2_output_size = model_params['clsa_conv2_output_size']
        self.clas_pool_kernel = model_params['clsa_pool_kernel']
        if ((self.day - self.clas_conv2_kernel + 1) - self.clas_pool_kernel) // 2 < 0:
            self.clas_conv2_kernel = 1
        # conv2d
        self.conv2 = torch.nn.Conv2d(1, self.clsa_conv2_output_size, self.clas_conv2_kernel, stride=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=self.clas_pool_kernel, stride=2)

        # linear
        self.lstm_input_size = model_params['lstm_input_size']
        h2 = ((self.day - self.clas_conv2_kernel + 1) - self.clas_pool_kernel) // 2 + 1
        w2 = ((self.a_field_size - self.clas_conv2_kernel + 1) - self.clas_pool_kernel) // 2 + 1
        input_size = self.clsa_conv2_output_size * h2 * w2
        self.fc_1 = torch.nn.Linear(input_size, self.day * self.lstm_input_size)

        # LSTM

        self.hidden_size = model_params['hidden_size']
        self.lstm = torch.nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.hidden_size, bidirectional=True,
                                  batch_first=True)

        # attention
        self.attention_input_size = model_params['attention_input_size']
        self.fc_2 = torch.nn.Linear(self.attention_input_size, self.attention_input_size)
        self.fc_3 = torch.nn.Linear(self.attention_input_size, 1, bias=False)

        # fc out
        self.fc_out = torch.nn.Linear(self.attention_input_size, 1)

        # activation function
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ui, uv, ai, av, y=None, time=None, loss_func='MSE'):
        # av: [batch_size, 1, day, a_field_size]
        av = torch.unsqueeze(av, dim=1)
        # av_conv: [batch_size, clsa_conv2_output_size, h1, w1]
        # h1: day - clas_conv2_kernel + 1
        # w1: a_field_size - clas_conv2_kernel + 1
        av_conv = self.relu(self.conv2(av))
        # av_conv: [batch_size, clsa_conv2_output_size, h2, w2]
        # h2: (h1 - self.clas_pool_kernel) // 2 + 1
        # w2: (w1 - self.clas_pool_kernel) // 2 + 1
        av_maxpool = self.maxpool(av_conv)
        # av_fc: [batch_size, clsa_conv2_output_size * h2 * w2]
        av_fc = av_maxpool.reshape(self.batch_size, -1)

        # linear
        # av_fc: [batch_size, day * lstm_input_size]
        av_fc = self.relu(self.fc_1(av_fc))
        # av_fc: [batch_size, day, lstm_input_size]
        av_fc = av_fc.reshape(self.batch_size, self.day, -1)

        # LSTM
        # output:　[batch_size, day, num_directions * hidden_size]
        output, (h_n, c_n) = self.lstm(av_fc)

        # attention
        # m: [batch_size, day, num_directions * hidden_size]
        m = self.tanh(self.fc_2(output))
        # s: [batch_size, day, 1]
        s = self.sigmoid(self.fc_3(m))
        # r: [batch_size, day, num_directions * hidden_size]
        r = output * s
        # y_deep: [batch_size, num_directions * hidden_size]
        y_deep = torch.sum(r, dim=1)

        # y_pred: [batch_size, 1]
        y_pred = torch.sigmoid(self.fc_out(y_deep))

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
