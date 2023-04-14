import torch


class RNN(torch.nn.Module):
    def __init__(self, model_params):
        super(RNN, self).__init__()
        # device
        self.device = model_params['device']

        # loss function
        # BCE
        self.criterion_1 = torch.nn.BCELoss()
        self.criterion_1.to(self.device)
        # MSE
        self.criterion_2 = torch.nn.MSELoss()
        self.criterion_2.to(self.device)

        # action user : one_hot_num
        self.a_feat_size = model_params['a_feat_size']
        self.u_feat_size = model_params['u_feat_size']
        # action user : feature_num
        self.a_field_size = model_params['a_field_size']
        self.u_field_size = model_params['u_field_size']

        # batch_size = 32, seq_length = day, input_size = action_type_num, hidden_size = 64
        self.batch_size = model_params['batch_size']
        self.seq_len = model_params['day']
        self.rnn_input_size = self.a_field_size
        self.rnn_hidden_size = model_params['rnn_hidden_size']
        self.rnn_num_layers = model_params['rnn_num_layers']

        self.rnn = torch.nn.RNN(input_size=self.rnn_input_size,
                                hidden_size=self.rnn_hidden_size,
                                num_layers=self.rnn_num_layers,
                                batch_first=True)

        self.fc_out = torch.nn.Linear(self.seq_len * self.rnn_num_layers * self.rnn_hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ui, uv, ai, av, y=None, time=None, loss_func='MSE'):
        # ui、uv: [batch_size, a_field_size]
        # ai、av: [batch_size, day, u_field_size]
        # y: [batch_size, 1]
        h0 = self.init_hidden().to(self.device)
        # out: [batch_size, day * D * hidden_size]
        output, h_n = self.rnn(av, h0)
        output = torch.reshape(output, [-1, self.seq_len * self.rnn_num_layers * self.rnn_hidden_size])
        # y_pred: [batch_size, 1]
        y_pred = self.sigmoid(self.fc_out(output))

        # y_true_bool (Active or not): [batch_size,  1]
        esp = 1e-5
        y_true_bool = y.clone()
        y_true_bool[y >= esp] = 1.0
        y_true_bool[y < esp] = 0.0
        y_true_bool = y_true_bool.to(self.device)
        # y_pred_bool (Active or not): [batch_size,  1]
        y_pred_bool = y_pred.clone()
        y_pred_bool[y_pred >= 0.5] = 1.0
        y_pred_bool[y_pred < 0.5] = 0.0
        y_pred_bool = y_pred_bool.to(self.device)

        if y is not None:
            if loss_func == 'BCE':
                loss = self.criterion_1(y_pred_bool, y_true_bool)
            else:
                loss = self.criterion_2(y_pred, y)

            return loss, y_pred
        else:
            return y_pred

    def init_hidden(self):
        return torch.zeros(self.rnn_num_layers, self.batch_size, self.rnn_hidden_size)
