import torch
import torch.nn
import torch.nn.functional as F


class DPCNN(torch.nn.Module):
    def __init__(self, model_params):
        super(DPCNN, self).__init__()
        # device
        self.device = model_params['device']

        # loss function
        # BCE
        self.criterion_1 = torch.nn.BCELoss()
        self.criterion_1.to(self.device)
        # MSE
        self.criterion_2 = torch.nn.MSELoss()
        self.criterion_2.to(self.device)

        # parameters
        self.dpcnn_conv2_kernel_1 = model_params['dpcnn_conv2_kernel_1']  # 3 * 3
        self.dpcnn_conv2_output_size_1 = model_params['dpcnn_conv2_output_size_1']  # 32
        self.dpcnn_conv2_kernel_2 = model_params['dpcnn_conv2_kernel_2']  # 3 * 3
        self.dpcnn_conv2_output_size_2 = model_params['dpcnn_conv2_output_size_2']  # 64
        self.batch_size = model_params['batch_size']

        self.day = model_params['day']
        self.a_field_size = model_params['a_field_size']

        # conv2d
        self.conv2_1 = torch.nn.Conv2d(1, self.dpcnn_conv2_output_size_1, self.dpcnn_conv2_kernel_1, padding='same',
                                       stride=1, bias=True)
        self.conv2_2 = torch.nn.Conv2d(self.dpcnn_conv2_output_size_1, self.dpcnn_conv2_output_size_2,
                                       self.dpcnn_conv2_kernel_2, padding='same', stride=1, bias=True)

        # linear parameters
        h2 = (self.day - self.dpcnn_conv2_kernel_1 + 3) - self.dpcnn_conv2_kernel_2 + 3
        w2 = (self.a_field_size - self.dpcnn_conv2_kernel_1 + 3) - self.dpcnn_conv2_kernel_2 + 3
        input_size = self.dpcnn_conv2_output_size_2 * h2 * w2
        self.liner_1_output_size = model_params['liner_1_output_size']
        # linear
        self.fc_1 = torch.nn.Linear(input_size, self.liner_1_output_size)
        self.fc_out = torch.nn.Linear(self.liner_1_output_size, 1, bias=False)

        # activation function
        self.dropout = torch.nn.Dropout(model_params['dropout_p'])
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ui, uv, ai, av, y, time=None, lossFun='MSE'):
        # av: [batch_size, 1, day, a_field_size]
        av = torch.unsqueeze(av, dim=1)
        # av_conv_1: [batch_size, dpcnn_conv2_output_size_1, h1, w1]
        # h1: self.day - self.dpcnn_conv2_kernel_1 + 3
        # w1: self.a_field_size - self.dpcnn_conv2_kernel_1 + 3
        av_conv_1 = self.conv2_1(av)
        av_conv_1 = self.relu(av_conv_1)
        # av_conv_1: [batch_size, dpcnn_conv2_output_size_2, h2, w2]
        # h2: h1 - self.dpcnn_conv2_kernel_2 + 3
        # w2: w1 - self.dpcnn_conv2_kernel_2 + 3
        av_conv_2 = self.conv2_2(av_conv_1)
        av_conv_2 = self.relu(av_conv_2)

        # av_fc: [av_conv:batch_size, dpcnn_conv2_output_size_2 * h2 * w2]
        av_fc = av_conv_2.reshape(self.batch_size, -1)
        # av_fc: [batch_size, liner_1_output_size]
        av_fc = self.dropout(av_fc)
        av_fc = self.relu(self.fc_1(av_fc))
        # y_pred: [batch_size, 1]
        y_pred = self.sigmoid(self.fc_out(av_fc))

        # y_true_bool (Active or not): [batch_size,  1](int)
        esp = 1e-5
        y_true_bool = y.clone()
        y_true_bool[y >= esp] = 1.0
        y_true_bool[y < esp] = 0.0
        y_true_bool = y_true_bool.to(self.device)

        if y is not None:
            if lossFun == 'BCE':
                loss = self.criterion_1(y_pred, y_true_bool)
            else:
                loss = self.criterion_2(y_pred, y)
            return loss, y_pred
        else:
            return y_pred
