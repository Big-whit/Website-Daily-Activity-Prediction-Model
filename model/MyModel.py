import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def BCE(
        input: Tensor,
        target: Tensor):
    # input:   [batch_size, future_day, label_size]
    # target:  [batch_size,future_day, label_size]
    # loss:    [batch_size, future_day,label_size]
    loss = F.binary_cross_entropy(input, target, reduction='none')
    return loss.mean()


class MyModel(torch.nn.Module):
    def __init__(self, model_params):
        super(MyModel, self).__init__()
        print('Start run MyModel')
        # device
        self.device = model_params['device']

        # BCE
        self.criterion_1 = torch.nn.BCELoss()
        self.criterion_1.to(self.device)
        # MSE
        self.criterion_2 = torch.nn.MSELoss()
        self.criterion_2.to(self.device)

        # batch_size
        self.batch_size = model_params['batch_size']
        # day、future_day
        self.day = model_params['day']
        self.future_day = model_params["future_day"]
        # week、day
        self.week_num = model_params['week_num']
        self.day_num = model_params['day_num']
        self.period_num = model_params['period_num']
        # action user : one_hot_num
        self.a_feat_size = model_params['a_feat_size']
        self.u_feat_size = model_params['u_feat_size']
        # action user : feature_num
        self.a_field_size = model_params['a_field_size']
        self.u_field_size = model_params['u_field_size']
        # Task purpose
        self.multi_task_enable = model_params['multi_task_enable']
        self.fine_grained = model_params['fine_grained']

        # Embedding parameters
        self.a_embedding_size = model_params['a_embedding_size']
        self.u_embedding_size = model_params['u_embedding_size']
        self.time_embedding_size = model_params["time_embedding_size"]
        # Embedding Layer
        self.a_embeddings = torch.nn.Embedding(self.a_feat_size, self.a_embedding_size)
        self.u_embeddings = torch.nn.Embedding(self.u_feat_size, self.u_embedding_size)
        self.week_embeddings = torch.nn.Embedding(self.week_num, self.time_embedding_size)
        self.day_embeddings = torch.nn.Embedding(self.day_num, self.time_embedding_size)

        # Linear Layer
        self.time_hidden_size = model_params['time_hidden_size']
        self.action_hidden_size = model_params['action_hidden_size']
        self.fc_u = torch.nn.Linear(self.u_field_size * self.u_embedding_size, self.u_embedding_size)
        self.fc_time = torch.nn.Linear(self.period_num, 1)
        self.fc_s = torch.nn.Linear(2 * self.time_hidden_size, 1)

        # pred_1 fc layer
        # if self.multi_task_enable != 0:
        #     self.fc_out_1 = torch.nn.Linear(self.future_day +
        #                                     self.future_day * 2 * self.action_hidden_size +
        #                                     self.future_day +
        #                                     self.u_embedding_size, 1)
        # else:
        #     self.fc_out_1 = torch.nn.Linear(self.day * 2 * self.action_hidden_size + self.u_embedding_size, 1)
        self.fc_out_1 = torch.nn.Linear(self.day * 2 * self.action_hidden_size + self.u_embedding_size, 1)

        # pred_2 fc layer
        if self.fine_grained == 1:
            self.fc_out_2 = torch.nn.Linear(
                2 * self.action_hidden_size + self.u_embedding_size + 2 * self.time_hidden_size, self.a_field_size)
            self.maxpool_1 = torch.nn.MaxPool1d(kernel_size=self.a_field_size)
        else:
            self.fc_out_2 = torch.nn.Linear(
                2 * self.action_hidden_size + self.u_embedding_size + 2 * self.time_hidden_size, 1)
            self.maxpool_1 = torch.nn.MaxPool1d(kernel_size=1)
        # self.maxpool_2 = torch.nn.MaxPool1d(kernel_size=2 * self.time_hidden_size)
        self.maxpool_2 = torch.nn.MaxPool1d(kernel_size=self.future_day)

        # Attention parameters
        self.u_attention_head_num = model_params['u_attention_head_num']
        self.time_attention_head_num = model_params['time_attention_head_num']
        self.a_attention_head_num = model_params['a_attention_head_num']
        self.pf_attention_head_num = model_params['pf_attention_head_num']
        # Attention
        self.u_multi_head_attention = torch.nn.MultiheadAttention(embed_dim=self.u_embedding_size,
                                                                  num_heads=self.u_attention_head_num,
                                                                  batch_first=True)
        self.time_multi_head_attention = torch.nn.MultiheadAttention(embed_dim=self.time_embedding_size,
                                                                     num_heads=self.time_attention_head_num,
                                                                     batch_first=True)
        self.a_multi_head_attention = torch.nn.MultiheadAttention(embed_dim=self.a_embedding_size,
                                                                  num_heads=self.a_attention_head_num,
                                                                  batch_first=True)
        self.pf_multi_head_attention = torch.nn.MultiheadAttention(embed_dim=2 * self.time_hidden_size,
                                                                   num_heads=self.pf_attention_head_num,
                                                                   kdim=2 * self.time_hidden_size,
                                                                   vdim=2 * self.action_hidden_size,
                                                                   batch_first=True)

        # LSTM parameters
        self.time_hidden_size = model_params['time_hidden_size']
        self.action_hidden_size = model_params['action_hidden_size']
        # LSTM
        self.lstm_time = torch.nn.LSTM(input_size=self.time_embedding_size,
                                       hidden_size=self.time_hidden_size,
                                       bidirectional=True, batch_first=True)
        self.lstm_action = torch.nn.LSTM(input_size=self.a_field_size * self.a_embedding_size,
                                         hidden_size=self.action_hidden_size,
                                         bidirectional=True, batch_first=True)

        # activation function
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        # BCE weight
        self.bce_weight = model_params['bce_weight']

        # Initial
        self.init_embeddings()
        self.init_param()

    def init_embeddings(self):
        print("Init embeddings!")
        nn.init.kaiming_normal_(self.a_embeddings.weight)
        nn.init.kaiming_normal_(self.u_embeddings.weight)
        nn.init.kaiming_normal_(self.week_embeddings.weight)
        nn.init.kaiming_normal_(self.day_embeddings.weight)

    def init_param(self):
        print("Initial parameters!")
        nn.init.kaiming_normal_(self.fc_u.weight)
        nn.init.constant_(self.fc_u.bias, 0)
        nn.init.kaiming_normal_(self.fc_time.weight)
        nn.init.constant_(self.fc_time.bias, 0)
        nn.init.kaiming_normal_(self.fc_s.weight)
        nn.init.constant_(self.fc_s.bias, 0)
        nn.init.kaiming_normal_(self.fc_out_1.weight)
        nn.init.constant_(self.fc_out_1.bias, 0)
        nn.init.kaiming_normal_(self.fc_out_2.weight)
        nn.init.constant_(self.fc_out_2.bias, 0)

    def forward(self, ui, uv, ai, av, y_1=None, y_2=None, time=None):
        """ ui、uv: [batch_size, u_field_size]
        ai、av: [batch_size, day, a_field_size]
        y_1: [batch_size, 1]
        y_2: [batch_size, future_day, a_field_size]
        time: [batch_size, day + future_day, 4] """

        """ Step 1 - Embedding user images. """
        # u_emb: [batch_size, u_field_size, u_embedding_size]
        u_emb = self.u_embeddings(ui)
        # u_emb: [batch_size, u_field_size, u_embedding_size]
        u_emb = torch.multiply(u_emb, uv.reshape(-1, self.u_field_size, 1))

        """ Step 2 - Extract user image features by self-Attention. """
        # u_inter: [batch_size, u_field_size, u_embedding_size]
        u_inter, _ = self.u_multi_head_attention(u_emb, u_emb, u_emb)
        # u_inter: [batch_size, u_field_size * u_embedding_size]
        u_inter = u_inter.reshape(self.batch_size, -1)
        # u_inter: [batch_size, u_embedding_size]
        u_inter = self.relu(self.fc_u(u_inter))

        """ Step 3 - Embedding week and day. """
        # weeks、day: [batch_size, day + future_day]
        weeks = (time[:, :, 3] - 1).cpu().numpy()
        days = (time[:, :, 2] - 1).cpu().numpy()
        weeks = torch.tensor(weeks).long()
        days = torch.tensor(days).long()
        weeks = weeks.to(self.device)
        days = days.to(self.device)
        # weeks_emb: [batch_size, day + future_day, time_embedding_size]
        weeks_emb = self.week_embeddings(weeks)
        # days_emb: [batch_size, day + future_day, time_embedding_size]
        days_emb = self.day_embeddings(days)
        # days_emb: [batch_size, day + future_day, period_num, time_embedding_size]
        period_emb = torch.cat((weeks_emb.unsqueeze(2), days_emb.unsqueeze(2)), dim=2)

        """ Step 4 - Extract week and day features by self-Attention. """
        # period_emb : [batch_size * (day + future_day), period_num, time_embedding_size]
        period_emb = period_emb.reshape(-1, self.period_num, self.time_embedding_size)
        # period_mul: [batch_size * (day + future_day), period_num, time_embedding_size]
        period_mul, period_weight = self.time_multi_head_attention(period_emb, period_emb, period_emb)
        # period_mul: [batch_size * (day + future_day), time_embedding_size, period_num]
        period_mul = period_mul.permute(0, 2, 1)
        # period_time: [batch_size, (day + future_day), time_embedding_size]
        period_time = self.relu(self.fc_time(period_mul)).squeeze(-1).reshape(self.batch_size, -1,
                                                                              self.time_embedding_size)

        """ Step 5 - Mining cycle information by Bi-LSTM. """
        # time_output: [batch_size, day + future_day, 2 * time_hidden_size]
        time_output, (h1_n, c1_n) = self.lstm_time(period_time)

        """ Step 6 - Embedding user action. """
        # a_emb: [batch_size, day, a_field_size, a_embedding_size]
        a_emb = self.a_embeddings(ai)
        # a_emb: [batch_size, day, a_field_size, a_embedding_size]
        a_emb = torch.multiply(a_emb, av.reshape(-1, self.day, self.a_field_size, 1))

        """ Step 7 - Extract user action features by self-Attention. """
        # a_emb: [batch_size * day, a_field_size, a_embedding_size]
        a_emb = a_emb.reshape(-1, self.a_field_size, self.a_embedding_size)
        # a_inter: [batch_size * day, a_field_size, a_embedding_size]
        a_inter, a_weights = self.a_multi_head_attention(a_emb, a_emb, a_emb)
        # a_inter: [batch_size, day, a_field_size * a_embedding_size]
        a_inter = a_inter.reshape(self.batch_size, self.day, -1)

        """ Step 8 - Mining action sequence features by Bi-LSTM. """
        # a_output: [batch_size, day, 2 * action_hidden_size]
        a_output, (h_n, c_n) = self.lstm_action(a_inter)

        """ Step 9 - Whether to start multi-task prediction. """
        if self.multi_task_enable != 0:
            # Calculate future_day and past_day time information weights
            # time_past: [batch_size, day, 2 * time_hidden_size]
            time_past = time_output[:, :self.day, :]
            # time_future: [batch_size, future_day, 2 * time_hidden_size]
            time_future = time_output[:, self.day:, :]
            # y_weights: [batch_size, future_day, day]
            _, y_weights = self.pf_multi_head_attention(time_future, time_past, a_output)

            # Calculate time interval weight
            days = torch.tensor([i for i in range(1, self.day + 1)])
            futures = torch.tensor([i for i in range(self.day + 1, self.day + self.future_day + 1)])
            # time_interval: [future_day, day]
            time_interval = torch.stack([i.item() - days for i in futures])
            # time_interval: [batch_size, future_day, day]
            time_interval = torch.unsqueeze(time_interval, 0).repeat(self.batch_size, 1, 1).to(self.device)
            # S: [batch_size, future_day, day]
            S = self.sigmoid(self.fc_s(time_future)).repeat(1, 1, self.day)
            # time_interval_weight: [batch_size, future_day, day]
            time_interval_weight = torch.exp((time_interval * S))

            # all_weight: [batch_size, future_day, day]
            all_weight = y_weights + time_interval_weight

            # y_deep_2: [batch_size, future_day, num_directions * time_hidden_size]
            y_deep_2 = torch.bmm(all_weight, a_output)
            # u_inter_us: [batch_size, future_day, u_embedding_size]
            u_inter_us = torch.unsqueeze(u_inter, 1).repeat(1, self.future_day, 1)
            # y_deep_2: [batch_size, future_day, num_directions * action_hidden_size +
            # u_embedding_size + num_directions * time_hidden_size]
            y_input_2_1 = torch.cat((y_deep_2, u_inter_us, time_future), dim=2)
            # pred_2: [batch_size, future_day, a_field_size || 1]
            pred_2 = self.sigmoid(self.fc_out_2(y_input_2_1))

            # # input_1_1: [batch_size, future_day]
            # input_1_1 = self.maxpool_1(pred_2).squeeze(-1)
            # # input_2_2: [batch_size, future_day * num_directions * action_hidden_size]
            # input_1_2 = y_deep_2.reshape((self.batch_size, -1))
            # # input_2_3: [batch_size, future_day, 1]
            # input_1_3 = self.maxpool_2(time_future).reshape(self.batch_size, -1)
            # # input_2_4: [batch_size, u_embedding_size]
            # input_1_4 = u_inter
            # input_cat = torch.cat((input_1_1, input_1_2, input_1_3, input_1_4), dim=1)
            # pred_1 = torch.sigmoid(self.fc_out_1(input_cat)).squeeze(-1)

            # input_1_1: [batch_size, future_day]
            y_input_1_1 = self.maxpool_1(pred_2).squeeze(-1)
            # pred_1: [batch_size, 1]
            pred_1 = self.maxpool_2(y_input_1_1)

        else:
            # a_output: [batch_size, day * num_directions * action_hidden_size]
            a_output = a_output.reshape(self.batch_size, -1)
            # y_deep: [batch_size, day * 2 * action_hidden_size + u_embedding_size]
            y_deep = torch.cat((a_output, u_inter), dim=1)
            # pred_1: [batch_size, 1]
            pred_1 = torch.sigmoid(self.fc_out_1(y_deep))

            if self.fine_grained == 1:
                pred_2 = torch.ones((self.batch_size, self.future_day, self.a_field_size))
            else:
                pred_2 = torch.ones((self.batch_size, self.future_day, 1))

        # y_2: [batch_size, future_day, a_field_size || 1]
        if self.fine_grained == 0:
            y_2 = y_2.sum(dim=2)
            one = torch.ones_like(y_2)
            zero = torch.zeros_like(y_2)
            y_2 = torch.where(y_2 == 0, zero, one)
            y_2 = y_2.reshape((self.batch_size, self.future_day, 1))

        if y_1 is not None:
            loss = self.criterion_2(pred_1, y_1)
            if self.multi_task_enable != 0:
                loss += self.bce_weight * BCE(pred_2, y_2[:, :, :])

            return loss, pred_1, pred_2
