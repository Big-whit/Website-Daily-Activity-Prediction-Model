import torch
import torch.nn
import torch.nn.functional as F


class LR(torch.nn.Module):
    def __init__(self, model_params):
        super(LR, self).__init__()
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

        # embedding size
        self.embedding_size = model_params['embedding_size']

        # setting embedding layers
        self.a_embeddings = torch.nn.Embedding(self.a_feat_size, self.embedding_size)
        self.u_embeddings = torch.nn.Embedding(self.u_feat_size, self.embedding_size)

        # input_size
        input_size = self.day * self.a_field_size * self.embedding_size + self.u_field_size * self.embedding_size
        # fc_out
        self.fc_out = torch.nn.Linear(input_size, 1)

    def forward(self, ui, uv, ai, av, y=None, time=None, lossFun='MSE'):
        # embedding_size: 32
        # ui、uv: [batch_size, u_field_size]
        # ai、av: [batch_size, day, a_field_size]
        # y: [batch_size, 1]
        # a_emb : [batch_size, day, u_field_size, embedding_size]
        # u_emb : [batch_size, a_field_size, embedding_size]

        a_emb = self.a_embeddings(ai)
        u_emb = self.u_embeddings(ui)
        # a_emb: [batch_size, day, field_size, embedding_size]
        a_emb = torch.multiply(a_emb, av.reshape(-1, self.day, self.a_field_size, 1))
        # u_emb: [batch_size, field_size, embedding_size]
        u_emb = torch.multiply(u_emb, uv.reshape(-1, self.u_field_size, 1))
        a_emb = a_emb.reshape(self.batch_size, -1)
        u_emb = u_emb.reshape(self.batch_size, -1)
        # deep_input: [batch_size, day * a_field_size * embedding_size + u_field_size * embedding_size]
        deep_input = torch.cat((a_emb, u_emb), 1)
        # y_pred: [batch_size, 1]
        y_pred = self.fc_out(deep_input)

        # y_true_bool (Active or not): [batch_size, 1](int)
        esp = 1e-5
        y_true_bool = y.clone()
        y_true_bool[y >= esp] = 1.0
        y_true_bool[y < esp] = 0.0
        y_true_bool = y_true_bool.to(self.device)

        if y is not None:
            if lossFun == 'BCE':
                y_pred = torch.sigmoid(y_pred)
                loss = self.criterion_1(y_pred, y_true_bool)
            else:
                loss = self.criterion_2(y_pred, y)
            return loss, y_pred
        else:
            return y_pred
