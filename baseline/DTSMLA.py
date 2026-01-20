import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.W_a = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_a = nn.Parameter(torch.Tensor(hidden_size))
        self.u_a = nn.Parameter(torch.Tensor(hidden_size, hidden_size))


    def forward(self, x):
        gru_out, _ = self.gru(x)

        h_t = gru_out[:, -1, :]

        a_t = torch.matmul(torch.tanh(torch.matmul(h_t, self.W_a) + self.b_a), self.u_a)
        a_t = F.softmax(a_t, dim=1)
        e_t = torch.cat((a_t, h_t), dim=1)

        return e_t


class DTSMLA(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads,stock_len):
        super(DTSMLA, self).__init__()
        self.gru = GRUModel(input_size, hidden_size, num_layers)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, batch_first=True)
        self.mlp=MLP(2*hidden_size,hidden_size,2*hidden_size)
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.W_p = nn.Linear(hidden_size*4, hidden_size)
        self.W_a = nn.Linear(hidden_size*2, hidden_size)

        self.lin_v1=nn.Linear(hidden_size, hidden_size)
        self.lin_v2=nn.Linear(hidden_size, hidden_size)
        self.lin_v3=nn.Linear(hidden_size, hidden_size)
        self.lin_A=nn.Linear(hidden_size, hidden_size)

        self.lin_y1=nn.Linear(stock_len*3, 1)
        self.lin_y2=nn.Linear(stock_len*3, 1)
        self.lin_y3=nn.Linear(stock_len*3, 1)
    def forward(self, x):
        e=self.gru(x)
        e_t = e.unsqueeze(1)
        attn_output, attn_output_weights = self.attention(e_t, e_t, e_t)
        attn_output = attn_output.squeeze(1)
        E_bar = attn_output * e
        E = torch.tanh(E_bar + e + self.mlp(E_bar + e))
        concat_input = torch.cat((E, e), dim=-1)
        r_i_t = torch.sigmoid(self.mlp2(concat_input))
        e_hat_i_t = torch.cat((E, r_i_t * e), dim=-1)
        q_i_t = self.W_p(e_hat_i_t)
        v1=self.lin_v1(q_i_t)
        v2=self.lin_v2(q_i_t)
        v3=self.lin_v3(q_i_t)
        P = torch.stack((v1, v2, v3))
        A = torch.zeros((3, 3,v1.size(0),v1.size(1))).to(v1.device)
        for i in range(3):
            for j in range(3):
                A[i, j] = self.similarity(P[i:i + 1], P[j:j + 1])
            A[i] = self.softmax(A[i])
        P=P.transpose(-1, -2)
        z1 = torch.matmul(A[0], P).permute(1, 0, 2).reshape(P.size(-1), P.size(-1)*3)
        z2 = torch.matmul(A[1], P).permute(1, 0, 2).reshape(P.size(-1), P.size(-1)*3)
        z3 = torch.matmul(A[2], P).permute(1, 0, 2).reshape(P.size(-1), P.size(-1)*3)
        return torch.tanh(self.lin_y1(z1)),torch.tanh(self.lin_y2(z2)),torch.tanh(self.lin_y3(z3))
    def similarity(self, p_i, p_j):
        return torch.tanh(self.W_a(torch.cat((p_i, p_j),dim=-1)) )

    def softmax(self, x):
        e_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        return e_x / e_x.sum(dim=-1, keepdim=True)

    def predict(self, data):
        with torch.no_grad():
            x = data['x'].view(data['x'].size(2), data['x'].size(1), -1)
            return self.forward(x)
    def loss(self, data ):
        x = data['x'].view(data['x'].size(2), data['x'].size(1), -1)
        y1,y2,y3 = self.forward(x)
        y_t_close =data['y'].view(-1,1)  # 当日收盘价
        y_t_1_close = data['x'][:, -1, :, -1].view(-1,1)  # 前一日收盘价
        y_t_open = data['y'].view(-1,1)  # 当日开盘价
        y_t_1_open = data['x'][:, -1, :, -1].view(-1,1)  # 前一日开盘价
        y_t_1 = (y_t_close - y_t_1_close) / y_t_1_open  # 任务1
        y_t_2 = (y_t_open - y_t_1_close) / y_t_1_close  # 任务2
        y_t_3 = (y_t_close - y_t_1_open) / y_t_1_close  # 任务3
        y = torch.cat([x[:, 1, :], y_t_1, y_t_2, y_t_3], dim=-1)
        mlp = nn.Sequential(
            nn.Linear(input_size + 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=-1)
        ).to(y.device)
        y = mlp(y)
        b1, b2, b3 = torch.tensor_split(y, 3)
        loss_fn = nn.MSELoss()
        loss = loss_fn(y1, y_t_1) * b1[:342]#672
        loss += loss_fn(y2, y_t_2) * b2
        loss += loss_fn(y3, y_t_3) * b3
        loss = torch.sum(loss) / stock_len
        return  loss


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

stock_len = 1026
sequence_length = 4
input_size = 9
input_data = torch.randn(stock_len, sequence_length, input_size)
hidden_size = 20
num_layers = 2
num_heads = 4
model = DTSMLA(input_size, hidden_size, num_layers, num_heads,stock_len)
output = model(input_data)



