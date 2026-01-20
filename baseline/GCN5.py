import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn  # 需要 PyTorch Geometric 库
import numpy as np

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, gcn_hidden_dim, output_dim, num_layers,data=None,adj=None,adj_index=None):
        super(GCN, self).__init__()
        # LSTM部分
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # GCN层
        self.gcn1 = pyg_nn.GCNConv(144, gcn_hidden_dim)
        self.gcn2 = pyg_nn.GCNConv(gcn_hidden_dim, output_dim)
        self.scaler = data['scaler']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.edge_index = adj_index.to(self.device)
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.MS = adj
    def forward(self, x):
        # print(self.MS)
        # print(np.sum(self.MS)/2)
        x = x['x']
        x = x.view(x.shape[2], -1)  # [batch_size, seq_len, input_dim]
        # 2. GCN层：处理节点之间的关系
        x = self.gcn1(x, self.edge_index)
        x = F.gelu(x)
        x = F.dropout(x,0.5)
        x = self.gcn2(x, self.edge_index)
        x =self.mlp(x)
        x = x.view(1, 1, -1, 1)

        return x  # 输出形状：[节点数, 输出维度]

    def _sample_latent(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        sampled_latent = mean + epsilon * std
        return sampled_latent
    def predict(self, data):
        with torch.no_grad():
            return self.forward(data)
    def cal_train_loss(self, data):
        # pred,a_factor,a_dygnn,loss_diffusion  = self.forward(data)
        # [B,1,N,1]
        pred = self.forward(data)
        pred = self.scaler.inverse_transform(pred).squeeze(-1)  # [B,1,N]
        truth = self.scaler.inverse_transform(data['y'])  # [B,1,N]
        data['x'] = self.scaler.inverse_transform(data['x'])
        mask = data['mask'].min(dim=1)[0].unsqueeze(1)
        mask /= torch.mean(mask)
        huber_loss = F.huber_loss(pred, truth, reduction='none')
        if torch.isnan(mask).any() or torch.isinf(mask).any():
            print("Warning: mask contains NaN or Inf values.")
        huber_loss.mul_(mask)
        huber_loss = torch.mean(huber_loss)
        rank_loss = []
        for i in range(pred.shape[0]):
            cur_mask = mask[i].view(-1).bool()
            cur_pred = pred[i].view(-1, 1)[cur_mask]
            cur_truth = truth[i].view(-1, 1)[cur_mask]
            last_price = data['x'][i, -1, :, -1].view(-1, 1)[cur_mask]
            return_ratio = torch.div(torch.sub(cur_pred, last_price), last_price)
            truth_ratio = torch.div(torch.sub(cur_truth, last_price), last_price)
            all_one = torch.ones(cur_pred.shape[0], 1, dtype=torch.float32).to(self.device)
            pre_pw_dif = torch.sub(return_ratio @ all_one.t(), all_one @ return_ratio.t())
            gt_pw_dif = torch.sub(all_one @ truth_ratio.t(), truth_ratio @ all_one.t())
            rank_loss.append(torch.mean(F.relu(pre_pw_dif * gt_pw_dif)))
        rank_loss = torch.mean(torch.stack(rank_loss))
        alpha = 1
        return huber_loss + 100*rank_loss
# # 模型参数
# input_dim = 9  # 每个节点每个时间步的特征数
# hidden_dim = 16  # LSTM的隐藏层维度
# gcn_hidden_dim = 8  # GCN的隐藏层维度
# output_dim = 1  # GCN输出的最终维度（1026x1）
# num_layers = 2  # LSTM层数
# # 初始化模型
# model = LSTM_GCN(input_dim, hidden_dim, gcn_hidden_dim, output_dim, num_layers)
#
# # 模拟输入数据
# nodes = 1026
# time_steps = 4
# features = 9
# x = torch.rand((nodes, time_steps, features))  # 输入张量 [节点数, 时间步, 特征数]
# edge_index = torch.randint(0, nodes, (2, 2000))  # 边的索引 (假设有2000条边)
#
# # 前向传播
# output = model(x, edge_index)
# print("Output shape:", output.shape)  # 应为 [1026, 1]
