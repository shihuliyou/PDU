import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTTS(nn.Module):
    def __init__(self, input_channels=9, transformer_hidden_dim=64, num_heads=4,
                 num_transformer_layers=2,data=None):
        super(CTTS, self).__init__()

        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
        )

        # 将CNN输出flatten，供Transformer输入
        self.flatten = nn.Flatten(start_dim=1)

        # Transformer部分
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=num_heads,
            dim_feedforward=transformer_hidden_dim,
            dropout=0.1,
            activation='relu',
            batch_first= True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

        # 最后一个线性层将Transformer的输出映射到单个值
        self.fc = nn.Linear(1024, 1)
        self.scaler = data['scaler']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        # x的输入大小应为 (num_nodes, num_time_windows, num_features)
        x = x['x']
        x = x.view(x.size(1), x.size(2), -1)
        # 1. CNN部分
        x = x.permute(2, 0, 1).unsqueeze(0)  # 转换维度，变为 (1, num_features, num_nodes, num_time_windows)
        x = self.cnn(x)  # CNN输出大小为 (1, out_channels, num_nodes, num_time_windows)

        # 2. Transformer部分
        x = x.view(x.size(-1),-1) # 变成 (out_channels, num_nodes)
        x = self.transformer_encoder(x) # Transformer输出大小为 (num_nodes, embedding_dim)

        # 3. 预测输出
        x = self.fc(x)  # 映射到单个值, 大小为 (num_nodes, 1)
        x = x.view(1,1,-1,1) # 最终输出为 (num_nodes)


        return x


# 模型实例

    def predict(self, data):
        with torch.no_grad():
            return self.forward(data)

    def cal_train_loss(self, data):
        # pred,a_factor,a_dygnn,loss_diffusion  = self.forward(data)
        # [B,1,N,1]

        pred= self.forward(data)
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
        return huber_loss+0.1*rank_loss
# 定义模型参数和随机输入的形状
# B, N, T, F = 16, 1026, 16, 9  # Batch size, Nodes, Time windows, Features
# in_channels = F
# hidden_channels = 64
# out_channels = 1
# num_heads = 8
# num_layers = 2
#
# # 实例化模型
# model = CTTS(in_channels, hidden_channels, out_channels, num_heads, num_layers,data=None)
#
# # 生成随机输入
# x = torch.randn(B, N, T, F)
#
# # 进行前向传播并打印输出形状
# output = model(x)
# print(output.shape)  # Expected output shape: [B, 1, N, 1]
