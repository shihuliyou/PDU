import torch
import torch.nn as nn
import torch.nn.functional as F

class StockGRUModel(nn.Module):
    def __init__(self,data):
        super(StockGRUModel, self).__init__()

        # gru层
        self.gru = nn.GRU(input_size=9, hidden_size=128, num_layers=2, batch_first=True)
        self.fc =nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        # 全连接层
        self.scaler = data['scaler']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def forward(self, x):
        x = x['x'].view(x['x'].shape[2], -1, x['x'].shape[-1])
        # x 的形状: (batch_size, 4, 9)，其中 batch_size=1026, 时间窗=4, 特征=9
        # 通过 gru 层
        gru_out, (hn, cn) = self.gru(x)  # gru_out 形状: (batch_size, 4, 64)
        # 取最后一个时间步的输出
        last_out = gru_out[:, -1, :]  # 取最后一个时间步，形状: (batch_size, 64)
        # 全连接层
        output = self.fc(last_out)  # 形状变为 (batch_size, 1).
        x = output.view(1, 1, -1, 1)
        return x

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