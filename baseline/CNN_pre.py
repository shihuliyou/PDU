import torch
import torch.nn as nn

import torch.nn.functional as F


class cnn_pre(nn.Module):
    def __init__(self,data=None):
        super(cnn_pre, self).__init__()
        self.conv1 = nn.Conv1d(9, 32, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 2, 32)  # 将第一个全连接层的输出维度改为32
        self.fc2 = nn.Linear(512, 1)
        self.scaler = data['scaler']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        x = x['x'].view(x['x'].shape[2], -1, 9)
        x=x.transpose(1, 2)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        # x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(1,1, -1, 1)
        return x
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
        return huber_loss+10*rank_loss