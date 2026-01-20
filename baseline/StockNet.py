import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class RNNContinuousLatentModel(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(RNNContinuousLatentModel, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.latent_mean = nn.Linear(hidden_size, latent_size)
        self.latent_logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        _, hidden = self.rnn(x)
        hidden = hidden[-1]
        latent_mean = self.latent_mean(hidden)
        latent_logvar = self.latent_logvar(hidden)
        return latent_mean, latent_logvar


class StockNet(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size,data=None):
        super(StockNet, self).__init__()
        self.rnn_model = RNNContinuousLatentModel(input_size, hidden_size, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.time_auxiliary = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.scaler = data['scaler']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def forward(self, x):
        x = x['x'].view(x['x'].shape[2], -1, x['x'].shape[-1])
        latent_mean, latent_logvar = self.rnn_model(x)
        sampled_latent = self._sample_latent(latent_mean, latent_logvar)
        prediction = self.decoder(sampled_latent)

        combined_prediction = prediction
        combined_prediction = combined_prediction.view(1, 1, -1, 1)  # 最终输出为 (num_nodes)
        return combined_prediction

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
        return huber_loss + 0.1 * rank_loss
# q=TimeAwareMixedObjectiveModel(9,16,1)
# input_data = torch.randn(1026, 16, 9)
# output = q(input_data)
# print(output.shape)
# model = TimeAwareMixedObjectiveModel(9, 65, 32).cuda()

