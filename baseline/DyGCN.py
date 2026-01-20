import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ==============================================================
# 1️⃣ 频谱动态图构建器 SpectralDynamicGraphBuilder
# ==============================================================
class SpectralDynamicGraphBuilder(nn.Module):
    def __init__(self, input_dim, k=10, n_bands=16, use_welch=True,
                 seg_len=None, overlap=0.5, temperature=0.07, ema=0.2,
                 eps=1e-8, row_stochastic=False):
        super().__init__()
        self.F = input_dim
        self.k = k
        self.n_bands = n_bands
        self.use_welch = use_welch
        self.seg_len = seg_len
        self.overlap = overlap
        self.temperature = temperature
        self.ema = ema
        self.eps = eps
        self.row_stochastic = row_stochastic

        self.feature_logits = nn.Parameter(torch.zeros(self.F))
        self.band_proj = None
        self.register_buffer("prev_A", None, persistent=False)

    def _welch_segments(self, x, T):
        if self.seg_len is None:
            L = max(8, T // 2)
        else:
            L = min(self.seg_len, T)
        step = max(1, int(L * (1 - self.overlap)))
        starts = list(range(0, max(1, T - L + 1), step))
        segs = [x[:, s:s+L, ...] for s in starts]
        segs = torch.stack(segs, dim=1)  # [B,S,L,N,F]
        return segs, L, len(starts)

    @staticmethod
    def _cosine_topk(sim, k):
        B, N, _ = sim.shape
        sim = sim.clone()
        eye = torch.eye(N, device=sim.device).unsqueeze(0)
        sim = sim.masked_fill(eye.bool(), 0.0)
        idx = torch.topk(sim, k, dim=-1).indices
        mask = torch.zeros_like(sim)
        mask.scatter_(dim=-1, index=idx, src=torch.ones_like(idx, dtype=sim.dtype))
        A = sim * mask
        A = 0.5 * (A + A.transpose(-1, -2))
        return A

    def forward(self, x):
        # x = x.permute(0, 3, 2, 1)
        B, T, N, F_ = x.shape
        device = x.device
        if self.use_welch and T >= 8:
            segs, L, S = self._welch_segments(x, T)
            window = torch.hann_window(L, device=device).view(1, 1, L, 1, 1)
            segs = segs * window
            spec = torch.fft.rfft(segs, dim=2)
            power = (spec.abs() ** 2).mean(dim=1)
        else:
            window = torch.hann_window(T, device=device).view(1, T, 1, 1)
            xw = x * window
            spec = torch.fft.rfft(xw, dim=1)
            power = (spec.abs() ** 2)

        feat_w = torch.softmax(self.feature_logits, dim=0)
        # power_agg = torch.einsum('bfni,f->bfni', power, feat_w).sum(dim=-1)
        power_agg = torch.einsum('bfni,i->bfni', power, feat_w).sum(dim=-1)
        freq_bins = power_agg.shape[1]
        if (self.band_proj is None) or (self.band_proj.in_features != freq_bins):
            self.band_proj = nn.Linear(freq_bins, self.n_bands, bias=False).to(device)
            with torch.no_grad():
                W = []
                xs = torch.arange(freq_bins, device=device).float() + 0.5
                for k in range(self.n_bands):
                    W.append(torch.cos(math.pi * (k + 0.5) * xs / freq_bins))
                W = torch.stack(W, dim=0)
                W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
                self.band_proj.weight.copy_(W)
        feat = torch.log(power_agg.clamp_min(self.eps))
        feat = feat.transpose(1, 2)
        feat = self.band_proj(feat)
        feat = F.layer_norm(feat, (self.n_bands,))
        sim = F.cosine_similarity(feat.unsqueeze(2), feat.unsqueeze(1), dim=-1)
        sim = sim / max(self.temperature, self.eps)
        sim = torch.softmax(sim, dim=-1)
        A = self._cosine_topk(sim, self.k)
        if self.row_stochastic:
            A = A / (A.sum(dim=-1, keepdim=True) + self.eps)
        if self.ema > 0:
            if self.prev_A is None or self.prev_A.shape != A.shape:
                self.prev_A = A.detach()
            A = self.ema * self.prev_A + (1 - self.ema) * A
            self.prev_A = A.detach()
        return A


# ==============================================================
# 2️⃣ Encoder + DyGCN + Decoder
# ==============================================================
class StationEncoder(nn.Module):
    def __init__(self, num_sites, encoder_dim, station_embed_dim, input_time, input_dim):
        super().__init__()
        self.input_time = input_time
        self.site_embedding = nn.Embedding(num_sites, station_embed_dim)
        self.flow_encoder = nn.Linear(input_time * input_dim, encoder_dim)
        self.embed_encoder = nn.Linear(station_embed_dim, encoder_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, flow):
        B, T, N, F = flow.shape
        site_idx = torch.arange(N).to(flow.device).unsqueeze(0).expand(B, N)
        site_emb = self.dropout(self.site_embedding(site_idx))
        flow = flow.permute(0, 2, 1, 3).reshape(B, N, T * F)
        flow_feat = self.flow_encoder(flow)
        site_feat = self.embed_encoder(site_emb)
        p = torch.sigmoid(site_feat)
        q = torch.tanh(flow_feat)
        return p * q + (1 - p) * site_feat


class GTUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        self.update = nn.Linear(dim, dim)

    def forward(self, x):
        p = torch.sigmoid(self.gate(x))
        q = torch.tanh(self.update(x))
        return p * q + (1 - p) * x


class Decoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gtu = GTUnit(dim_in)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return self.fc(self.gtu(x))


class DyGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheby_k, embed_dim, node_num):
        super().__init__()
        self.cheby_k = cheby_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheby_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.node_num = node_num
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.xavier_uniform_(self.bias_pool)

    def _normalize_adj(self, A):
        A = A.clamp(min=0)
        I = torch.eye(A.size(-1), device=A.device).unsqueeze(0).expand_as(A)
        A = A + I
        D = A.sum(-1).clamp(min=1e-6)
        D_inv_sqrt = D.pow(-0.5)
        return D_inv_sqrt.unsqueeze(-1) * A * D_inv_sqrt.unsqueeze(-2)

    def forward(self, x, emb, A):
        B, N, F = x.shape
        K = self.cheby_k
        A_hat = self._normalize_adj(A)
        t0 = torch.eye(N, device=A.device).unsqueeze(0).expand(B, -1, -1)
        t1 = A_hat
        support_set = [t0, t1]
        for k in range(2, K):
            support_set.append(2 * A_hat @ support_set[-1] - support_set[-2])
        supports = torch.stack(support_set, dim=1)
        x_g = torch.einsum('bknm,bmf->bknf', supports, x)
        weights = torch.einsum('bne,ekio->bknio', emb, self.weights_pool)
        bias = torch.matmul(emb, self.bias_pool)
        x_g_conv = torch.einsum('bknf,bknfo->bno', x_g, weights) + bias
        return x_g_conv


# ==============================================================
# 3️⃣ 主模型 MyModel：频谱动态图 + Huber&Rank Loss
# ==============================================================
class MyModel(nn.Module):
    def __init__(self, config, data_feature, adj, adj_index):
        super().__init__()
        self.node_num = data_feature['num_nodes']
        self.input_dim = data_feature['input_dim']
        self.output_dim = data_feature['output_dim']
        self.input_time = config['input_time']
        self.output_time = config['output_time']
        self.scaler = data_feature['scaler']

        self.encoder_dim = config['encoder_dim']
        self.gcn_dim = config['gcn_dim']
        self.cheby_k = config['cheby_k']

        self.station_encoder = StationEncoder(
            num_sites=self.node_num,
            encoder_dim=self.encoder_dim,
            station_embed_dim=config['station_embed_dim'],
            input_time=self.input_time,
            input_dim=self.input_dim
        )

        self.spectral_builder = SpectralDynamicGraphBuilder(
            input_dim=self.input_dim, k=10, n_bands=16,
            use_welch=True, temperature=0.07, ema=0.2
        )

        self.dygcn = DyGCN(
            dim_in=self.input_time * self.input_dim,
            dim_out=self.gcn_dim,
            cheby_k=self.cheby_k,
            embed_dim=self.encoder_dim,
            node_num=self.node_num
        )
        self.decoder = Decoder(self.gcn_dim, self.output_dim * self.output_time)
        self.gcn_activation = nn.GELU()
        self.station_norm = nn.LayerNorm(self.encoder_dim)
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self.register_buffer('MS', torch.tensor(adj, dtype=torch.float32))
        self.MS_index = adj_index
        self._init_parameters()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def _init_parameters(self):
        for name, p in self.named_parameters():
            if 'fusion_alpha' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, data):
        flow = data['x']  # [B,T,N,F]
        A_dyn = self.spectral_builder(flow)
        noise = torch.randn_like(self.MS) * 0.05
        A_static_noisy = torch.clamp(self.MS + noise, 0, 1)
        alpha = torch.sigmoid(self.fusion_alpha)
        A_fused = alpha * A_dyn + (1 - alpha) * A_static_noisy
        A_fused = 0.5 * (A_fused + A_fused.transpose(-1, -2))

        flow_station = self.station_encoder(flow)
        flow_station = self.station_norm(flow_station)
        flow_flat = flow.permute(0, 2, 1, 3).reshape(flow.shape[0], flow.shape[2], -1)
        gcn_output = self.dygcn(flow_flat, flow_station, A_fused)
        gcn_output = self.gcn_activation(gcn_output)
        output = self.decoder(gcn_output)
        output = output.reshape(output.shape[0], output.shape[1],
                                self.output_time, self.output_dim)
        output = output.permute(0, 2, 1, 3)
        return output, A_static_noisy, A_dyn, A_fused

    def predict(self, data):
        with torch.no_grad():
            return self.forward(data)

    def cal_train_loss(self, data):
        pred, A_static, A_dyn, A_fused = self.forward(data)
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
        alpha = 10
        beta = 2
        gamma = 5.0  # 可调超参数
        # beta_adaptive = beta * torch.exp(-gamma * delta)
        # self.px.append(loss_graph.detach().item())
        # np.save(f"./graph/graph_data/loss_graph/loss_graph_epoch_{self.epoch_counter}.npy", self.px)
        # self.epoch_counter += 1

        return huber_loss + alpha * rank_loss ,1.0

