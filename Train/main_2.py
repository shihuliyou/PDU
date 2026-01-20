import copy
import json
import os
import random
import sys
import time
import pickle
from copy import deepcopy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from Train.evaluator import evaluate
from .load_data import *
from .PDU import PDUModel
from .utils import DataGraph
from .utils import get_graph_MS

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
device = 'cuda'


def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)


def trr_loss_mse_rank(pred, post_cov, history_price_batch, history_gt_batch, base_price, ground_truth, future_price,
                      mask, alpha, no_stocks):
    return_ratio = torch.div(future_price - base_price, base_price.clamp_min(1e-8))
    pred_pos = F.softplus(pred) + 1e-6
    base_pos = base_price.clamp_min(1e-8)
    realized_volatility = torch.abs(torch.log(pred_pos) - torch.log(base_pos))

    reg_loss = weighted_mse_loss(realized_volatility, ground_truth, mask)
    all_ones = torch.ones(no_stocks, 1).to(device)

    pre_pw_dif = (torch.matmul(realized_volatility, torch.transpose(all_ones, 0, 1))
                  - torch.matmul(all_ones, torch.transpose(realized_volatility, 0, 1)))
    gt_batch = ground_truth
    gt_pw_dif = (torch.matmul(gt_batch, torch.transpose(all_ones, 0, 1)) -
                 torch.matmul(all_ones, torch.transpose(gt_batch, 0, 1)))
    mask_pw = torch.matmul(mask, torch.transpose(mask, 0, 1))

    rank_loss = torch.mean(F.relu((-(pre_pw_dif * gt_pw_dif) * mask_pw)))
    loss = rank_loss + reg_loss + 0.5 * post_cov
    del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
    return loss, rank_loss, reg_loss, realized_volatility, return_ratio


class Stock_PDU:
    def __init__(self, data_path, market_name, tickers_fname, n_node,
                 parameters, steps=1, epochs=None, early_stop_count=0, early_stop_n=3, indicators=None, flat=False,
                 gpu=True, in_pro=False, seed=0, weight=0.6, num_basis=15, args=None, hidden=None):
        self.hidden = hidden
        self.args = args
        self.seed = seed
        self.weight = weight
        self.num_basis = num_basis
        self.early_stop_count = early_stop_count
        self.early_stop_n = early_stop_n
        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)
        self.train_data = pickle.load(open('data/relation/NYSE_File.txt', 'rb'))
        self.n_node = n_node
        self.graph_data = DataGraph(self.train_data, len(self.tickers))
        self.pyg_graph = get_graph_MS(self.graph_data.graph)
        print('tickers selected:', len(self.tickers))
        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(data_path, market_name, self.tickers, steps)
        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
        if indicators is None:
            self.indicators = len(self.tickers)
        else:
            self.indicators = indicators
        self.Stock_num = len(self.tickers)
        self.in_dim = 64
        self.emb_size = 64
        self.days = 4
        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5
        self.gpu = gpu

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.eod_data[:, offset:offset + seq_len, :], \
            np.expand_dims(mask_batch, axis=1), \
            np.expand_dims(self.price_data[:, offset + seq_len - 1], axis=1), \
            np.expand_dims(self.price_data[:, offset + seq_len], axis=1), \
            np.expand_dims(self.price_data[:, offset:offset + seq_len], axis=1), \
            np.expand_dims(self.gt_data[:, offset + seq_len + self.steps - 1], axis=1), \
            np.expand_dims(self.gt_data[:, offset + self.steps:offset + seq_len + self.steps], axis=1), \
            np.expand_dims(self.gt_data[:, offset:offset + seq_len], axis=1)

    def train(self):
        global df
        model = PDUModel(in_feature=4, out_feature=1, node_num=1737, revin=False, adj=self.graph_data.graph,
                         adj_strg=None, hidden_feature=self.hidden).cuda()

        index = 0
        for p in model.parameters():
            index += 1
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        optimizer_hgat = optim.Adam(model.parameters(),
                                    lr=self.parameters['lr'],
                                    weight_decay=1e-3)

        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)

        get_hist = self.eod_data[:, :756, :]
        get_hist = torch.FloatTensor(get_hist).cuda()

        metric_mode = {
            'valid_loss': 'min',
            'mse': 'min',
            'sharpe1': 'max',
            'sharpe5': 'max',
            'sharpe10': 'max',
            'sharpe20': 'max',
            'ndcg5': 'max',
            'mrr_top1': 'max',
            'dd1': 'min',
            'dd5': 'min',
            'dd10': 'min',
            'dd20': 'min',
        }

        best = {}
        for k, mode in metric_mode.items():
            best[k] = {
                'mode': mode,
                'best_valid': float('inf') if mode == 'min' else -float('inf'),
                'best_epoch': -1,
                'test_value': None,
                'valid_value': None,
            }

        def better(mode, new, old):
            return (new < old) if mode == 'min' else (new > old)

        # ====== 早停机制：4 个指标各自早停，全部早停则停止训练 ======
        es_metrics = ['sharpe5', 'ndcg5', 'mrr_top1', 'dd5']
        es_cfg = {
            'sharpe5':  {'mode': 'max', 'warmup': 1,  'min_delta': 0.05,   'patience': 15},
            'ndcg5':    {'mode': 'max', 'warmup': 1, 'min_delta': 0.003,  'patience': 15},
            'mrr_top1': {'mode': 'max', 'warmup': 1, 'min_delta': 0.0008, 'patience': 15},
            'dd5':      {'mode': 'min', 'warmup': 1,  'min_delta': 0.003,  'patience': 15},
        }
        es_state = {}
        for m in es_metrics:
            mode = es_cfg[m]['mode']
            es_state[m] = {
                'best': float('inf') if mode == 'min' else -float('inf'),
                'bad_count': 0,
                'stopped': False,
                'stop_epoch': None,
            }

        def is_better_with_delta(mode, new, old, min_delta):
            if mode == 'min':
                return new <= old - min_delta
            else:
                return new >= old + min_delta

        best_save_path = os.path.join('..', 'output_best_metrics_NYSE.json')
        ckpt_dir = os.path.join('..', 'output_ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)

        for i in range(self.epochs):
            print('epoch:', i)
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0

            model.train()
            for j in tqdm(range(self.valid_index - self.parameters['seq'] - self.steps + 1)):
                emb_batch, mask_batch, price_batch, future_price, history_price_batch, gt_batch, future_gt_batch, history_gt_batch = self.get_batch(
                    batch_offsets[j])

                optimizer_hgat.zero_grad()
                output, post_cov = model.forward(torch.FloatTensor(emb_batch).to(device),
                                                 torch.FloatTensor(self.graph_data.graph).cuda(),
                                                 history_gt_batch,
                                                 h_t=get_hist, pyg_graph=self.pyg_graph, type='train', e=j)

                cur_loss, cur_rank_loss, cur_reg_loss, curr_rv_train, cur_rr_train = trr_loss_mse_rank(
                    output, post_cov,
                    torch.squeeze(torch.FloatTensor(history_price_batch).to(device)),
                    torch.squeeze(torch.FloatTensor(future_gt_batch).to(device)),
                    torch.FloatTensor(price_batch).to(device),
                    torch.FloatTensor(gt_batch).to(device),
                    torch.FloatTensor(future_price).to(device),
                    torch.FloatTensor(mask_batch).to(device),
                    self.parameters['alpha'],
                    self.indicators
                )

                all_loss = cur_loss
                all_loss.backward()
                optimizer_hgat.step()
                model.update_moving_average()

                tra_loss += all_loss.detach().cpu().item()
                tra_reg_loss += cur_reg_loss.detach().cpu().item()
                tra_rank_loss += cur_rank_loss.detach().cpu().item()

            print('Train Loss:',
                  tra_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1))

            with torch.no_grad():
                cur_valid_pred = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
                cur_valid_gt = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
                cur_valid_mask = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
                cur_valid_rr = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)

                val_loss = 0.0
                val_reg_loss = 0.0
                val_rank_loss = 0.0

                model.eval()
                for cur_offset in tqdm(range(
                        self.valid_index - self.parameters['seq'] - self.steps + 1,
                        self.test_index - self.parameters['seq'] - self.steps + 1
                )):
                    emb_batch, mask_batch, price_batch, future_price, history_price_batch, gt_batch, future_gt_batch, history_gt_batch = self.get_batch(
                        cur_offset)

                    output_val, post_cov = model(torch.FloatTensor(emb_batch).to(device),
                                                 torch.FloatTensor(self.graph_data.graph).cuda(),
                                                 history_gt_batch=history_gt_batch, h_t=get_hist,
                                                 pyg_graph=self.pyg_graph, type='vel', e=cur_offset)

                    cur_loss, cur_rank_loss, cur_reg_loss, cur_rv, cur_rr = trr_loss_mse_rank(
                        output_val, post_cov,
                        torch.squeeze(torch.FloatTensor(history_price_batch).to(device)),
                        torch.squeeze(torch.FloatTensor(future_gt_batch).to(device)),
                        torch.FloatTensor(price_batch).to(device),
                        torch.FloatTensor(gt_batch).to(device),
                        torch.FloatTensor(future_price).to(device),
                        torch.FloatTensor(mask_batch).to(device),
                        self.parameters['alpha'],
                        self.indicators
                    )

                    cur_rv = cur_rv.detach().cpu().numpy().reshape((len(self.tickers), 1))
                    cur_rr = cur_rr.detach().cpu().numpy().reshape((len(self.tickers), 1))

                    val_loss += cur_loss.detach().cpu().item()
                    val_reg_loss += cur_reg_loss.detach().cpu().item()
                    val_rank_loss += cur_rank_loss.detach().cpu().item()

                    idx = cur_offset - (self.valid_index - self.parameters['seq'] - self.steps + 1)
                    cur_valid_pred[:, idx] = copy.copy(cur_rv[:, 0])
                    cur_valid_gt[:, idx] = copy.copy(gt_batch[:, 0])
                    cur_valid_mask[:, idx] = copy.copy(mask_batch[:, 0])
                    cur_valid_rr[:, idx] = copy.copy(cur_rr[:, 0])

                print('Valid LOSS:', val_loss / (self.test_index - self.valid_index))
                cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask, cur_valid_rr, report=False)

                print('\t Valid performance:',
                      'sharpe1:', cur_valid_perf.get('sharpe1', 0.0),
                      'sharpe5:', cur_valid_perf.get('sharpe5', 0.0),
                      'sharpe10:', cur_valid_perf.get('sharpe10', 0.0),
                      'sharpe20:', cur_valid_perf.get('sharpe20', 0.0),
                      'mse_rv:', cur_valid_perf.get('mse', 0.0),
                      'ndcg5:', cur_valid_perf.get('ndcg5', 0.0),
                      'mrr_top1:', cur_valid_perf.get('mrr_top1', 0.0),
                      'dd1:', cur_valid_perf.get('dd1', 0.0),
                      'dd5:', cur_valid_perf.get('dd5', 0.0),
                      'dd10:', cur_valid_perf.get('dd10', 0.0),
                      'dd20:', cur_valid_perf.get('dd20', 0.0))

                # ===================== TEST =====================
                cur_test_pred = np.zeros([len(self.tickers), self.trade_dates - self.test_index], dtype=float)
                cur_test_gt = np.zeros([len(self.tickers), self.trade_dates - self.test_index], dtype=float)
                cur_test_mask = np.zeros([len(self.tickers), self.trade_dates - self.test_index], dtype=float)
                cur_test_rr = np.zeros([len(self.tickers), self.trade_dates - self.test_index], dtype=float)

                test_loss = 0.0
                test_reg_loss = 0.0
                test_rank_loss = 0.0

                model.eval()
                for cur_offset in tqdm(range(
                        self.test_index - self.parameters['seq'] - self.steps + 1,
                        self.trade_dates - self.parameters['seq'] - self.steps + 1
                )):
                    emb_batch, mask_batch, price_batch, future_price, history_price_batch, gt_batch, future_gt_batch, history_gt_batch = self.get_batch(
                        cur_offset)

                    output_test, post_cov = model(torch.FloatTensor(emb_batch).to(device),
                                                  torch.FloatTensor(self.graph_data.graph).cuda(),
                                                  history_gt_batch=history_gt_batch, h_t=get_hist,
                                                  pyg_graph=self.pyg_graph, type='test', e=cur_offset)

                    cur_loss, cur_rank_loss, cur_reg_loss, cur_rv, cur_rr = trr_loss_mse_rank(
                        output_test, post_cov,
                        torch.squeeze(torch.FloatTensor(history_price_batch).to(device)),
                        torch.squeeze(torch.FloatTensor(future_gt_batch).to(device)),
                        torch.FloatTensor(price_batch).to(device),
                        torch.FloatTensor(gt_batch).to(device),
                        torch.FloatTensor(future_price).to(device),
                        torch.FloatTensor(mask_batch).to(device),
                        self.parameters['alpha'],
                        self.indicators
                    )

                    cur_rv = cur_rv.detach().cpu().numpy().reshape((len(self.tickers), 1))
                    cur_rr = cur_rr.detach().cpu().numpy().reshape((len(self.tickers), 1))

                    test_loss += cur_loss.detach().cpu().item()
                    test_reg_loss += cur_reg_loss.detach().cpu().item()
                    test_rank_loss += cur_rank_loss.detach().cpu().item()

                    idx = cur_offset - (self.test_index - self.parameters['seq'] - self.steps + 1)
                    cur_test_pred[:, idx] = copy.copy(cur_rv[:, 0])
                    cur_test_gt[:, idx] = copy.copy(gt_batch[:, 0])
                    cur_test_mask[:, idx] = copy.copy(mask_batch[:, 0])
                    cur_test_rr[:, idx] = copy.copy(cur_rr[:, 0])

            print('Test LOSS:', test_loss / (self.trade_dates - self.test_index))
            cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask, cur_test_rr,
                                     self.parameters['unit'], i, self.market_name, report=True)

            valid_loss_avg = val_loss / (self.test_index - self.valid_index)
            test_loss_avg = test_loss / (self.trade_dates - self.test_index)
            cur_valid_pack = dict(cur_valid_perf)
            cur_test_pack = dict(cur_test_perf)
            cur_valid_pack['valid_loss'] = float(valid_loss_avg)
            cur_test_pack['valid_loss'] = float(test_loss_avg)

            for metric, rec in best.items():
                if metric in es_state and es_state[metric]['stopped']:
                    continue

                mode = rec['mode']
                v = cur_valid_pack.get(metric, None)
                t = cur_test_pack.get(metric, None)

                if v is None:
                    continue

                if better(mode, float(v), rec['best_valid']):
                    rec['best_valid'] = float(v)
                    rec['valid_value'] = float(v)
                    rec['best_epoch'] = int(i)
                    rec['test_value'] = None if t is None else float(t)

                    torch.save(model.state_dict(), os.path.join(ckpt_dir, f'NYSE_best_by_valid_{metric}.pt'))

                    with open(best_save_path, 'w', encoding='utf-8') as f:
                        json.dump(best, f, ensure_ascii=False, indent=2)

                    print(f'[BEST] metric={metric} valid={rec["valid_value"]} test={rec["test_value"]} epoch={rec["best_epoch"]}')

            for m in es_metrics:
                if es_state[m]['stopped']:
                    continue

                v = cur_valid_pack.get(m, None)
                if v is None:
                    continue

                cfg = es_cfg[m]
                st = es_state[m]

                raw_v = float(v)

                if i < cfg['warmup']:
                    if is_better_with_delta(cfg['mode'], raw_v, st['best'], 0.0):
                        st['best'] = raw_v
                    continue

                if is_better_with_delta(cfg['mode'], raw_v, st['best'], cfg['min_delta']):
                    st['best'] = raw_v
                    st['bad_count'] = 0
                else:
                    st['bad_count'] += 1
                    if st['bad_count'] >= cfg['patience']:
                        st['stopped'] = True
                        st['stop_epoch'] = i
                        print(f'[EARLY_STOP] metric={m} stop at epoch={i} (best={st["best"]:.6f})')

            if all(es_state[m]['stopped'] for m in es_metrics):
                print(f'[EARLY_STOP][ALL] stop training at epoch={i} because all {es_metrics} stopped.')
                break

            print('\t Test performance:',
                  'sharpe1:', cur_test_perf.get('sharpe1', 0.0),
                  'sharpe5:', cur_test_perf.get('sharpe5', 0.0),
                  'sharpe10:', cur_test_perf.get('sharpe10', 0.0),
                  'sharpe20:', cur_test_perf.get('sharpe20', 0.0),
                  'mse_rv:', cur_test_perf.get('mse', 0.0),
                  'ndcg5:', cur_test_perf.get('ndcg5', 0.0),
                  'mrr_top1:', cur_test_perf.get('mrr_top1', 0.0),
                  'dd1:', cur_test_perf.get('dd1', 0.0),
                  'dd5:', cur_test_perf.get('dd5', 0.0),
                  'dd10:', cur_test_perf.get('dd10', 0.0),
                  'dd20:', cur_test_perf.get('dd20', 0.0))

            np.set_printoptions(threshold=sys.maxsize)

    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='path of EOD data', default='data/2013-01-01-3')
    parser.add_argument('-m', help='market name', default='NYSE')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=4, help='length of historical sequence for feature')
    parser.add_argument('-u', default=64, help='number of hidden units in lstm')
    parser.add_argument('-s', default=1, help='steps to make prediction')
    parser.add_argument('-r', default=0.001, help='learning rate')
    parser.add_argument('-a', default=1, help='alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument('-rn', '--rel_name', type=str, default='sector_industry',
                        help='relation type: sector_industry or wikidata')
    parser.add_argument('-ip', '--inner_prod', type=int, default=0)
    parser.add_argument('-node', default=1026, help='n_node')
    parser.add_argument('-seed', default=57, help='seed')
    args = parser.parse_args()

    args.gpu = (args.gpu == 1)
    args.inner_prod = (args.inner_prod == 1)

    market_name = 'NYSE'
    args.t = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'

    parameters = {'seq': int(4), 'unit': int(args.u), 'lr': float(0.001), 'alpha': float(args.a)}

    PDU_NET = Stock_PDU(
        data_path=args.p,
        market_name=market_name,
        tickers_fname=args.t,
        n_node=args.node,
        parameters=parameters,
        steps=1, epochs=100,
        early_stop_count=0,
        early_stop_n=500,
        indicators=None, gpu=args.gpu,
        in_pro=args.inner_prod,
        seed=64, args=args,
        hidden=64
    )
    PDU_NET.train()
