import numpy as np
from sklearn.metrics import ndcg_score


def evaluate(prediction, ground_truth, mask, return_truth, epoch=None, unit=None, market='hope', report=None):
    """
    prediction:     (N, T)  预测的单日RV（用于排名，越小越好）
    ground_truth:   (N, T)  真实的单日RV（用于MSE、用于评估排序）
    mask:           (N, T)  有效掩码
    return_truth:   (N, T)  真实收益率（用于Sharpe / DD，真实收益）
    """
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    assert return_truth.shape == prediction.shape, 'return_truth shape mis-match'

    # 这里把 return_truth 转成“超额收益”（按日无风险利率 3% 年化）
    return_truth = return_truth - ((1.0 + 0.03) ** (1.0 / 252.0) - 1.0)

    performance = {
        'mse': np.linalg.norm((prediction - ground_truth) * mask) ** 2 / np.sum(mask)
    }

    sharpe_li1, sharpe_li5, sharpe_li10, sharpe_li20 = [], [], [], []
    ann = 15.87  # sqrt(252)

    # MRR（按“预测top1在真实排序中的名次”）
    mrr_sum = 0.0
    mrr_cnt = 0
    mrr_miss_days = 0

    # NDCG@5（每天算一个，最后平均）
    ndcg5_sum = 0.0
    ndcg5_cnt = 0

    N, T = prediction.shape

    for i in range(T):
        valid_idx = np.where(mask[:, i] >= 0.5)[0]
        if valid_idx.size == 0:
            mrr_miss_days += 1
            continue

        # -----------------------
        # 1) 按预测RV选 topK（越大越好）
        # -----------------------
        pred_day = prediction[valid_idx, i]
        order_pred = np.argsort(-pred_day)  # 升序：RV大的在前

        def pick_topk(order, k):
            k = min(k, order.size)
            return valid_idx[order[:k]]

        pre_top1 = pick_topk(order_pred, 1)
        pre_top5 = pick_topk(order_pred, 5)
        pre_top10 = pick_topk(order_pred, 10)
        pre_top20 = pick_topk(order_pred, 20)

        # -----------------------
        # 2) Sharpe：用真实收益序列（等权topK）
        # -----------------------
        r1 = return_truth[pre_top1[0], i]
        r5 = np.mean(return_truth[pre_top5, i]) if pre_top5.size > 0 else np.nan
        r10 = np.mean(return_truth[pre_top10, i]) if pre_top10.size > 0 else np.nan
        r20 = np.mean(return_truth[pre_top20, i]) if pre_top20.size > 0 else np.nan

        if np.isfinite(r1):
            sharpe_li1.append(float(r1))
        if np.isfinite(r5):
            sharpe_li5.append(float(r5))
        if np.isfinite(r10):
            sharpe_li10.append(float(r10))
        if np.isfinite(r20):
            sharpe_li20.append(float(r20))

        # -----------------------
        # 3) MRR：预测top1在真实RV排序中的名次倒数
        #    真实RV也“越大越好”，所以同样升序
        # -----------------------
        gt_day = ground_truth[valid_idx, i]
        order_gt = np.argsort(-gt_day)  # 升序：RV大的在前

        top1_global = int(pre_top1[0])
        pos = np.where(valid_idx[order_gt] == top1_global)[0]
        if pos.size == 0:
            mrr_miss_days += 1
        else:
            rank_pos = int(pos[0]) + 1
            mrr_sum += 1.0 / rank_pos
            mrr_cnt += 1

        # -----------------------
        # 4) NDCG@5（改法B：用“真实RV的排名”当 relevance，天然非负）
        #    y_score 越大越好：用 -prediction（RV越小得分越高）
        #    y_true_rel 越大越好：用 真实RV的“好排名” -> 大relevance（最好=最大）
        # -----------------------
        n_items = valid_idx.size
        k = min(5, n_items)

        if k >= 1:
            # y_score: 分数越大越好（RV越小越好，所以取负）
            y_score = pred_day.reshape(1, -1)

            # y_true_rel: 用排名做relevance（非负）
            # gt_day 越大越好：最大的给最大relevance
            order = np.argsort(-gt_day)                 # 从大到小（最好在最前）
            ranks = np.empty_like(order)
            ranks[order] = np.arange(n_items)          # 最好=0
            y_true_rel_1d = (n_items - 1 - ranks).astype(float)  # 最好 -> n_items-1，最差 -> 0
            y_true_rel = y_true_rel_1d.reshape(1, -1)

            nd = float(ndcg_score(y_true_rel, y_score, k=k))
            ndcg5_sum += nd
            ndcg5_cnt += 1

    def safe_sharpe(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0
        s = np.std(x)
        if s <= 1e-12:
            return 0.0
        return float((np.mean(x) / s) * ann)

    def safe_downside_deviation(x, tau=0.0, ann_factor=np.sqrt(252.0)):
        """
        年化 Downside Deviation (DD):
          DD_daily = sqrt(E[min(x - tau, 0)^2])
          DD_ann   = DD_daily * sqrt(252)

        - x: 日频收益（建议用“日超额收益”）；会自动过滤 nan/inf
        - tau: 阈值（tau=0 表示只惩罚负收益；若 x 不是超额收益而想扣 rf，可设 tau=rf_daily）
        - ann_factor: 年化因子，默认 sqrt(252)
        """
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0

        downside = np.minimum(x - tau, 0.0)  # 只保留下行部分（<=0）
        dd_daily = float(np.sqrt(np.mean(downside ** 2)))
        return dd_daily * float(ann_factor)

    performance['sharpe1'] = safe_sharpe(sharpe_li1)
    performance['sharpe5'] = safe_sharpe(sharpe_li5)
    performance['sharpe10'] = safe_sharpe(sharpe_li10)
    performance['sharpe20'] = safe_sharpe(sharpe_li20)

    performance['mrr_top1'] = float(mrr_sum / mrr_cnt) if mrr_cnt > 0 else 0.0
    performance['mrr_miss_days'] = int(mrr_miss_days)

    performance['ndcg5'] = float(ndcg5_sum / ndcg5_cnt) if ndcg5_cnt > 0 else 0.0

    # 下行偏差：基于各组合的“日收益序列”
    performance['dd1'] = safe_downside_deviation(sharpe_li1, tau=0.0)
    performance['dd5'] = safe_downside_deviation(sharpe_li5, tau=0.0)
    performance['dd10'] = safe_downside_deviation(sharpe_li10, tau=0.0)
    performance['dd20'] = safe_downside_deviation(sharpe_li20, tau=0.0)

    # 方便你调试/画图
    performance['data1_series'] = sharpe_li1
    performance['data5_series'] = sharpe_li5
    performance['data10_series'] = sharpe_li10
    performance['data20_series'] = sharpe_li20

    return performance
