import copy
import numpy as np
import os


def load_EOD_data(data_path, market_name, tickers, steps=1):
    eod_data = []
    masks = []
    ground_truth = []
    base_price = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ':
            # remove the last day since lots of missing data
            single_EOD = single_EOD[:-1, :]
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0],
                                 single_EOD.shape[1] - 1], dtype=np.float32)
            masks = np.ones([len(tickers), single_EOD.shape[0]],
                            dtype=np.float32)
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
                                    dtype=np.float32)
            base_price = np.zeros([len(tickers), single_EOD.shape[0]],
                                  dtype=np.float32)
        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                masks[index][row] = 0.0
            elif row > steps - 1 and abs(single_EOD[row - steps][-1] + 1234) > 1e-8:
                p_t = single_EOD[row, -1]
                p_tm = single_EOD[row - 1, -1]
                if (abs(p_t + 1234) < 1e-8) or (abs(p_tm + 1234) < 1e-8) or (p_t <= 0) or (p_tm <= 0):
                    masks[index][row] = 0.0
                    continue
                r = np.log(p_t) - np.log(p_tm)
                ground_truth[index][row] = abs(r)
            for col in range(single_EOD.shape[1] - 1):
                if abs(single_EOD[row][col] + 1234) < 1e-8:
                    single_EOD[row][col] = 1.1
        eod_data[index, :, :] = single_EOD[:, 1:]
        base_price[index, :] = single_EOD[:, -1]
    return eod_data, masks, ground_truth, base_price
