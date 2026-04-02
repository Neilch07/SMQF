from .factor import factor
import numpy as np
import pandas as pd
from .analysis import *

class alpha191_003(factor):
    def set_factor_property(self):
        factor_property={'factor_name': 'alpha191_003','data_needed': ['close','high','low'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        close = self.data['close']  
        high = self.data['high']
        low = self.data['low']
        delay_close = close.shift(1)  
        term = np.zeros_like(close)
        condition1 = (close == delay_close)
        condition2 = (close > delay_close)
        min_val = np.minimum(low, delay_close)
        max_val = np.maximum(high, delay_close)
        term = np.where(condition2, close - min_val, term)
        term = np.where((~condition1) & (~condition2), close - max_val, term)
        term_df = pd.DataFrame(term, index=close.index, columns=close.columns)
        self.factor_data = term_df.rolling(6, min_periods=1).sum()

class alpha191_008(factor):
    def set_factor_property(self):
        factor_property={'factor_name': 'alpha191_008','data_needed': ['high', 'low', 'amount', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        amounts = self.data['amount']
        volumes = self.data['volume']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        highs = self.data['high']
        lows = self.data['low']
        weighted_price = ((highs + lows) / 2) * 0.2 + vwaps * 0.8
        delta_val = weighted_price.diff(4)
        self.factor_data = cs_rank(-1 * delta_val)

class alpha191_012(factor):
    def set_factor_property(self):
        factor_property={'factor_name': 'alpha191_012','data_needed': ['open', 'close', 'amount', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        amounts = self.data['amount']
        volumes = self.data['volume']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        opens = self.data['open']
        closes = self.data['close']
        vwap_sma10 = vwaps.rolling(10, min_periods=5).mean()
        rank_part1 = cs_rank(opens - vwap_sma10)
        rank_part2 = cs_rank(np.abs(closes - vwaps))
        self.factor_data = rank_part1 * (-1 * rank_part2)

class alpha191_014(factor):
    def set_factor_property(self):
        factor_property={'factor_name': 'alpha191_014','data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        self.factor_data = closes - ts_delay(closes, 5)

class alpha191_015(factor):
    def set_factor_property(self):
        factor_property={'factor_name': 'alpha191_015','data_needed': ['open', 'close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        opens = self.data['open']
        closes = self.data['close']
        prev_close = ts_delay(closes, 1)
        self.factor_data = opens / prev_close - 1

class alpha191_016(factor):
    def set_factor_property(self):
        factor_property={'factor_name': 'alpha191_016','data_needed': ['volume', 'amount'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        volumes = self.data['volume']
        amounts = self.data['amount']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        rank_volume = cs_rank(volumes)
        rank_vwap = cs_rank(vwaps)
        corr_val = ts_corr1(rank_volume, rank_vwap, 5)
        rank_corr = cs_rank(corr_val)
        tsmax_val = rank_corr.rolling(5, min_periods=2).max()
        self.factor_data = -1 * tsmax_val

class alpha191_018(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_018', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        self.factor_data = closes / ts_delay(closes, 5)

class alpha191_019(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_019', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        prev_close_5 = ts_delay(closes, 5)
        factor_values = np.where(closes < prev_close_5, (closes - prev_close_5) / prev_close_5, np.where(closes == prev_close_5, 0, (closes - prev_close_5) / closes))
        self.factor_data = pd.DataFrame(factor_values, index=closes.index, columns=closes.columns)

class alpha191_020(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_020', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        prev_close_6 = ts_delay(closes, 6)
        self.factor_data = (closes - prev_close_6) / prev_close_6 * 100

class alpha191_022(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_022', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        mean_close_6 = ts_mean(closes, 6)
        x = (closes - mean_close_6) / mean_close_6
        y = x - ts_delay(x, 3)
        self.factor_data = ts_mean(y, 12)

class alpha191_024(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_024', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        price_diff = closes - closes.shift(5)
        self.factor_data = price_diff.rolling(window=5, min_periods=5).apply(
            lambda x: (x[-1] + np.mean(x[:-1]) * 4) / 5, raw=True
        )

class alpha191_025(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_025', 'data_needed': ['close', 'volume', 'return'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        volumes = self.data['volume']
        returns = self.data['return']
        decay_w = np.arange(1, 10)
        decay_val = wma(volumes / ts_mean(volumes, 20), decay_w)
        part_a = -1 * cs_rank(ts_delta(closes, 7) * (1 - cs_rank(decay_val)))
        part_b = 1 + cs_rank(ts_sum(returns, 250))
        self.factor_data = part_a * part_b

class alpha191_026(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_026', 'data_needed': ['close', 'amount', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        amounts = self.data['amount']
        volumes = self.data['volume']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        part_a = ts_mean(closes, 7) - closes
        part_b = ts_corr1(vwaps, ts_delay(closes, 5), 230)
        self.factor_data = part_a + part_b

class alpha191_028(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_028', 'data_needed': ['close', 'low', 'high'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        lows = self.data['low']
        highs = self.data['high']
        tsmin_low_9 = lows.rolling(9, min_periods=4).min()
        tsmax_high_9 = highs.rolling(9, min_periods=4).max()
        stoch_k = ((closes - tsmin_low_9) / (tsmax_high_9 - tsmin_low_9)) * 100
        stoch_d = ts_mean(stoch_k, 3)
        self.factor_data = 3 * stoch_d - 2 * ts_mean(stoch_d, 3)

class alpha191_029(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_029', 'data_needed': ['close', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        volumes = self.data['volume']
        self.factor_data = (closes - ts_delay(closes, 6)) / ts_delay(closes, 6) * volumes

class alpha191_032(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_032', 'data_needed': ['high', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        highs = self.data['high']
        volumes = self.data['volume']
        corr_val = ts_corr1(cs_rank(highs), cs_rank(volumes), 3)
        self.factor_data = -1 * ts_sum(cs_rank(corr_val), 3)

class alpha191_034(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_034', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        self.factor_data = ts_mean(closes, 12) / closes

class alpha191_036(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_036', 'data_needed': ['volume', 'amount'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        volumes = self.data['volume']
        amounts = self.data['amount']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        corr_val = ts_corr1(cs_rank(volumes), cs_rank(vwaps), 6)
        self.factor_data = cs_rank(ts_sum(corr_val, 2))

class alpha191_039(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_039', 'data_needed': ['close', 'amount', 'volume', 'open'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        amounts = self.data['amount']
        volumes = self.data['volume']
        opens = self.data['open']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        decay_w8 = np.arange(1, 9)
        part_a = cs_rank(wma(ts_delta(closes, 2), decay_w8))
        x = (vwaps * 0.3) + (opens * 0.7)
        y = ts_sum(ts_mean(volumes, 180), 37)
        corr_val = ts_corr1(x, y, 14)
        decay_w12 = np.arange(1, 13)
        part_b = cs_rank(wma(corr_val, decay_w12))
        self.factor_data = (part_b - part_a)

class alpha191_041(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_041', 'data_needed': ['amount', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        amounts = self.data['amount']
        volumes = self.data['volume']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        delta_vwap = ts_delta(vwaps, 3)
        tsmax_val = delta_vwap.rolling(5, min_periods=2).max()
        self.factor_data = -1 * cs_rank(tsmax_val)

class alpha191_042(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_042', 'data_needed': ['high', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        highs = self.data['high']
        volumes = self.data['volume']
        part_a = -1 * cs_rank(ts_std(highs, 10))
        part_b = ts_corr1(highs, volumes, 10)
        self.factor_data = part_a * part_b

class alpha191_045(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_045', 'data_needed': ['close', 'open', 'amount', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        opens = self.data['open']
        amounts = self.data['amount']
        volumes = self.data['volume']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        weighted_price = (closes * 0.6) + (opens * 0.4)
        part_a = cs_rank(ts_delta(weighted_price, 1))
        mean_vol = ts_mean(volumes, 150)
        part_b = cs_rank(ts_corr1(vwaps, mean_vol, 15))
        self.factor_data = part_a * part_b

class alpha191_046(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_046', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        mean3 = ts_mean(closes, 3)
        mean6 = ts_mean(closes, 6)
        mean12 = ts_mean(closes, 12)
        mean24 = ts_mean(closes, 24)
        self.factor_data = (mean3 + mean6 + mean12 + mean24) / (4 * closes)

class alpha191_054(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_054', 'data_needed': ['close', 'open'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        opens = self.data['open']
        close_open_diff = closes - opens
        inner_expr = ts_std(np.abs(close_open_diff), 10) + close_open_diff + ts_corr1(closes, opens, 10)
        self.factor_data = -1 * cs_rank(inner_expr)

class alpha191_055(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_055', 'data_needed': ['high', 'low', 'close', 'open'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        highs = self.data['high']
        lows = self.data['low']
        closes = self.data['close']
        opens = self.data['open']
        prev_close = ts_delay(closes, 1)
        prev_open = ts_delay(opens, 1)
        h_pc = np.abs(highs - prev_close)
        l_pc = np.abs(lows - prev_close)
        h_l = np.abs(highs - lows)
        pc_po = np.abs(prev_close - prev_open)
        cond1 = (h_pc > l_pc) & (h_pc > h_l)
        val1 = h_pc + l_pc / 2.0 + pc_po / 4.0
        cond2 = (l_pc > h_pc) & (l_pc > h_l)
        val2 = l_pc + h_pc / 2.0 + pc_po / 4.0
        val3 = h_l + pc_po / 4.0
        i = np.where(cond1, val1, np.where(cond2, val2, val3))
        term_max = np.maximum(h_pc, l_pc)
        self.factor_data = ts_sum(i * term_max, 20)

class alpha191_057(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_057', 'data_needed': ['close', 'low', 'high'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        lows = self.data['low']
        highs = self.data['high']
        tsmin_low_9 = lows.rolling(9, min_periods=4).min()
        tsmax_high_9 = highs.rolling(9, min_periods=4).max()
        stoch_k = ((closes - tsmin_low_9) / (tsmax_high_9 - tsmin_low_9)) * 100
        self.factor_data = ts_mean(stoch_k, 3)

class alpha191_061(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_061', 'data_needed': ['amount', 'volume', 'low'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        amounts = self.data['amount']
        volumes = self.data['volume']
        lows = self.data['low']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        decay_w12 = np.arange(1, 13)
        part_a = cs_rank(wma(ts_delta(vwaps, 1), decay_w12))
        corr_val = ts_corr1(lows, ts_mean(volumes, 80), 8)
        decay_w17 = np.arange(1, 18)
        part_b = cs_rank(wma(cs_rank(corr_val), decay_w17))
        self.factor_data = -1 * np.maximum(part_a, part_b)

class alpha191_062(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_062', 'data_needed': ['high', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        highs = self.data['high']
        volumes = self.data['volume']
        self.factor_data = -1 * ts_corr1(highs, cs_rank(volumes), 5)

class alpha191_063(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_063', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        delta1 = ts_delta(closes, 1)
        up_move = ts_mean(np.maximum(delta1, 0), 6)
        down_move = ts_mean(np.abs(delta1), 6)
        self.factor_data = up_move / down_move * 100

class alpha191_064(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_064', 'data_needed': ['amount', 'volume', 'close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        amounts = self.data['amount']
        volumes = self.data['volume']
        closes = self.data['close']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        decay_w4 = np.arange(1, 5)
        corr_a = ts_corr1(cs_rank(vwaps), cs_rank(volumes), 4)
        part_a = cs_rank(wma(corr_a, decay_w4))
        corr_b = ts_corr1(cs_rank(closes), cs_rank(ts_mean(volumes, 60)), 4)
        tsmax_b = corr_b.rolling(4, min_periods=2).max()
        decay_w13 = np.arange(1, 14)
        part_b = cs_rank(wma(tsmax_b, decay_w13))
        self.factor_data = -1 * np.maximum(part_a, part_b)

class alpha191_065(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_065', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        self.factor_data = ts_mean(closes, 6) / closes

class alpha191_066(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_066', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        mean_close_6 = ts_mean(closes, 6)
        self.factor_data = (closes - mean_close_6) / mean_close_6 * 100

class alpha191_070(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_070', 'data_needed': ['amount'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        amounts = self.data['amount']
        self.factor_data = ts_std(amounts, 6)

class alpha191_071(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_071', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        mean_close_24 = ts_mean(closes, 24)
        self.factor_data = (closes - mean_close_24) / mean_close_24 * 100

class alpha191_074(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_074', 'data_needed': ['low', 'amount', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        lows = self.data['low']
        amounts = self.data['amount']
        volumes = self.data['volume']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        x = ts_sum((lows * 0.35) + (vwaps * 0.65), 20)
        y = ts_sum(ts_mean(volumes, 40), 20)
        part_a = cs_rank(ts_corr1(x, y, 7))
        part_b = cs_rank(ts_corr1(cs_rank(vwaps), cs_rank(volumes), 6))
        self.factor_data = part_a + part_b

class alpha191_079(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_079', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        delta1 = ts_delta(closes, 1)
        up_move = ts_mean(np.maximum(delta1, 0), 12)
        down_move = ts_mean(np.abs(delta1), 12)
        self.factor_data = up_move / down_move * 100

class alpha191_081(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_081', 'data_needed': ['volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        volumes = self.data['volume']
        self.factor_data = ts_mean(volumes, 21)

class alpha191_083(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_083', 'data_needed': ['high', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        highs = self.data['high']
        volumes = self.data['volume']
        rank_high = cs_rank(highs)
        rank_volume = cs_rank(volumes)
        self.factor_data = -1 * cs_rank(rank_high.rolling(5, min_periods=2).cov(rank_volume))

class alpha191_087(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_087', 'data_needed': ['amount', 'volume', 'low', 'open', 'high'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        amounts = self.data['amount']
        volumes = self.data['volume']
        lows = self.data['low']
        opens = self.data['open']
        highs = self.data['high']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        decay_w7 = np.arange(1, 8)
        part_a = cs_rank(wma(ts_delta(vwaps, 4), decay_w7))
        numerator = (lows - vwaps)
        denominator = (opens - (highs + lows) / 2.0)
        x = numerator / denominator
        decay_w11 = np.arange(1, 12)
        part_b = ts_rank(wma(x, decay_w11), 7)
        self.factor_data = -1 * (part_a + part_b)

class alpha191_088(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_088', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        prev_close_20 = ts_delay(closes, 20)
        self.factor_data = (closes - prev_close_20) / prev_close_20 * 100

class alpha191_090(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_090', 'data_needed': ['amount', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        amounts = self.data['amount']
        volumes = self.data['volume']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        corr_val = ts_corr1(cs_rank(vwaps), cs_rank(volumes), 5)
        self.factor_data = -1 * cs_rank(corr_val)

class alpha191_092(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_092', 'data_needed': ['close', 'amount', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        amounts = self.data['amount']
        volumes = self.data['volume']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        weighted_price = (closes * 0.35) + (vwaps * 0.65)
        decay_w3 = np.arange(1, 4)
        part_a = cs_rank(wma(ts_delta(weighted_price, 2), decay_w3))
        corr_val = ts_corr1(ts_mean(volumes, 180), closes, 13)
        decay_w5 = np.arange(1, 6)
        part_b = ts_rank(wma(np.abs(corr_val), decay_w5), 15)
        self.factor_data = -1 * np.maximum(part_a, part_b)

class alpha191_094(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_094', 'data_needed': ['close', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        volumes = self.data['volume']
        delta1 = ts_delta(closes, 1)
        signed_volume = np.sign(delta1) * volumes
        self.factor_data = ts_sum(signed_volume, 30)

class alpha191_095(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_095', 'data_needed': ['amount'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        amounts = self.data['amount']
        self.factor_data = ts_std(amounts, 20)

class alpha191_097(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_097', 'data_needed': ['volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        volumes = self.data['volume']
        self.factor_data = ts_std(volumes, 10)

class alpha191_099(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_099', 'data_needed': ['close', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        volumes = self.data['volume']
        rank_close = cs_rank(closes)
        rank_volume = cs_rank(volumes)
        self.factor_data = -1 * cs_rank(rank_close.rolling(5, min_periods=2).cov(rank_volume))

class alpha191_100(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_100', 'data_needed': ['volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        volumes = self.data['volume']
        self.factor_data = ts_std(volumes, 20)

class alpha191_102(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_102', 'data_needed': ['volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        volumes = self.data['volume']
        delta1 = ts_delta(volumes, 1)
        up_move = ts_mean(np.maximum(delta1, 0), 6)
        down_move = ts_mean(np.abs(delta1), 6)
        self.factor_data = up_move / down_move * 100

class alpha191_106(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_106', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        self.factor_data = closes - ts_delay(closes, 20)

class alpha191_109(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_109', 'data_needed': ['high', 'low'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        highs = self.data['high']
        lows = self.data['low']
        x = ts_mean(highs - lows, 10)
        self.factor_data = x / ts_mean(x, 10)

class alpha191_113(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_113', 'data_needed': ['close', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        volumes = self.data['volume']
        part_a = cs_rank(ts_mean(ts_delay(closes, 5), 20))
        part_b = ts_corr1(closes, volumes, 2)
        part_c = cs_rank(ts_corr1(ts_sum(closes, 5), ts_sum(closes, 20), 2))
        self.factor_data = -1 * part_a * part_b * part_c

class alpha191_121(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_121', 'data_needed': ['amount', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        amounts = self.data['amount']
        volumes = self.data['volume']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        tsmin_vwap_12 = vwaps.rolling(12, min_periods=5).min()
        part_a = cs_rank(vwaps - tsmin_vwap_12)
        corr_val = ts_corr1(ts_rank(vwaps, 20), ts_rank(ts_mean(volumes, 60), 2), 18)
        part_b = ts_rank(corr_val, 3)
        self.factor_data = -1 * (part_a / part_b)

class alpha191_122(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_122', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        x = ts_mean(np.log(closes), 13)
        y = ts_mean(x, 13)
        z = ts_mean(y, 13)
        self.factor_data = ts_delta(z, 1) / ts_delay(z, 1)

class alpha191_124(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_124', 'data_needed': ['close', 'amount', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        amounts = self.data['amount']
        volumes = self.data['volume']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        decay_w2 = np.arange(1, 3)
        tsmax_val = closes.rolling(30, min_periods=15).max()
        denominator = wma(cs_rank(tsmax_val), decay_w2)
        self.factor_data = (closes - vwaps) / denominator

class alpha191_132(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_132', 'data_needed': ['amount'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        amounts = self.data['amount']
        self.factor_data = ts_mean(amounts, 20)

class alpha191_134(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_134', 'data_needed': ['close', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        volumes = self.data['volume']
        self.factor_data = (closes - ts_delay(closes, 12)) / ts_delay(closes, 12) * volumes

class alpha191_135(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_135', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        x = closes / ts_delay(closes, 20)
        self.factor_data = ts_mean(ts_delay(x, 1), 20)

class alpha191_137(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_137', 'data_needed': ['close', 'open', 'high', 'low'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        opens = self.data['open']
        highs = self.data['high']
        lows = self.data['low']
        prev_close = ts_delay(closes, 1)
        prev_open = ts_delay(opens, 1)
        numerator = (closes - prev_close) + (closes - opens) / 2.0 + (prev_close - prev_open)
        h_pc = np.abs(highs - prev_close)
        l_pc = np.abs(lows - prev_close)
        h_l = np.abs(highs - lows)
        pc_po = np.abs(prev_close - prev_open)
        cond1 = (h_pc > l_pc) & (h_pc > h_l)
        val1 = h_pc + l_pc / 2.0 + pc_po / 4.0
        cond2 = (l_pc > h_pc) & (l_pc > h_l)
        val2 = l_pc + h_pc / 2.0 + pc_po / 4.0
        val3 = h_l + pc_po / 4.0
        denominator = np.where(cond1, val1, np.where(cond2, val2, val3))
        multiplier = np.maximum(h_pc, l_pc)
        self.factor_data = 16 * numerator / denominator * multiplier

class alpha191_140(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_140', 'data_needed': ['open', 'low', 'high', 'close', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        opens = self.data['open']
        lows = self.data['low']
        highs = self.data['high']
        closes = self.data['close']
        volumes = self.data['volume']
        x = (cs_rank(opens) + cs_rank(lows)) - (cs_rank(highs) + cs_rank(closes))
        decay_w8 = np.arange(1, 9)
        part_a = cs_rank(wma(x, decay_w8))
        corr_val = ts_corr1(ts_rank(closes, 8), ts_rank(ts_mean(volumes, 60), 20), 8)
        decay_w7 = np.arange(1, 8)
        part_b = ts_rank(wma(corr_val, decay_w7), 3)
        self.factor_data = np.minimum(part_a, part_b)

class alpha191_142(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_142', 'data_needed': ['close', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        volumes = self.data['volume']
        part_a = cs_rank(ts_rank(closes, 10))
        part_b = cs_rank(ts_delta(closes, 1))
        part_c = cs_rank(ts_rank(volumes / ts_mean(volumes, 20), 5))
        self.factor_data = -1 * part_a * part_b * part_c

class alpha191_150(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_150', 'data_needed': ['close', 'high', 'low', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        highs = self.data['high']
        lows = self.data['low']
        volumes = self.data['volume']
        self.factor_data = (closes + highs + lows) / 3 * volumes

class alpha191_156(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_156', 'data_needed': ['amount', 'volume', 'open', 'low'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        amounts = self.data['amount']
        volumes = self.data['volume']
        opens = self.data['open']
        lows = self.data['low']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        decay_w3 = np.arange(1, 4)
        part_a = cs_rank(wma(ts_delta(vwaps, 5), decay_w3))
        y = (opens * 0.15) + (lows * 0.85)
        roc = ts_delta(y, 2) / y
        part_b = cs_rank(wma(-1 * roc, decay_w3))
        self.factor_data = -1 * np.maximum(part_a, part_b)

class alpha191_157(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_157', 'data_needed': ['close', 'return'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        returns = self.data['return']
        inner_val = cs_rank(cs_rank(-1 * cs_rank(ts_delta(closes, 5))))
        tsmin_val = inner_val.rolling(2, min_periods=1).min()
        log_val = np.log(tsmin_val)
        rank_log_val = cs_rank(cs_rank(log_val))
        prod_val = rank_log_val.rolling(5, min_periods=1).apply(np.prod, raw=True)
        part_a = np.minimum(prod_val, 5)
        part_b = ts_rank(ts_delay(-1 * returns, 6), 5)
        self.factor_data = part_a + part_b

class alpha191_158(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_158', 'data_needed': ['high', 'low', 'close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        highs = self.data['high']
        lows = self.data['low']
        closes = self.data['close']
        self.factor_data = (highs - lows) / closes

class alpha191_163(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_163', 'data_needed': ['return', 'volume', 'amount', 'high', 'close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        returns = self.data['return']
        volumes = self.data['volume']
        amounts = self.data['amount']
        highs = self.data['high']
        closes = self.data['close']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        inner_expr = -1 * returns * ts_mean(volumes, 20) * vwaps * (highs - closes)
        self.factor_data = cs_rank(inner_expr)

class alpha191_176(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_176', 'data_needed': ['close', 'low', 'high', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        lows = self.data['low']
        highs = self.data['high']
        volumes = self.data['volume']
        tsmin_low_12 = lows.rolling(12, min_periods=5).min()
        tsmax_high_12 = highs.rolling(12, min_periods=5).max()
        stoch_k = (closes - tsmin_low_12) / (tsmax_high_12 - tsmin_low_12)
        self.factor_data = -ts_corr1(cs_rank(stoch_k), cs_rank(volumes), 6)

class alpha191_179(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_179', 'data_needed': ['amount', 'volume', 'low'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        amounts = self.data['amount']
        volumes = self.data['volume']
        lows = self.data['low']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        part_a = cs_rank(ts_corr1(vwaps, volumes, 4))
        rank_low = cs_rank(lows)
        rank_mean_vol = cs_rank(ts_mean(volumes, 50))
        part_b = cs_rank(ts_corr1(rank_low, rank_mean_vol, 12))
        self.factor_data = part_a * part_b

class alpha191_184(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_184', 'data_needed': ['open', 'close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        opens = self.data['open']
        closes = self.data['close']
        open_close_diff = opens - closes
        part_a = cs_rank(ts_corr1(ts_delay(open_close_diff, 1), closes, 200))
        part_b = cs_rank(open_close_diff)
        self.factor_data = part_a + part_b

class alpha191_188(factor):
    def set_factor_property(self):
        factor_property = {'factor_name': 'alpha191_188', 'data_needed': ['high', 'low'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        highs = self.data['high']
        lows = self.data['low']
        x = highs - lows
        y = ts_mean(x, 11)
        self.factor_data = (x - y) / y * 100
