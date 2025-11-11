from .factor import factor
import numpy as np
import pandas as pd
from .analysis import *


class alpha101_003(factor):
    def set_factor_property(self):
        """
        Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
        """
        factor_property = {'factor_name': 'alpha101_003', 'data_needed': ['open', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        self.factor_data = -1 * ts_corr1(cs_rank(self.data['open']), cs_rank(self.data['volume']), 10)

class alpha101_004(factor):
    def set_factor_property(self):
        """
        Alpha#4: (-1 * Ts_Rank(rank(low), 9))
        """
        factor_property = {'factor_name': 'alpha101_004', 'data_needed': ['low'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        self.factor_data = -1 * ts_rank(cs_rank(self.data['low']), 9)

class alpha101_005(factor):
    def set_factor_property(self):
        """
        Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
        """
        factor_property = {'factor_name': 'alpha101_005', 'data_needed': ['open', 'close', 'amount', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        vwap = self.data['amount'] / self.data['volume'].where(self.data['volume'] > 0, np.nan)
        self.factor_data = cs_rank(self.data['open'] - ts_mean(vwap, 10)) * (-1 * np.abs(cs_rank(self.data['close'] - vwap)))

class alpha101_008(factor):
    def set_factor_property(self):
        """
        Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
        """
        factor_property = {'factor_name': 'alpha101_008', 'data_needed': ['open', 'return'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        x = ts_sum(self.data['open'], 5) * ts_sum(self.data['return'], 5)
        self.factor_data = -1 * cs_rank(ts_delta(x, 10))

class alpha101_010(factor):
    def set_factor_property(self):
        """
        Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
        """
        factor_property = {'factor_name': 'alpha101_010', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property

    def calc_factor(self):
        closes = self.data['close']
        delta1 = ts_delta(closes, 1)
        rank1 = cs_rank(delta1)
        rank2 = cs_rank(rank1)
        tsmax5 = rank2.rolling(5, min_periods=3).max()
        self.factor_data = cs_rank(tsmax5)

class alpha101_011(factor):
    def set_factor_property(self):
        """
        Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
        """
        factor_property = {'factor_name': 'alpha101_011', 'data_needed': ['amount', 'volume', 'close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        vwap = self.data['amount'] / self.data['volume'].where(self.data['volume'] > 0, np.nan)
        x = vwap - self.data['close']
        part_a = cs_rank(x.rolling(3, min_periods=2).max())
        part_b = cs_rank(x.rolling(3, min_periods=2).min())
        part_c = cs_rank(ts_delta(self.data['volume'], 3))
        self.factor_data = (part_a + part_b) * part_c

class alpha101_013(factor):
    def set_factor_property(self):
        """
        Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))
        """
        factor_property = {'factor_name': 'alpha101_013', 'data_needed': ['close', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        rank_close = cs_rank(self.data['close'])
        rank_volume = cs_rank(self.data['volume'])
        self.factor_data = -1 * cs_rank(rank_close.rolling(5, min_periods=3).cov(rank_volume))

class alpha101_015(factor):
    def set_factor_property(self):
        """
        Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
        """
        factor_property = {'factor_name': 'alpha101_015', 'data_needed': ['high', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        corr = ts_corr1(cs_rank(self.data['high']), cs_rank(self.data['volume']), 3)
        self.factor_data = -1 * ts_sum(cs_rank(corr), 3)

class alpha101_016(factor):
    def set_factor_property(self):
        """
        Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))
        """
        factor_property = {'factor_name': 'alpha101_016', 'data_needed': ['high', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        rank_high = cs_rank(self.data['high'])
        rank_volume = cs_rank(self.data['volume'])
        self.factor_data = -1 * cs_rank(rank_high.rolling(5, min_periods=3).cov(rank_volume))

class alpha101_017(factor):
    def set_factor_property(self):
        """
        Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
        """
        factor_property = {'factor_name': 'alpha101_017', 'data_needed': ['close', 'volume', 'amount'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        adv20 = ts_mean(self.data['amount'], 20)
        vol_adv20 = self.data['volume'] / adv20.where(adv20 > 0, np.nan)
        part_a = -1 * cs_rank(ts_rank(self.data['close'], 10))
        part_b = cs_rank(ts_delta(ts_delta(self.data['close'], 1), 1))
        part_c = cs_rank(ts_rank(vol_adv20, 5))
        self.factor_data = part_a * part_b * part_c

class alpha101_018(factor):
    def set_factor_property(self):
        """
        Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
        """
        factor_property = {'factor_name': 'alpha101_018', 'data_needed': ['close', 'open'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        part_a = ts_std(np.abs(self.data['close'] - self.data['open']), 5)
        part_b = self.data['close'] - self.data['open']
        part_c = ts_corr1(self.data['close'], self.data['open'], 10)
        self.factor_data = -1 * cs_rank(part_a + part_b + part_c)

class alpha101_019(factor):
    def set_factor_property(self):
        """
        Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
        """
        factor_property = {'factor_name': 'alpha101_019', 'data_needed': ['close', 'return'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        # The term (close - delay(close, 7)) is equivalent to delta(close, 7). [cite: 304, 308]
        # So, ((close - delay(close, 7)) + delta(close, 7)) simplifies to 2 * delta(close, 7).
        # The sign of 2 * delta(close, 7) is the same as sign of delta(close, 7).
        part_a = -1 * np.sign(ts_delta(self.data['close'], 7))
        part_b = 1 + cs_rank(1 + ts_sum(self.data['return'], 250))
        self.factor_data = part_a * part_b # [cite: 203]

class alpha101_024(factor):
    def set_factor_property(self):
        """
        Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
        """
        factor_property = {'factor_name': 'alpha101_024', 'data_needed': ['close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        closes = self.data['close']
        mean100 = ts_mean(closes, 100)

        condition_val = ts_delta(mean100, 100) / ts_delay(closes, 100)
        condition = condition_val <= 0.05

        true_val = -1 * (closes - closes.rolling(100, min_periods=50).min())
        false_val = -1 * ts_delta(closes, 3)

        # np.where 的返回结果是一个 NumPy 数组
        factor_array = np.where(condition, true_val, false_val)

        # 关键修正：将 NumPy 数组转换回 Pandas DataFrame，以确保后续函数能正确处理
        self.factor_data = pd.DataFrame(factor_array, index=closes.index, columns=closes.columns)

class alpha101_025(factor):
    def set_factor_property(self):
        """
        Alpha#25: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
        """
        factor_property = {'factor_name': 'alpha101_025', 'data_needed': ['return', 'amount', 'volume', 'high', 'close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        adv20 = ts_mean(self.data['amount'], 20)
        vwap = self.data['amount'] / self.data['volume'].where(self.data['volume'] > 0, np.nan)
        inner_expr = -1 * self.data['return'] * adv20 * vwap * (self.data['high'] - self.data['close'])
        self.factor_data = cs_rank(inner_expr)

class alpha101_029(factor):
    def set_factor_property(self):
        """
        Alpha#29: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
        """
        factor_property = {'factor_name': 'alpha101_029', 'data_needed': ['close', 'return'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        # 'scale(x)' rescales x such that sum(abs(x)) = 1 [cite: 307]
        def scale(df):
            return df.div(df.abs().sum(axis=1), axis=0)

        # Breakdown of the first part of the formula [cite: 215]
        # Note: delta((close-1), 5) is just delta(close, 5)
        # sum(..., 1) and product(..., 1) are just the identity of the inner value
        p1 = ts_delta(self.data['close'], 5)
        p2 = cs_rank(p1)
        p3 = -1 * p2
        p4 = cs_rank(p3)
        p5 = cs_rank(p4)
        p6 = p5.rolling(2, min_periods=1).min()
        # p7 = sum(p6, 1) is just p6
        p8 = np.log(p6)
        p9 = scale(p8)
        p10 = cs_rank(p9)
        p11 = cs_rank(p10)
        # p12 = product(p11, 1) is just p11
        part_a = p11.rolling(5, min_periods=3).min()
        part_b = ts_rank(ts_delay(-1 * self.data['return'], 6), 5)

        self.factor_data = part_a + part_b

class alpha101_032(factor):
    def set_factor_property(self):
        """
        Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))
        """
        factor_property = {'factor_name': 'alpha101_032', 'data_needed': ['close', 'amount', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        def scale(df):
            return df.div(df.abs().sum(axis=1), axis=0)
        vwap = self.data['amount'] / self.data['volume'].where(self.data['volume'] > 0, np.nan)
        part_a = scale(ts_mean(self.data['close'], 7) - self.data['close'])
        part_b = 20 * scale(ts_corr1(vwap, ts_delay(self.data['close'], 5), 230))
        self.factor_data = part_a + part_b

class alpha101_033(factor):
    def set_factor_property(self):
        """
        Alpha#33: rank((-1 * ((1 - (open / close))^1)))
        """
        factor_property = {'factor_name': 'alpha101_033', 'data_needed': ['open', 'close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        self.factor_data = cs_rank(-1 * (1 - (self.data['open'] / self.data['close'])))

class alpha101_037(factor):
    def set_factor_property(self):
        """
        Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
        """
        factor_property = {'factor_name': 'alpha101_037', 'data_needed': ['open', 'close'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        open_close_diff = self.data['open'] - self.data['close']
        part_a = cs_rank(ts_corr1(ts_delay(open_close_diff, 1), self.data['close'], 200))
        part_b = cs_rank(open_close_diff)
        self.factor_data = part_a + part_b

class alpha101_038(factor):
    def set_factor_property(self):
        """
        Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
        """
        factor_property = {'factor_name': 'alpha101_038', 'data_needed': ['close', 'open'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        self.factor_data = -1 * cs_rank(ts_rank(self.data['close'], 10)) * cs_rank(self.data['close'] / self.data['open'])

class alpha101_040(factor):
    def set_factor_property(self):
        """
        Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
        """
        factor_property = {'factor_name': 'alpha101_040', 'data_needed': ['high', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        self.factor_data = -1 * cs_rank(ts_std(self.data['high'], 10)) * ts_corr1(self.data['high'], self.data['volume'], 10)

class alpha101_044(factor):
    def set_factor_property(self):
        """
        Alpha#44: (-1 * correlation(high, rank(volume), 5))
        """
        factor_property = {'factor_name': 'alpha101_044', 'data_needed': ['high', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        self.factor_data = -1 * ts_corr1(self.data['high'], cs_rank(self.data['volume']), 5)

class alpha101_050(factor):
    def set_factor_property(self):
        """
        Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
        """
        factor_property = {'factor_name': 'alpha101_050', 'data_needed': ['volume', 'amount'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        vwap = self.data['amount'] / self.data['volume'].where(self.data['volume'] > 0, np.nan)
        corr = ts_corr1(cs_rank(self.data['volume']), cs_rank(vwap), 5)
        self.factor_data = -1 * cs_rank(corr).rolling(5, min_periods=3).max()

class alpha101_055(factor):
    def set_factor_property(self):
        """
        Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
        """
        factor_property = {'factor_name': 'alpha101_055', 'data_needed': ['close', 'low', 'high', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        tsmin_low = self.data['low'].rolling(12, min_periods=6).min()
        tsmax_high = self.data['high'].rolling(12, min_periods=6).max()
        stoch = (self.data['close'] - tsmin_low) / (tsmax_high - tsmin_low)
        self.factor_data = -1 * ts_corr1(cs_rank(stoch), cs_rank(self.data['volume']), 6)

class alpha101_057(factor):
    def set_factor_property(self):
        """
        Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
        """
        factor_property = {'factor_name': 'alpha101_057', 'data_needed': ['close', 'amount', 'volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        vwap = self.data['amount'] / self.data['volume'].where(self.data['volume'] > 0, np.nan)
        decay_weights = np.arange(1, 2 + 1)
        numerator = self.data['close'] - vwap
        denominator = wma(cs_rank(ts_argmax(self.data['close'], 30)), decay_weights)
        self.factor_data = -1 * (numerator / denominator)

class alpha101_066(factor):
    def set_factor_property(self):
        """
        Alpha#66: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1-0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
        """
        factor_property = {'factor_name': 'alpha101_066', 'data_needed': ['amount', 'volume', 'low', 'open', 'high'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        d1, d2, d3, d4 = 3, 7, 11, 6
        vwap = self.data['amount'] / self.data['volume'].where(self.data['volume'] > 0, np.nan)

        # The expression (low * 0.96633) + (low * (1-0.96633)) simplifies to 'low'
        # Formula breakdown [cite: 255]
        part_a = cs_rank(wma(ts_delta(vwap, d1), np.arange(1, d2 + 1)))

        numerator = self.data['low'] - vwap
        denominator = self.data['open'] - (self.data['high'] + self.data['low']) / 2
        inner_expr = numerator / denominator.where(denominator != 0, np.nan)

        part_b = ts_rank(wma(inner_expr, np.arange(1, d3 + 1)), d4)

        self.factor_data = -1 * (part_a + part_b)

class alpha101_083(factor):
    def set_factor_property(self):
        """
        Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
        """
        factor_property = {'factor_name': 'alpha101_083', 'data_needed': ['high', 'low', 'close', 'volume', 'amount'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        vwap = self.data['amount'] / self.data['volume'].where(self.data['volume'] > 0, np.nan)
        x = (self.data['high'] - self.data['low']) / ts_mean(self.data['close'], 5)
        numerator = cs_rank(ts_delay(x, 2)) * cs_rank(cs_rank(self.data['volume']))
        denominator = x / (vwap - self.data['close'])

        self.factor_data = numerator / denominator.where(denominator != 0, np.nan)

class alpha101_088(factor):
    def set_factor_property(self):
        """
        Alpha#88: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
        """
        factor_property = {'factor_name': 'alpha101_088', 'data_needed': ['open', 'low', 'high', 'close', 'amount'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        d1, d2, d3, d4, d5, d6 = 8, 8, 20, 8, 6, 2
        adv60 = ts_mean(self.data['amount'], 60)
        x = (cs_rank(self.data['open']) + cs_rank(self.data['low'])) - (cs_rank(self.data['high']) + cs_rank(self.data['close']))
        part_a = cs_rank(wma(x, np.arange(1, d1 + 1)))

        corr = ts_corr1(ts_rank(self.data['close'], d2), ts_rank(adv60, d3), d4)
        part_b = ts_rank(wma(corr, np.arange(1, d5 + 1)), d6)

        self.factor_data = np.minimum(part_a, part_b)