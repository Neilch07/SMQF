from hchen.SMQF.quant_lib.factor import factor
import numpy as np
import pandas as pd
from hchen.SMQF.quant_lib.analysis import *

class vol_std20(factor):
    def set_factor_property(self):
        """
        因子：vol_std20
        说明：计算每只标的过去 20 个交易日的成交量标准差（rolling std）。
        数据需求：['volume']，其中 self.data['volume'] 为 DataFrame（index=日期，columns=标的）。
        """
        factor_property = {
            'factor_name': 'vol_std20',
            'data_needed': ['volume'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        vol = self.data.get('volume')
        if vol is None:
            raise ValueError("vol_std20: required data 'volume' not found in self.data")
        # 计算 20 日标准差；如需严格满窗，可将 min_periods=20
        self.factor_data = - vol.rolling(window=20, min_periods=1).std()


class ret_sum60_rinv(factor):
    def set_factor_property(self):
        """
        ret_sum60_rinv
        说明：1 - cs_rank(ts_sum(return, 60))，即过去60日收益之和的横截面分位数的反向值。
        数据需求：['return']，self.data['return'] 为 DataFrame（index=日期，columns=标的）。
        """
        factor_property = {
            'factor_name': 'ret_sum60_rinv',
            'data_needed': ['return'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        ret = self.data.get('return')
        if ret is None:
            raise ValueError("ret_sum60_rinv: required data 'return' not found in self.data")
        self.factor_data = 1.0 - cs_rank(ts_sum(ret, 60))


class vol_priceSum_rinv(factor):
    def set_factor_property(self):
        factor_property = {
            'factor_name': 'vol_priceSum_rinv',
            'data_needed': ['volume', 'close', 'high', 'low', 'pct_change'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        # 需要的字段
        vol = self.data.get('volume')
        close = self.data.get('close')
        high = self.data.get('high')
        low = self.data.get('low')
        pct = self.data.get('pct_change')
        for name, df in [('volume', vol), ('close', close), ('high', high), ('low', low), ('pct_change', pct)]:
            if df is None:
                raise ValueError(f"vol_priceSum_rinv: required data '{name}' not found in self.data")

        volume_rank = cs_rank(vol, pct=False)
        price_sum = close + high - low
        price_sum_rank = cs_rank(price_sum, pct=False)
        returns_rank = cs_rank(pct, pct=False)

        self.factor_data = volume_rank * price_sum_rank * (1 - returns_rank)

class ret_20(factor):
    def set_factor_property(self):
        factor_property = {
            'factor_name': 'ret_20',
            'data_needed': ['close'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        self.factor_data = - (self.data['close'] / ts_delay(self.data['close'], 20))
    
class std_20(factor):
    def set_factor_property(self):
        factor_property = {
            'factor_name': 'std_20',
            'data_needed': ['close'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        close = self.data.get('close')
        self.factor_data = - close.rolling(20).std()


# Helper functions for GP factors
def sign(df):
    return np.sign(df)

def add(df1, df2):
    return df1 + df2

def ts_min_5(df):
    return ts_min(df, 5)

def abs(df):
    return np.abs(df)


# GP Factor 1: ts_min_5(sign(add(sign(abs(sign(low))), cs_rank(abs(low)))))
class gp_factor_1(factor):
    def set_factor_property(self):
        factor_property = {
            'factor_name': 'gp_factor_1',
            'data_needed': ['low'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        low = self.data.get('low')
        if low is None:
            raise ValueError("gp_factor_1: required data 'low' not found in self.data")
        # ts_min_5(sign(add(sign(abs(sign(low))), cs_rank(abs(low)))))
        self.factor_data = ts_min_5(sign(add(sign(abs(sign(low))), cs_rank(abs(low)))))


# GP Factor 2: ts_min_5(sign(add(sign(sign(sign(low))), cs_rank(sign(low)))))
class gp_factor_2(factor):
    def set_factor_property(self):
        factor_property = {
            'factor_name': 'gp_factor_2',
            'data_needed': ['low'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        low = self.data.get('low')
        if low is None:
            raise ValueError("gp_factor_2: required data 'low' not found in self.data")
        # ts_min_5(sign(add(sign(sign(sign(low))), cs_rank(sign(low)))))
        self.factor_data = ts_min_5(sign(add(sign(sign(sign(low))), cs_rank(sign(low)))))


# GP Factor 3: ts_min_5(sign(add(sign(abs(sign(open))), cs_rank(sign(low)))))
class gp_factor_3(factor):
    def set_factor_property(self):
        factor_property = {
            'factor_name': 'gp_factor_3',
            'data_needed': ['open', 'low'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        open_price = self.data.get('open')
        low = self.data.get('low')
        if open_price is None or low is None:
            raise ValueError("gp_factor_3: required data 'open' or 'low' not found in self.data")
        # ts_min_5(sign(add(sign(abs(sign(open))), cs_rank(sign(low)))))
        self.factor_data = ts_min_5(sign(add(sign(abs(sign(open_price))), cs_rank(sign(low)))))


# GP Factor 4: ts_min_5(sign(add(sign(cs_rank(sign(low))), cs_rank(sign(low)))))
class gp_factor_4(factor):
    def set_factor_property(self):
        factor_property = {
            'factor_name': 'gp_factor_4',
            'data_needed': ['low'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        low = self.data.get('low')
        if low is None:
            raise ValueError("gp_factor_4: required data 'low' not found in self.data")
        # ts_min_5(sign(add(sign(cs_rank(sign(low))), cs_rank(sign(low)))))
        self.factor_data = ts_min_5(sign(add(sign(cs_rank(sign(low))), cs_rank(sign(low)))))


# GP Factor 5: ts_min_5(sign(add(sign(abs(sign(low))), cs_rank(sign(high)))))
class gp_factor_5(factor):
    def set_factor_property(self):
        factor_property = {
            'factor_name': 'gp_factor_5',
            'data_needed': ['low', 'high'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        low = self.data.get('low')
        high = self.data.get('high')
        if low is None or high is None:
            raise ValueError("gp_factor_5: required data 'low' or 'high' not found in self.data")
        # ts_min_5(sign(add(sign(abs(sign(low))), cs_rank(sign(high)))))
        self.factor_data = ts_min_5(sign(add(sign(abs(sign(low))), cs_rank(sign(high)))))


# GP Factor 6: ts_min_5(sign(add(sign(abs(sign(low))), sign(abs(sign(low))))))
class gp_factor_6(factor):
    def set_factor_property(self):
        factor_property = {
            'factor_name': 'gp_factor_6',
            'data_needed': ['low'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        low = self.data.get('low')
        if low is None:
            raise ValueError("gp_factor_6: required data 'low' not found in self.data")
        # ts_min_5(sign(add(sign(abs(sign(low))), sign(abs(sign(low))))))
        self.factor_data = ts_min_5(sign(add(sign(abs(sign(low))), sign(abs(sign(low))))))


# GP Factor 7: ts_min_5(sign(add(sign(abs(sign(low))), cs_rank(sign(sign(low))))))
class gp_factor_7(factor):
    def set_factor_property(self):
        factor_property = {
            'factor_name': 'gp_factor_7',
            'data_needed': ['low'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        low = self.data.get('low')
        if low is None:
            raise ValueError("gp_factor_7: required data 'low' not found in self.data")
        # ts_min_5(sign(add(sign(abs(sign(low))), cs_rank(sign(sign(low))))))
        self.factor_data = ts_min_5(sign(add(sign(abs(sign(low))), cs_rank(sign(sign(low))))))


# GP Factor 8: ts_min_5(sign(add(sign(sign(low)), cs_rank(sign(sign(volume))))))
class gp_factor_8(factor):
    def set_factor_property(self):
        factor_property = {
            'factor_name': 'gp_factor_8',
            'data_needed': ['low', 'volume'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        low = self.data.get('low')
        volume = self.data.get('volume')
        if low is None or volume is None:
            raise ValueError("gp_factor_8: required data 'low' or 'volume' not found in self.data")
        # ts_min_5(sign(add(sign(sign(low)), cs_rank(sign(sign(volume))))))
        self.factor_data = ts_min_5(sign(add(sign(sign(low)), cs_rank(sign(sign(volume))))))


# GP Factor 9: ts_min_5(sign(add(sign(abs(sign(low))), sign(sign(sign(low))))))
class gp_factor_9(factor):
    def set_factor_property(self):
        factor_property = {
            'factor_name': 'gp_factor_9',
            'data_needed': ['low'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        low = self.data.get('low')
        if low is None:
            raise ValueError("gp_factor_9: required data 'low' not found in self.data")
        # ts_min_5(sign(add(sign(abs(sign(low))), sign(sign(sign(low))))))
        self.factor_data = ts_min_5(sign(add(sign(abs(sign(low))), sign(sign(sign(low))))))


# GP Factor 10: ts_min_5(sign(add(sign(abs(sign(low))), cs_rank(sign(low)))))
# Same as GP Factor 1
class gp_factor_10(factor):
    def set_factor_property(self):
        factor_property = {
            'factor_name': 'gp_factor_10',
            'data_needed': ['low'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        low = self.data.get('low')
        if low is None:
            raise ValueError("gp_factor_10: required data 'low' not found in self.data")
        # ts_min_5(sign(add(sign(abs(sign(low))), cs_rank(sign(low)))))
        self.factor_data = ts_min_5(sign(add(sign(abs(sign(low))), cs_rank(sign(low)))))
