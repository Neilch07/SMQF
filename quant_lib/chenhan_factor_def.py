from factor import factor
import numpy as np
import pandas as pd
from analysis import *

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
    