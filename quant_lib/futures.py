from typing import Dict, List, Optional, Union
from attrs import define, field
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import copy
import itertools
import time
from matplotlib import pyplot as plt
from factor import factor

@define
class future_factor(factor):
    __future_params = {'trading_cost': 0.0001, # trading costs in terms of two way costs in returns
                       'n_group': 5, # if 0, no group return will be calculated
                       'data_dir': 'D:\\OneDrive - SAIF\\学校\\量化\\data\\cn\\futures\\futures\\',
                       'factor_dir': 'D:\\OneDrive - SAIF\\学校\\量化\\data\\cn\\futures\\factor\\'
                       }  
    
    def __attrs_post_init__(self):
        if self.params is not None:
            self.__future_params.update(self.params)
        self.params = self.__future_params
        super().__attrs_post_init__()
    

   
if __name__ == '__main__':

    from analysis import *

    class my_factor1(future_factor):
        def set_factor_property(self):
            factor_property={'factor_name': 'alpha101-03',
                             'data_needed': ['open','volume'],
                             'lag_periods': 2, #1,
                             'ic_return_horizon': 1, # 1,20,65
                             'universe': 'top_4000', # None,'zz500',
                             'benchmark': 'zz500', #None,
                             'smooth_periods': 0, #10,
                             'sector_neutral': False #True
                            }
            return factor_property

        def calc_factor(self):
            opens=self.data['open']
            volumes=self.data['volume']
            self.factor_data=-ts_corr1(cs_rank(opens,pct=True),cs_rank(volumes,pct=True),10)

    class my_factor2(future_factor):
        def set_factor_property(self):
            factor_property={'factor_name': 'strange',
                             'data_needed': ['returns'],
                            }
            return factor_property

        def calc_factor(self):
            self.factor_data=self.returns.rolling(10).sum()
            
         
    #params={'ic_return_horizons': [5,20,60], 'ic_delay_periods': [1,2,5]}
    #params={'start_os_date': '2022-01-01','ir_details': True}
    params={}
    f=my_factor2(params=params)
    f.run()