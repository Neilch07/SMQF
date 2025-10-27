import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'quant_lib')))

from factor import factor
from factors import multi_factor,factor_params,factor_context
from analysis import *
from alpha101 import *
import copy
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

class my_factor_params(factor_params):
    def set_factor_property(self):
        factor_property={'factor_name': 'test_parmas','data_needed': []}
        return factor_property
    def calc_factor(self):
        n1=self.factor_params['n1']
        n2=self.factor_params['n2']
        n3=self.factor_params['n3']
        self.factor_data=self.returns.rolling(n1).sum()+self.returns.rolling(n3).sum()-2*self.returns.rolling(n2).sum()
            
class my_factor_context(factor_context):
    def set_factor_property(self):
        factor_property={'factor_name': 'test_context'}
        return factor_property
    def calc_factor(self):
        self.factor_data=self.returns.rolling(10).sum()
         



#params={}
all_factor_params={'n1':[5,7],'n2':[10,15],'n3':[20,25]}
f=my_factor_params(all_factor_params=all_factor_params)
f.run()

#f=my_factor_context()
#f.run()
    
