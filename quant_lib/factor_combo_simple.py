from typing import Dict, List, Optional, Union
from attrs import define, field
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from factor_combo_base import factor_combo_base
import time



@define
class factor_combo_simple(factor_combo_base):
    __params = {'testing_periods': 'rolling', # 0,1,'rolling', default 0 if 'weighting_method' is 'equal'
                'testing_period_length': 63, # int
                'training_period_length': 125, # 'auto', int, 'all'
                }  
    __combo_property = {'weighting_method': 'equal', # 'equal','equal_vol','max_ir','min_vol'
                        'min_weight': 'auto',
                        'max_weight': 'auto'
                        }        
    
    def __attrs_post_init__(self):
        self.__combo_property.update(self.set_combo_property())
        if self.combo_property is not None:
            self.__combo_property.update(self.combo_property)
        self.combo_property = self.__combo_property
        if self.combo_property['weighting_method']=='equal':
            self.__params['testing_periods']=0
        if self.params is not None:
            self.__params.update(self.params)
        self.params = self.__params
        super().__attrs_post_init__()

    def equal_weight_portfolio(self,training_data,testing_data):
        training_pos=None
        testing_pos=None
        nf = len(training_data)
        for f in training_data:
            lag_periods=self.factors_property[f]['lag_periods']
            if training_pos is None:
                training_pos=self.calc_rank_pos(lag_periods,training_data[f])/nf
            else:
                training_pos=training_pos+self.calc_rank_pos(lag_periods,training_data[f])/nf
        for f in testing_data:
            lag_periods=self.factors_property[f]['lag_periods']
            if testing_pos is None:
                testing_pos=self.calc_rank_pos(lag_periods,testing_data[f])/nf
            else:
                testing_pos=testing_pos+self.calc_rank_pos(lag_periods,testing_data[f])/nf
        return (training_pos,testing_pos)

    def equal_vol_portfolio(self,training_data,testing_data,training_returns):
        training_pos=None
        testing_pos=None
        training_positions={}
        testing_positions={}
        wts={}

        for f in training_data:
            lag_periods=self.factors_property[f]['lag_periods']
            training_positions[f]=self.calc_rank_pos(lag_periods,training_data[f])
            wts[f]=1.0/(1e-10+self.calc_factor_returns(training_positions[f],training_returns).std())
            testing_positions[f]=self.calc_rank_pos(lag_periods,testing_data[f])
        wts = self.clip_weights(wts)

        for f in training_data:
            if training_pos is None:
                training_pos=training_positions[f]*wts[f]
            else:
                training_pos=training_pos+training_positions[f]*wts[f]
            if testing_pos is None:
                testing_pos=testing_positions[f]*wts[f]
            else:
                testing_pos=testing_pos+testing_positions[f]*wts[f]

        return (training_pos,testing_pos)
    
    def max_ir_weight(self,f_returns):
        cov_matrix=f_returns.cov().values
        avg_rets = f_returns.mean().values

        wts={}
        for f in f_returns.columns:
            wts[f]=1
        return wts
    
    def min_vol_weight(self,f_returns):
        cov_matrix=f_returns.cov().values

        wts={}
        for f in f_returns.columns:
            wts[f]=1
        return wts

    def optimal_portfolio(self,training_data,testing_data,training_returns,opt_type):
        training_pos=None
        testing_pos=None
        f_names=list(training_data.keys())
        training_positions={}
        training_factor_returns=pd.DataFrame(columns=f_names)
        testing_positions={}

        for f in training_data:
            lag_periods=self.factors_property[f]['lag_periods']
            training_positions[f]=self.calc_rank_pos(lag_periods,training_data[f])
            training_factor_returns[f]=self.calc_factor_returns(training_positions[f],training_returns)
            testing_positions[f]=self.calc_rank_pos(lag_periods,testing_data[f])
        if opt_type=='max_ir':
            wts=self.max_ir_weight(training_factor_returns)
        elif opt_type=='min_vol':
            wts=self.min_vol_weight(training_factor_returns)
        else:
            raise ValueError("opt_type is unrecoganized")
        wts = self.clip_weights(wts)

        for f in training_data:
            if training_pos is None:
                training_pos=training_positions[f]*wts[f]
            else:
                training_pos=training_pos+training_positions[f]*wts[f]
            if testing_pos is None:
                testing_pos=testing_positions[f]*wts[f]
            else:
                testing_pos=testing_pos+testing_positions[f]*wts[f]

        return (training_pos,testing_pos)

    def calc_subperiod_combo_positions(self,i0,i1,i2):
        training_factors_data={}
        testing_factors_data={}     
        for tk in self.factors_data:
            training_factors_data[tk]=self.factors_data[tk].iloc[i0:i1,:]
            testing_factors_data[tk]=self.factors_data[tk].iloc[i1:i2,:]
        if self.params['weighting_method'] == 'equal':
            return self.equal_weight_portfolio(training_factors_data,testing_factors_data)
        elif self.params['weighting_method'] == 'equal_vol':
            training_returns=self.returns[i0:i1]
            return self.equal_vol_portfolio(training_factors_data,testing_factors_data,training_returns)
        elif self.params['weighting_method'] == 'max_ir':
            training_returns=self.returns[i0:i1]
            return self.optimal_portfolio(training_factors_data,testing_factors_data,training_returns,'max_ir')
        elif self.params['weighting_method'] == 'min_vol':
            training_returns=self.returns[i0:i1]
            return self.optimal_portfolio(training_factors_data,testing_factors_data,training_returns,'min_vol')
        else:
            raise ValueError("weighting_method not recoganized")


@define
class factor_combo_linear(factor_combo_base):
    __params = {'testing_periods': 'rolling', # 0,1,'rolling', default 0 if 'weighting_method' is 'equal'
                'testing_period_length': 63, # int
                'training_period_length': 125, # 'auto', int, 'all'
                }  
    __combo_property = {'weighting_method': 'equal', # 'equal','by_ir','max_icir'
                        'normalization_method': 'zscore', # 'zscore','minmax','tanh'
                        'scale_coeff': 1, # used when 'normalization_method' is 'tanh'
                        'min_weight': 'auto',
                        'max_weight': 'auto'
                        }        
    
    def __attrs_post_init__(self):
        self.__combo_property.update(self.set_combo_property())
        if self.combo_property is not None:
            self.__combo_property.update(self.combo_property)
        self.combo_property = self.__combo_property
        if self.combo_property['weighting_method']=='equal':
            self.__params['testing_periods']=0
        if self.params is not None:
            self.__params.update(self.params)
        self.params = self.__params
        super().__attrs_post_init__()

    def equal_weight_linear(self,training_data,testing_data):
        training_combo=None
        testing_combo=None
        nf = len(training_data)
        for f in training_data:
            if training_combo is None:
                training_combo = training_data[f]/nf
            else:
                training_combo = training_combo+training_data[f]/nf
        for f in testing_data:
            if testing_combo is None:
                testing_combo = testing_data[f]/nf
            else:
                testing_combo = testing_combo+testing_data[f]/nf
        lag_periods=self.combo_property['lag_periods']
        training_pos=self.calc_rank_pos(lag_periods,training_combo)
        testing_pos=self.calc_rank_pos(lag_periods,testing_combo)
        return (training_pos,testing_pos)
        
    def weight_by_ir(training_data,testing_data,training_returns):
        pass

    def max_icir_weight(training_data,testing_data,training_returns):
        pass
        
    def optimal_factor(self,training_data,testing_data,training_returns,opt_type):
        f_names=list(training_data.keys())
        training_positions={}
        training_factor_returns=pd.DataFrame(columns=f_names)
        testing_positions={}

        for f in training_data:
            lag_periods=self.factors_property[f]['lag_periods']
            training_positions[f]=self.calc_rank_pos(lag_periods,training_data[f])
            training_factor_returns[f]=self.calc_factor_returns(training_positions[f],training_returns)
            testing_positions[f]=self.calc_rank_pos(lag_periods,testing_data[f])
        if opt_type=='by_ir':
            wts=self.weight_by_ir(training_factor_returns)
        elif opt_type=='max_icir':
            wts=self.max_icir_weight(training_factor_returns)
        else:
            raise ValueError("opt_type is unrecoganized")
        wts = self.clip_weights(wts)

        training_combo=None
        testing_combo=None
        nf = len(training_data)
        for f in training_data:
            if training_combo is None:
                training_combo = training_data[f]*wts[f]
            else:
                training_combo = training_combo+training_data[f]*wts[f]
        for f in testing_data:
            if testing_combo is None:
                testing_combo = testing_data[f]*wts[f]
            else:
                testing_combo = testing_combo+testing_data[f]*wts[f]
        lag_periods=self.combo_property['lag_periods']
        training_pos=self.calc_rank_pos(lag_periods,training_combo)
        testing_pos=self.calc_rank_pos(lag_periods,testing_combo)

        return (training_pos,testing_pos)
    
    def calc_subperiod_combo_positions(self,i0,i1,i2):
        training_factors_data={}
        testing_factors_data={}     
        for tk in self.factors_data:
            training_factors_data[tk]=self.factors_data[tk].iloc[i0:i1,:]
            testing_factors_data[tk]=self.factors_data[tk].iloc[i1:i2,:]
        if self.params['weighting_method'] == 'equal':
            return self.equal_weight_linear(training_factors_data,testing_factors_data)
        elif self.params['weighting_method'] == 'by_ir':
            training_returns=self.returns[i0:i1]
            return self.optimal_factor(training_factors_data,testing_factors_data,training_returns,'by_ir')
        elif self.params['weighting_method'] == 'max_icir':
            training_returns=self.returns[i0:i1]
            return self.optimal_factor(training_factors_data,testing_factors_data,training_returns,'max_icir')
        else:
            raise ValueError("weighting_method not recoganized")
        


        
if __name__ == '__main__':



    params={'testing_periods': 'rolling', 'testing_period_length': 120}
    #combo_property={'factors_needed':['alpha101_04','alpha101_05'],'benchmark':'zz500','ir_details':True,'weighting_method':'equal_vol'}
    #combo_property={'factors_needed':['alpha101_04','alpha101_05'],'weighting_method':'max_ir'}
    #f = factor_combo_simple(params=params,combo_property=combo_property)

    combo_property={'factors_needed':['alpha101_04','alpha101_05'],'normalization_method': 'zscore'}
    f = factor_combo_linear(combo_property=combo_property)
    f.run()
