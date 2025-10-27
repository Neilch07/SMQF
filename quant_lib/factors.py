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
class multi_factor(factor):
    all_factors: Dict[str,factor] = field(default={}, init=False)
    all_results: Dict[str,Union[Dict, str]] = field(default={}, init=False)
    

    def run(self):
        self.calc_all_factors()
        self.calc_all_performance()
        if self.params['display'] == True:
            self.display()

    def get_all_results(self):
        return self.all_results
    
    def calc_all_performance(self):
        for ft in self.all_factors:
            print('Start calculating %s ...' % ft)
            t1=time.perf_counter()
            self.all_factors[ft].run(turnoff_display=True)
            self.all_results[ft]=self.all_factors[ft].get_performance()
            t2=time.perf_counter()
            print('%-20s%20s%10.4f%4s' % (ft,"calculating time:",t2-t1,"sec"))

    def set_factor(self,factor_class):
        self.all_factors[factor_class.factor_property['factor_name']]=copy.deepcopy(factor_class)

    def display(self):
        print('%s' % '='*80)
        print('%-20s%10s%10s%10s%10s%10s' % ('Name','Anal_rets','Anal_IR','MDrawdown','Net_rets','Net_IR'))
        for ft in self.all_results:
            results = self.all_results[ft]
            print('%-20s%10.4f%10.4f%10.4f%10.4f%10.4f' % (ft,results['Annualized Return'],results['Annualized IR'],results['Maximum Drawdown'],
                                                           results['Annualized Net Return'],results['Annualized Net IR']))
            
        print('%s' % '='*80)

    @abstractmethod
    def calc_all_factors(self):
        raise NotImplementedError
    
@define
class factor_params(factor):
    all_factor_params: Dict[str,list] = field(default={}, kw_only=True)
    all_factor_data: Dict[str,pd.DataFrame] = field(default={}, init=False)
    all_results: Dict[str,Dict] = field(default={}, init=False)
    param_names: list[str] = field(default=[], init=False)
    all_param_values: list[list] = field(default=[], init=False)

    def run(self):
        if self.data == {}:
            self.load_data() 
        self.calc_all_factors()
        self.calc_all_performance()
        if self.params['display'] == True:
            self.display()
        return [self.all_factor_data,self.all_results]

    def calc_all_performance(self):
        for pm in self.all_param_values:
            print('%-30s%20s' % ('Start calculating param set ...',pm))
            t1=time.perf_counter()
            self.factor_data = self.all_factor_data[pm]
            self.calc_factor_performance()
            self.all_results[pm]=copy.deepcopy(self.results)
            t2=time.perf_counter()
            print('%-30s%20s%2s%10.4f%4s' % ("calculating time for",pm,": ",t2-t1,"sec"))

    def calc_all_factors(self):
        lists = list(self.all_factor_params.values())
        self.all_param_values = list(itertools.product(*lists))
        self.param_names=list(self.all_factor_params.keys())
        for pm in self.all_param_values:
            t31=time.perf_counter()
            self.factor_params = dict(zip(self.param_names, pm))
            self.calc_factor()
            self.all_factor_data[pm]=copy.deepcopy(self.factor_data)
            t32=time.perf_counter()
            print('%-20s%20s%5s%10.4f%4s' % ("calculating time for",pm,":",t32-t31,"sec"))
            
    def display(self):
        print('%s' % '='*80)
        print('%-20s%10s%10s%10s%10s%10s' % ('parameters','Anal_rets','Anal_IR','MDrawdown','Net_rets','Net_IR'))
        for pm in self.all_param_values:
            results = self.all_results[pm]
            print('%-20s%10.4f%10.4f%10.4f%10.4f%10.4f' % (pm,results['Annualized Return'],results['Annualized IR'],results['Maximum Drawdown'],
                                                           results['Annualized Net Return'],results['Annualized Net IR']))
        print('%s' % '='*80)


@define

class factor_context(factor):
    __factor_context = {'market_value': True,
                        'momentum': False, 
                        'momentum_horizon': 20,
                        'volatility': False, 
                        'volatility_horizon': 60,
                        'trade_amount': False, 
                        'trade_amount_horizon': 20,
                        'value_growth': False, # use P/B
                        'bucket': [0.3,0.7] 
                        }
    factor_context: Dict[str,Union[int, float, str]] = field(default=None, kw_only=True)
    all_factor_data: Dict[str,Dict] = field(default={}, init=False)
    all_results: Dict[str,Dict] = field(default={}, init=False)

    def __attrs_post_init__(self):
        if self.factor_context is not None:
            self.__factor_context.update(self.factor_context)
        self.factor_context = self.__factor_context
        super().__attrs_post_init__()

    def run(self):
        if self.data == {}:
            if self.factor_context['market_value']==True:
                if 'mktcap' not in self.data_needed:
                    self.data_needed.append('mktcap')
            if self.factor_context['trade_amount']==True:
                if 'amount' not in self.data_needed:
                    self.data_needed.append('amount')
            if self.factor_context['value_growth']==True:
                pass
            self.load_data() 
        self.set_context_data()
        self.calc_factor()
        self.calc_all_performance()
        if self.params['display'] == True:
            self.display()
        return [self.all_factor_data,self.all_results]
    
    def set_context_data(self):
        if self.factor_context['market_value']==True:
            self.market_value=self.data['mktcap']
        if self.factor_context['momentum']==True:
            n=self.factor_context['momentum_horizon']
            self.momentum=self.returns.rolling(n,int(min_periods=n/2).sum())
        if self.factor_context['volatility']==True:
            n=self.factor_context['volatility_horizon']
            self.momentum=self.returns.rolling(n,int(min_periods=n/2).std())
        if self.factor_context['trade_amount']==True:
            n=self.factor_context['trade_amount_horizon']
            self.momentum=self.amount.rolling(n,int(min_periods=n/2).median())
        if self.factor_context['value_growth']==True:
            pass

    def factor_context_data(self,factor_data,context_value,bucket):
        ranking = context_value.rank(axis=1, pct=True)
        return factor_data.where(((ranking>bucket[0]) & (ranking<bucket[1])),np.nan,inplace=False)

    def calc_all_performance(self):
        bucket=self.factor_context['bucket']
        all_contexts = list(zip(bucket[:-1], bucket[1:])) 
        if bucket[0]>0.01:
            all_contexts = [(0,bucket[0])]+all_contexts 
        if bucket[-1]<0.99:
            all_contexts = all_contexts +[(bucket[-1],1)]

        factor_data = copy.deepcopy(self.factor_data)
        if self.factor_context['market_value']==True:
            print('Start calculating market_value context set ...')
            for b in all_contexts:
                t1=time.perf_counter()
                self.factor_data = self.factor_context_data(factor_data,self.market_value,b)
                self.calc_factor_performance()
                self.all_results['market_value_'+str(b)]=copy.deepcopy(self.results)
                t2=time.perf_counter()
                print('%-30s%20s%2s%10.4f%4s' % ("  calculating market_value context ",b,": ",t2-t1,"sec"))
        if self.factor_context['momentum']==True:
            print('Start calculating momentum context set ...')
            for b in all_contexts:
                t1=time.perf_counter()
                self.factor_data = self.factor_context_data(factor_data,self.data['momentum'],b)
                self.calc_factor_performance()
                self.all_results['momentum_'+str(b)]=copy.deepcopy(self.results)
                print('%-30s%20s%2s%10.4f%4s' % ("  calculating momentum context ",b,": ",t2-t1,"sec"))
        if self.factor_context['volatility']==True:
            print('Start calculating volatility context set ...')
            for b in all_contexts:
                t1=time.perf_counter()
                self.factor_data = self.factor_context_data(factor_data,self.data['volatility'],b)
                self.calc_factor_performance()
                self.all_results['mvolatility_'+str(b)]=copy.deepcopy(self.results)
                print('%-30s%20s%2s%10.4f%4s' % ("  calculating volatility context ",b,": ",t2-t1,"sec"))
        if self.factor_context['trade_amount']==True:
            print('Start calculating trade_amount context set ...')
            for b in all_contexts:
                t1=time.perf_counter()
                self.factor_data = self.factor_context_data(factor_data,self.data['trade_amount'],b)
                self.calc_factor_performance()
                self.all_results['trade_amount'+str(b)]=copy.deepcopy(self.results)
                print('%-30s%20s%2s%10.4f%4s' % ("  calculating trade_amount context ",b,": ",t2-t1,"sec"))
        if self.factor_context['value_growth']==True:
            print('Start calculating value_growth context set ...')
            for b in all_contexts:
                t1=time.perf_counter()
                self.factor_data = self.factor_context_data(factor_data,self.value_growth,b)
                self.calc_factor_performance()
                self.all_results['value_growth_'+str(b)]=copy.deepcopy(self.results)
                print('%-30s%20s%2s%10.4f%4s' % ("  calculating value_growth ",b,": ",t2-t1,"sec"))
        

    def display(self):
        print('%s' % '='*80)
        print('%-30s%10s%10s%10s%10s%10s' % ('context','Anal_rets','Anal_IR','MDrawdown','Net_rets','Net_IR'))
        for cv in self.all_results:
            results = self.all_results[cv]
            print('%-30s%10.4f%10.4f%10.4f%10.4f%10.4f' % (cv,results['Annualized Return'],results['Annualized IR'],results['Maximum Drawdown'],
                                                           results['Annualized Net Return'],results['Annualized Net IR']))
        print('%s' % '='*80)
    
if __name__ == '__main__':

    class my_factor1(factor):
        def set_factor_property(self):
            factor_property={'factor_name': 'test_01','data_needed': ['pe'],'lag_periods': 2, 'ic_return_horizon': 60}
            return factor_property
        def calc_factor(self):
            self.factor_data=1/self.data['pe']

    class my_factor2(factor):
        def set_factor_property(self):
            factor_property={'factor_name': 'test_02','data_needed': ['returns'],'lag_periods': 1, 'ic_return_horizon': 1}
            return factor_property
        def calc_factor(self):
            self.factor_data=self.data['returns'].rolling(5).sum()


    class my_factors(multi_factor):
        def calc_all_factors(self):     
            factor_class=my_factor1()    
            self.all_factors[factor_class.factor_property['factor_name']]=copy.deepcopy(factor_class)
          
            factor_class=my_factor2()    
            self.all_factors[factor_class.factor_property['factor_name']]=copy.deepcopy(factor_class)
       

    class my_factor_params(factor_params):
        def set_factor_property(self):
            factor_property={'factor_name': 'test_parmas', 'data_needed': []}
            return factor_property
        
        def calc_factor(self):
            n1=self.factor_params['n1']
            n2=self.factor_params['n2']
            n3=self.factor_params['n3']
            self.factor_data=self.returns.rolling(n1).sum()+self.returns.rolling(n3).sum()-2*self.returns.rolling(n2).sum()
            
    class my_factor_context(factor_context):
        def set_factor_property(self):
            factor_property={'factor_name': 'test_context','data_needed': []}
            return factor_property
        def calc_factor(self):
            self.factor_data=self.returns.rolling(10).sum()
         
    #params={'ic_return_horizons': [5,20,60], 'ic_delay_periods': [1,2,5]}
    #params={'start_os_date': '2022-01-01','ir_details': True}
    #params={}
    #f=my_factors()
    #f.run()


    all_factor_params={'n1':[5,7],'n2':[10,15],'n3':[20,25]}
    f=my_factor_params(all_factor_params=all_factor_params)
    f.run()

    #f=my_factor_context()
    #f.run()
    