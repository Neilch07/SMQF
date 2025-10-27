from typing import Dict, List, Optional, Union
from attrs import define, field
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import pickle
import gzip
from matplotlib import pyplot as plt
import time
import os


@define
class factor_combo_base():
    __params0 = {'start_date': None, # if None, using the start of the whole data
                'end_date': None, # if None, using the end of the whole data      
                'trading_cost': 0.0012, # trading costs in terms of two way costs in returns
                'n_trading_days': 243, # number of trading days per year        
                'quantile': 0.2,      
                'display': True, 
                'save': False, # save the results to files
                'ir_details': False, # if True, calculate IR for each year, in-sample and out-sample
                'data_dir': '.\\data\\cn\\equity\\data\\',
                'factor_dir': '.\\data\\cn\\equity\\factor\\',
                'testing_periods': 0, # 0,1,'rolling'
                'testing_period_length': 20, # int
                'training_period_length': 'auto', # 'auto', int, 'all'
                'first_training_period_length': 'auto' # 'auto', int, used only when 'training_period_length' is 'all'
                }  
    __combo_property0 = {'combo_name': '',
                        'factors_needed': [],
                        'universe': None, # options are None, 'Top80','Top60', 'Top40', 'HS300','ZZ500','ZZ800','ZZ1000'
                        'lag_periods': 2, # number of delay periods to forecast, suggesting 1 for prices/volumes, 2 for fundamentals
                        'benchmark': None, # options are None,'HS300','ZZ500','ZZ1000',
                        'normalization_method': None, # None, 'zscore','minmax','tanh'
                        'scale_coeff': 1 # used when 'normalization_method' is 'tanh'
                        }    
    params: Dict[str,Union[int, float, str]] = field(default=None, kw_only=True)
    combo_property: Dict[str,Union[int, float, str]] = field(default=None, kw_only=True)
    factors_needed: List[str] = field(default=[], kw_only=True)
    factors_data: Dict[str,pd.DataFrame] = field(default={}, init=False)
    factors_property: Dict[str,Dict] = field(default={}, init=False)
    returns: pd.DataFrame = field(default=None, init=False)
    index_returns: pd.DataFrame = field(default=None, init=False)
    results: Dict[str,Union[int, float, str]] = field(default={}, init=False)
    testing_period_indices: List[tuple] = field(default=[], init=False)
    training_period_indices: List[tuple] = field(default=[], init=False)
    testing_period_dates: List[tuple] = field(default=[], init=False)
    training_period_dates: List[tuple] = field(default=[], init=False)
    testing_positions: pd.DataFrame = field(default=None, init=False)
    training_positions: pd.DataFrame = field(default=None, init=False)
    training_returns: pd.DataFrame = field(default=None, init=False)
    testing_returns: pd.DataFrame = field(default=None, init=False)
    training_net_returns: pd.DataFrame = field(default=None, init=False)
    testing_net_returns: pd.DataFrame = field(default=None, init=False)
    training_ex_returns: pd.DataFrame = field(default=None, init=False)
    testing_ex_returns: pd.DataFrame = field(default=None, init=False)
    training_net_ex_returns: pd.DataFrame = field(default=None, init=False)
    testing_net_ex_returns: pd.DataFrame = field(default=None, init=False)

    def __attrs_post_init__(self):
        if self.params is not None:
            self.__params0.update(self.params)
        self.__combo_property0.update(self.set_combo_property())
        if self.combo_property is not None:
            self.__combo_property0.update(self.combo_property)
        self.params = self.__params0
        self.combo_property = self.__combo_property0
        self.factors_needed=self.combo_property['factors_needed']
        self.params.update(self.combo_property)
        
    def run(self,turnoff_display=False):
        t1=time.perf_counter()
        self.load_data() 
        self.set_testing_periods()
        t2=time.perf_counter()
        print('%4s%-40s%10.4f%4s' % (" ","load_data time:",t2-t1,"sec"))
        if self.params['normalization_method'] is not None:
            t1=time.perf_counter()
            self.normalize_factors()
            t2=time.perf_counter()
            print('%4s%-40s%10.4f%4s' % (" ","normalize_factors calculating time:",t2-t1,"sec"))
        t1=time.perf_counter()
        self.calc_combo_positions()
        t2=time.perf_counter()
        print('%4s%-40s%10.4f%4s' % (" ","calc_combo_positions calculating time:",t2-t1,"sec"))
        t1=time.perf_counter()
        self.calc_combo_returns()
        t2=time.perf_counter()
        print('%4s%-40s%10.4f%4s' % (" ","calc_combo_returns calculating time:",t2-t1,"sec"))
        self.calc_combo_performance()
        if self.params['save'] == True:
            self.save()
        if self.params['display'] == True and turnoff_display==False:
            self.display()
        return self.results

    def load_data(self):
        self.returns = pd.read_pickle(os.path.join(self.params['data_dir'],'returns.pkl'))
        self.returns.index=pd.to_datetime(self.returns.index)
        if len(self.factors_needed)==0:
            raise ValueError("factors_needed needs at least one element")
        for fld in self.factors_needed:
            with gzip.open(os.path.join(self.params['factor_dir'], 'factor_data_'+fld+'.pkl.gz'), 'rb') as f:
                loaded_data = pickle.load(f)
            self.factors_data[fld] = loaded_data['factor_data']
            self.factors_data[fld].index=pd.to_datetime(self.factors_data[fld].index)
            self.factors_property[fld] = loaded_data['factor_property']
        if self.combo_property['universe'] is not None:
            self.universe = pd.read_pickle(os.path.join(self.params['data_dir'],self.factor_property['universe']+'.pkl'))
            self.universe.index=pd.to_datetime(self.universe.index)
        if self.combo_property['benchmark'] is not None:
            index_data = pd.read_pickle(os.path.join(self.params['data_dir'],'index_data.pkl'))
            index_data.index=pd.to_datetime(index_data.index)
            self.index_returns = index_data[self.combo_property['benchmark']+'_returns']
        self.align_data()

    def align_data(self):
        if self.params['start_date'] is None:
            start_date = self.returns.index[0]
        else:
            start_date = pd.to_datetime(self.params['start_date'])
        if self.params['end_date'] is None:
            end_date = self.returns.index[-1]
        else:
            end_date = pd.to_datetime(self.params['end_date'])

        if (self.params['start_date'] is not None) | (self.params['end_date'] is not None):
            for fld in self.factors_needed:
                self.data[fld] = self.data[fld][start_date:end_date]
            if self.returns is not None:
                self.returns=self.returns[start_date:end_date]
            if self.universe is not None:
                self.universe=self.universe[start_date:end_date]
            if self.index_returns is not None:
                self.index_returns=self.index_returns[start_date:end_date]
        self.params['start_date'] = start_date
        self.params['end_date'] = end_date
 

    def set_testing_periods(self):
        nd,_ = self.returns.shape
        dates = list(self.returns.index.date)
        if (self.params['testing_periods'] is None) or (self.params['testing_periods']==0):
            self.testing_period_indices.append((nd-1,nd))
            self.training_period_indices.append((0,nd))
            self.testing_period_dates.append((np.nan,np.nan))
            self.training_period_dates.append((dates[0],dates[nd-1]))
        elif self.params['testing_periods']==1:
            testing_period_length=self.params['testing_period_length']
            if testing_period_length>nd-20:
                raise ValueError("testing_period_length is too large, not enough data for training")
            self.testing_period_indices.append((nd-testing_period_length,nd))
            self.training_period_indices.append((0,nd-testing_period_length))
            self.testing_period_dates.append((dates[nd-testing_period_length],dates[nd-1]))
            self.training_period_dates.append((dates[0],dates[nd-testing_period_length-1]))
        elif self.params['testing_periods']=='rolling':
            testing_period_length=self.params['testing_period_length']
            if testing_period_length>nd-20:
                raise ValueError("testing_period_length is too large, not enough data for training")
            training_period_length=self.params['training_period_length']
            if training_period_length=='all':
                first_training_period_length=self.params['first_training_period_length']
                if first_training_period_length=='auto':
                    first_training_period_length=min(3*testing_period_length,nd-testing_period_length)
                if first_training_period_length>nd-20:
                    raise ValueError("first_training_period_length is too large, not enough data")
                i0=i1=i2=0
                while i2<nd:
                    if i1==0:
                        i1=i0+first_training_period_length
                    i2=min(i1+testing_period_length,nd)
                    self.testing_period_indices.append((i1,i2))
                    self.training_period_indices.append((i0,i1))
                    self.testing_period_dates.append((dates[i1],dates[i2-1]))
                    self.training_period_dates.append((dates[i0],dates[i1-1]))
                    i1=i2
            else:
                if training_period_length=='auto':
                    training_period_length=min(3*testing_period_length,nd-testing_period_length)
                if training_period_length>nd-20:
                    raise ValueError("training_period_length is too large, not enough data")
                i0=i1=i2=0
                while i2<nd:
                    if i1==0:
                        i1=i0+training_period_length
                    i2=min(i1+testing_period_length,nd)
                    self.testing_period_indices.append((i1,i2))
                    self.training_period_indices.append((i0,i1))
                    self.testing_period_dates.append((dates[i1],dates[i2-1]))
                    self.training_period_dates.append((dates[i0],dates[i1-1]))
                    i0=i0+testing_period_length
                    i1=i2
        else:
            raise ValueError("testing_periods must be 0, 1, or \'rolling\'")


    def display_testing_periods(self):
        if self.testing_period_indices==[]:
            self.returns = pd.read_pickle(os.path.join(self.params['data_dir'],'returns.pkl'))
            self.returns.index=pd.to_datetime(self.returns.index)
            self.set_testing_periods()
        print('%8s%46s%46s' % (' ','training period','testing period'))
        print('%8s%8s%8s%15s%15s%8s%8s%15s%15s' % (' ','start','end','start','end','start','end','start','end'))
        for i in range(len(self.testing_period_indices)):
            print('%8d%8d%8d%15s%15s%8d%8d%15s%15s' % (i,self.training_period_indices[i][0],self.training_period_indices[i][1]-1,
                                                       self.training_period_dates[i][0],self.training_period_dates[i][1],
                                                       self.testing_period_indices[i][0],self.testing_period_indices[i][1]-1,
                                                       self.testing_period_dates[i][0],self.testing_period_dates[i][1]))


    def normalize_factors(self):
        if self.params['normalization_method'] == 'zscore':
            for ft in self.factors_data:
                factors_data = self.factors_data[ft]
                self.factors_data[ft]=(factors_data.sub(factors_data.mean(axis=1),axis=0)).div(1e-10+factors_data.std(axis=1),axis=0)
        elif self.params['normalization_method'] == 'minmax':
            for ft in self.factors_data:
                factors_data = self.factors_data[ft]
                factor_min = factors_data.min(axis=1)
                self.factors_data[ft]=(factors_data.sub(factor_min,axis=0)).div(1e-10+factors_data.max(axis=1)-factor_min,axis=0)
        elif self.params['normalization_method'] == 'tanh':
            for ft in self.factors_data:
                factors_data = self.factors_data[ft]
                zscore=(factors_data.sub(factors_data.mean(axis=1),axis=0)).div(1e-10+factors_data.std(axis=1),axis=0)
                self.factors_data[ft]=np.tanh(self.params['scale_coeff']*zscore)
        else:
            raise ValueError("normalization_method is unrecoganized")
            

    def calc_rank_pos(self,lag_periods,factor_data):
        factors = factor_data.shift(lag_periods)
        ranking = factors.rank(axis=1, pct=True)
        qt=self.params['quantile']
        top_qt = ranking<qt
        bot_qt = ranking>1-qt
        top_pos = top_qt.astype(int).divide(top_qt.sum(axis=1),axis=0)
        bot_pos = bot_qt.astype(int).divide(bot_qt.sum(axis=1),axis=0)
        return bot_pos.fillna(0)-top_pos.fillna(0)
    
    def calc_factor_returns(self,factor_pos,returns):
        return (factor_pos.shift(1)*returns).sum(axis=1)

    def calc_combo_positions(self):
        self.testing_positions=pd.DataFrame(np.nan,index=self.returns.index,columns=self.returns.columns)
        self.training_positions=pd.DataFrame(np.nan,index=self.returns.index,columns=self.returns.columns)

        for i in range(len(self.testing_period_indices)):
            i0=self.training_period_indices[i][0]
            i1=self.training_period_indices[i][1]
            i2=self.testing_period_indices[i][1]
            training_pos,testing_pos=self.calc_subperiod_combo_positions(i0,i1,i2)
            self.training_positions.iloc[i0:i1,:]=training_pos
            self.testing_positions.iloc[i1:i2,:]=testing_pos


    def calc_combo_returns(self):
        training_trade = self.training_positions.fillna(0).diff(axis=0).abs().sum(axis=1)
        testing_trade = self.testing_positions.fillna(0).diff(axis=0).abs().sum(axis=1)
        self.training_returns=(self.training_positions.shift(1)*self.returns/2.0).sum(axis=1)
        self.training_net_returns=self.training_returns-0.5*self.params['trading_cost']*training_trade/2.0
        self.testing_returns=(self.testing_positions.shift(1)*self.returns/2.0).sum(axis=1)
        self.testing_net_returns=self.testing_returns-0.5*self.params['trading_cost']*testing_trade/2.0


        if self.params['benchmark'] is not None:
            long_pos = self.training_positions.where(self.training_positions>0, np.nan)
            long_training_rets=(long_pos.shift(1)*self.returns).sum(axis=1,min_count=1)
            long_training_trade = long_pos.fillna(0).diff(axis=0).abs().sum(axis=1)
            long_training_trade_cost_rets = 0.5*self.params['trading_cost']*long_training_trade
            self.training_ex_returns=(long_training_rets-self.index_returns)/1.2
            self.training_net_ex_returns=(long_training_rets-self.index_returns-long_training_trade_cost_rets)/1.2
            long_pos = self.testing_positions.where(self.testing_positions>0, np.nan)
            long_testing_rets=(long_pos.shift(1)*self.returns).sum(axis=1,min_count=1)
            long_testing_trade = long_pos.fillna(0).diff(axis=0).abs().sum(axis=1)
            long_testing_trade_cost_rets = 0.5*self.params['trading_cost']*long_testing_trade
            self.testing_ex_returns=(long_testing_rets-self.index_returns)/1.2
            self.testing_net_ex_returns=(long_testing_rets-self.index_returns-long_testing_trade_cost_rets)/1.2

    def calc_ir(self):
        self.results['Annualized Training Return']=self.params['n_trading_days']*self.training_returns.mean()
        self.results['Annualized Training Volatility']=np.sqrt(self.params['n_trading_days'])*self.training_returns.std()
        if self.results['Annualized Training Volatility']>0:
            self.results['Annualized Training IR']=self.results['Annualized Training Return']/self.results['Annualized Training Volatility']
        else:
            self.results['Annualized Training IR']=np.nan
        nav=(1+self.training_returns).cumprod()
        drawdown=nav/nav.cummax()-1
        self.results['Maximum Training Drawdown']=-drawdown.min()
        
        self.results['Annualized Training Net Return']=self.params['n_trading_days']*self.training_net_returns.mean()
        self.results['Annualized Training Net Volatility']=np.sqrt(self.params['n_trading_days'])*self.training_net_returns.std()
        if self.results['Annualized Training Net Volatility']>0:
            self.results['Annualized Training Net IR']=self.results['Annualized Training Net Return']/self.results['Annualized Training Net Volatility']
        else:
            self.results['Annualized Training Net IR']=np.nan
        net_nav=(1+self.training_net_returns).cumprod()
        drawdown=net_nav/net_nav.cummax()-1
        self.results['Maximum Training Net Drawdown']=-drawdown.min()

        self.results['Annualized Testing Return']=self.params['n_trading_days']*self.testing_returns.mean()
        self.results['Annualized Testing Volatility']=np.sqrt(self.params['n_trading_days'])*self.testing_returns.std()
        if self.results['Annualized Testing Volatility']>0:
            self.results['Annualized Testing IR']=self.results['Annualized Testing Return']/self.results['Annualized Testing Volatility']
        else:
            self.results['Annualized Testing IR']=np.nan
        nav=(1+self.testing_returns).cumprod()
        drawdown=nav/nav.cummax()-1
        self.results['Maximum Testing Drawdown']=-drawdown.min()
        
        self.results['Annualized Testing Net Return']=self.params['n_trading_days']*self.testing_net_returns.mean()
        self.results['Annualized Testing Net Volatility']=np.sqrt(self.params['n_trading_days'])*self.testing_net_returns.std()
        if self.results['Annualized Testing Net Volatility']>0:
            self.results['Annualized Testing Net IR']=self.results['Annualized Testing Net Return']/self.results['Annualized Testing Net Volatility']
        else:
            self.results['Annualized Testing Net IR']=np.nan
        net_nav=(1+self.testing_net_returns).cumprod()
        drawdown=net_nav/net_nav.cummax()-1
        self.results['Maximum Testing Net Drawdown']=-drawdown.min()

        if self.params['benchmark'] is not None:
            self.results['Annualized Training Ex Return']=self.params['n_trading_days']*self.training_ex_returns.mean()
            vol=np.sqrt(self.params['n_trading_days'])*self.training_ex_returns.std()
            if vol>0:
                self.results['Annualized Training Ex IR']=self.results['Annualized Training Ex Return']/vol
            else:
                self.results['Annualized Training Ex IR']=np.nan
            self.results['Annualized Training Net Ex Return']=self.params['n_trading_days']*self.training_net_ex_returns.mean()
            vol=np.sqrt(self.params['n_trading_days'])*self.training_net_ex_returns.std()
            if vol>0:
                self.results['Annualized Training Net Ex IR']=self.results['Annualized Training Net Ex Return']/vol
            else:
                self.results['Annualized Training Net Ex IR']=np.nan

            self.results['Annualized Testing Ex Return']=self.params['n_trading_days']*self.testing_ex_returns.mean()
            vol=np.sqrt(self.params['n_trading_days'])*self.testing_ex_returns.std()
            if vol>0:
                self.results['Annualized Testing Ex IR']=self.results['Annualized Testing Ex Return']/vol
            else:
                self.results['Annualized Testing Ex IR']=np.nan
            self.results['Annualized Testing Net Ex Return']=self.params['n_trading_days']*self.testing_net_ex_returns.mean()
            vol=np.sqrt(self.params['n_trading_days'])*self.testing_net_ex_returns.std()
            if vol>0:
                self.results['Annualized Testing Net Ex IR']=self.results['Annualized Testing Net Ex Return']/vol
            else:
                self.results['Annualized Testing Net Ex IR']=np.nan


    def calc_ir_details(self):
        self.results['training_return_details']=[]
        self.results['training_ir_details']=[]
        self.results['testing_return_details']=[]
        self.results['testing_ir_details']=[]
        for i in range(len(self.testing_period_indices)):
            i0=self.training_period_indices[i][0]
            i1=self.training_period_indices[i][1]
            i2=self.testing_period_indices[i][1]
            rets=self.params['n_trading_days']*self.training_returns.iloc[i0:i1].mean()
            vol=np.sqrt(self.params['n_trading_days'])*self.training_returns.iloc[i0:i1].std()
            self.results['training_return_details'].append(rets)
            if vol>0:
                self.results['training_ir_details'].append(rets/vol)
            else:
                self.results['training_ir_details'].append(np.nan)
            rets=self.params['n_trading_days']*self.testing_returns.iloc[i1:i2].mean()
            vol=np.sqrt(self.params['n_trading_days'])*self.testing_returns.iloc[i1:i2].std()
            self.results['testing_return_details'].append(rets)
            if vol>0:
                self.results['testing_ir_details'].append(rets/vol)
            else:
                self.results['testing_ir_details'].append(np.nan)


    def calc_combo_performance(self):
        t1=time.perf_counter()
        self.calc_ir()
        t2=time.perf_counter()
        print('%4s%-40s%10.4f%4s' % (" ","calc_ir calculating time:",t2-t1,"sec"))
        if self.params['ir_details'] == True:
            self.calc_ir_details()
            t3=time.perf_counter()
            print('%4s%-40s%10.4f%4s' % (" ","calc_ir_details calculating time:",t3-t2,"sec"))

    def save(self):
        combo_name=self.params['combo_name']
        self.testing_positions.to_pickle(os.path.join(self.params['factor_dir'], 'combo_position_'+combo_name+'.pkl'))
        self.results['testing_returns']=self.testing_returns
        self.results['training_returns']=self.training_returns
        self.results['testing_net_returns']=self.testing_net_returns
        self.results['training_net_returns']=self.training_net_returns
        self.results.to_pickle(os.path.join(self.params['factor_dir'], 'combo_results_'+combo_name+'.pkl'))

    def display(self):
        print('%s' % '='*100)
        if self.params['universe'] is None:
            universe='All'
        else:
            universe=self.params['universe']
        if self.params['benchmark'] is None:
            benchmark='None'
        else:
            benchmark=self.params['benchmark']
        print('%-15s%15s' % ('Combo Name',self.params['combo_name']))
        print('%-15s%15s%11s%-22s%15s' % ('Start Date',self.params['start_date'].strftime("%Y-%m-%d"),' ','End Date',self.params['end_date'].strftime("%Y-%m-%d")))
        print('%-15s%15s%11s%-22s%15s' % ('Universe',universe,' ','Benchmark',benchmark))

        print('%s' % '-'*100)
        print('%-32s %8.4f %9s %-35s %8.4f' % ('Annualized Training Return',self.results['Annualized Training Return'],' ','Annualized Training Net Rerturn',self.results['Annualized Training Net Return']))
        print('%-32s %8.4f %9s %-35s %8.4f'  % ('Annualized Training IR',self.results['Annualized Training IR'],' ','Annualized Training Net IR',self.results['Annualized Training Net IR']))
        print('%-32s %8.4f %9s %-35s %8.4f'  % ('Maximum Training Drawdown',self.results['Maximum Training Drawdown'],' ','Maximum Training Net Drawdown',self.results['Maximum Training Net Drawdown']))
        print('%-32s %8.4f %9s %-35s %8.4f' % ('Annualized Testing Return',self.results['Annualized Testing Return'],' ','Annualized Testing Net Rerturn',self.results['Annualized Testing Net Return']))
        print('%-32s %8.4f %9s %-35s %8.4f'  % ('Annualized Testing IR',self.results['Annualized Testing IR'],' ','Annualized Testing Net IR',self.results['Annualized Testing Net IR']))
        print('%-32s %8.4f %9s %-35s %8.4f'  % ('Maximum Testing Drawdown',self.results['Maximum Testing Drawdown'],' ','Maximum Testing Net Drawdown',self.results['Maximum Testing Net Drawdown']))

        if self.params['benchmark'] is not None:
            print('%-32s %8.4f %9s %-35s %8.4f' % ('Annualized Training Ex Return',self.results['Annualized Training Ex Return'],' ','Annualized Training Net Ex Rerturn',self.results['Annualized Training Net Ex Return']))
            print('%-32s %8.4f %9s %-35s %8.4f'  % ('Annualized Training Ex IR',self.results['Annualized Training Ex IR'],' ','Annualized Training Net Ex IR',self.results['Annualized Training Net Ex IR']))
            print('%-32s %8.4f %9s %-35s %8.4f' % ('Annualized Testing Ex Return',self.results['Annualized Testing Ex Return'],' ','Annualized Testing Net Ex Rerturn',self.results['Annualized Testing Net Ex Return']))
            print('%-32s %8.4f %9s %-35s %8.4f'  % ('Annualized Testing Ex IR',self.results['Annualized Testing Ex IR'],' ','Annualized Testing Net Ex IR',self.results['Annualized Testing Net Ex IR']))
        print('%s' % '-'*100)
 
        if self.params['ir_details']==True:
            print('%-14s%-14s%-14s%-14s%10s%10s%10s%10s' % ('Start','End','Start','End','Training','Training','Testing','Testing'))
            print('%-14s%-14s%-14s%-14s%10s%10s%10s%10s' % ('Training','Training','Testing','Testing','Return','IR','Return','IR'))
            for i in range(len(self.testing_period_indices)):
                dt1=self.training_period_dates[i][0]
                dt2=self.training_period_dates[i][1]
                dt3=self.testing_period_dates[i][0]
                dt4=self.testing_period_dates[i][1]
                training_ret=self.results['training_return_details'][i]
                training_ir=self.results['training_ir_details'][i]
                testing_ret=self.results['testing_return_details'][i]
                testing_ir=self.results['testing_ir_details'][i]
                print('%-14s%-14s%-14s%-14s%10.4f%10.4f%10.4f%10.4f' % (dt1,dt2,dt3,dt4,training_ret,training_ir,testing_ret,testing_ir))
            print('%s' % '-'*100)

    def clip_weights(self,weights): 
        tw=0
        nf = len(weights)
        for f in weights:                 
            tw = tw+weights[f]
        if self.params['min_weight']=='auto':
            min_weight=0.5/nf
        else:
            min_weight=self.params['min_weight']
        if self.params['max_weight']=='auto':
            max_weight=2.0/nf
        else:
            max_weight=self.params['max_weight']
        for f in weights:
            weights[f]=weights[f]/tw
            if weights[f]<min_weight:
                weights[f]=min_weight
            if weights[f]>max_weight:
                weights[f]=max_weight
        tw=0
        for f in weights:
            tw=tw+weights[f]
        for f in weights:
            weights[f]=weights[f]/tw
        return weights

    def set_combo_property(self):
        return {}
 

    @abstractmethod
    def calc_subperiod_combo_positions(self,i0,i1,i2):
        raise NotImplementedError



        

        
if __name__ == '__main__':

    f = factor_combo_base(params={'testing_periods': 'rolling', # 0,1,'rolling'
                                'testing_period_length': 20,
                                'training_period_length': 'all', # 'auto', int, 'all'
                                'first_training_period_length': 40 # 'auto', int, used only when 'training_period_length' is 'all'
                                })
    f.display_testing_periods()
  