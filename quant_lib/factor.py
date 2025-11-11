from typing import Dict, List, Optional, Union
from attrs import define, field
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
try:
    from matplotlib import pyplot as plt
except Exception:  # matplotlib may be unavailable if binary wheels mismatch local numpy
    plt = None
import time
from datetime import datetime, timedelta
import os
import pickle
import gzip

@define
class factor():
    __params = {'start_date': None, # if None, using the start of the whole data
                'end_date': None, # if None, using the end of the whole data
                'start_os_date': None, # out of sample start date, if None, there is no os               
                'trading_cost': 0.0012, # trading costs in terms of two way costs in returns
                'n_group': 10, # if 0, no group return will be calculated
                'quantile': 0.2,
                'ic_return_horizons': None, # list of number of periods to calculate returns for ic, for example [5,20,60]
                'ic_delay_periods': None, # list of periods for delayed ic, for example, [1,2,5]
                'n_trading_days': 243, # number of trading days per year              
                'display': True, 
                'save': False, # save the results to files
                'ir_details': False, # if True, calculate IR for each year, in-sample and out-sample
                'data_dir': '.\\data\\cn\\equity\\data\\',
                'factor_dir': '.\\data\\cn\\equity\\factor\\'
                }  
    __factor_property = {'factor_name': None,
                         'factor_type': None, # 'market','fundamental','hybrid','alternative'，‘intraday'
                         'data_needed': [],
                         'universe': None, # options are None, 'top60pct','top75pct', 'hs300','cs500','cs800','cs1000','cs1800'
                         'lag_periods': 2, # number of delay periods to forecast
                         'ic_return_horizon': 1, # number of periods to calculate returns for ic
                         'benchmark': None, # options are None,'hs300','cs500','cs1000'
                         'smooth_periods': 0, # number of periods to calculate moving average of factor 
                         'sector_neutral': False,
                         }
    data: Dict[str,pd.DataFrame] = field(factory=dict, kw_only=True)  
    params: Dict[str,Union[int, float, str]] = field(default=None, kw_only=True)
    factor_property: Dict[str,Union[int, float, str]] = field(default=None, kw_only=True)
    data_needed: List[str] = field(factory=list, init=False)
    factor_data: pd.DataFrame = field(default=None, init=False)
    returns: pd.DataFrame = field(default=None, init=False)
    sectors: pd.DataFrame = field(default=None, init=False)
    universe: pd.DataFrame = field(default=None, init=False)
    index_returns: pd.DataFrame = field(default=None, init=False)
    results: Dict[str,Union[int, float, str]] = field(factory=dict, init=False)
   
    def __attrs_post_init__(self):
        if self.params is not None:
            self.__params.update(self.params)
        self.__factor_property.update(self.set_factor_property())
        if self.factor_property is not None:
            self.__factor_property.update(self.factor_property)
        self.params = self.__params
        self.factor_property = self.__factor_property
        self.data_needed=self.factor_property['data_needed']
        self.params.update(self.factor_property)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        for key in ('data_dir', 'factor_dir'):
            path_val = self.params.get(key)
            if path_val:
                normalized = path_val.replace('\\', os.sep)
                if not os.path.isabs(normalized):
                    normalized = os.path.join(base_dir, normalized)
                self.params[key] = os.path.normpath(normalized)
        
    def run(self,turnoff_display=False):
        if self.data == {}:
            t1=time.perf_counter()
            self.load_data() 
            t2=time.perf_counter()
            print('%4s%-40s%10.4f%4s' % (" ","load_data time:",t2-t1,"sec"))
        t1=time.perf_counter()
        self.calc_factor()
        t2=time.perf_counter()
        print('%4s%-40s%10.4f%4s' % (" ","calc_factor calculating time:",t2-t1,"sec"))
        self.calc_factor_performance()
        if self.params['save'] == True:
            self.save()
        if self.params['display'] == True and turnoff_display==False:
            self.display()
        return [self.factor_data,self.results]

    def get_factor(self):
        return self.factor_data
    
    def get_performance(self):
        return self.results
    
    def available_data(self):
        dir_name=os.path.join(self.params['data_dir'],"2021-current")
        all_items = os.listdir(dir_name) 
        data_names = [item[:-4] for item in all_items if os.path.isfile(os.path.join(dir_name, item))]
        return data_names

    def read_pickle(self,fld):
        # 字段名映射：
        # 对于2007-2014和2015-2020: 因子定义 -> 单数形式
        # 对于2021-current: 因子定义 -> 复数形式
        field_mapping_old = {  # 旧目录(2007-2014, 2015-2020)使用单数
            'close': 'close',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'volume': 'volume',
            'amount': 'turnover_amount',  # amount映射到turnover_amount
            'return': 'pct_change',  # return映射到pct_change
            'industry': 'industry',
            'index_data': 'index_data',
            'top75pct': 'top75pct',
            'top60pct': 'top60pct',
            'cs500': 'cs500',
            'cs800': 'cs800',
            'cs1000': 'cs1000',
            'cs1800': 'cs1800',
            'hs300': 'hs300',
            'sse50': 'sse50',
        }
        
        field_mapping_new = {  # 新目录(2021-current)使用复数
            'close': 'closes',
            'open': 'opens',
            'high': 'highs',
            'low': 'lows',
            'volume': 'volumes',
            'amount': 'turnover_amounts',
            'return': 'pct_changes',
            'industry': 'industrys',
            'index_data': 'index_datas',
            'top75pct': 'top75pcts',
            'top60pct': 'top60pcts',
            'cs500': 'cs500s',
            'cs800': 'cs800s',
            'cs1000': 'cs1000s',
            'cs1800': 'cs1800s',
            'hs300': 'hs300s',
            'sse50': 'sse50s',
        }
        
        start_date = self.params['start_date']
        if (start_date is not None):
            start_date = pd.to_datetime(start_date)
        date1 = pd.to_datetime('2015-01-01')
        date2 = pd.to_datetime('2021-01-01')
        dir_name = self.params['data_dir']
        data = None
        
        if (start_date is None) or (start_date<date1):
            # 2007-2014 目录使用旧映射
            actual_fld = field_mapping_old.get(fld, fld)
            f_name=os.path.join(dir_name,"2007-2014",actual_fld+'1.pkl')
            data=pd.read_pickle(f_name)
            data.index=pd.to_datetime(data.index)
            
        if (start_date is None) or (start_date<date2):
            # 2015-2020 目录使用旧映射
            actual_fld = field_mapping_old.get(fld, fld)
            f_name=os.path.join(dir_name,"2015-2020",actual_fld+'2.pkl')
            if data is None:
                data=pd.read_pickle(f_name)
                data.index=pd.to_datetime(data.index)
            else:
                data1=pd.read_pickle(f_name)
                data1.index=pd.to_datetime(data1.index)
                data=pd.concat([data,data1],join='outer')
                
        # 2021-current 目录使用新映射
        actual_fld = field_mapping_new.get(fld, fld)
        f_name=os.path.join(dir_name,"2021-current",actual_fld+'.pkl')
        if data is None:
            data=pd.read_pickle(f_name)
            data.index=pd.to_datetime(data.index)
        else:
            data1=pd.read_pickle(f_name)
            data1.index=pd.to_datetime(data1.index)
            data=pd.concat([data,data1],join='outer')
        return data
                                

    def load_data(self):
        self.returns = self.read_pickle('return')  # 使用统一的字段名'return',会自动映射
        for fld in self.data_needed:
            self.data[fld] = self.read_pickle(fld)
        if self.factor_property['sector_neutral']==True:
            if 'industry' not in self.data_needed:
                self.sectors = self.read_pickle('industry')  # 使用统一的字段名
            else:
                self.sectors = self.data['industry']
        if self.factor_property['universe'] is not None:
            if self.factor_property['universe'] not in self.data_needed:
                self.universe = self.read_pickle(self.factor_property['universe'])
            else:
                self.universe = self.data['universe']
        if self.factor_property['benchmark'] is not None:
            if 'index_data' not in self.data_needed:
                index_data = self.read_pickle('index_data')  # 使用统一的字段名
            else:
                index_data = self.data['index_data']
            self.index_returns = index_data[self.factor_property['benchmark']+'_pct']
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
        if self.params['start_os_date'] is not None:
            self.params['start_os_date'] = pd.to_datetime(self.params['start_os_date'])

        if (self.params['start_date'] is not None) | (self.params['end_date'] is not None):
            for d in self.data_needed:
                self.data[d] = self.data[d][start_date:end_date]
            if self.returns is not None:
                self.returns=self.returns[start_date:end_date]
            if self.sectors is not None:
                self.sectors=self.sectors[start_date:end_date]
            if self.universe is not None:
                self.universe=self.universe[start_date:end_date]
            if self.index_returns is not None:
                self.index_returns=self.index_returns[start_date:end_date]
        self.params['start_date'] = start_date
        self.params['end_date'] = end_date
        

    def calc_sector_neutral(self):
        sectors = self.sectors.values
        sector_ids=np.unique(sectors)
        sec_factors=0
        for sec in sector_ids:
            sec_data = pd.DataFrame(np.where(sectors==sec,0,np.nan),index=self.sectors.index,columns=self.sectors.columns)
            sec_factor=self.factor_data+sec_data
            sec_factor.subtract(sec_factor.mean(axis=1),axis=0)
            sec_factors=sec_factors+sec_factor.where(sectors==sec,other=0,inplace=False)
        self.factor_data=sec_factors
       


    def calc_factor_returns(self):
        factors = self.factor_data.shift(self.params['lag_periods'])
        ranking = factors.rank(axis=1, pct=True)
        n_bins=self.params['n_group']
        qt=self.params['quantile']
        if n_bins>1:
            nd = self.returns.shape[0]
            grp_names=[('group%d' % i) for i in range(1,n_bins+1)]
            self.results['group_returns']=pd.DataFrame(np.full((nd,n_bins),np.nan),index=factors.index,columns=grp_names)
            for i in range(n_bins):
                l,u=i/n_bins,(i+1)/n_bins
                rets=self.returns[((ranking<u) & (ranking>l))]
                self.results['group_returns'].iloc[:,i]=rets.mean(axis=1)  
            self.results['group_returns']=self.results['group_returns'].sub(self.results['group_returns'].mean(axis=1),axis=0)

        top_qt = ranking<qt
        bot_qt = ranking>1-qt
        self.top_returns=self.returns[top_qt].mean(axis=1)
        self.bot_returns=self.returns[bot_qt].mean(axis=1)
        self.results['ls_returns']=(self.bot_returns-self.top_returns)/2.0

        top_pos = top_qt.astype(int).divide(top_qt.sum(axis=1),axis=0)
        top_trade = top_pos.fillna(0).diff(axis=0).abs().sum(axis=1)
        bot_pos = bot_qt.astype(int).divide(bot_qt.sum(axis=1),axis=0)
        bot_trade = bot_pos.fillna(0).diff(axis=0).abs().sum(axis=1)
        top_trade_cost_rets = 0.5*self.params['trading_cost']*top_trade
        bot_trade_cost_rets = 0.5*self.params['trading_cost']*bot_trade
        self.results['ls_net_returns']=self.results['ls_returns']-(top_trade_cost_rets+bot_trade_cost_rets)/2.0
        self.results['turn_over']=(top_trade+bot_trade)/2.0
        self.results['avg_turn_over']=self.results['turn_over'].mean()

        if self.params['benchmark'] is not None:
            self.results['ex_returns']=(self.bot_returns-self.index_returns)/1.2
            self.results['ex_net_returns']=self.results['ex_returns']-bot_trade_cost_rets/1.2
        if self.params['ir_details'] == True:
            self.ranking=ranking

    def calc_ir(self):
        self.results['Annualized Return']=self.params['n_trading_days']*self.results['ls_returns'].mean()
        self.results['Annualized Volatility']=np.sqrt(self.params['n_trading_days'])*self.results['ls_returns'].std()
        if self.results['Annualized Volatility']>0:
            self.results['Annualized IR']=self.results['Annualized Return']/self.results['Annualized Volatility']
        else:
            self.results['Annualized IR']=np.nan
        nav=(1+self.results['ls_returns']).cumprod()
        drawdown=nav/nav.cummax()-1
        self.results['Maximum Drawdown']=-drawdown.min()
        
        self.results['Annualized Net Return']=self.params['n_trading_days']*self.results['ls_net_returns'].mean()
        self.results['Annualized Net Volatility']=np.sqrt(self.params['n_trading_days'])*self.results['ls_net_returns'].std()
        if self.results['Annualized Net Volatility']>0:
            self.results['Annualized Net IR']=self.results['Annualized Net Return']/self.results['Annualized Net Volatility']
        else:
            self.results['Annualized Net IR']=np.nan
        net_nav=(1+self.results['ls_net_returns']).cumprod()
        drawdown=net_nav/net_nav.cummax()-1
        self.results['Maximum Net Drawdown']=-drawdown.min()

        if self.params['benchmark'] is not None:
            self.results['Annualized Ex Return']=self.params['n_trading_days']*self.results['ex_returns'].mean()
            vol=np.sqrt(self.params['n_trading_days'])*self.results['ex_returns'].std()
            if vol>0:
                self.results['Annualized Ex IR']=self.results['Annualized Ex Return']/vol
            else:
                self.results['Annualized Ex IR']=np.nan
            self.results['Annualized Net Ex Return']=self.params['n_trading_days']*self.results['ex_net_returns'].mean()
            vol=np.sqrt(self.params['n_trading_days'])*self.results['ex_net_returns'].std()
            if vol>0:
                self.results['Annualized Net Ex IR']=self.results['Annualized Net Ex Return']/vol
            else:
                self.results['Annualized Net Ex IR']=np.nan


    def calc_ir_details(self):
        ls_returns = self.results['ls_returns']
        years = ls_returns.index.year
        yr_list = np.unique(years)

        self.results['return_details']={}
        self.results['ir_details']={}
        for yr in yr_list:
            rets = ls_returns[years==yr]
            if len(rets)>2:
                label='year%d' % yr
                self.results['return_details'][label]=self.params['n_trading_days']*rets.mean()
                vol=np.sqrt(self.params['n_trading_days'])*rets.std()
                if vol>0:
                    self.results['ir_details'][label]=self.results['return_details'][label]/vol
                else:
                    self.results['ir_details'][label]=np.nan

        if self.params['start_os_date'] is not None:
            rets = self.results['ls_returns'][self.params['start_os_date']:self.params['end_date']]
            if len(rets)>2:
                self.results['return_details']['os']=self.params['n_trading_days']*rets.mean()
                vol=np.sqrt(self.params['n_trading_days'])*rets.std()
                if vol>0:
                    self.results['ir_details']['os']=self.results['return_details']['os']/vol
                else:
                    self.results['ir_details']['os']=np.nan
                rets = self.results['ls_returns'][self.params['start_date']:self.params['start_os_date']]
                self.results['return_details']['is']=self.params['n_trading_days']*rets.mean()
                vol=np.sqrt(self.params['n_trading_days'])*rets.std()
                if vol>0:
                    self.results['ir_details']['is']=self.results['return_details']['is']/vol
                else:
                    self.results['ir_details']['is']=np.nan

    def calc_ic(self):
        n_horizon = self.params['ic_return_horizon']
        factors=self.factor_data.shift(self.params['lag_periods'])
        if n_horizon>1:
            factors=factors.shift(n_horizon)
        if n_horizon>1:
            returns=self.returns.rolling(n_horizon,min_periods=int(n_horizon/2)).sum()
        else:
            returns=self.returns
        self.results['ic'] = factors.T.corrwith(returns.T, axis=0)

        self.results['avg_ic']=self.results['ic'].mean()
        self.results['daily_avg_ic']=self.results['avg_ic']/self.params['ic_return_horizon']
        self.results['std_ic']=self.results['ic'].std()
        self.results['icir']=self.results['avg_ic']/self.results['std_ic']
        self.results['anual_icir']=np.sqrt(self.params['n_trading_days']/self.params['ic_return_horizon'])*self.results['icir']

        if self.params['ir_details'] == True:
            ranking = self.ranking.shift(n_horizon)
            return_ranking = returns.rank(axis=1, pct=True)
            self.results['rank_ic'] = ranking.T.corrwith(return_ranking.T, axis=0)
            factors_long = factors.where(ranking>0.5,other=np.nan,inplace=False)
            factors_short = factors.where(ranking<0.5,other=np.nan,inplace=False)
            self.results['ic_long'] = factors_long.T.corrwith(returns.T, axis=0)
            self.results['ic_short'] = factors_short.T.corrwith(returns.T, axis=0)
            self.results['avg_rank_ic']=self.results['rank_ic'].mean()
            self.results['daily_avg_rank_ic']=self.results['rank_ic'].mean()/self.params['ic_return_horizon']
            self.results['std_rank_ic']=self.results['rank_ic'].std()
            self.results['rank_icir']=self.results['avg_rank_ic']/self.results['std_rank_ic']
            self.results['anual_rank_icir']=np.sqrt(self.params['n_trading_days']/self.params['ic_return_horizon'])*self.results['rank_icir']


        if (self.params['ic_return_horizons'] is not None) or (self.params['ic_delay_periods'] is not None):
            ic_return_horizons = [self.params['ic_return_horizon']]
            if self.params['ic_return_horizons'] is not None:
                ic_return_horizons.extend(self.params['ic_return_horizons'])
            ic_delay_periods = [0]
            if self.params['ic_delay_periods'] is not None:
                ic_delay_periods.extend(self.params['ic_delay_periods'])
            self.results['ic_list']={}
            for n_horizon in ic_return_horizons:
                returns=self.returns.rolling(n_horizon,min_periods=int(n_horizon/2)).sum()
                ic_list = {}
                for n_delay in ic_delay_periods:
                    factors=self.factor_data.shift(self.params['lag_periods']+n_delay+n_horizon)
                    ic = factors.T.corrwith(returns.T, axis=0)
                    ic_list['Delay%d' % n_delay] = ic.mean()/self.params['ic_return_horizon']
                self.results['ic_list']['Horizon%d' % n_horizon]=ic_list


    def calc_factor_performance(self):
        if self.params['smooth_periods']>1:
            self.factor_data=self.factor_data.rolling(self.params['smooth_periods'],min_periods=1).mean()
        if self.params['sector_neutral'] == True:
            t1=time.perf_counter()
            self.calc_sector_neutral()
            t2=time.perf_counter()
            print('%4s%-40s%10.4f%4s' % (" ","calc_sector_neutral calculating time:",t2-t1,"sec"))
        if self.params['universe'] is not None:
            self.factor_data = self.factor_data.where(self.universe==1,other=np.nan)
        t1=time.perf_counter()
        self.results['Effective Stocks'] = (~np.isnan(self.factor_data)).sum(axis=1)
        self.calc_factor_returns()
        t3=time.perf_counter()
        print('%4s%-40s%10.4f%4s' % (" ","calc_factor_returns calculating time:",t3-t1,"sec"))
        self.calc_ic()
        t4=time.perf_counter()
        print('%4s%-40s%10.4f%4s' % (" ","calc_ic calculating time:",t4-t3,"sec"))
        self.calc_ir()
        t5=time.perf_counter()
        print('%4s%-40s%10.4f%4s' % (" ","calc_ir calculating time:",t5-t4,"sec"))
        if self.params['ir_details'] == True:
            self.calc_ir_details()
            t6=time.perf_counter()
            print('%4s%-40s%10.4f%4s' % (" ","calc_ir_details calculating time:",t6-t5,"sec"))


    def save(self):
        self.results['params']=self.params
        factor_name=self.params['factor_name']
        factor_data={'factor_property':self.factor_property,'factor_data':self.factor_data}
        with gzip.open(os.path.join(self.params['factor_dir'], 'factor_data_'+factor_name+'.pkl.gz'), 'wb') as file:
            pickle.dump(factor_data, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.params['factor_dir'], 'factor_results_'+factor_name+'.pkl'), 'wb') as file:  
            pickle.dump(self.results, file)

    def display(self):
        print('%s' % '='*82)
        if self.params['factor_type'] is None:
            factor_type='Unknown'
        elif (self.params['factor_type'] == 'market') | (self.params['factor_type'] == 'fundamental') | (self.params['factor_type']  == 'alternative'):
            factor_type=self.params['factor_type']
        else:
            raise ValueError("factor_type not recoganized")
        if self.params['universe'] is None:
            universe='All'
        else:
            universe=self.params['universe']
        if self.params['benchmark'] is None:
            benchmark='None'
        else:
            benchmark=self.params['benchmark']
        if self.params['smooth_periods'] == 0:
            smooth_periods='None'
        else:
            smooth_periods=self.params['smooth_periods']
        print('%-18s%15s%11s%-22s%15s' % ('Factor Name',self.params['factor_name'],' ','factor_type',factor_type))
        if self.params['start_os_date'] is not None:
            start_os_date=datetime.strptime(self.params['start_os_date'].strftime("%Y-%m-%d"),"%Y-%m-%d")
            end_is_date=(start_os_date - timedelta(days=1)).strftime("%Y-%m-%d")
            print('%-18s%15s%11s%-22s%15s' % ('Start IS Date',self.params['start_date'].strftime("%Y-%m-%d"),' ','End IS Date',end_is_date))
            print('%-18s%15s%11s%-22s%15s' % ('Start OS Date',self.params['start_os_date'].strftime("%Y-%m-%d"),' ','End OS Date',self.params['end_date'].strftime("%Y-%m-%d")))
        else:
            print('%-18s%15s%11s%-22s%15s' % ('Start Date',self.params['start_date'].strftime("%Y-%m-%d"),' ','End Date',self.params['end_date'].strftime("%Y-%m-%d")))
        print('%-18s%15s%11s%-22s%15s' % ('Universe',universe,' ','Benchmark',benchmark))
        print('%-18s%15s%11s%-22s%15s' % ('Lag Periods',self.params['lag_periods'],' ','Returns Periods for IC',self.params['ic_return_horizon']))
        print('%-18s%15s%11s%-22s%15s' % ('Smoothing Periods',smooth_periods,' ','Sector Neutral',self.params['sector_neutral']))
    
        print('%s' % '-'*82)
        print('%-21s %8.4f %9s %-28s %8.4f' % ('Annualized Return',self.results['Annualized Return'],' ','Annualized Net Rerturn',self.results['Annualized Net Return']))
        print('%-21s %8.4f %9s %-28s %8.4f'  % ('Annualized IR',self.results['Annualized IR'],' ','Annualized Net IR',self.results['Annualized Net IR']))
        print('%-21s %8.4f %9s %-28s %8.4f'  % ('Maximum Drawdown',self.results['Maximum Drawdown'],' ','Maximum Net Drawdown',self.results['Maximum Net Drawdown']))
        print('%-21s %8.4f %9s '  % ('Average Turnover',self.results['avg_turn_over'],' '))
        if self.params['benchmark'] is not None:
            print('%-21s %8.4f %9s %-28s %8.4f' % ('Annualized Ex Return',self.results['Annualized Ex Return'],' ','Annualized Net Ex Rerturn',self.results['Annualized Net Ex Return']))
            print('%-21s %8.4f %9s %-28s %8.4f'  % ('Annualized Ex IR',self.results['Annualized Ex IR'],' ','Annualized Net Ex IR',self.results['Annualized Net Ex IR']))
        print('%-21s %8.4f %9s %-28s %8.4f'  % ('Daily Average IC',self.results['daily_avg_ic'],' ','Annualized ICIR',self.results['anual_icir']))
        if self.params['ir_details']==True:
            print('%-21s %8.4f %9s %-28s %8.4f'  % ('Daily Average Rank IC',self.results['daily_avg_rank_ic'],' ','Annualized Rank ICIR',self.results['anual_rank_icir']))
        print('%s' % '-'*82)
        if (self.params['ic_return_horizons'] is not None) or (self.params['ic_delay_periods'] is not None):
            s='%-20s' % 'Daily Average IC'
            h = list(self.results['ic_list'].keys())[0]
            for d in self.results['ic_list'][h]:
                s = s+('%12s'  % d)
            print(s)
            for h in self.results['ic_list']:
                s='%-20s'  % h
                for d in self.results['ic_list'][h]:
                    s = s+'%12.4f'  % (self.results['ic_list'][h][d])
                print(s)
            print('%s' % '-'*82)
        if self.params['ir_details']==True:
            print('%15s%20s%20s' % ('','Annualized Return','Annualized IR'))
            for k in sorted(self.results['ir_details']):
                print('%-15s%20.4f%20.4f' % (k,self.results['return_details'][k],self.results['ir_details'][k]))
            print('%s' % '-'*82)

        if plt is None:
            print('matplotlib plotting skipped (binary not available).')
            return

        fig=plt.figure(figsize=(10,6))
        ax1=fig.add_subplot(221)
        ax2=fig.add_subplot(222)
        ax3=fig.add_subplot(223)
        ax4=fig.add_subplot(224)
        ax1.plot(self.results['ls_returns'].cumsum(),label='QT Cum Returns')
        ax1.plot(self.results['ls_net_returns'].cumsum(),label='QT Net Cum Returns')
        ax1.legend()
        if self.params['n_group']>1:
            ax2.plot(self.results['group_returns'].cumsum(),label=list(self.results['group_returns'].columns))
            ax2.legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,1))
        ax3.plot(self.results['ic'].cumsum(),label='Cum IC')
        if self.params['ir_details'] == True:
            ax3.plot(self.results['ic_long'].cumsum(),label='Cum IC Long')
            ax3.plot(self.results['ic_short'].cumsum(),label='Cum IC Short')
            ax3.plot(self.results['rank_ic'].cumsum(),label='Cum Rank IC')
        ax4.plot(self.results['Effective Stocks'],label='Number of Effective Data')
        ax3.legend()
        ax4.legend()
        plt.show()

    def set_factor_property(self):
        return {}
 
    @abstractmethod
    def calc_factor(self):
        raise NotImplementedError
    
   
if __name__ == '__main__':

    from analysis import *

    class my_factor1(factor):
        def set_factor_property(self):
            factor_property={'factor_name': 'alpha101-03',
                             'factor_type': 'market',
                             'data_needed': ['open','volume'],
                             'lag_periods': 2, #1,
                             'ic_return_horizon': 1, # 1,20,65
                             'universe': 'top75pct', # None,'cs500',
                             'benchmark': None,#'cs500', #None,
                             'smooth_periods': 0, #10,
                             'sector_neutral': False #True
                            }
            return factor_property

        def calc_factor(self):
            opens=self.data['open']
            volumes=self.data['volume']
            self.factor_data=-ts_corr1(cs_rank(opens,pct=True),cs_rank(volumes,pct=True),10)

    class my_factor2(factor):
        def set_factor_property(self):
            factor_property={'factor_name': 'strange',
                             'factor_type': 'market',
                             'data_needed': ['pe'],
                            }
            return factor_property

        def calc_factor(self):
            self.factor_data=1/self.data['pe']
            
         
    #params={'ic_return_horizons': [1,5,60], 'ic_delay_periods': [1,2,5],'ir_details': True}
    #params={'start_os_date': '2022-01-01','ir_details': True}
    params={'ir_details': True}
    #params={}
    f=my_factor1(params=params)
    f.run()
