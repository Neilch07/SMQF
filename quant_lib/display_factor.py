from typing import Dict, List, Optional, Union
from attrs import define, field
import pickle
import os
from matplotlib import pyplot as plt

@define
class display_factor():
    __stock_factor_dir='.\\data\\cn\\equity\\factor\\'
    __future_factor_dir='.\\data\\cn\\futures\\factor\\'
    stock_factor_dir: str = field(default=None, kw_only=True)
    future_factor_dir: str = field(default=None, kw_only=True)
    stock_future: bool = field(default='stock', kw_only=True)  # 'stock' or 'future'
    factor_displayed: Union[str,List[str]] = field(default=[], kw_only=True)
    factor_dir: str = field(default=None, init=False)
    factor_results: Dict[str,Dict] = field(default={}, init=False)
    params: Dict[str,Union[int, float, str]] = field(default=None, kw_only=True)
    results: Dict[str,Union[int, float, str]] = field(default={}, init=False)

    def __attrs_post_init__(self):
        if self.stock_factor_dir is None:
            self.stock_factor_dir = self.__stock_factor_dir
        if self.future_factor_dir is None:
            self.future_factor_dir = self.__future_factor_dir
        if self.stock_future == 'stock':
            self.factor_dir = self.__stock_factor_dir
        elif self.stock_future == 'future':
            self.factor_dir = self.__future_factor_dir
        else:
            raise ValueError("stock_future must be 'stock' or 'future'")

    def load_data(self):
        if type(self.factor_displayed) is str:
            if self.factor_displayed=='all':
                pass
            else:
                file_name = os.path.join(self.factor_dir, 'factor_results_'+self.factor_displayed+'.pkl')
                with open(file_name, 'rb') as file:
                    self.factor_results[self.factor_displayed] = pickle.load(file)
                    print((self.factor_results[self.factor_displayed].keys()))
        elif type(self.factor_displayed) is list:
            pass
    
    def display_one_factor(self,factor_displayed=None):
        if factor_displayed is not None:
            self.factor_displayed = factor_displayed
        self.load_data()
        self.params = self.factor_results[self.factor_displayed]['params']
        self.results = self.factor_results[self.factor_displayed]
        print('%s' % '='*80)
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
        print('%-15s%15s%11s%-22s%15s' % ('Factor Name',self.params['factor_name'],' ','Universe',universe))
        print('%-15s%15s%11s%-22s%15s' % ('Start Date',self.params['start_date'].strftime("%Y-%m-%d"),' ','Benchmark',benchmark))
        if self.params['start_os_date'] is not None:
            print('%-15s%15s%11s%-22s%15s' % ('Start OS Date',self.params['start_os_date'].strftime("%Y-%m-%d"),' ','Smoothing Periods',smooth_periods))
        else:
            print('%-15s%15s%11s%-22s%15s' % ('Start OS Date','None',' ','Smoothing Periods',smooth_periods))
        print('%-15s%15s%11s%-22s%15s' % ('End Date',self.params['end_date'].strftime("%Y-%m-%d"),' ','Sector Neutral',self.params['sector_neutral']))
        print('%-15s%15s%11s%-22s%15s' % ('Lag Periods',self.params['lag_periods'],' ','Returns Periods for IC',self.params['ic_return_horizon']))

        print('%s' % '-'*80)
        print('%-21s %8.4f %9s %-28s %8.4f' % ('Annualized Return',self.results['Annualized Return'],' ','Annualized Net Rerturn',self.results['Annualized Net Return']))
        print('%-21s %8.4f %9s %-28s %8.4f'  % ('Annualized IR',self.results['Annualized IR'],' ','Annualized Net IR',self.results['Annualized Net IR']))
        print('%-21s %8.4f %9s %-28s %8.4f'  % ('Maximum Drawdown',self.results['Maximum Drawdown'],' ','Maximum Net Drawdown',self.results['Maximum Net Drawdown']))
        if self.params['benchmark'] is not None:
            print('%-21s %8.4f %9s %-28s %8.4f' % ('Annualized Ex Return',self.results['Annualized Ex Return'],' ','Annualized Net Ex Rerturn',self.results['Annualized Net Ex Return']))
            print('%-21s %8.4f %9s %-28s %8.4f'  % ('Annualized Ex IR',self.results['Annualized Ex IR'],' ','Annualized Net Ex IR',self.results['Annualized Net Ex IR']))
        print('%-21s %8.4f %9s %-28s %8.4f'  % ('Daily Average IC',self.results['daily_avg_ic'],' ','Annualized ICIR',self.results['anual_icir']))
        if self.params['ir_details']==True:
            print('%-21s %8.4f %9s %-28s %8.4f'  % ('Daily Average Rank IC',self.results['daily_avg_rank_ic'],' ','Annualized Rank ICIR',self.results['anual_rank_icir']))
        print('%s' % '-'*80)
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
            print('%s' % '-'*80)
        if self.params['ir_details']==True:
            print('%15s%20s%20s' % ('','Annualized Return','Annualized IR'))
            for k in sorted(self.results['ir_details']):
                print('%-15s%20.4f%20.4f' % (k,self.results['return_details'][k],self.results['ir_details'][k]))
            print('%s' % '-'*80)

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


if __name__ == '__main__':

    df = display_factor(factor_displayed='alpha101_04')
    df.display_one_factor()