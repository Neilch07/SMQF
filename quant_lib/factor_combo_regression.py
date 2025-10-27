from typing import Dict, List, Optional, Union
from attrs import define, field
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import time
from factor_combo_base import factor_combo_base
from sklearn.linear_model import LinearRegression,ElasticNet,ElasticNetCV,Ridge,Lasso
from linearmodels import FamaMacBeth

    
        
@define
class factor_combo_regression(factor_combo_base):
    __params = {'testing_periods': 'rolling', # 0,1,'rolling'
                'testing_period_length': 63, # int
                'training_period_length': 125, # 'auto', int, 'all'
                }  
    __combo_property = {'regression_method': 'pooled_ols', # 'pooled_ols','avg_ols','fama_macb_ols','pooled_elastic_net','avg_elastic_net','fama_macb_elastic_net'
                        'normalization_method': 'zscore', # None,'zscore','minmax','tanh','sigmoid'
                        'alpha': 1, # coefficient for elastic_net， float or list of possible values to choose
                        'l1_ratio': 0.1, # mix of L1/L2 penalties (1 = LASSO, 0 = Ridge), float bwtween 0 and 1, or list of possible values to choose
                        }        
    
    def __attrs_post_init__(self):
        self.__combo_property.update(self.set_combo_property())
        if self.combo_property is not None:
            self.__combo_property.update(self.combo_property)
        self.combo_property = self.__combo_property
        if self.params is not None:
            self.__params.update(self.params)
        self.params = self.__params
        super().__attrs_post_init__()

    def dict_to_df(self,df_data):
        df_data = pd.DataFrame({k: v.stack(dropna=False) for k, v in df_data.items()})
        df_data.index.names = ['date', 'stock']
        return df_data

    def df_to_dict(self,df_data):
        df_data = {col: df_data[col].unstack(level='date').T for col in df_data.columns}
        return df_data
    
    def pooled_ols(self,training_factors_data,testing_factors_data):
        x_list=list(training_factors_data.columns)
        x_list.remove('training_returns')
        x_data=training_factors_data.dropna()
        X=x_data[x_list]
        y=x_data['training_returns']
        model = LinearRegression()
        reg_results = model.fit(X, y)
        train_pred = reg_results.predict(X)
        train_pred = pd.DataFrame(train_pred,index=x_data.index,columns=['predicted_return'])
        train_pred=train_pred.reindex(training_factors_data.index)
        x_data=testing_factors_data.dropna()
        X=x_data[x_list]
        test_pred = reg_results.predict(X)
        test_pred = pd.DataFrame(test_pred,index=x_data.index,columns=['predicted_return'])
        test_pred=test_pred.reindex(testing_factors_data.index)
        train_pred=self.df_to_dict(train_pred)
        test_pred=self.df_to_dict(test_pred)
        return (train_pred['predicted_return'],test_pred['predicted_return'])
    
    def avg_ols(self,training_factors_data,testing_factors_data):
        x_data=training_factors_data.dropna()
        cross_section = x_data.groupby('stock').mean().reset_index()
        x_list=list(training_factors_data.columns)
        x_list.remove('training_returns')
        X=cross_section[x_list]
        y=cross_section['training_returns']
        model = LinearRegression()
        reg_results = model.fit(X, y)
        X=x_data[x_list]
        train_pred = reg_results.predict(X)
        train_pred = pd.DataFrame(train_pred,index=x_data.index,columns=['predicted_return'])
        train_pred=train_pred.reindex(training_factors_data.index)
        x_data=testing_factors_data.dropna()
        X=x_data[x_list]
        test_pred = reg_results.predict(X)
        test_pred = pd.DataFrame(test_pred,index=x_data.index,columns=['predicted_return'])
        test_pred=test_pred.reindex(testing_factors_data.index)
        train_pred=self.df_to_dict(train_pred)
        test_pred=self.df_to_dict(test_pred)
        return (train_pred['predicted_return'],test_pred['predicted_return'])
    
    def fama_macb_ols2(self,training_factors_data,testing_factors_data): # using linearmodels.FamaMacBeth
        training_factors_data=training_factors_data.swaplevel(0, 1)
        testing_factors_data=testing_factors_data.swaplevel(0, 1)
        x_data=training_factors_data.dropna()
        x_list=list(training_factors_data.columns)
        x_list.remove('training_returns')
        fm = 'training_returns ~ ' + ' + '.join(x_list)
        model = FamaMacBeth.from_formula(fm, data=x_data)
        reg_results = model.fit(cov_type='kernel')
        X=x_data[x_list]
        if 'Intercept' in reg_results.params:
            X['const'] = 1
        train_pred = reg_results.predict(data=X)
        train_pred = train_pred.reindex(x_data.index)
        train_pred = train_pred.rename(columns={'predictions': 'predicted_return'})
        train_pred=train_pred.swaplevel(0, 1)
        train_pred=train_pred.reindex(training_factors_data.index)
        train_pred=self.df_to_dict(train_pred)

        x_data=testing_factors_data.dropna()
        X=x_data[x_list]
        if 'Intercept' in reg_results.params:
            X['const'] = 1
        test_pred = reg_results.predict(X)
        test_pred = test_pred.reindex(x_data.index)
        test_pred = test_pred.rename(columns={'predictions': 'predicted_return'})
        test_pred=test_pred.swaplevel(0, 1)
        test_pred=test_pred.reindex(testing_factors_data.index)
        test_pred=self.df_to_dict(test_pred)

        return (train_pred['predicted_return'],test_pred['predicted_return'])
    
    def fama_macb_ols(self,training_factors_data,testing_factors_data):
        x_list=list(training_factors_data.columns)
        x_list.remove('training_returns')
        x_data=training_factors_data.dropna()
        coefs=[]
        for dt in x_data.index.get_level_values('date').unique():
            df_date = x_data.xs(dt, level='date')
            X=df_date[x_list]
            y=df_date['training_returns']
            model = LinearRegression()
            reg_results = model.fit(X, y)
            coefs.append(pd.Series(model.coef_, index=X.columns, name=dt))
        coef_df = pd.concat(coefs, axis=1).T
        fm_coef = coef_df.mean()
        X=x_data[x_list]
        train_pred = X.dot(fm_coef) + y.mean()
        train_pred = pd.DataFrame(train_pred,index=x_data.index,columns=['predicted_return'])
        train_pred=train_pred.reindex(training_factors_data.index)
        x_data=testing_factors_data.dropna()
        X=x_data[x_list]
        test_pred = X.dot(fm_coef) + y.mean()
        test_pred = pd.DataFrame(test_pred,index=x_data.index,columns=['predicted_return'])
        test_pred=test_pred.reindex(testing_factors_data.index)
        train_pred=self.df_to_dict(train_pred)
        test_pred=self.df_to_dict(test_pred)
        return (train_pred['predicted_return'],test_pred['predicted_return'])
    
    def pooled_elastic_net(self,training_factors_data,testing_factors_data):
        alpha=self.params['alpha']
        l1_ratio=self.params['l1_ratio']
        if (type(alpha) == float or type(alpha) == int) and type(l1_ratio) == list:
            alpha = [alpha]
        elif (type(l1_ratio) == float or type(l1_ratio) == int) and type(alpha) == list:
            l1_ratio = [l1_ratio]

        x_list=list(training_factors_data.columns)
        x_list.remove('training_returns')
        x_data=training_factors_data.dropna()
        X=x_data[x_list]
        y=x_data['training_returns']
        if (type(alpha) == float or type(alpha) == int) and (type(l1_ratio) == float or type(l1_ratio) == int):
            if l1_ratio == 0:
                model = Ridge(alpha=alpha)
            elif l1_ratio == 1:
                alpha = alpha/len(y)   # objective function is different from Ridge, divided by sample size
                model = Lasso(alpha=alpha)
            else:
                n = len(y)
                alpha = [x / n for x in alpha]  # objective function is different from Ridge, divided by sample size
                model = ElasticNet(alpha=alpha,l1_ratio=l1_ratio)
        elif type(alpha) == list and type(l1_ratio) == list:
            model = ElasticNetCV(alphas=alpha,l1_ratio=l1_ratio,cv=2)
        else:
            raise ValueError("alpha or l1_ratio is incorrect type")
        reg_results = model.fit(X, y)
        if (type(alpha) == float or type(alpha) == int) and type(l1_ratio) == float or type(l1_ratio) == int:
            self.combo_property['alpha_used']=alpha
            self.combo_property['l1_ratio_used']=l1_ratio
        else:
            self.combo_property['alpha_used']=reg_results.alpha_
            self.combo_property['l1_ratio_used']=reg_results.l1_ratio_
        train_pred = reg_results.predict(X)
        train_pred = pd.DataFrame(train_pred,index=x_data.index,columns=['predicted_return'])
        train_pred=train_pred.reindex(training_factors_data.index)
        x_data=testing_factors_data.dropna()
        X=x_data[x_list]
        test_pred = reg_results.predict(X)
        test_pred = pd.DataFrame(test_pred,index=x_data.index,columns=['predicted_return'])
        test_pred=test_pred.reindex(testing_factors_data.index)
        train_pred=self.df_to_dict(train_pred)
        test_pred=self.df_to_dict(test_pred)
        return (train_pred['predicted_return'],test_pred['predicted_return'])
    
    def avg_elastic_net(self,training_factors_data,testing_factors_data):
        alpha=self.params['alpha']
        l1_ratio=self.params['l1_ratio']
        if (type(alpha) == float or type(alpha) == int) and type(l1_ratio) == list:
            alpha = [alpha]
        elif (type(l1_ratio) == float or type(l1_ratio) == int) and type(alpha) == list:
            l1_ratio = [l1_ratio]

        x_data=training_factors_data.dropna()
        cross_section = x_data.groupby('stock').mean().reset_index()
        x_list=list(training_factors_data.columns)
        x_list.remove('training_returns')
        X = cross_section[x_list]
        y = cross_section['training_returns']
        if (type(alpha) == float or type(alpha) == int) and (type(l1_ratio) == float or type(l1_ratio) == int):
            if l1_ratio == 0:
                model = Ridge(alpha=alpha)
            elif l1_ratio == 1:
                alpha=alpha/len(y)    # objective function is different from Ridge, divided by sample size
                model = Lasso(alpha=alpha)
            else:
                n = len(y)
                alpha = [x / n for x in alpha]     # objective function is different from Ridge, divided by sample size
                model = ElasticNet(alpha=alpha,l1_ratio=l1_ratio)
        elif type(alpha) == list and type(l1_ratio) == list:
            model = ElasticNetCV(alphas=alpha,l1_ratio=l1_ratio,cv=2)
        else:
            raise ValueError("alpha or l1_ratio is incorrect type")
        reg_results = model.fit(X, y)
        self.combo_property['alpha_used']=reg_results.alpha_
        self.combo_property['l1_ratio_used']=reg_results.l1_ratio_
        X=x_data[x_list]
        train_pred = reg_results.predict(X)
        train_pred = pd.DataFrame(train_pred,index=x_data.index,columns=['predicted_return'])
        train_pred=train_pred.reindex(training_factors_data.index)
        x_data=testing_factors_data.dropna()
        X=x_data[x_list]
        test_pred = reg_results.predict(X)
        test_pred = pd.DataFrame(test_pred,index=x_data.index,columns=['predicted_return'])
        test_pred=test_pred.reindex(testing_factors_data.index)
        train_pred=self.df_to_dict(train_pred)
        test_pred=self.df_to_dict(test_pred)
        return (train_pred['predicted_return'],test_pred['predicted_return'])
    
    def fama_macb_elastic_net(self,training_factors_data,testing_factors_data):
        alpha=self.params['alpha']
        l1_ratio=self.params['l1_ratio']
        x_list=list(training_factors_data.columns)
        x_list.remove('training_returns')
        x_data=training_factors_data.dropna()
        coefs=[]
        for dt in x_data.index.get_level_values('date').unique():
            df_date = x_data.xs(dt, level='date')
            X=df_date[x_list]
            y=df_date['training_returns']
            if (type(alpha) == float or type(alpha) == int) and (type(l1_ratio) == float or type(l1_ratio) == int):
                if l1_ratio == 0:
                    model = Ridge(alpha=alpha)
                elif l1_ratio == 1:
                    alpha=alpha/len(y)
                    model = Lasso(alpha=alpha)
                else:
                    n = len(y)
                    alpha = [x / n for x in alpha]
                    model = ElasticNet(alpha=alpha,l1_ratio=l1_ratio)
            else:
                raise ValueError("alpha or l1_ratio is incorrect type for fama_macb_elastic_net")
            reg_results = model.fit(X, y)
            coefs.append(pd.Series(model.coef_, index=X.columns, name=dt))
        coef_df = pd.concat(coefs, axis=1).T
        fm_coef = coef_df.mean()
        X_scaled = (X - X.mean()) / (1.0e-10+X.std()) 
        train_pred = X_scaled.dot(fm_coef) + y.mean()
        train_pred = pd.DataFrame(train_pred,index=x_data.index,columns=['predicted_return'])
        train_pred=train_pred.reindex(training_factors_data.index)
        x_data=testing_factors_data.dropna()
        X=x_data[x_list]
        X_scaled = (X - X.mean()) / (1.0e-10+X.std()) 
        test_pred = X_scaled.dot(fm_coef) + y.mean()
        test_pred = pd.DataFrame(test_pred,index=x_data.index,columns=['predicted_return'])
        test_pred=test_pred.reindex(testing_factors_data.index)
        train_pred=self.df_to_dict(train_pred)
        test_pred=self.df_to_dict(test_pred)
        return (train_pred['predicted_return'],test_pred['predicted_return'])

    def calc_subperiod_combo_positions(self,i0,i1,i2):
        training_factors_data={}
        testing_factors_data={}     
        lag_periods=self.combo_property['lag_periods']
        for tk in self.factors_data:
            training_factors_data[tk]=self.factors_data[tk].iloc[i0:i1,:].shift(lag_periods)
            testing_factors_data[tk]=self.factors_data[tk].iloc[i1:i2,:].shift(lag_periods)
        training_factors_data['training_returns']=self.returns[i0:i1]
        training_factors_data=self.dict_to_df(training_factors_data)
        testing_factors_data=self.dict_to_df(testing_factors_data)

        if self.params['regression_method'] == 'pooled_ols':
            training_pred_rets,testing_pred_rets = self.pooled_ols(training_factors_data,testing_factors_data)
        elif self.params['regression_method'] == 'avg_ols':
            training_pred_rets,testing_pred_rets = self.avg_ols(training_factors_data,testing_factors_data)
        elif self.params['regression_method'] == 'fama_macb_ols':
            training_pred_rets,testing_pred_rets = self.fama_macb_ols(training_factors_data,testing_factors_data)
        elif self.params['regression_method'] == 'pooled_elastic_net':
            training_pred_rets,testing_pred_rets = self.pooled_elastic_net(training_factors_data,testing_factors_data)
        elif self.params['regression_method'] == 'avg_elastic_net':
            training_pred_rets,testing_pred_rets = self.avg_elastic_net(training_factors_data,testing_factors_data)
        elif self.params['regression_method'] == 'fama_macb_elastic_net':
            training_pred_rets,testing_pred_rets = self.fama_macb_elastic_net(training_factors_data,testing_factors_data)
        else:
            raise ValueError("regression_method not recoganized")
        lag_periods=self.combo_property['lag_periods']
        training_pos=self.calc_rank_pos(lag_periods,training_pred_rets)
        testing_pos=self.calc_rank_pos(lag_periods,testing_pred_rets)
        return (training_pos,testing_pos)
    

            
if __name__ == '__main__':


    params={'testing_periods': 'rolling', 'testing_period_length': 120}
    #combo_property={'factors_needed':['alpha101_04','alpha101_05'],'benchmark':'zz500','ir_details':True}

    alphas=[0.1,1,10]
    l1_ratios=[0,0.5,1]
    combo_property={'factors_needed':['alpha101_04','alpha101_05']} 
    combo_property['regression_method']='fama_macb_elastic_net'
    # 'pooled_ols','avg_ols','fama_macb_ols','pooled_elastic_net','avg_elastic_net','fama_macb_elastic_net'
    combo_property['alpha']=0.1
    combo_property['l1_ratios']=0
    f = factor_combo_regression(combo_property=combo_property)
    f.run()
