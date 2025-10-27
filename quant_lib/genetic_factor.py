from typing import Dict, List, Optional, Union
from attrs import define, field
import operator
import random
import os
import numpy as np
import pandas as pd
import time
from math import isnan,sqrt
from functools import partial
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import geppy
from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness
import matplotlib.pyplot as plt

# Define new functions
def add(x,y):
    return x+y

def sub(x,y):
    return x-y

def mul(x,y):
    return x*y

def div(x,y):
    if type(y) == int or type(y) == float:
        if y==0:
            return np.nan
    else:
        y[y==0] = 1e-10
    return x/y

def neg(x):
    return -x

def rank(df):
    if type(df) == int or type(df) == float:
        return np.nan
    return df.rank(axis=1, pct=True)

def rank_gp(x):
    global nd,ns
    if type(x) == int or type(x) == float:
        return np.nan
    if nd*ns != len(x):
        return x
    df = pd.DataFrame(x.reshape(nd,ns))
    y = df.rank(axis=1, pct=True).values.reshape(nd*ns,)
    return y

def diff5(df):
    if type(df) == int or type(df) == float:
        return np.nan
    return df-df.shift(5)

def diff5_gp(x):
    global nd,ns
    if type(x) == int or type(x) == float:
        return np.nan
    if nd*ns != len(x):
        return x
    df = pd.DataFrame(x.reshape(nd,ns))
    y = (df-df.shift(5)).values.reshape(nd*ns,)
    return y

def ts_max5(df):
    if type(df) == int or type(df) == float:
        return np.nan
    return df.rolling(5).max()

def ts_max5_gp(x):
    global nd,ns
    if type(x) == int or type(x) == float:
        return np.nan
    if nd*ns != len(x):
        return x
    df = pd.DataFrame(x.reshape(nd,ns))
    y = (df.rolling(5).max()).values.reshape(nd*ns,)
    return y

def ts_min5(df):
    if type(df) == int or type(df) == float:
        return np.nan
    return df.rolling(5).min()

def ts_min5_gp(x):
    global nd,ns
    if type(x) == int or type(x) == float:
        return np.nan
    if nd*ns != len(x):
        return x
    df = pd.DataFrame(x.reshape(nd,ns))
    y = (df.rolling(5).min()).values.reshape(nd*ns,)
    return y

def ts_ma5(df):
    if type(df) == int or type(df) == float:
        return np.nan
    return df.rolling(5).mean()

def ts_ma5_gp(x):
    global nd,ns
    if type(x) == int or type(x) == float:
        return np.nan
    if nd*ns != len(x):
        return x
    df = pd.DataFrame(x.reshape(nd,ns))
    y = (df.rolling(5).mean()).values.reshape(nd*ns,)
    return y

@define
class genetic_factor():
    __params = {'gen_package': 'deap', #
                'start_date': None, # if None, using the start of the whole data
                'end_date': None, # if None, using the end of the whole data    
                'data_needed': ['close','open','high','low','amount'],
                'use_small_sample': False, # use a small sample for debug, if True,  start_date is ignored          
                'lag_periods': 2, # number of delay periods to forecast
                'ic_return_horizon': 1, # number of periods to calculate returns for ic
                'universe': None, # options are None, 'Top80','Top60', 'Top40', 'HS300','ZZ500','ZZ800','ZZ1000'
                'benchmark': None, # options are None,'HS300','ZZ500','ZZ1000'
                'trading_cost': 0.0012, # trading costs in terms of two way costs in returns
                'quantile': 0.2,
                'n_trading_days': 243, # number of trading days per year              
                'display': True, 
                'save': False, # save the results to files
                'data_dir': '.\\data\\cn\\equity\\data\\',
                'factor_dir': '.\\data\\cn\\equity\\factor\\genetic\\',
                'fitness': 'ir', # 'ir','net_ir','excess_ir','excess_net_ir','icir','returns','net_returns'
                'number_population': 100,
                'number_generation': 10,
                'number_hall_of_fame': 10,
                'head_length': 20, # for geppy
                'rnc_array_length': 20, # for geppy
                'number_gene': 2, # for geppy
                'display_fitness': False
                }  
    params: Dict[str,Union[int, float, str]] = field(default=None, kw_only=True)
    data_needed: List[str] = field(default=[], init=False)
    data: Dict[str,pd.DataFrame] = field(default={}, init=False)  
    factor_data: pd.DataFrame = field(default=None, init=False)
    returns: pd.DataFrame = field(default=None, init=False)
    universe: pd.DataFrame = field(default=None, init=False)
    index_returns: pd.DataFrame = field(default=None, init=False)
    results: Dict[str,Union[int, float, str]] = field(default={}, init=False)
    factor_formula: List[str] = field(default=[], init=False)
    fits: List[Union[int, float]] = field(default=[], init=False)
    factors: List[pd.DataFrame] = field(default=[], init=False)
    fit_log: object = field(default=None, init=False)

    def __attrs_post_init__(self):
        if self.params is not None:
            self.__params.update(self.params)
        self.params = self.__params

    def load_data(self):
        for fld in self.params['data_needed']:
            self.data[fld]=pd.read_pickle(os.path.join(self.params['data_dir'], fld+'.pkl'))
            self.data[fld].index=pd.to_datetime(self.data[fld].index)
        if 'returns' not in self.params['data_needed']:
            self.returns = pd.read_pickle(os.path.join(self.params['data_dir'],'returns.pkl'))
            self.returns.index=pd.to_datetime(self.returns.index)
        else:
            self.returns = self.data['returns']
        if self.params['universe'] is not None:
            if self.params['universe'] not in self.data_needed:
                self.universe = pd.read_pickle(os.path.join(self.params['data_dir'],self.factor_property['universe']+'.pkl'))
                self.universe.index=pd.to_datetime(self.universe.index)
            else:
                self.universe = self.data['universe']
        if self.params['benchmark'] is not None:
            if 'index_data' not in self.data_needed:
                index_data = pd.read_pickle(os.path.join(self.params['data_dir'],'index_data.pkl'))
                index_data.index=pd.to_datetime(index_data.index)
            else:
                index_data = self.data['index_data']
            self.index_returns = index_data[self.factor_property['benchmark']+'_returns']

        if self.params['use_small_sample'] == True:
            for fld in self.params['data_needed']:
                self.data[fld] = self.data[fld].iloc[-100:,:10]
            self.returns = self.returns.iloc[-100:,:10]
            if self.params['universe'] is not None:
                self.universe = self.universe.iloc[-100:,:10]
            if self.params['benchmark'] is not None:
                self.index_returns = self.index_returns.iloc[-100:,:10]
        else:
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
            for d in self.self.params['data_needed']:
                self.data[d] = self.data[d][start_date:end_date]
            if self.returns is not None:
                self.returns=self.returns[start_date:end_date]
            if self.universe is not None:
                self.universe=self.universe[start_date:end_date]
            if self.index_returns is not None:
                self.index_returns=self.index_returns[start_date:end_date]
        self.params['start_date'] = start_date
        self.params['end_date'] = end_date

    def calc_performance(self,factor_data):
        factors = factor_data.shift(self.params['lag_periods'])
        ranking = factors.rank(axis=1, pct=True)
        qt=self.params['quantile']

        top_qt = ranking<qt
        bot_qt = ranking>1-qt
        top_returns=self.returns[top_qt].mean(axis=1)
        bot_returns=self.returns[bot_qt].mean(axis=1)
        ls_returns=(bot_returns-top_returns)/2.0
        nav=(1+ls_returns).cumprod()
        drawdown=nav/nav.cummax()-1
        max_drawdown=-drawdown.min()

        top_pos = top_qt.astype(int).divide(top_qt.sum(axis=1),axis=0)
        top_trade = top_pos.fillna(0).diff(axis=0).abs().sum(axis=1)
        bot_pos = bot_qt.astype(int).divide(bot_qt.sum(axis=1),axis=0)
        bot_trade = bot_pos.fillna(0).diff(axis=0).abs().sum(axis=1)
        top_trade_cost_rets = 0.5*self.params['trading_cost']*top_trade
        bot_trade_cost_rets = 0.5*self.params['trading_cost']*bot_trade
        ls_net_returns=ls_returns-(top_trade_cost_rets+bot_trade_cost_rets)/2.0
        nav=(1+ls_returns).cumprod()
        drawdown=nav/nav.cummax()-1
        max_net_drawdown=-drawdown.min()

        annualized_return=self.params['n_trading_days']*ls_returns.mean()
        annualized_volatility=np.sqrt(self.params['n_trading_days'])*ls_returns.std()
        if annualized_volatility>0:
            annualized_ir=annualized_return/annualized_volatility
        else:
            annualized_ir=np.nan
        annualized_net_return=self.params['n_trading_days']*ls_net_returns.mean()
        annualized_net_volatility=np.sqrt(self.params['n_trading_days'])*ls_net_returns.std()
        if annualized_net_volatility>0:
            annualized_net_ir=annualized_net_return/annualized_net_volatility
        else:
            annualized_net_ir=np.nan
        return annualized_return,annualized_ir,max_drawdown,annualized_net_return,annualized_net_ir,max_net_drawdown

    
    def calc_ir(self,factor_data):
        factors = factor_data.shift(self.params['lag_periods'])
        ranking = factors.rank(axis=1, pct=True)
        qt=self.params['quantile']
        
        top_qt = ranking<qt
        bot_qt = ranking>1-qt
        top_returns=self.returns[top_qt].mean(axis=1)
        bot_returns=self.returns[bot_qt].mean(axis=1)
        ls_returns=(bot_returns-top_returns)/2.0

        annualized_return=self.params['n_trading_days']*ls_returns.mean()
        annualized_volatility=np.sqrt(self.params['n_trading_days'])*ls_returns.std()
        if annualized_volatility>0:
            annualized_ir=annualized_return/annualized_volatility
        else:
            annualized_ir=-1.0
        return annualized_return,annualized_ir
    
    def calc_net_ir(self,factor_data):
        factors = factor_data.shift(self.params['lag_periods'])
        ranking = factors.rank(axis=1, pct=True)
        qt=self.params['quantile']

        top_qt = ranking<qt
        bot_qt = ranking>1-qt
        top_returns=self.returns[top_qt].mean(axis=1)
        bot_returns=self.returns[bot_qt].mean(axis=1)
        ls_returns=(bot_returns-top_returns)/2.0

        top_pos = top_qt.astype(int).divide(top_qt.sum(axis=1),axis=0)
        top_trade = top_pos.fillna(0).diff(axis=0).abs().sum(axis=1)
        bot_pos = bot_qt.astype(int).divide(bot_qt.sum(axis=1),axis=0)
        bot_trade = bot_pos.fillna(0).diff(axis=0).abs().sum(axis=1)
        top_trade_cost_rets = 0.5*self.params['trading_cost']*top_trade
        bot_trade_cost_rets = 0.5*self.params['trading_cost']*bot_trade
        ls_net_returns=ls_returns-(top_trade_cost_rets+bot_trade_cost_rets)/2.0

        annualized_return=self.params['n_trading_days']*ls_net_returns.mean()
        annualized_volatility=np.sqrt(self.params['n_trading_days'])*ls_net_returns.std()
        if annualized_volatility>0:
            annualized_ir=annualized_return/annualized_volatility
        else:
            annualized_ir=-1.0
        return annualized_return,annualized_ir

    def calc_excess_ir(self,factor_data):
        factors = factor_data.shift(self.params['lag_periods'])
        ranking = factors.rank(axis=1, pct=True)
        qt=self.params['quantile']

        bot_qt = ranking>1-qt
        bot_returns=self.returns[bot_qt].mean(axis=1)

        ex_returns=(bot_returns-self.index_returns)/1.2
        annualized_return=self.params['n_trading_days']*ex_returns.mean()
        annualized_volatility=np.sqrt(self.params['n_trading_days'])*ex_returns.std()
        if annualized_volatility>0:
            annualized_ir=annualized_return/annualized_volatility
        else:
            annualized_ir=-1.0
        return annualized_return,annualized_ir

    def calc_excess_net_ir(self,factor_data):
        factors = factor_data.shift(self.params['lag_periods'])
        ranking = factors.rank(axis=1, pct=True)
        qt=self.params['quantile']

        bot_qt = ranking>1-qt
        bot_returns=self.returns[bot_qt].mean(axis=1)

        bot_pos = bot_qt.astype(int).divide(bot_qt.sum(axis=1),axis=0)
        bot_trade = bot_pos.fillna(0).diff(axis=0).abs().sum(axis=1)
        bot_trade_cost_rets = 0.5*self.params['trading_cost']*bot_trade

        ex_returns=(bot_returns-self.index_returns)/1.2
        ex_net_returns=ex_returns-bot_trade_cost_rets/1.2
        annualized_return=self.params['n_trading_days']*ex_net_returns.mean()
        annualized_volatility=np.sqrt(self.params['n_trading_days'])*ex_net_returns.std()
        if annualized_volatility>0:
            annualized_ir=annualized_return/annualized_volatility
        else:
            annualized_ir=-1.0
        return annualized_return,annualized_ir
    
    def calc_icir(self,factor_data):
        n_horizon = self.params['ic_return_horizon']
        factors=factor_data.shift(self.params['lag_periods'])
        if n_horizon>1:
            factors=factors.shift(n_horizon)
        if n_horizon>1:
            returns=self.returns.rolling(n_horizon,min_periods=int(n_horizon/2)).sum()
        else:
            returns=self.returns
        ic = factors.T.corrwith(returns.T, axis=0)

        avg_ic=ic.mean()
        std_ic=ic.std()
        if std_ic>0:
            icir=np.sqrt(self.params['n_trading_days']/self.params['ic_return_horizon'])*avg_ic/std_ic
        else:
            icir=-1.0
        return avg_ic,icir


    def calc_fitness(self, factor_data):
        if self.params['fitness'] == 'ir':
            analRet,perf=self.calc_ir(factor_data)
        elif self.params['fitness'] == 'net_ir':
            analRet,perf=self.calc_net_ir(factor_data)
        elif self.params['fitness'] == 'excess_ir':
            analRet,perf=self.calc_excess_ir(factor_data)
        elif self.params['fitness'] == 'excess_net_ir':
            analRet,perf=self.calc_excess_net_ir(factor_data)
        elif self.params['fitness'] == 'icir':
            analRet,perf=self.calc_icir(factor_data)
        elif self.params['fitness'] == 'returns':
            perf,analIR=self.calc_ir(factor_data)
        elif self.params['fitness'] == 'net_returns':
            perf,analIR=self.calc_net_ir(factor_data)
        return perf

    def deap_factor(self):

        def evaluate(individual):
            func = toolbox.compile(individual)
            factor_data = func(closes,highs,lows,opens,amounts)
            if type(factor_data) == float or type(factor_data) == int:
                return -1,
            perf = self.calc_fitness(factor_data)
            if isnan(perf):
                return -1,
            return perf,

        t1=time.perf_counter()
        self.load_data()
        closes = self.data['close']
        highs = self.data['high']
        lows = self.data['low']
        opens = self.data['open']
        amounts = self.data['amount']
        returns = self.returns
        t2=time.perf_counter()
        print('%4s%-40s%10.4f%4s' % (" ","load_data time:",t2-t1,"sec"))
        t1=time.perf_counter()
  
        pset = gp.PrimitiveSet('Main', 5)
        pset.addPrimitive(add,2)
        pset.addPrimitive(sub,2)
        pset.addPrimitive(mul,2)
        pset.addPrimitive(div,2)
        pset.addPrimitive(rank,1)
        pset.addPrimitive(neg,1)
        pset.addPrimitive(diff5,1)
        pset.addPrimitive(ts_max5,1)
        pset.addPrimitive(ts_min5,1)
        pset.addPrimitive(ts_ma5,1)

        pset.addEphemeralConstant("rand", lambda: random.randint(-10, 10))

        #pset.renameArguments(**{'x0':'closes','x1':'highs','x2':'lows','x3':'opens','x4':'amounts'})
        #pset.renameArguments(x0='closes',x1='highs',x2='lows',x3='opens',x4='amounts')
        pset.renameArguments(ARG0='closes',ARG1='highs',ARG2='lows',ARG3='opens',ARG4='amounts')

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        toolbox.register("evaluate", evaluate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


        pop = toolbox.population(n=self.params['number_population'])
        hof = tools.HallOfFame(self.params['number_hall_of_fame'])
        ngen=self.params['number_generation']


    
        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        #stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        #stats_size = tools.Statistics(len)
        #mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, ngen, stats=stats,halloffame=hof, verbose=True)
        self.fit_log = log

        fits = []
        factors=[]
        factor_formula=[]
        for ind in hof:
            func = toolbox.compile(ind)
            factors.append(func(closes,highs,lows,opens,amounts))
            factor_formula.append(str(ind))
            fits.append(ind.fitness.values[0])
        self.factor_formula=factor_formula
        self.fits=fits
        self.factors=factors

        t2=time.perf_counter()
        print('%4s%-40s%10.4f%4s' % (" ","calculating time:",t2-t1,"sec"))

        return


    def geppy_factor(self):

        def evaluate(individual):
            func = toolbox.compile(individual)
            factor_data = func(closes,highs,lows,opens,amounts)
            if type(factor_data) == float or type(factor_data) == int:
                return -1,
            perf = self.calc_fitness(factor_data)
            if isnan(perf):
                return -1,
            return perf,

        t1=time.perf_counter()
        self.load_data()
        closes = self.data['close']
        highs = self.data['high']
        lows = self.data['low']
        opens = self.data['open']
        amounts = self.data['amount']
        returns = self.returns
        
        t2=time.perf_counter()
        print('%4s%-40s%10.4f%4s' % (" ","load_data time:",t2-t1,"sec"))
        t1=time.perf_counter()
  
        creator.create("FitnessMax", base.Fitness, weights=(1,))
        creator.create("Individual", geppy.Chromosome, fitness=creator.FitnessMax)

        pset = geppy.PrimitiveSet('Main', input_names=['closes','highs','lows','opens','amounts'])
        pset.add_function(add,2)
        pset.add_function(sub,2)
        pset.add_function(mul,2)
        pset.add_function(div,2)
        pset.add_function(rank,1)
        pset.add_function(neg,1)
        pset.add_function(diff5,1)
        pset.add_function(ts_max5,1)
        pset.add_function(ts_min5,1)
        pset.add_function(ts_ma5,1)
        pset.add_rnc_terminal()

        head_length = self.params['head_length']
        rnc_array_length = self.params['rnc_array_length']
        number_gene = self.params['number_gene']
        number_population = self.params['number_population']
        number_generation = self.params['number_generation']
        number_hof = self.params['number_hall_of_fame']

        toolbox = geppy.Toolbox()
        toolbox.register('rnc_gen', random.randint, a=-10, b=10)  # each RNC is random integer within [-10, 10]
        toolbox.register('gene_gen', geppy.GeneDc, pset=pset, head_length=head_length, rnc_gen=toolbox.rnc_gen, rnc_array_length=rnc_array_length)

        if number_gene > 1:
            toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=number_gene, linker=operator.add)
        else:
            toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=1)
            
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register('compile', geppy.compile_, pset=pset)

        toolbox.register('cx_1p', geppy.crossover_one_point, pb=0.3)
        toolbox.register('cx_2p', geppy.crossover_two_point, pb=0.2)
        toolbox.register('cx_gene', geppy.crossover_gene, pb=0.1)
        toolbox.register('evaluate', evaluate)
        toolbox.register('select', tools.selTournament, tournsize=3)
        toolbox.register('mut_uniform', geppy.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
        toolbox.register('mut_invert', geppy.invert, pb=0.1)
        toolbox.register('mut_is_transpose', geppy.is_transpose, pb=0.1)
        toolbox.register('mut_ris_transpose', geppy.ris_transpose, pb=0.1)
        toolbox.register('mut_gene_transpose', geppy.gene_transpose, pb=0.1)
        toolbox.register('mut_dc', geppy.mutate_uniform_dc, ind_pb=0.05, pb=1)
        toolbox.register('mut_invert_dc', geppy.invert_dc, pb=0.1)
        toolbox.register('mut_transpose_dc', geppy.transpose_dc, pb=0.1)
        toolbox.register('mut_rnc_array_dc', geppy.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='0.5p', pb=1)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop = toolbox.population(n=number_population)  
        hof = tools.HallOfFame(number_hof)  
        pop, log = geppy.gep_simple(pop, toolbox, n_generations=number_generation, n_elites=2, stats=stats, hall_of_fame=hof, verbose=True)
        self.fit_log = log
        
        fits = []
        factors=[]
        factor_formula=[]
        for ind in hof:
            func = toolbox.compile(ind)
            factors.append(func(closes,highs,lows,opens,amounts))
            factor_formula.append(str(ind))
            fits.append(ind.fitness.values[0])
        self.factor_formula=factor_formula
        self.fits=fits
        self.factors=factors
        t2=time.perf_counter()
        print('%4s%-40s%10.4f%4s' % (" ","calculating time:",t2-t1,"sec"))

        return
    
    def gplearn_factor(self):
        global index_list,col_list,nd,ns
        def evaluate(y,x,w):
            global index_list,col_list,nd,ns
            if type(x) == float or type(x) == int:
                return -1
            if nd*ns != len(x):
                return -1
            factor_data = pd.DataFrame(x.reshape(nd,ns),index=index_list,columns=col_list)
            perf = self.calc_fitness(factor_data)
            if isnan(perf):
                return -1
            return perf

        t1=time.perf_counter()
        self.load_data()
        closes = self.data['close']
        highs = self.data['high']
        lows = self.data['low']
        opens = self.data['open']
        amounts = self.data['amount']
        returns = self.returns
        index_list=list(self.returns.index)
        col_list=list(self.returns.columns)
        nd = len(index_list)
        ns = len(col_list)
        x_dict={}
        x_dict['closes']=closes
        x_dict['highs']=highs
        x_dict['lows']=lows
        x_dict['opens']=opens
        x_dict['amounts']=amounts
        x_array = np.array(list(x_dict.values()))
        x_array = np.transpose(x_array, axes=(1, 2, 0)) #ndate x nstock x nfeature
        nd,ns,nf = x_array.shape
        x_factors = x_array[0,:,:]
        for i in range(1,nd):
            x_factors = np.vstack((x_factors,x_array[i,:,:]))   
        y_returns=returns.values.reshape(nd*ns,)

        t2=time.perf_counter()
        print('%4s%-40s%10.4f%4s' % (" ","load_data time:",t2-t1,"sec"))
        t1=time.perf_counter()

        addf = make_function(function=add, name='add', arity=2)
        subf = make_function(function=sub, name='sub', arity=2)
        mulf = make_function(function=mul, name='mul', arity=2)
        divf = make_function(function=div, name='div', arity=2)
        negf = make_function(function=neg, name='neg', arity=1)
        rank = make_function(function=rank_gp, name='rank', arity=1)
        diff5 = make_function(function=diff5_gp, name='diff5', arity=1)
        ts_max5 = make_function(function=ts_max5_gp, name='ts_max5', arity=1)
        ts_min5 = make_function(function=ts_min5_gp, name='ts_min5', arity=1)
        ts_ma5 = make_function(function=ts_ma5_gp, name='ts_ma5', arity=1)
        function_set = [addf,subf,mulf,divf,negf,rank,diff5,ts_max5,ts_min5,ts_ma5]

        fitness_func = make_fitness(function=evaluate,greater_is_better=True)
        generations=self.params['number_generation']
        population_size=self.params['number_population']
        hall_of_fame=self.params['number_hall_of_fame']

        est_gp = SymbolicTransformer(feature_names = ['close','high','low','open','amount'],
                                     function_set = function_set,
                                     generations = generations, 
                                     population_size = population_size,
                                     tournament_size = 3,
                                     metric= fitness_func,
                                     hall_of_fame = hall_of_fame,
                                     n_components = 5,
                                     init_method = 'grow',
                                     init_depth = (1,4),
                                     stopping_criteria= 10,
                                     const_range = (-1,1),
                                     p_crossover=0.7, 
                                     p_subtree_mutation=0.01,
                                     p_hoist_mutation=0.05, 
                                     p_point_mutation=0.01,
                                     p_point_replace = 0.1,
                                     max_samples = 1, 
                                     parsimony_coefficient=0.00, 
                                     verbose = 1)
        est_gp.fit(x_factors, y_returns)
        self.fit_log = est_gp
   
        factors=[]
        fits=[]
        formulas = []
        for p in est_gp._best_programs:
            factor_data = p.execute(x_factors)
            factor_data = pd.DataFrame(factor_data.reshape(nd,ns),index=index_list,columns=col_list)
            factors.append(factor_data)
            formulas.append(str(p))
            fits.append(p.fitness_)

        self.factor_formula=formulas
        self.fits=fits
        self.factors=factors
        t2=time.perf_counter()
        print('%4s%-40s%10.4f%4s' % (" ","calculating time:",t2-t1,"sec"))
    
    def calc_results(self):
        fits = self.fits
        factors = self.factors
        print(fits)
        mean = sum(fits) / len(fits)
        print("Min Fitness %8.4f" % min(fits))
        print("Max Fitness %8.4f" % max(fits))
        print("Avg Fitness %8.4f" % mean)

        annualized_returns = []
        annualized_irs = []
        max_drawdowns = []
        annualized_net_returns = []
        annualized_net_irs = []
        max_net_drawdowns = []
        for i in range(len(factors)):
            annualized_return,annualized_ir,max_drawdown,annualized_net_return,annualized_net_ir,max_net_drawdown = self.calc_performance(factors[i])
            annualized_returns.append(annualized_return)
            annualized_irs.append(annualized_ir)
            max_drawdowns.append(max_drawdown)
            annualized_net_returns.append(annualized_net_return)
            annualized_net_irs.append(annualized_net_ir)
            max_net_drawdowns.append(max_net_drawdown)
            print(self.factor_formula[i])
            print("%5d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f" % (i,fits[i],annualized_return,annualized_ir,max_drawdown,
                                                                     annualized_net_return,annualized_net_ir,max_net_drawdown))
        self.results['factor_formula']=self.factor_formula
        self.results['fit_values']=fits
        self.results['annualized_return']=annualized_returns
        self.results['annualized_ir']=annualized_irs
        self.results['max_drawdown']=max_drawdowns
        self.results['annualized_net_return']=annualized_net_returns
        self.results['annualized_net_ir']=annualized_net_irs
        self.results['max_net_drawdown']=max_net_drawdowns

    def display_fitness(self):
        if self.params['gen_package']=='deap' or self.params['gen_package']=='geppy':
            max_fits = self.fit_log.select("max") 
            mean_fits = self.fit_log.select("avg") 
            plt.figure(figsize=(6,4))
            plt.plot(max_fits,label='max fit')
            plt.plot(mean_fits,label='average fit')
            plt.legend()
            plt.show()
        elif self.params['gen_package']=='gplearn':
            df_statistics = pd.DataFrame(self.fit_log.run_details_)
            x = df_statistics['generation']
            plt.figure(figsize=(6,4))
            plt.plot(x, df_statistics['average_fitness'], label='average')
            plt.plot(x, df_statistics['best_fitness'], label='best')
            plt.legend()
            plt.show()

    def run(self):
        if self.params['gen_package']=='deap':
            self.deap_factor()
        elif self.params['gen_package']=='geppy':
            self.geppy_factor()
        elif self.params['gen_package']=='gplearn':
            self.gplearn_factor()
        else:
            raise ValueError("gen_package not recoganized")
        self.calc_results()
        if self.params['display_fitness'] == True:
            self.display_fitness()



if __name__ == '__main__':

    params={'use_small_sample':True,'gen_package':'geppy','display_fitness': True}  # 'deap','geppy,'gplearn'
    gf=genetic_factor(params=params)
    gf.run()


    

