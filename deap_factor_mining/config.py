"""
DEAP遗传算法因子挖掘配置文件
"""

# 遗传算法参数
GA_CONFIG = {
    'population_size': 100,        # 种群大小
    'n_generations': 50,           # 进化代数
    'crossover_prob': 0.7,         # 交叉概率
    'mutation_prob': 0.2,          # 变异概率
    'tournament_size': 3,          # 锦标赛选择大小
    'elite_size': 5,               # 精英保留数量
    'max_depth': 8,                # 最大树深度（增加以支持更复杂的因子）
    'min_depth': 3,                # 最小树深度（增加以生成更复杂的因子）
}

# 因子评价参数
FACTOR_CONFIG = {
    'start_date': '2015-01-01',    # 开始日期
    'end_date': '2023-12-31',      # 结束日期
    'start_os_date': '2022-01-01', # 样本外开始日期
    'universe': 'top75pct',        # 股票池
    'lag_periods': 2,              # 延迟期数
    'ic_return_horizon': 5,        # IC收益期数
    'n_group': 10,                 # 分组数量
    'quantile': 0.2,               # 多空分位数
    'trading_cost': 0.0012,        # 交易成本
}

# 适应度权重（用于多目标优化）
# 用户最关心的是IC，所以给IC极高的权重
FITNESS_WEIGHTS = {
    'icir': 5.0,                   # ICIR权重
    'ir': 2.0,                     # IR权重
    'ic': 20.0,                    # 平均IC权重 - 最重要！
    'turnover_penalty': -0.05,     # 换手率惩罚权重（降低）
}

# GP原语配置
PRIMITIVE_CONFIG = {
    'window_sizes': [5, 10, 20, 40, 60],  # 时间窗口参数
    'data_fields': ['close', 'open', 'high', 'low', 'volume', 'return'],  # 可用数据字段
    'enable_division': True,       # 是否启用除法（可能产生inf/nan）
    'enable_high_complexity': True, # 是否启用高复杂度函数（如ts_corr, ts_beta）
}

# 输出配置
OUTPUT_CONFIG = {
    'output_dir': './deap_results',     # 输出目录
    'save_best_n': 10,                  # 保存前N个最佳因子
    'save_population': True,            # 是否保存整个种群
    'save_log': True,                   # 是否保存运行日志
    'plot_evolution': True,             # 是否绘制进化曲线
}
