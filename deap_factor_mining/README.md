# DEAP 遗传算法因子挖掘系统

## 快速开始

```bash
cd /path/to/SMQF/deap_factor_mining
python run_mining.py
```

这将运行完整的因子挖掘流程。

## 项目简介

使用DEAP遗传算法自动挖掘量化因子，完全集成SMQF因子框架：

- ✅ 完整的因子评价体系（IC、ICIR、IR、多空收益等）
- ✅ 遗传算法优化（选择、交叉、变异、精英保留）
- ✅ 表达式深度限制（≤5层）
- ✅ 自动保存结果和检查点

## 系统架构

```
deap_factor_mining/
├── config.py              # 配置文件
├── gp_primitives.py       # GP原语定义（数据字段、分析函数）
├── gp_factor.py          # GP因子类（继承自factor基类）
├── fitness_evaluator.py  # 适应度评估器
├── genetic_algorithm.py  # 主遗传算法程序
├── utils.py              # 工具函数
├── __init__.py           # 包初始化
└── README.md             # 本文档
```

## 核心特性

### 1. 量纲对齐

系统实现了完整的量纲系统，确保生成的因子表达式在物理意义上合理：

- **Price**: close, open, high, low
- **Volume**: volume
- **Return**: return, pct_change
- **Ratio**: 无量纲比率（如排序结果，0-1之间）
- **Factor**: 最终因子值

### 2. 表达式复杂度控制

- 最大深度限制为 8 层（可配置）
- 最小深度要求为 3 层，避免过于简单的因子
- 使用深度限制装饰器自动裁剪过深的树
- 奖励中等复杂度的因子（深度3-6，大小10-30）

### 3. 完整的因子评价

集成 SMQF 因子框架，自动计算：

- **IC 指标**: IC, Rank IC, ICIR, 年化ICIR
- **收益指标**: 年化收益率, 年化IR, 最大回撤
- **多空组合**: 多空收益, 换手率, 交易成本
- **分组收益**: 10分组收益分析

### 4. 智能适应度函数（IC优先）

**IC是最重要的指标**，适应度函数给予IC极高的权重：

```python
适应度 = 20.0 × |IC| × 1000 + 5.0 × |ICIR| × 10 + 2.0 × IR × 5
         - 0.05 × 换手率惩罚 + 复杂度奖励 - 回撤惩罚 + 高收益奖励
```

**权重设置**：
- **IC权重**: 20.0（放大1000倍） - **最重要！**
- **ICIR权重**: 5.0（放大10倍）
- **IR权重**: 2.0（放大5倍）
- 换手率惩罚: -0.05
- 复杂度: 奖励中等复杂度，惩罚过于简单或复杂的因子

## 快速开始

### 1. 安装依赖

```bash
pip install deap numpy pandas matplotlib networkx
```

### 2. 运行因子挖掘

```bash
cd /Users/neilchen/Library/Mobile\ Documents/com~apple~CloudDocs/CodeField/Python/SAIF/SMQF/deap_factor_mining

python genetic_algorithm.py
```

### 3. 自定义配置

编辑 `config.py` 文件，修改参数：

```python
# 遗传算法参数
GA_CONFIG = {
    'population_size': 100,     # 种群大小
    'n_generations': 50,        # 进化代数
    'max_depth': 8,             # 最大树深度（增加以支持更复杂因子）
    'min_depth': 3,             # 最小树深度（避免过于简单的因子）
    ...
}

# 因子评价参数
FACTOR_CONFIG = {
    'start_date': '2015-01-01',
    'end_date': '2023-12-31',
    'universe': 'top75pct',     # 股票池
    ...
}

# 适应度权重（IC优先！）
FITNESS_WEIGHTS = {
    'icir': 5.0,                # ICIR权重
    'ir': 2.0,                  # IR权重
    'ic': 20.0,                 # 平均IC权重 - 最重要！
    'turnover_penalty': -0.05,  # 换手率惩罚权重
}
```

## 输出结果

运行完成后，会在 `deap_results/run_YYYYMMDD_HHMMSS/` 目录下生成：

1. **best_factors.json**: 最佳因子列表（包含表达式、性能指标）
2. **factor_data_N.pkl**: 各因子的完整数据
3. **evolution_log.csv**: 进化过程日志
4. **evolution_curve.png**: 适应度进化曲线图
5. **config.json**: 运行配置
6. **evaluation_stats.json**: 评估统计信息
7. **checkpoint_genN.pkl**: 中间检查点（每10代保存一次）

## 使用示例

### 示例1: 基础使用

```python
from genetic_algorithm import GeneticAlgorithmFactorMiner

# 创建挖掘器
miner = GeneticAlgorithmFactorMiner()

# 设置
miner.setup()

# 运行
population, logbook, hall_of_fame = miner.run()

# 查看最佳因子
for i, ind in enumerate(hall_of_fame[:5]):
    print(f"{i+1}. {ind}, Fitness: {ind.fitness.values[0]:.4f}")
```

### 示例2: 自定义配置

```python
from genetic_algorithm import GeneticAlgorithmFactorMiner

# 自定义配置
custom_config = {
    'ga': {
        'population_size': 200,
        'n_generations': 100,
        'max_depth': 4,
    },
    'factor': {
        'start_date': '2020-01-01',
        'universe': 'hs300',
    }
}

miner = GeneticAlgorithmFactorMiner(config=custom_config)
miner.setup()
miner.run()
```

### 示例3: 加载和分析结果

```python
from utils import load_best_factors, compare_factors, create_factor_report

# 加载最佳因子
run_dir = './deap_results/run_20231201_120000'
best_factors = load_best_factors(run_dir)

# 比较因子性能
df = compare_factors(best_factors, top_n=10)
print(df)

# 生成HTML报告
create_factor_report(run_dir, output_file='report.html')
```

### 示例4: 可视化表达式树

```python
from utils import visualize_tree
from utils import load_best_factors
import pickle

# 加载检查点
with open('./deap_results/run_20231201_120000/checkpoint_gen50.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

# 可视化最佳个体
best_ind = checkpoint['hall_of_fame'][0]
visualize_tree(best_ind, filename='best_factor_tree.png')
```

## 可用的原语（Primitives）

### 数据字段（Terminals）

- **close**: 收盘价
- **open**: 开盘价
- **high**: 最高价
- **low**: 最低价
- **volume**: 成交量
- **return**: 收益率

### 时序函数（Time Series）

对于每个窗口大小 `w` ∈ {5, 10, 20, 40, 60}：

- `ts_mean{w}(x)`: w日均值
- `ts_std{w}(x)`: w日标准差
- `ts_sum{w}(x)`: w日求和
- `ts_delay{w}(x)`: 延迟w期
- `ts_delta{w}(x)`: w期差分
- `ts_rank{w}(x)`: w日时序排序
- `ts_max{w}(x)`: w日最大值
- `ts_min{w}(x)`: w日最小值
- `ema{w}(x)`: w日指数移动平均

### 横截面函数（Cross Section）

- `cs_rank(x)`: 横截面排序（输出0-1）
- `cs_zscore(x)`: 横截面标准化

### 数学函数

- `add(x, y)`: 加法
- `sub(x, y)`: 减法
- `mul(x, y)`: 乘法
- `div(x, y)`: 除法（安全，避免除零）
- `neg(x)`: 取负
- `abs(x)`: 绝对值
- `sign(x)`: 符号函数

## 高级功能

### 1. 恢复训练

```python
import pickle

# 加载检查点
with open('./deap_results/run_XXX/checkpoint_gen30.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

# 恢复状态
population = checkpoint['population']
logbook = checkpoint['logbook']
random.setstate(checkpoint['random_state'])
np.random.set_state(checkpoint['numpy_random_state'])

# 继续训练...
```

### 2. 并行评估（实验性）

```python
from fitness_evaluator import ParallelFitnessEvaluator

evaluator = ParallelFitnessEvaluator(
    pset=pset,
    factor_params=factor_params,
    n_jobs=-1  # 使用所有CPU核心
)
```

### 3. 导出因子为Python类

```python
from utils import export_best_factor_as_class

# 导出最佳因子
export_best_factor_as_class(
    individual=hall_of_fame[0],
    pset=pset,
    factor_name='AutoGenFactor001',
    output_file='auto_gen_factor_001.py'
)
```

## 性能优化建议

1. **预加载共享数据**: 系统已自动实现，避免重复加载数据
2. **缓存机制**: 自动缓存已评估的个体，避免重复计算
3. **调整种群大小**: 根据计算资源调整 `population_size`
4. **限制表达式复杂度**: 较小的 `max_depth` 能加快评估速度
5. **使用较短的数据范围**: 在开发阶段使用较短的时间范围测试

## 常见问题

### Q1: 为什么评估速度慢？

A: 因子评估需要完整的回测计算。建议：
- 减少种群大小
- 使用较短的数据范围
- 减少进化代数

### Q2: 如何避免过拟合？

A: 系统已内置防过拟合机制：
- 使用样本外测试（设置 `start_os_date`）
- 惩罚过于复杂的表达式
- 考虑换手率和交易成本

### Q3: 生成的因子可以直接使用吗？

A: 建议进一步验证：
- 检查因子的经济学意义
- 在更长的时间范围上验证
- 测试不同市场环境下的稳定性
- 与现有因子的相关性分析

### Q4: 如何与现有因子比较？

A: 使用 `compare_factors` 函数：
```python
from utils import compare_factors, load_best_factors

# 加载GP挖掘的因子
gp_factors = load_best_factors('./deap_results/run_XXX')

# 手动添加现有因子的性能数据
existing_factors = [
    {'rank': 0, 'fitness': X, 'performance': {...}},
    ...
]

# 合并比较
all_factors = existing_factors + gp_factors
df = compare_factors(all_factors)
print(df)
```

## 技术细节

### 遗传操作

1. **选择（Selection）**: 锦标赛选择（Tournament Selection）
2. **交叉（Crossover）**: 单点交叉（One-Point Crossover）
3. **变异（Mutation）**: 均匀变异（Uniform Mutation）
4. **精英保留（Elitism）**: 保留前5个最佳个体

### 适应度评估流程

```
1. 创建GPFactor对象
2. 加载共享数据
3. 计算因子值
4. 运行完整回测
5. 提取性能指标
6. 计算综合适应度分数
7. 缓存结果
```

### 表达式深度限制

使用两层机制限制深度：
1. 初始生成时限制最大深度
2. 遗传操作后使用装饰器裁剪超深树

## 参考文献

- DEAP文档: https://deap.readthedocs.io/
- WorldQuant Alpha101: https://arxiv.org/abs/1601.00991
- 遗传编程入门: Koza, J. R. (1992). Genetic Programming

## 贡献与反馈

如有问题或建议，请联系项目维护者。

## 许可证

本项目为 SMQF 系统的一部分，仅供研究和学习使用。

---

**注意**: 本系统生成的因子仅供研究参考，不构成投资建议。实际投资请谨慎评估风险。
