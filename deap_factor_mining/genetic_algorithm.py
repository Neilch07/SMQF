"""
DEAP遗传算法主程序
使用遗传编程（Genetic Programming）挖掘量化因子
"""

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools, gp
import random
import time
from datetime import datetime
import os
import pickle
import json
import sys

# 添加父目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 添加quant_lib到路径
quant_lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'quant_lib'))
if quant_lib_dir not in sys.path:
    sys.path.insert(0, quant_lib_dir)

from gp_primitives import create_primitive_set, get_tree_depth, limit_depth
from fitness_evaluator import FitnessEvaluator, load_shared_data
from gp_factor import GPFactor
from config import GA_CONFIG, FACTOR_CONFIG, FITNESS_WEIGHTS, PRIMITIVE_CONFIG, OUTPUT_CONFIG


class GeneticAlgorithmFactorMiner:
    """
    遗传算法因子挖掘器

    使用DEAP框架实现遗传编程，自动挖掘量化因子
    """

    def __init__(self, config=None):
        """
        初始化遗传算法

        Args:
            config: 配置字典，可以覆盖默认配置
        """
        # 合并配置
        self.ga_config = {**GA_CONFIG, **(config.get('ga', {}) if config else {})}
        self.factor_config = {**FACTOR_CONFIG, **(config.get('factor', {}) if config else {})}
        self.fitness_weights = {**FITNESS_WEIGHTS, **(config.get('fitness', {}) if config else {})}
        self.primitive_config = {**PRIMITIVE_CONFIG, **(config.get('primitive', {}) if config else {})}
        self.output_config = {**OUTPUT_CONFIG, **(config.get('output', {}) if config else {})}

        # 创建输出目录
        self.output_dir = self.output_config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        # 创建时间戳子目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.output_dir, f'run_{timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)

        print(f"输出目录: {self.run_dir}")

        # 初始化组件
        self.pset = None
        self.toolbox = None
        self.evaluator = None
        self.shared_data = None

        # 统计信息
        self.stats = None
        self.logbook = None
        self.hall_of_fame = None

        # 设置随机种子
        random.seed(42)
        np.random.seed(42)

    def setup(self):
        """
        设置遗传算法的所有组件
        """
        print("\n" + "="*80)
        print("初始化遗传算法因子挖掘器")
        print("="*80)

        # 1. 创建原语集
        print("\n[1/5] 创建GP原语集...")
        self.pset = create_primitive_set(
            window_sizes=self.primitive_config['window_sizes'],
            data_fields=self.primitive_config['data_fields'],
            enable_division=self.primitive_config['enable_division'],
            enable_high_complexity=self.primitive_config['enable_high_complexity']
        )
        print(f"  - 函数节点数: {len(self.pset.primitives[object])}")
        print(f"  - 终端节点数: {len(self.pset.terminals[object])}")

        # 2. 创建DEAP类型
        print("\n[2/5] 创建DEAP类型...")
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        print("  - FitnessMax和Individual类型已创建")

        # 3. 创建工具箱
        print("\n[3/5] 创建DEAP工具箱...")
        self._create_toolbox()
        print("  - 工具箱已配置完成")

        # 4. 预加载共享数据
        print("\n[4/5] 预加载共享数据...")
        self.shared_data = load_shared_data(self.factor_config)
        print("  - 共享数据已加载")

        # 5. 创建适应度评估器
        print("\n[5/5] 创建适应度评估器...")
        self.evaluator = FitnessEvaluator(
            pset=self.pset,
            factor_params=self.factor_config,
            fitness_weights=self.fitness_weights,
            shared_data=self.shared_data,
            verbose=True  # 开启详细模式查看错误
        )
        print("  - 评估器已创建")

        print("\n" + "="*80)
        print("初始化完成！")
        print("="*80 + "\n")

    def _create_toolbox(self):
        """创建DEAP工具箱，注册所有遗传操作"""
        self.toolbox = base.Toolbox()

        # 表达式生成：使用HalfAndHalf方法（一半grow，一半full）
        self.toolbox.register(
            "expr",
            gp.genHalfAndHalf,
            pset=self.pset,
            min_=self.ga_config['min_depth'],
            max_=self.ga_config['max_depth']
        )

        # 个体生成
        self.toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            self.toolbox.expr
        )

        # 种群生成
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )

        # 评估函数（稍后会设置为evaluator.evaluate）
        # 这里先注册一个占位符
        self.toolbox.register("evaluate", lambda x: (0.0,))

        # 选择：锦标赛选择
        self.toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.ga_config['tournament_size']
        )

        # 交叉：单点交叉
        self.toolbox.register("mate", gp.cxOnePoint)

        # 变异：均匀变异
        self.toolbox.register(
            "expr_mut",
            gp.genFull,
            min_=0,
            max_=2
        )
        self.toolbox.register(
            "mutate",
            gp.mutUniform,
            expr=self.toolbox.expr_mut,
            pset=self.pset
        )

        # 装饰器：限制深度
        self.toolbox.decorate(
            "mate",
            gp.staticLimit(
                key=lambda x: x.height,
                max_value=self.ga_config['max_depth']
            )
        )
        self.toolbox.decorate(
            "mutate",
            gp.staticLimit(
                key=lambda x: x.height,
                max_value=self.ga_config['max_depth']
            )
        )

    def run(self):
        """
        运行遗传算法主循环
        """
        print("\n" + "="*80)
        print("开始遗传算法进化")
        print("="*80)

        # 更新评估函数
        self.toolbox.register("evaluate", self.evaluator.evaluate)

        # 初始化种群
        print(f"\n初始化种群（大小: {self.ga_config['population_size']}）...")
        population = self.toolbox.population(n=self.ga_config['population_size'])

        # 名人堂（保存最佳个体）
        self.hall_of_fame = tools.HallOfFame(self.output_config['save_best_n'])

        # 统计信息
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        stats_height = tools.Statistics(lambda ind: ind.height)

        self.stats = tools.MultiStatistics(
            fitness=stats_fit,
            size=stats_size,
            height=stats_height
        )

        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        # 日志
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + self.stats.fields

        # 初始评估
        print("\n评估初始种群...")
        start_time = time.time()

        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        self.hall_of_fame.update(population)

        record = self.stats.compile(population)
        self.logbook.record(gen=0, nevals=len(population), **record)

        print(f"第 0 代: {self.logbook[-1]}")
        print(f"评估耗时: {time.time() - start_time:.2f}秒")

        # 进化循环
        print(f"\n开始进化 {self.ga_config['n_generations']} 代...")
        print("-" * 80)

        for gen in range(1, self.ga_config['n_generations'] + 1):
            gen_start_time = time.time()

            # 选择
            offspring = self.toolbox.select(population, len(population) - self.ga_config['elite_size'])
            offspring = list(map(self.toolbox.clone, offspring))

            # 交叉和变异
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.ga_config['crossover_prob']:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.ga_config['mutation_prob']:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 评估需要评估的个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 精英保留
            elite = tools.selBest(population, self.ga_config['elite_size'])
            population[:] = elite + offspring

            # 更新名人堂
            self.hall_of_fame.update(population)

            # 记录统计信息
            record = self.stats.compile(population)
            self.logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            # 打印进度
            gen_time = time.time() - gen_start_time
            best_fitness = self.hall_of_fame[0].fitness.values[0]

            print(f"第 {gen:3d} 代 | "
                  f"最佳适应度: {best_fitness:8.4f} | "
                  f"平均适应度: {record['fitness']['avg']:8.4f} | "
                  f"评估数: {len(invalid_ind):4d} | "
                  f"耗时: {gen_time:5.1f}秒")

            # 每10代保存一次中间结果
            if gen % 10 == 0:
                self._save_checkpoint(gen, population)

        print("-" * 80)
        print("\n进化完成！")

        # 保存最终结果
        self._save_results(population)

        return population, self.logbook, self.hall_of_fame

    def _save_checkpoint(self, gen, population):
        """保存检查点"""
        checkpoint_path = os.path.join(self.run_dir, f'checkpoint_gen{gen}.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'generation': gen,
                'population': population,
                'hall_of_fame': self.hall_of_fame,
                'logbook': self.logbook,
                'random_state': random.getstate(),
                'numpy_random_state': np.random.get_state()
            }, f)

    def _save_results(self, population):
        """保存最终结果"""
        print("\n" + "="*80)
        print("保存结果")
        print("="*80)

        # 1. 保存最佳个体
        print("\n保存最佳因子...")
        best_factors = []

        for rank, ind in enumerate(self.hall_of_fame):
            factor_info = {
                'rank': rank + 1,
                'fitness': ind.fitness.values[0],
                'expression': str(ind),
                'size': len(ind),
                'depth': ind.height
            }

            # 详细评估最佳因子
            try:
                gp_factor = GPFactor(
                    individual=ind,
                    pset=self.pset,
                    factor_name=f'best_factor_{rank+1}',
                    params=self.factor_config,
                    data=self.shared_data
                )
                gp_factor.run(turnoff_display=True)
                results = gp_factor.get_performance()

                # 添加性能指标
                factor_info['performance'] = {
                    'icir': results.get('anual_icir', np.nan),
                    'ir': results.get('Annualized IR', np.nan),
                    'ic': results.get('daily_avg_ic', np.nan),
                    'annual_return': results.get('Annualized Return', np.nan),
                    'max_drawdown': results.get('Maximum Drawdown', np.nan),
                    'turnover': results.get('avg_turn_over', np.nan)
                }

                # 保存因子数据
                factor_data_path = os.path.join(self.run_dir, f'factor_data_{rank+1}.pkl')
                with open(factor_data_path, 'wb') as f:
                    pickle.dump(gp_factor.factor_data, f)

            except Exception as e:
                print(f"  警告：无法详细评估第{rank+1}个因子: {str(e)}")
                factor_info['performance'] = {}

            best_factors.append(factor_info)

            print(f"  第 {rank+1:2d} 名: 适应度 {factor_info['fitness']:8.4f} | "
                  f"ICIR {factor_info['performance'].get('icir', np.nan):6.4f} | "
                  f"深度 {factor_info['depth']}")

        # 保存为JSON
        best_factors_path = os.path.join(self.run_dir, 'best_factors.json')
        with open(best_factors_path, 'w', encoding='utf-8') as f:
            json.dump(best_factors, f, indent=2, ensure_ascii=False, default=str)

        # 2. 保存日志
        print("\n保存进化日志...")
        logbook_df = pd.DataFrame(self.logbook)
        logbook_df.to_csv(os.path.join(self.run_dir, 'evolution_log.csv'), index=False)

        # 3. 保存配置
        print("保存配置...")
        config_path = os.path.join(self.run_dir, 'config.json')
        full_config = {
            'ga': self.ga_config,
            'factor': self.factor_config,
            'fitness': self.fitness_weights,
            'primitive': self.primitive_config,
            'output': self.output_config
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=2, ensure_ascii=False, default=str)

        # 4. 保存种群（可选）
        if self.output_config['save_population']:
            print("保存最终种群...")
            population_path = os.path.join(self.run_dir, 'final_population.pkl')
            with open(population_path, 'wb') as f:
                pickle.dump(population, f)

        # 5. 绘制进化曲线
        if self.output_config['plot_evolution']:
            print("绘制进化曲线...")
            try:
                self._plot_evolution()
            except Exception as e:
                print(f"  警告：绘图失败: {str(e)}")

        # 6. 保存评估统计
        print("保存评估统计...")
        eval_stats = self.evaluator.get_statistics()
        eval_stats_path = os.path.join(self.run_dir, 'evaluation_stats.json')
        with open(eval_stats_path, 'w', encoding='utf-8') as f:
            json.dump(eval_stats, f, indent=2)

        print("\n所有结果已保存到:", self.run_dir)
        print("="*80 + "\n")

    def _plot_evolution(self):
        """绘制进化曲线"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  警告：matplotlib未安装，跳过绘图")
            return

        gen = self.logbook.select("gen")
        fit_maxs = [record['fitness']['max'] for record in self.logbook]
        fit_avgs = [record['fitness']['avg'] for record in self.logbook]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(gen, fit_maxs, 'b-', label='Maximum Fitness', linewidth=2)
        ax.plot(gen, fit_avgs, 'r--', label='Average Fitness', linewidth=2)
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.set_title('Evolution of Fitness over Generations', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(self.run_dir, 'evolution_curve.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数：运行完整的因子挖掘流程"""
    print("\n" + "="*80)
    print(" "*20 + "DEAP 遗传算法因子挖掘系统")
    print("="*80 + "\n")

    # 创建挖掘器
    miner = GeneticAlgorithmFactorMiner()

    # 设置
    miner.setup()

    # 运行
    population, logbook, hall_of_fame = miner.run()

    print("\n" + "="*80)
    print("因子挖掘完成！")
    print("="*80)

    # 打印最佳因子
    print("\n最佳因子Top 5:")
    for i, ind in enumerate(hall_of_fame[:5]):
        print(f"\n{i+1}. 适应度: {ind.fitness.values[0]:.4f}")
        print(f"   表达式: {str(ind)}")
        print(f"   深度: {ind.height}, 大小: {len(ind)}")


if __name__ == '__main__':
    main()
