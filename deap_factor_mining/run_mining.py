#!/usr/bin/env python
"""
快速启动脚本
运行DEAP遗传算法因子挖掘
"""

import argparse
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 添加quant_lib到路径
quant_lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'quant_lib'))
if quant_lib_dir not in sys.path:
    sys.path.insert(0, quant_lib_dir)

from genetic_algorithm import GeneticAlgorithmFactorMiner
from config import GA_CONFIG, FACTOR_CONFIG, FITNESS_WEIGHTS, PRIMITIVE_CONFIG, OUTPUT_CONFIG


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='DEAP遗传算法因子挖掘',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 遗传算法参数
    parser.add_argument('--population', type=int, default=GA_CONFIG['population_size'],
                       help='种群大小')
    parser.add_argument('--generations', type=int, default=GA_CONFIG['n_generations'],
                       help='进化代数')
    parser.add_argument('--max-depth', type=int, default=GA_CONFIG['max_depth'],
                       help='最大树深度')
    parser.add_argument('--crossover-prob', type=float, default=GA_CONFIG['crossover_prob'],
                       help='交叉概率')
    parser.add_argument('--mutation-prob', type=float, default=GA_CONFIG['mutation_prob'],
                       help='变异概率')

    # 因子参数
    parser.add_argument('--start-date', type=str, default=FACTOR_CONFIG['start_date'],
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=FACTOR_CONFIG['end_date'],
                       help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--universe', type=str, default=FACTOR_CONFIG['universe'],
                       help='股票池 (top75pct, hs300, cs500等)')
    parser.add_argument('--ic-horizon', type=int, default=FACTOR_CONFIG['ic_return_horizon'],
                       help='IC收益期数')

    # 输出参数
    parser.add_argument('--output-dir', type=str, default=OUTPUT_CONFIG['output_dir'],
                       help='输出目录')
    parser.add_argument('--save-best-n', type=int, default=OUTPUT_CONFIG['save_best_n'],
                       help='保存前N个最佳因子')

    # 其他
    parser.add_argument('--quick-test', action='store_true',
                       help='快速测试模式（小种群、少代数、短数据）')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 构建配置
    config = {
        'ga': {
            'population_size': args.population,
            'n_generations': args.generations,
            'max_depth': args.max_depth,
            'crossover_prob': args.crossover_prob,
            'mutation_prob': args.mutation_prob,
            **GA_CONFIG
        },
        'factor': {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'universe': args.universe,
            'ic_return_horizon': args.ic_horizon,
            **FACTOR_CONFIG
        },
        'fitness': FITNESS_WEIGHTS,
        'primitive': PRIMITIVE_CONFIG,
        'output': {
            'output_dir': args.output_dir,
            'save_best_n': args.save_best_n,
            **OUTPUT_CONFIG
        }
    }

    # 快速测试模式
    if args.quick_test:
        print("\n*** 快速测试模式 ***")
        config['ga']['population_size'] = 20
        config['ga']['n_generations'] = 10
        config['factor']['start_date'] = '2022-01-01'
        config['factor']['end_date'] = '2023-12-31'
        config['factor']['start_os_date'] = None
        print(f"种群大小: {config['ga']['population_size']}")
        print(f"进化代数: {config['ga']['n_generations']}")
        print(f"数据范围: {config['factor']['start_date']} ~ {config['factor']['end_date']}")
        print()

    # 创建挖掘器
    miner = GeneticAlgorithmFactorMiner(config=config)

    # 设置
    miner.setup()

    # 运行
    try:
        population, logbook, hall_of_fame = miner.run()

        # 打印总结
        print("\n" + "="*80)
        print("运行完成！")
        print("="*80)
        print(f"\n结果保存在: {miner.run_dir}")
        print("\n最佳因子Top 5:")
        for i, ind in enumerate(hall_of_fame[:5]):
            print(f"\n{i+1}. 适应度: {ind.fitness.values[0]:.4f}")
            print(f"   表达式: {str(ind)[:100]}...")
            print(f"   深度: {ind.height}, 大小: {len(ind)}")

        print("\n" + "="*80)

    except KeyboardInterrupt:
        print("\n\n用户中断！正在保存当前状态...")
        # TODO: 保存中断时的状态
        print("状态已保存")
        sys.exit(0)

    except Exception as e:
        print(f"\n\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
