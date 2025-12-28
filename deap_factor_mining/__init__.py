"""
DEAP遗传算法因子挖掘模块

使用遗传编程（Genetic Programming）自动挖掘量化因子
"""

__version__ = '1.0.0'
__author__ = 'SMQF Team'

from .gp_primitives import create_primitive_set
from .gp_factor import GPFactor, create_gp_factor_from_individual
from .fitness_evaluator import FitnessEvaluator, load_shared_data
from .genetic_algorithm import GeneticAlgorithmFactorMiner
from .utils import (
    expression_to_string,
    visualize_tree,
    compare_factors,
    load_best_factors,
    load_factor_data,
    create_factor_report,
    export_best_factor_as_class
)

__all__ = [
    'create_primitive_set',
    'GPFactor',
    'create_gp_factor_from_individual',
    'FitnessEvaluator',
    'load_shared_data',
    'GeneticAlgorithmFactorMiner',
    'expression_to_string',
    'visualize_tree',
    'compare_factors',
    'load_best_factors',
    'load_factor_data',
    'create_factor_report',
    'export_best_factor_as_class'
]
