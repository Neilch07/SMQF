import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'quant_lib')))

from genetic_factor import genetic_factor

params={'use_small_sample':True,'gen_package':'deap','display_fitness': True}  # 'deap','geppy,'gplearn'
gf=genetic_factor(params=params)
gf.run()

