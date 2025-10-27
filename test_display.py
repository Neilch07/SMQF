import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'quant_lib')))

from display_factor import display_factor


df = display_factor(factor_displayed='alpha101_04')
df.display_one_factor()