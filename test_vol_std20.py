import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'quant_lib')))

from factor import factor
from analysis import *
from chenhan_factor import vol_std20

import warnings
warnings.filterwarnings("ignore")

f=factor()
print(f.available_data())

params={'ir_details': True}
factor_property={'universe':'top75pct'}

f=vol_std20(params=params, factor_property=factor_property)
f.run()
