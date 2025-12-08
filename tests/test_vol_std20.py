import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
QLIB_DIR = os.path.join(PROJECT_DIR, "quant_lib")
if QLIB_DIR not in sys.path:
    sys.path.append(QLIB_DIR)

from factor import factor
from analysis import *
from chenhan_factor import vol_std20

import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings


f=factor()
print(f.available_data())

params={'ir_details': True}
factor_property={'universe':'top75pct'}

f=vol_std20(params=params, factor_property=factor_property)
f.run()
