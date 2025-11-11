import sys
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QLIB_DIR = os.path.join(THIS_DIR, "quant_lib")
if QLIB_DIR not in sys.path:
    sys.path.append(QLIB_DIR)

from quant_lib.alpha101 import alpha101_004

# Simulate model_train.py parameters
params = {
    'start_date': None,  # Use None like model_train when no args provided
    'end_date': None,
    'display': False,
    'save': False,
}
factor_property = {
    'universe': None,  # 'None' string or None value?
    'benchmark': None,
}

print("Testing with params:", params)
print("Testing with factor_property:", factor_property)

f = alpha101_004(params=params, factor_property=factor_property)
print('data_needed:', f.data_needed)
print('Before load_data, self.data keys:', list(f.data.keys()))
f.load_data()
print('After load_data, self.data keys:', list(f.data.keys()))
if 'low' in f.data:
    print('low data shape:', f.data['low'].shape)
print('Data loaded successfully!')

# Test calc_factor
print('\nTesting calc_factor...')
try:
    f.calc_factor()
    print('calc_factor succeeded!')
    print('factor_data shape:', f.factor_data.shape)
except Exception as e:
    print(f'calc_factor failed: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
