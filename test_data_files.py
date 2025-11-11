#!/usr/bin/env python
"""测试数据文件加载"""
import pandas as pd
import os

data_dir = "/Users/neilchen/Library/Mobile Documents/com~apple~CloudDocs/CodeField/Python/SAIF/SMQF/data/cn/equity/data/2021-current"

# 测试加载各个数据文件
files_to_test = ['closes', 'opens', 'highs', 'lows', 'volumes', 'turnover_amounts', 'pct_changes']

for f in files_to_test:
    fpath = os.path.join(data_dir, f + '.pkl')
    if os.path.exists(fpath):
        print(f"\n{f}.pkl exists:")
        try:
            df = pd.read_pickle(fpath)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns[:5])}...")
            print(f"  Index: {df.index[:3]}")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        except Exception as e:
            print(f"  Error loading: {e}")
    else:
        print(f"\n{f}.pkl NOT FOUND")
