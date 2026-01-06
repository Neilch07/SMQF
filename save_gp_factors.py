"""
Script to compute and save GP-generated alpha factors
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "quant_lib"))

from quant_lib.gp_alphas import *


def save_gp_factors():
    """
    Compute and save all GP alpha factors to disk
    """
    # Parameters for saving factors
    params = {
        "save": True,
        "display": True,
        "start_date": None,  # Use all available data
        "end_date": None,
    }
    factor_property = {"universe": "top75pct"}

    # List of all GP alpha classes (30 unique implementations)
    gp_alphas = [
        gp_alpha_029,
        gp_alpha_030,
    ]

    print("=" * 80)
    print("Computing and saving GP Alpha Factors")
    print("=" * 80)

    for alpha_class in gp_alphas:
        try:
            print(f"\nProcessing {alpha_class.__name__}...")
            factor = alpha_class(params=params, factor_property=factor_property)
            factor.run()

            factor_name = factor.params["factor_name"]
            print(f"✓ Successfully saved {factor_name}")
            print(f"  - Factor data: factor_data_{factor_name}.pkl.gz")
            print(f"  - Results: factor_results_{factor_name}.pkl")

        except Exception as e:
            print(f"✗ Error processing {alpha_class.__name__}: {e}")
            continue

    print("\n" + "=" * 80)
    print("All GP factors processed!")
    print("=" * 80)


if __name__ == "__main__":
    save_gp_factors()
