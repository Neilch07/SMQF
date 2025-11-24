"""
Data Leakage Fix 전후 비교 스크립트
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 한글 폰트 설정
rcParams['font.family'] = 'AppleGothic'
rcParams['axes.unicode_minus'] = False

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# 기존 결과 (Data Leakage 있음) - 이전 실험 결과
BEFORE_RESULTS = {
    'lgbm_3factors': {
        'train': {'ic_mean': 0.0554, 'ic_ir': 1.01, 'ls_ann_ret': 1.12, 'ls_ann_ir': 13.19},
        'val': {'ic_mean': 0.0432, 'ic_ir': 0.85, 'ls_ann_ret': 0.71, 'ls_ann_ir': 9.06},
        'test': {'ic_mean': 0.0386, 'ic_ir': 0.62, 'ls_ann_ret': 0.40, 'ls_ann_ir': 4.94}
    },
    'xgb_3factors': {
        'train': {'ic_mean': 0.0626, 'ic_ir': 1.33, 'ls_ann_ret': 1.42, 'ls_ann_ir': 18.58},
        'val': {'ic_mean': 0.0347, 'ic_ir': 0.83, 'ls_ann_ret': 0.59, 'ls_ann_ir': 8.85},
        'test': {'ic_mean': 0.0323, 'ic_ir': 0.65, 'ls_ann_ret': 0.33, 'ls_ann_ir': 4.16}
    }
}

def load_after_results():
    """수정 후 결과 로드"""
    results = {}

    # LightGBM 결과
    lgbm_path = os.path.join(THIS_DIR, 'artifacts', 'metrics_lgbm_top3_train20211231_val20221231.json')
    if os.path.exists(lgbm_path):
        with open(lgbm_path, 'r') as f:
            results['lgbm_3factors'] = json.load(f)
    else:
        # 기본 파일 체크
        lgbm_path = os.path.join(THIS_DIR, 'artifacts', 'metrics_lgbm.json')
        if os.path.exists(lgbm_path):
            with open(lgbm_path, 'r') as f:
                results['lgbm_3factors'] = json.load(f)

    # XGBoost 결과
    xgb_path = os.path.join(THIS_DIR, 'artifacts', 'metrics_xgb_top3_train20211231_val20221231.json')
    if os.path.exists(xgb_path):
        with open(xgb_path, 'r') as f:
            results['xgb_3factors'] = json.load(f)
    else:
        xgb_path = os.path.join(THIS_DIR, 'artifacts', 'metrics_xgb.json')
        if os.path.exists(xgb_path):
            with open(xgb_path, 'r') as f:
                results['xgb_3factors'] = json.load(f)

    return results

def plot_comparison(before, after):
    """전후 비교 시각화"""
    if not after:
        print("⚠️ 수정 후 결과 파일이 없습니다. 먼저 model_train.py를 실행하세요.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Leakage Fix 전후 비교', fontsize=16, fontweight='bold')

    splits = ['train', 'val', 'test']
    x = np.arange(len(splits))
    width = 0.35

    # LightGBM만 비교 (가장 중요한 모델)
    model_key = 'lgbm_3factors'

    if model_key in before and model_key in after:
        # 1. IC Mean
        ic_before = [before[model_key][s]['ic_mean'] for s in splits]
        ic_after = [after[model_key][s]['ic_mean'] for s in splits]

        axes[0, 0].bar(x - width/2, ic_before, width, label='Before (Leakage)', alpha=0.8, color='red')
        axes[0, 0].bar(x + width/2, ic_after, width, label='After (Fixed)', alpha=0.8, color='green')
        axes[0, 0].set_ylabel('IC Mean')
        axes[0, 0].set_title('Information Coefficient 비교')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(splits)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 2. IC IR
        ir_before = [before[model_key][s]['ic_ir'] for s in splits]
        ir_after = [after[model_key][s]['ic_ir'] for s in splits]

        axes[0, 1].bar(x - width/2, ir_before, width, label='Before (Leakage)', alpha=0.8, color='red')
        axes[0, 1].bar(x + width/2, ir_after, width, label='After (Fixed)', alpha=0.8, color='green')
        axes[0, 1].set_ylabel('IC IR')
        axes[0, 1].set_title('IC Information Ratio 비교')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(splits)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 3. Annual Return
        ret_before = [before[model_key][s]['ls_ann_ret'] * 100 for s in splits]
        ret_after = [after[model_key][s]['ls_ann_ret'] * 100 for s in splits]

        axes[1, 0].bar(x - width/2, ret_before, width, label='Before (Leakage)', alpha=0.8, color='red')
        axes[1, 0].bar(x + width/2, ret_after, width, label='After (Fixed)', alpha=0.8, color='green')
        axes[1, 0].set_ylabel('연간 수익률 (%)')
        axes[1, 0].set_title('Long-Short 연간 수익률 비교')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(splits)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 4. Sharpe Ratio
        sharpe_before = [before[model_key][s]['ls_ann_ir'] for s in splits]
        sharpe_after = [after[model_key][s]['ls_ann_ir'] for s in splits]

        axes[1, 1].bar(x - width/2, sharpe_before, width, label='Before (Leakage)', alpha=0.8, color='red')
        axes[1, 1].bar(x + width/2, sharpe_after, width, label='After (Fixed)', alpha=0.8, color='green')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].set_title('Long-Short Sharpe Ratio 비교')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(splits)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()

        output_file = os.path.join(THIS_DIR, 'artifacts', 'leakage_fix_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ 저장: {output_file}")

def print_summary(before, after):
    """요약 통계 출력"""
    print("\n" + "="*80)
    print("Data Leakage Fix 전후 비교 요약")
    print("="*80)

    model_key = 'lgbm_3factors'

    if model_key not in after:
        print("⚠️ 수정 후 결과가 없습니다.")
        return

    for split in ['train', 'val', 'test']:
        print(f"\n[{split.upper()}]")
        print("-"*80)

        # Before
        b = before[model_key][split]
        print(f"Before (Data Leakage 있음):")
        print(f"  IC Mean:  {b['ic_mean']:>8.4f}")
        print(f"  IC IR:    {b['ic_ir']:>8.4f}")
        print(f"  Return:   {b['ls_ann_ret']*100:>8.2f}%")
        print(f"  Sharpe:   {b['ls_ann_ir']:>8.4f}")

        # After
        a = after[model_key][split]
        print(f"\nAfter (Data Leakage 수정 후):")
        print(f"  IC Mean:  {a['ic_mean']:>8.4f}")
        print(f"  IC IR:    {a['ic_ir']:>8.4f}")
        print(f"  Return:   {a['ls_ann_ret']*100:>8.2f}%")
        print(f"  Sharpe:   {a['ls_ann_ir']:>8.4f}")

        # 변화율
        print(f"\n변화율:")
        ic_change = ((a['ic_mean'] - b['ic_mean']) / abs(b['ic_mean'])) * 100 if b['ic_mean'] != 0 else 0
        ir_change = ((a['ic_ir'] - b['ic_ir']) / abs(b['ic_ir'])) * 100 if b['ic_ir'] != 0 else 0
        ret_change = ((a['ls_ann_ret'] - b['ls_ann_ret']) / abs(b['ls_ann_ret'])) * 100 if b['ls_ann_ret'] != 0 else 0
        sharpe_change = ((a['ls_ann_ir'] - b['ls_ann_ir']) / abs(b['ls_ann_ir'])) * 100 if b['ls_ann_ir'] != 0 else 0

        print(f"  IC Mean:  {ic_change:>+8.1f}%")
        print(f"  IC IR:    {ir_change:>+8.1f}%")
        print(f"  Return:   {ret_change:>+8.1f}%")
        print(f"  Sharpe:   {sharpe_change:>+8.1f}%")

    print("\n" + "="*80)
    print("주요 발견:")
    print("="*80)

    test_before = before[model_key]['test']
    test_after = after[model_key]['test']

    sharpe_drop = test_before['ls_ann_ir'] - test_after['ls_ann_ir']
    sharpe_drop_pct = (sharpe_drop / test_before['ls_ann_ir']) * 100

    print(f"1. Test Sharpe Ratio: {test_before['ls_ann_ir']:.2f} → {test_after['ls_ann_ir']:.2f}")
    print(f"   ({sharpe_drop_pct:+.1f}% 변화)")
    print(f"\n2. Data leakage 제거 후 성능 {' 향상' if test_after['ls_ann_ir'] > test_before['ls_ann_ir'] else '하락'}")

    if test_after['ls_ann_ir'] < 0:
        print(f"\n⚠️  경고: Test Sharpe가 음수입니다!")
        print(f"   이는 모델이 val/test 기간에서 실제로 작동하지 않음을 의미합니다.")
        print(f"   추가 분석이 필요합니다:")
        print(f"   - 더 많은 팩터 사용 (top 10, 20 등)")
        print(f"   - 하이퍼파라미터 튜닝")
        print(f"   - Feature engineering 개선")
        print(f"   - 다른 기간에서의 성능 확인 (rolling backtest)")

    print("\n" + "="*80)

def main():
    """메인 실행"""
    print("Data Leakage Fix 전후 비교 분석")
    print("="*80)

    # 결과 로드
    after = load_after_results()

    if not after:
        print("\n⚠️ 수정 후 결과가 없습니다.")
        print("다음 명령어로 모델을 먼저 학습하세요:")
        print("  python3 model_train.py --rank-ic --top-n 3 --engine lgbm")
        return

    # 요약 출력
    print_summary(BEFORE_RESULTS, after)

    # 시각화
    print("\n그래프 생성 중...")
    plot_comparison(BEFORE_RESULTS, after)

    print("\n✅ 분석 완료!")

if __name__ == "__main__":
    main()
