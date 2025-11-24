"""
Rolling Window Backtest
시간에 따른 모델 성능 평가 - 다양한 기간에서 IR 측정
"""
import os
import sys
import json
import subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정
rcParams['font.family'] = 'AppleGothic'
rcParams['axes.unicode_minus'] = False

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_rolling_windows(
    start_date: str,
    end_date: str,
    train_months: int = 24,  # 24개월 학습
    val_months: int = 12,    # 12개월 검증
    test_months: int = 12,   # 12개월 테스트
    step_months: int = 6     # 6개월씩 이동
) -> List[Tuple[str, str, str, str, str, str]]:
    """Rolling window 기간 생성

    Returns:
        List of (start, end, train_end, val_end, train_start, val_start)
    """
    windows = []

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    current_train_start = start

    while True:
        train_end = current_train_start + pd.DateOffset(months=train_months)
        val_end = train_end + pd.DateOffset(months=val_months)
        test_end = val_end + pd.DateOffset(months=test_months)

        # 테스트 종료일이 전체 기간을 넘으면 중단
        if test_end > end:
            break

        windows.append((
            current_train_start.strftime('%Y-%m-%d'),
            test_end.strftime('%Y-%m-%d'),
            train_end.strftime('%Y-%m-%d'),
            val_end.strftime('%Y-%m-%d'),
            current_train_start.strftime('%Y-%m-%d'),
            train_end.strftime('%Y-%m-%d')
        ))

        # 다음 윈도우로 이동
        current_train_start += pd.DateOffset(months=step_months)

    return windows


def run_single_window(
    window_idx: int,
    start_date: str,
    end_date: str,
    train_end: str,
    val_end: str,
    top_n: int,
    engine: str
) -> Dict:
    """단일 윈도우 실험 실행"""
    print(f"\n{'='*80}")
    print(f"Window {window_idx}: {start_date} ~ {end_date}")
    print(f"  Train: {start_date} ~ {train_end}")
    print(f"  Val:   {train_end} ~ {val_end}")
    print(f"  Test:  {val_end} ~ {end_date}")
    print(f"{'='*80}")

    cmd = [
        'python3', 'model_train.py',
        '--rank-ic',
        '--top-n', str(top_n),
        '--engine', engine,
        '--start-date', start_date,
        '--end-date', end_date,
        '--train-end', train_end,
        '--val-end', val_end
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=THIS_DIR,
            capture_output=True,
            text=True,
            timeout=600
        )

        # JSON 파싱
        output_lines = result.stdout.split('\n')
        for i, line in enumerate(output_lines):
            if line.strip().startswith('{'):
                json_start = i
                json_text = '\n'.join(output_lines[json_start:])

                depth = 0
                json_end = 0
                for j, char in enumerate(json_text):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            json_end = j + 1
                            break

                if json_end > 0:
                    metrics = json.loads(json_text[:json_end])
                    return {
                        'window_idx': window_idx,
                        'start_date': start_date,
                        'end_date': end_date,
                        'train_end': train_end,
                        'val_end': val_end,
                        'metrics': metrics,
                        'success': True
                    }
    except Exception as e:
        print(f"❌ 실험 실패: {e}")
        return {'success': False, 'error': str(e)}

    return {'success': False, 'error': 'Failed to parse output'}


def run_rolling_backtest(
    top_n: int = 10,
    engine: str = 'lgbm',
    train_months: int = 24,
    val_months: int = 12,
    test_months: int = 12,
    step_months: int = 6
):
    """Rolling backtest 실행"""

    # 윈도우 생성
    windows = generate_rolling_windows(
        start_date='2018-01-01',
        end_date='2023-12-31',
        train_months=train_months,
        val_months=val_months,
        test_months=test_months,
        step_months=step_months
    )

    print(f"\n{'='*80}")
    print(f"Rolling Backtest 설정")
    print(f"{'='*80}")
    print(f"총 윈도우 수: {len(windows)}")
    print(f"학습 기간: {train_months}개월")
    print(f"검증 기간: {val_months}개월")
    print(f"테스트 기간: {test_months}개월")
    print(f"이동 간격: {step_months}개월")
    print(f"{'='*80}\n")

    results = []

    for idx, (start, end, train_end, val_end, _, _) in enumerate(windows, 1):
        result = run_single_window(
            window_idx=idx,
            start_date=start,
            end_date=end,
            train_end=train_end,
            val_end=val_end,
            top_n=top_n,
            engine=engine
        )

        if result['success']:
            results.append(result)

            # 중간 저장
            output_file = os.path.join(
                THIS_DIR,
                'artifacts',
                f'rolling_backtest_{engine}_top{top_n}.json'
            )
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def plot_rolling_results(results: List[Dict], engine: str, top_n: int):
    """Rolling backtest 결과 시각화"""

    if not results:
        print("결과가 없습니다.")
        return

    # 데이터 추출
    test_periods = []
    train_irs = []
    val_irs = []
    test_irs = []
    test_returns = []
    test_ics = []

    for r in results:
        if not r.get('success'):
            continue

        # Test 기간 중간 날짜
        val_end = pd.to_datetime(r['val_end'])
        end = pd.to_datetime(r['end_date'])
        test_mid = val_end + (end - val_end) / 2
        test_periods.append(test_mid)

        metrics = r['metrics']
        train_irs.append(metrics['train']['ls_ann_ir'])
        val_irs.append(metrics['val']['ls_ann_ir'])
        test_irs.append(metrics['test']['ls_ann_ir'])
        test_returns.append(metrics['test']['ls_ann_ret'] * 100)
        test_ics.append(metrics['test']['ic_mean'])

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Rolling Backtest 결과: {engine.upper()} (Top {top_n} Factors)',
                 fontsize=16, fontweight='bold')

    # 1. Sharpe Ratio 시계열
    axes[0, 0].plot(test_periods, train_irs, 'o-', label='Train', alpha=0.7, linewidth=2)
    axes[0, 0].plot(test_periods, val_irs, 's-', label='Val', alpha=0.7, linewidth=2)
    axes[0, 0].plot(test_periods, test_irs, '^-', label='Test', alpha=0.7, linewidth=2)
    axes[0, 0].axhline(y=np.mean(test_irs), color='red', linestyle='--',
                       label=f'Test 평균: {np.mean(test_irs):.2f}')
    axes[0, 0].set_xlabel('Test 기간')
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].set_title('시간에 따른 Sharpe Ratio 변화')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Test Return
    axes[0, 1].bar(range(len(test_returns)), test_returns, alpha=0.7, color='skyblue')
    axes[0, 1].axhline(y=np.mean(test_returns), color='red', linestyle='--',
                       label=f'평균: {np.mean(test_returns):.1f}%')
    axes[0, 1].set_xlabel('Window')
    axes[0, 1].set_ylabel('연간 수익률 (%)')
    axes[0, 1].set_title('Window별 Test 수익률')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Test IC
    axes[1, 0].plot(test_periods, test_ics, 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].axhline(y=np.mean(test_ics), color='red', linestyle='--',
                       label=f'평균: {np.mean(test_ics):.4f}')
    axes[1, 0].set_xlabel('Test 기간')
    axes[1, 0].set_ylabel('IC Mean')
    axes[1, 0].set_title('시간에 따른 IC 변화')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. 성능 안정성 (Train vs Test)
    axes[1, 1].scatter(train_irs, test_irs, s=100, alpha=0.6, color='purple')

    # 1:1 라인
    min_ir = min(min(train_irs), min(test_irs))
    max_ir = max(max(train_irs), max(test_irs))
    axes[1, 1].plot([min_ir, max_ir], [min_ir, max_ir], 'r--', alpha=0.5, label='1:1 라인')

    # 추세선
    z = np.polyfit(train_irs, test_irs, 1)
    p = np.poly1d(z)
    axes[1, 1].plot(train_irs, p(train_irs), "b-", alpha=0.5,
                    label=f'추세선: y={z[0]:.2f}x+{z[1]:.2f}')

    axes[1, 1].set_xlabel('Train Sharpe Ratio')
    axes[1, 1].set_ylabel('Test Sharpe Ratio')
    axes[1, 1].set_title('과적합 분석 (Train vs Test)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(
        THIS_DIR,
        'artifacts',
        f'rolling_backtest_{engine}_top{top_n}.png'
    )
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 저장: {output_file}")

    # 요약 통계
    print(f"\n{'='*80}")
    print(f"Rolling Backtest 요약 통계")
    print(f"{'='*80}")
    print(f"Test Sharpe Ratio:")
    print(f"  평균:     {np.mean(test_irs):.4f}")
    print(f"  표준편차:  {np.std(test_irs):.4f}")
    print(f"  최소:     {np.min(test_irs):.4f}")
    print(f"  최대:     {np.max(test_irs):.4f}")
    print(f"\nTest 연간 수익률:")
    print(f"  평균:     {np.mean(test_returns):.2f}%")
    print(f"  표준편차:  {np.std(test_returns):.2f}%")
    print(f"\nTest IC:")
    print(f"  평균:     {np.mean(test_ics):.4f}")
    print(f"  표준편차:  {np.std(test_ics):.4f}")
    print(f"\n성능 저하율 (Train → Test):")
    avg_degradation = (np.mean(test_irs) - np.mean(train_irs)) / np.mean(train_irs) * 100
    print(f"  평균:     {avg_degradation:.1f}%")
    print(f"{'='*80}\n")


def main():
    """메인 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="Rolling Window Backtest")
    parser.add_argument('--top-n', type=int, default=10, help='Number of top factors')
    parser.add_argument('--engine', type=str, default='lgbm', choices=['lgbm', 'xgb', 'catboost'])
    parser.add_argument('--train-months', type=int, default=24, help='Training period in months')
    parser.add_argument('--val-months', type=int, default=12, help='Validation period in months')
    parser.add_argument('--test-months', type=int, default=12, help='Test period in months')
    parser.add_argument('--step-months', type=int, default=6, help='Step size in months')

    args = parser.parse_args()

    print("=" * 80)
    print("Rolling Window Backtest 시작")
    print("=" * 80)

    # artifacts 디렉토리 생성
    os.makedirs(os.path.join(THIS_DIR, 'artifacts'), exist_ok=True)

    # Rolling backtest 실행
    results = run_rolling_backtest(
        top_n=args.top_n,
        engine=args.engine,
        train_months=args.train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        step_months=args.step_months
    )

    # 결과 시각화
    plot_rolling_results(results, args.engine, args.top_n)

    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
