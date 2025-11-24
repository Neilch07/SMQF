# 프로젝트 문제점 진단 및 해결 방안

## 🔴 발견된 문제점

### 1. 비현실적으로 높은 Train IR (13.19)
- **증상**: Train IR=13.19 → Val IR=9.06 → Test IR=4.94
- **원인**: Cross-sectional normalization이 과적합을 유발

### 2. Cross-sectional Z-score의 문제점

**현재 구현** (model_train.py Line 301):
```python
train_stats = X_train.groupby(level='date').agg(['mean', 'std'])
```

**문제**:
1. 매일 종목들을 정규화 → 상대적 순위가 너무 명확
2. 모델이 "어떤 종목이 항상 강하다"는 패턴을 쉽게 학습
3. Train 데이터에 과적합 (out-of-sample에서 성능 급락)

**증거**:
- Train IC: 0.0554 (높음)
- Test IC: 0.0386 (30% 감소)
- Train IR: 13.19 (비현실적)
- Test IR: 4.94 (62% 감소)

## ✅ 해결 방안

### Option 1: Time-series Normalization (권장)
각 팩터를 시계열로 정규화:

```python
def normalize_time_series(X, train_idx, clip=5.0):
    """각 팩터별로 시계열 정규화 (cross-section 아님)"""
    X_norm = X.copy()
    X_train = X.loc[train_idx]
    
    # 각 컬럼(팩터)별로 정규화
    for col in X.columns:
        mean = X_train[col].mean()
        std = X_train[col].std()
        
        if std > 1e-8:
            X_norm[col] = (X[col] - mean) / std
            X_norm[col] = X_norm[col].clip(-clip, clip)
    
    return X_norm
```

### Option 2: Rolling Window Normalization
과거 N일 이동평균/표준편차 사용:

```python
def normalize_rolling(X, window=60, clip=5.0):
    """Rolling window로 정규화 (look-ahead bias 방지)"""
    X_norm = X.copy()
    
    for col in X.columns:
        # Expanding window (훈련 기간 동안)
        rolling_mean = X[col].expanding(min_periods=window).mean()
        rolling_std = X[col].expanding(min_periods=window).std()
        
        X_norm[col] = (X[col] - rolling_mean) / rolling_std
        X_norm[col] = X_norm[col].clip(-clip, clip)
    
    return X_norm
```

### Option 3: Cross-sectional Rank (대안)
Z-score 대신 percentile rank 사용:

```python
def cross_sectional_rank(X):
    """매일 종목별 순위를 percentile로 변환"""
    X_ranked = X.copy()
    
    for date in X.index.get_level_values('date').unique():
        date_mask = X.index.get_level_values('date') == date
        X_ranked.loc[date_mask] = X.loc[date_mask].rank(pct=True)
    
    return X_ranked
```

## 📊 추가 확인 사항

### 1. Factor IC 분포 확인
```bash
# IC가 너무 높은 팩터는 의심
# IC > 0.03은 비정상적으로 높을 수 있음
```

### 2. Train/Test 기간 확인
- Train: 2021-01 ~ 2022-06 (18개월)
- Val: 2022-07 ~ 2023-03 (9개월)
- Test: 2023-04 ~ 2023-12 (9개월)

→ 기간이 너무 짧을 수 있음 (최소 3년 이상 권장)

### 3. Horizon 설정
- 현재: horizon=5 (5일)
- 너무 짧으면 노이즈가 많음
- 추천: horizon=10 또는 20

## 🎯 권장 조치

1. **즉시 적용**:
   - Time-series normalization으로 변경
   - Train/Test 기간 연장 (2018-2023)

2. **검증**:
   - Walk-forward validation 실행
   - IC 분포 히스토그램 확인
   - Feature importance 분석

3. **장기 개선**:
   - 더 많은 팩터 테스트
   - Ensemble 모델 구축
   - Transaction cost 고려

## 📝 예상 결과

Time-series normalization 적용 후:
- Train IR: 3~5 (현실적)
- Test IR: 2~3 (안정적)
- Train/Test IC 차이: <20%
