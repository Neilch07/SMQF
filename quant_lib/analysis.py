import numpy as np
import pandas as pd

def ts_delay(df,n):
    return df.shift(n)

def ts_delta(df,n):
    return df-df.shift(n)

def ts_rank(df,n,min_periods=None,pct=True):
    if min_periods is None:
        min_periods=int(n/2)
    return df.rolling(n,min_periods=min_periods).rank(pct=pct)

def cs_rank(df,pct=True):
    return df.rank(pct=pct,axis=1)

def ts_sum(df,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    return df.rolling(n,min_periods=min_periods).sum()

def cs_sum(df):
    return df.sum(axis=1)

def ts_argmax(df,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    return df.rolling(n,min_periods=min_periods).apply(lambda x:n-x.argmax(),raw=True)

def ts_argmin(df,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    return df.rolling(n,min_periods=min_periods).apply(lambda x:n-x.argmin(),raw=True)

def ema1(df,n,min_periods=None):   # alpha=2/(1+n)
    if min_periods is None:
        min_periods=int(n/2)
    return df.ewm(span=n,min_periods=min_periods)

def ema2(df,n,min_periods=None):   # alpha=1/(1+n)
    if min_periods is None:
        min_periods=int(n/2)
    return df.ewm(com=n,min_periods=min_periods)

def ema3(df,a,min_periods=None):   # alpha is given by a (0<a<1)
    if min_periods is None:
        min_periods=2
    return df.ewm(alpha=a,min_periods=min_periods)

def wma(df,w):
    n=len(w) 
    w=np.array(w).reshape(-1,1)
    nd,ns=df.shape
    avg=np.full((nd,ns),np.nan)
    dv=df.values
    for i in range(n-1,nd):
        v=dv[i-n+1:i+1,:] 
        w_nan=np.where(np.isnan(v),np.nan,w)
        avg[i,:]=np.nansum(v * w, axis=0)/np.nansum(w_nan,axis=0)
    return pd.DataFrame(avg,index=df.index,columns=df.columns)

def ts_mean(df,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    return df.rolling(n,min_periods=min_periods).mean()

def cs_mean(df):
    return df.mean(axis=1)

def ts_std(df,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    return df.rolling(n,min_periods=min_periods).std()

def cs_std(df):
    return df.std(axis=1)

def ts_corr1(df1,df2,n,min_periods=None):  # df1 and df2 have the same shape
    if min_periods is None:
        min_periods=int(n/2)
    return df1.rolling(n,min_periods=min_periods).corr(df2)

def ts_corr2(df1,df2,n,min_periods=None):  # df2 is a DataFrame with 1 column
    if min_periods is None:
        min_periods=int(n/2)
    df=df2.values+np.zeros(df1.shape)
    df=pd.DataFrame(df,index=df1.index,columns=df1.columns)   
    return df1.rolling(n,min_periods=min_periods).corr(df)

def cs_corr(df1,df2):
    return df1.T.corrwith(df2.T)

def ts_median(df,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    return df.rolling(n,min_periods=min_periods).median()

def cs_median(df):
    return df.median(axis=1)

def ts_zscore(df,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    std_df=df.rolling(n,min_periods=min_periods).std()
    std_df=std_df.where(std_df>0,other=np.nan)
    return (df-df.rolling(n,min_periods=min_periods).mean())/std_df

def cs_zscore(df):
    return df.sub(df.mean(axis=1),axis=0).div(df.std(axis=1),axis=0)

def ts_beta(x,y,n,min_periods=None): # x is a DataFrame with 1 column
    if min_periods is None:
        min_periods=int(n/2)
    xx=x*x
    xy=pd.DataFrame(x.values*y.values,index=y.index,columns=y.columns)
    x_bar=x.rolling(n,min_periods=min_periods).mean()
    y_bar=y.rolling(n,min_periods=min_periods).mean()
    xx_bar=xx.rolling(n,min_periods=min_periods).mean()
    xy_bar=xy.rolling(n,min_periods=min_periods).mean()
    xy_bar2=pd.DataFrame(x_bar.values*y_bar.values,index=y.index,columns=y.columns)
    z=xx_bar-x_bar*x_bar
    z=z.where(z>0,np.nan)
    b=(xy_bar-xy_bar2).values/z.values
    return pd.DataFrame(b,index=y.index,columns=y.columns)

def ts_regression(x,y,n): # x is a DataFrame with 1 column
    nd,ns=y.shape
    xv=x.values
    yv=y.values

    slopes=np.full((nd,ns),np.nan)
    intcepts=np.full((nd,ns),np.nan)
    eps=np.full((nd,ns),np.nan)
    for i in range(n-1,nd):
        xs=xv[i-n+1:i+1,:]
        ys=yv[i-n+1:i+1,:]
        xm=np.nanmean(xs,axis=0)
        ym=np.nanmean(ys,axis=0)
        x_bar=xs-xm
        y_bar=ys-ym
        b=np.nanmean(x_bar*y_bar,axis=0)/np.nanmean(x_bar*x_bar,axis=0)
        a=ym-b*xm
        ep=np.nanstd(a+b*xs-ys,axis=0)
        slopes[i,:]=b
        intcepts[i,:]=a
        eps[i,:]=ep
    
    slopes=pd.DataFrame(slopes,index=y.index,columns=y.columns)
    intcepts=pd.DataFrame(intcepts,index=y.index,columns=y.columns)
    eps=pd.DataFrame(eps,index=y.index,columns=y.columns)
    return (slopes,intcepts,eps)

if __name__ == '__main__':
    import time

    nd,ns=2000,5000
    dx= pd.DataFrame(np.random.randn(nd,1))
    dy= pd.DataFrame(np.random.randn(nd,ns))
    dy2= pd.DataFrame(np.random.randn(nd,ns))

    n=20
    t1=time.perf_counter()
    z=ts_sum(dy,n)
    t2=time.perf_counter()
    print('ts_sum time:',t2-t1,"sec")
    z=cs_sum(dy)
    t3=time.perf_counter()
    print('cs_sum time:',t3-t2,"sec")
    t1=time.perf_counter()
    z=ts_rank(dy,n)
    t2=time.perf_counter()
    print('ts_rank time:',t2-t1,"sec")
    z=cs_rank(dy)
    t3=time.perf_counter()
    print('cs_rank time:',t3-t2,"sec")
    z=ts_argmax(dy,n)
    t4=time.perf_counter()
    print('ts_argmax time:',t4-t3,"sec")
    z=ema1(dy,n)
    t5=time.perf_counter()
    print('ema1 time:',t5-t4,"sec")
    z=wma(dy,w=np.arange(1,n+1))
    t6=time.perf_counter()
    print('wma time:',t6-t5,"sec")
    z=ts_median(dy,n)
    t7=time.perf_counter()
    print('ts_median time:',t7-t6,"sec")
    z=cs_median(dy)
    t8=time.perf_counter()
    print('cs_median time:',t8-t7,"sec")
    z=ts_std(dy,n)
    t9=time.perf_counter()
    print('ts_std time:',t9-t8,"sec")
    z=cs_std(dy)
    t10=time.perf_counter()
    print('cs_std time:',t10-t9,"sec")
    z=ts_zscore(dy,n)
    t11=time.perf_counter()
    print('ts_zscore time:',t11-t10,"sec")
    z=cs_zscore(dy)
    t12=time.perf_counter()
    print('cs_zscore time:',t12-t11,"sec")
    z=ts_corr1(dy,dy2,n)
    t13=time.perf_counter()
    print('ts_corr1 time:',t13-t12,"sec")
    z=ts_corr2(dy,dx,n)
    t14=time.perf_counter()
    print('ts_corr2 time:',t14-t13,"sec")
    z=cs_corr(dy,dy2)
    t15=time.perf_counter()
    print('cs_corr time:',t15-t14,"sec")
    z=ts_beta(dx,dy,n)
    t16=time.perf_counter()
    print('ts_beta time:',t16-t15,"sec")
    z=ts_regression(dx,dy,n)
    t17=time.perf_counter()
    print('ts_regression time:',t17-t16,"sec")


