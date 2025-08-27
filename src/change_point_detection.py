import polars as pl
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np

def clean_up_time_series_data(
        df: pl.DataFrame, 
        topic_col:str = "dominant_topic", 
        topic_values:list = [0],
        duration: str = "1d"):
    """
    Cleaning up and calculating percentage and count per topic and aggregation level: 
    duration from ["1d", "1w", "2w", "1mo", "2mo", "3mo", "6mo"]
    """
    # ------------------------------------
    # 1. Datetime formatting
    date_col = 'date'
    df = df.with_columns([pl.col(date_col).cast(pl.Datetime("ms")).cast(pl.Date()).alias(date_col)]).sort("date")
    df = df.with_columns(pl.col(date_col).dt.truncate(duration).alias('period'))

    # 2. Seperate to values fro list to others
    if topic_values is not None: 
        df = df.with_columns(
            pl.when(pl.col(topic_col).is_in(topic_values))
            .then(pl.col(topic_col))
            .otherwise(-1)
            .alias(topic_col)
            .cast(pl.Int8))

    # 3. Calculating the counts per topic and the respective percentages
    tot = df.group_by('period').agg(pl.len().alias("total"))

    tmp = (
        df.group_by(['period', topic_col])
        .agg(pl.len().alias("cnt"))
        .join(tot, on='period')
        .with_columns((pl.col("cnt") / pl.col("total") * 100).alias("pct"))
    )

    wide = (
        tmp
        .pivot(                      
            index="period",
            on=topic_col,
            values=["cnt", "pct"], 
            aggregate_function=None
        )
        .join(
            tmp.select(["period", "total"]).unique(),
            on="period"
        )
        .with_columns(                # zero-fill only the new cnt/pct columns
            pl.col("^cnt_.*$").fill_null(0),
            pl.col("^pct_.*$").fill_null(0),
        )
        .sort("period")
    )
    wide = wide[['period', 'total'] + sorted([x for x in wide.columns if x not in  ['period', 'total']])]
    # Adding missed dates with 0
    daterange_df = pl.date_range(wide['period'].min(), wide['period'].max(), duration, eager=True).alias("period").to_frame()
    wide = daterange_df.join(wide, left_on = 'period', right_on="period", how="left")
    wide = wide.fill_null(0) 
    wide = wide.sort('period')
    return wide

def _add_dur(ts: pd.Timestamp, dur: str) -> pd.Timestamp:
    if dur.endswith("mo"):                  # months
        return ts + relativedelta(months=int(dur[:-2]))
    unit = dur[-1]                          # 'd' or 'w'
    n = int(dur[:-1])
    return ts + (pd.Timedelta(days=n) if unit == "d" else pd.Timedelta(weeks=n))

def create_time_series_events(df:pl.DataFrame, count_col:str = 'cnt_0', duration:str = '1d', period_col:str = 'period',):
    # 1. Expand count into individual event times
    events = []
    for start, cnt in zip(df[period_col].to_numpy(), df[count_col].to_numpy()):
        if cnt == 0:
            continue
        start = pd.Timestamp(start)
        end   = _add_dur(start, duration)
        step  = (end - start) / (cnt + 1)
        events.extend(start + step * (k + 1) for k in range(cnt))

    events = np.sort(np.array(events, dtype="datetime64[ns]"))
    return events

def calculate_cumsum_statistics(events:np.ndarray, plot = False, **galeano_kwargs):
    """
    Calculating the Cumulative Sum Statistics based on the paper by Galeano 2006 (doi:10.1016/j.csda.2006.12.042)
    
    Returns
    -------
    Tuple
        (event_days: days since the very first paper, Vector of the Galeano test statistic, galeano_crit_value)
    """ 
    t0 = events[0]
    event_days = (events - t0).astype("timedelta64[s]").astype(float) / 86400.0
    n = len(event_days)
    D = np.sqrt(n) * (event_days / event_days[-1] - np.arange(1, n + 1) / n)
    critical_value = galeano_crit_value(n, **galeano_kwargs)

    if plot: 
        plt.plot(event_days, D)
        plt.axhline(y=critical_value, color='grey', linestyle='--', linewidth=1)
        plt.title("CUSUM deviations")
        plt.xlabel("days since first paper")
        plt.ylabel("D_n")
        plt.show()

    return (event_days, D, critical_value)

def galeano_crit_value(n, alpha=0.05, nsim=10_000, seed=42):
    """
    Monte-Carlo critical value (lambda_max statistic) calculation based on the paper by Galeano 2006 (doi:10.1016/j.csda.2006.12.042).
    
    Parameters
    ----------
    n : int 
        Segment length (number of events).
    alpha : float, default 0.05
        Test size. 0.05 corresponds to the p=0.95 column in Galeano Table 1.
    nsim : int, default 10_000
        Number of Monte-Carlo replications.
    seed : int or None
        RNG seed for reproducibility.
        
    Returns
    -------
    float
        Critical value c such that P(lambda_max <= c) = 1-alpha.
    """
    rng = np.random.default_rng(seed)
    lam_max = np.empty(nsim)

    for k in range(nsim):
        gaps = rng.exponential(scale=1.0, size=n)         # d_1 ... d_n
        t    = np.cumsum(gaps)                            # t_1 ... t_n
        D    = np.sqrt(n) * (t / t[-1] - (np.arange(1, n+1) / n))
        lam_max[k] = np.abs(D).max()

    return np.quantile(lam_max, 1 - alpha)

def _segment_stat(times):
    """times: 1-D float array, time since segment start (same unit)"""
    n = len(times)
    D = np.sqrt(n) * (times / times[-1] - np.arange(1, n+1) / n)
    absD = np.abs(D)
    return absD.max(), absD.argmax()            # lambda_max, argmax local index

def galeano_binary_segmentation(events:np.ndarray, alpha0=0.05, nsim=10_000, min_size=30):
    """
    events: numpy datetime64[ns] array, strictly increasing
    alpha0: family-wise error rate (e.g. 0.05)
    nsim: MC replications for every critical value
    min_size: don't split segments shorter than this

    example: 
    cps, info = galeano_binary_seg(events, alpha0=0.05, nsim=20_000)
    for d in info:
        print(f"change @ idx {d['index']:>5}  {d['date']}  "
            f"lambda={d['stat']:.3f}  crit={d['crit']:.3f}")
    """
    # convert once to float seconds for cheap arithmetic
    ev_sec = events.astype('datetime64[s]').astype(float)

    change_idx = []   # global indices of accepted changepoints
    change_info = []   # (index, datetime, lambda_max, critical)

    def recurse(lo, hi, m_found):
        n_seg = hi - lo
        if n_seg < min_size:
            return

        seg_times = ev_sec[lo:hi] - ev_sec[lo]     # relative seconds
        lam, arg = _segment_stat(seg_times)

        # Bonferroni-style adjusted alpha per Galeano
        alpha_r = 1.0 - (1.0 - alpha0)**(1.0 / (m_found + 1))
        crit    = galeano_crit_value(n_seg, alpha=alpha_r, nsim=nsim)

        if lam > crit:                             # significant change
            cp = lo + arg
            change_idx.append(cp)
            change_info.append(
                dict(index=cp,
                     date=str(events[cp]),
                     stat=lam,
                     crit=crit,
                     seg_start=lo,
                     seg_end=hi)
            )
            recurse(lo,  cp + 1, len(change_idx))  # left segment
            recurse(cp + 1, hi, len(change_idx))   # right segment

    recurse(0, len(events), 0)
    change_idx.sort()
    change_info.sort(key=lambda d: d['index'])
    return change_idx, change_info
