import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import datetime as dt

def pullYahooFinanceData(series: str, col_name: str) -> pd.DataFrame:
    ''' Use yfinance API to pull data from Yahoo Finance for given series.
    
    Args:
        series (str): Yahoo Finance series to pull.
        col_name (str): the column name to label the requested data.
    
    Returns:
        df (pd.DataFrame): df with date column and column of requested data.
    '''
    # pull data
    data = yf.Ticker(series)
    df = data.history(period='max')

    # clean columns
    df = df.reset_index()
    df = df.rename(columns={'Date': 'date', 'Close': col_name})
    df = df[['date', col_name]]
    df['date'] = pd.to_datetime(df.date)

    # adjust window and time stamp
    df = df[df.date >= '2015-10-25']
    df['date'] = df.date + pd.DateOffset(hours=21)
    df['date'] = df.date.dt.tz_localize('UTC')

    # clean up index
    df = df.reset_index(drop=True)

    # TODO check the latency on this data to adjust accordingly
    # TODO need to learn proper how to handle date times in pandas; maybe a quick 2 hour course on it

    return df

def pullFredData(series: str, col_name: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    '''' Use pandas datareader API to pull data from FRED for given series.
    
    Args:
        series (str): FRED series to pull.
        col_name (str): the column name to label the requested data.
        start (dt.datetime): start of window of interest.
        end (dt.datetime): end of window.
    
    Returns:
        df (pd.DataFrame): df with date column and column of requested data.
    '''
    # pull data
    df = web.DataReader(series, 'fred', start, end)

    # clean columns
    df = df.reset_index()
    df = df.rename(columns = {'DATE': 'date', series: col_name})
    df = df[['date', col_name]]
    df['date'] = pd.to_datetime(df.date)

    # adjust time stamp
    if series=='WM2NS':
        df['date'] = df.date + pd.DateOffset(days=60)
    elif series in ['DGS2', 'DGS10', 'T5YIE']:
        df['date'] = df.date + pd.DateOffset(days=4)
        df['date'] = df.date + pd.DateOffset(hours=21.25)
    else:
        assert(1==0),('adjust time offset for a new series!')

    df['date'] = df.date.dt.tz_localize('UTC')

    # clean up index
    df = df.reset_index(drop=True)

    return df

def mergeAndClean(dfs: list) -> pd.DataFrame:
    ''' Clean the macro data.

    Args:
        dfs (list): list of dataframes from yfinance or FRED.
    
    Returns:
        df (pd.DataFrame): clean data.    
    '''
    # merge the dfs together
    df = dfs[0].merge(dfs[1],
                    on='date',
                    how='outer',
                    validate='one_to_one')
    for new_df in dfs[2:]:
        df = df.merge(new_df,
                    on='date',
                    how='outer',
                    validate='one_to_one')

    # resort and interpolate
    df = df.sort_values(by='date')
    df = df.set_index('date')
    df = df.interpolate(method='linear')
    df = df['2015-10-31':]

    # resample to 1 minute windows
    df = df.resample("min").interpolate(method='linear')

    # cut window down
    df = df['2015-11-01':'2022-12-15']

    # confirm missingness and range of values
    assert(0==df.isnull().sum().sum()),('missing values')
    assert(df.max().max() < 1e9),('values seem too large')
    assert(df.min().min() > 0),('values seem too large')

    # ensure sorted and reset index
    df = df.sort_values(by='date')
    df = df.reset_index()

    return df

if __name__ == "__main__":
    # pull yfinance data
    snp_df  = pullYahooFinanceData('^SP500-20', 'snp_indust_t')
    ixic_df = pullYahooFinanceData('^IXIC', 'ixic_t')
    vix_df  = pullYahooFinanceData('^VIX', 'vix_t')

    # pull fred data
    start = dt.datetime(2015, 8, 1)
    end   = dt.datetime(2022, 12, 15)
    m2_df    = pullFredData('WM2NS', 'm2_t', start, end)
    t2yr_df  = pullFredData('DGS2', 'trsy_2yr_t', start, end)
    t10yr_df = pullFredData('DGS10', 'trsy_10yr_t', start, end)
    be5yr_df = pullFredData('T5YIE', 'brkevn_5yr_t', start, end)

    # merge and clean the data
    df = mergeAndClean([snp_df, ixic_df, vix_df, m2_df, t2yr_df, t10yr_df, be5yr_df])

    # save
    df.to_pickle('../1-data/raw/macro.pkl')