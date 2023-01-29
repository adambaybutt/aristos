import pandas as pd
import numpy as np

def initialClean(df: pd.DataFrame) -> pd.DataFrame:
    ''' Performs initial checks and clean of the trade level data.
    
    Args:
        df (pd.DataFrame): trade level data with columns for date, price, and trades.
    
    Returns:
        df (pd.DataFrame): trade level data with columns for date, price, volume and trades.
    '''
    # name columns
    df.columns = ['date', 'close', 'volume']

    # keep only relevant columns
    df = df[['date', 'close', 'volume']]

    # convert/confirm data types and compress size
    df['date'] = pd.to_datetime(df.date, unit='s')
    df = df.rename(columns = {'close': 'price'})
    df['price']  = df.price.astype('float32')
    df['volume'] = df.volume.astype('float32')
    df['trades'] = 1
    df['trades'] = df.trades.astype('float32')

    # confirm range of values
    assert(df.shape[0] == np.sum(df.volume >= 0)),('volume has negative values.')
    assert(df.shape[0] == np.sum(df.volume < 1e9)),('volume is implausibly high.')
    assert(df.shape[0] == np.sum(df.price >= 0)),('price has a negative value.')
    assert(df.shape[0] == np.sum(df.price < 1e6)),('price reported is implausibly high.')

    # subset to time range of interest
    df = df[(df.date.dt.year >= 2015) & (df.date.dt.year <= 2022)]

    # ensure no duplicate observations
    df = df.drop_duplicates()

    return df

def collapseToMinuteLevel(df: pd.DataFrame) -> pd.DataFrame:
    ''' Collapse trade level data to minute level.
    
    Args:
        df (pd.DataFrame): clean trade level data with columns for date, price, volume and trades.
    
    Returns:
        df (pd.DataFrame): minute level data with columns for date, price, volume and trades.
    '''
    # round time up to nearest minute
    df['date'] = df.date.dt.ceil(freq='min')

    # collapse down to one minute level
    df = df.sort_values(by='date')
    avg_df  = df.groupby('date')[['price']].last()
    sum_df  = df.groupby('date')[['volume', 'trades']].sum()
    df = avg_df.merge(sum_df, on='date', how='inner', validate='one_to_one')
    assert(df.shape[0] == len(np.unique(df.index))),('dates not unique on date after dropping dups')
    df = df.sort_values(by='date')

    # interpolate missing observations
    price_df = df[['price']].copy()
    price_df = price_df.resample('1min').ffill()
    df = df.drop('price', axis=1)
    df = df.merge(price_df, on='date', how='outer', validate='one_to_one')
    df = df.fillna(0)
    
    # ensure time is unique
    assert(df.shape[0] == len(np.unique(df.index))),("date column is not unique.")

    # ensure no missing values
    assert(0==df.isnull().sum().sum())

    # sort values
    df = df.reset_index().sort_values(by='date', ignore_index=True)

    # sort columns
    df = df[['date', 'price', 'volume', 'trades']]
    
    return df

if __name__ == "__main__":
    # set fps
    ethbtc_fp = '../1-data/raw/ETHXBT_1.csv'
    out_fp = '../1-data/clean/panel_ethbtc_1min.pkl'

    # import
    ethbtc_df = pd.read_csv(ethbtc_fp, header=None)

    # clean
    df = initialClean(ethbtc_df)
    df = collapseToMinuteLevel(df)

    # save
    df.to_pickle(out_fp)
