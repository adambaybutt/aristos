import pandas as pd
import numpy as np

def initialClean(df: pd.DataFrame) -> pd.DataFrame:
    ''' Performs initial checks and clean of the OHLCV data.
    
    Args:
        df (pd.DataFrame): trade level data with columns for date, price, volume, and buy/sell.
    
    Returns:
        df (pd.DataFrame): trade level data with columns for X Y AND Z # TODO FINISH
    '''
    # name columns
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'trades']

    # keep only relevant columns
    df = df[['date', 'close', 'volume', 'trades']]

    # convert/confirm data types and compress size
    df['date'] = pd.to_datetime(df.date, unit='s')
    df = df.rename(columns = {'close': 'price'})
    df['price']  = df.price.astype('float32')
    df['volume'] = df.volume.astype('float32')
    df['trades'] = df.volume.astype('float32')

    # confirm range of values
    assert(df.shape[0] == np.sum(df.trades >= 0)),('number of trades has negative values.')
    assert(df.shape[0] == np.sum(df.trades < 1e6)),('number of trades is implausibly high.')
    assert(df.shape[0] == np.sum(df.volume >= 0)),('volume has negative values.')
    assert(df.shape[0] == np.sum(df.volume < 1e9)),('volume reported is implausibly high.')
    assert(df.shape[0] == np.sum(df.price >= 0)),('price has a negative value.')
    assert(df.shape[0] == np.sum(df.price < 1e6)),('price reported is implausibly high.')

    # subset to time range of interest
    df = df[(df.date.dt.year >= 2015) & (df.date.dt.year <= 2022)]

    # ensure no duplicate observations
    df = df.drop_duplicates()

    # interpolate missing observations
    price_df = df[['date', 'price']].set_index('date')
    price_df = price_df.resample('1min').ffill()
    df = df.drop('price', axis=1)
    df = df.merge(price_df, on='date', how='outer', validate='one_to_one')
    df = df.fillna(0)
    
    # ensure time is unique
    assert(df.shape[0] == len(np.unique(df.date))),("date column is not unique.")

    # ensure no missing values
    assert(0==df.isnull().sum().sum())

    # sort values
    df = df.sort_values(by='date', ignore_index=True)
    
    return df

def merge(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> pd.DataFrame:
    ''' Concatenate minute-level price data into panel.
    
    Args:
        btc_df (pd.DataFrame): minute level price, volume, and trades for BTC.
        eth_df (pd.DataFrame): minute level price, volume, and trades for ETH.
    
    Returns:
        df (pd.DataFrame): minute level panel data of price, volume, and trades.
    '''
    # add asset names
    btc_df['asset'] = 'btc'
    eth_df['asset'] = 'eth'

    # concatenatte
    df = pd.concat((btc_df, eth_df))

    # set column order
    df = df[['date', 'asset', 'price', 'volume', 'trades']]

    # sort values
    df = df.sort_values(by=['date', 'asset'], ignore_index=True)

    return df

if __name__ == "__main__":
    # set fps
    eth_fp = '../1-data/raw/ETHUSD_1.csv'
    btc_fp = '../1-data/raw/XBTUSD_1.csv'

    # import
    btc_df = pd.read_csv(btc_fp, header=None)
    eth_df = pd.read_csv(eth_fp, header=None)

    # clean
    btc_df = initialClean(btc_df)
    eth_df = initialClean(eth_df)

    # merge
    df = merge(btc_df, eth_df)

    # save
    df.to_pickle('../1-data/clean/panel_btceth_1min.pkl')
