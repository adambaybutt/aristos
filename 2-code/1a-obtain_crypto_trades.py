import pandas as pd
import time
import requests

def makeKrakenApiTradeCall(trade_url: str, params: dict) -> tuple:
    ''' Pull Kraken trades for specified start time.
    
    Args:
        trade_url (str): url of kraken rest api for trade history.
        params (dict): api call parameters.
    
    Returns:
        (tuple):
            - (list): trade lists of format [pricce, volume, time
                      buy/sell, market/limit, misc., trade_id]
            - (str): unix time of last trade
    '''
    resp = requests.get(trade_url, params=params)
    assert(resp.status_code==200),"bad api call"
    try:
        result = resp.json()['result']
    except:
        print("Bad result:")
        print(resp.json())
    return (result[params['pair']], result['last'])
    
def pullTrades(base_url: str, pair: str, start_time: int, end_time: int) -> list:
    ''' Pull Karken trades for given pair and window.

    Args:
        base_url (str):   base Kraken url to use to build specific api url.
        pair (str):       pair to pull.
        start_time (int): beginning of time window of interest. 
        end_time (int):   end of time window of interest.

    Returns:
        trades (list): list of lists of trades.
    '''
    # build args
    trade_url  = base_url+'Trades'
    trades = []

    while start_time < end_time:
        # pull new trades
        params = {'pair': pair, 'since': str(start_time)}
        new_trades, start_time = makeKrakenApiTradeCall(trade_url, params)

        # adjust unix time
        start_time = int(int(start_time)/1e9)

        # update data
        trades  += new_trades

        # space out calls
        time.sleep(0.3)

        # update on progress
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time)))

    return trades
    
def clean(trades) -> pd.DataFrame:
    ''' Clean the raw trade data into usable raw data.

    Args:
        trades (list): trade lists of format [pricce, volume, time
                       buy/sell, market/limit, misc., trade_id].

    Returns:
        df (pd.DataFrame): trade level date with pd date, price, 
                           volume, buy indicator.
    '''   
    df = pd.DataFrame(trades)
    df.columns = ['price', 'volume', 'time', 'buy_sell', 'market_limit', 'misc', 'trade_id']
    df = df.drop_duplicates()
    assert(len(np.unique(df.trade_id)) == len(df.trade_id))
    df['buy'] = 0
    df.loc[df.buy_sell=='b', 'buy'] = 1
    df['date'] = pd.to_datetime(df.time, unit='s').dt.tz_localize(None)
    df = df[['date', 'price', 'volume', 'buy']]
    return df
    
if __name__ == "__main__":
    # build common objects
    base_url   = 'https://api.kraken.com/0/public/'
    start_time = 1451606400 # jan 1 2016 midnight gmt
    end_time   = 1671062400 # dec 15 2022 midnight gmt

    # build btc trades
    pair       = 'XXBTZUSD'
    btc_fp     = '../1-data/raw/trades-kraken-btc.pkl'
    btc_trades = pullTrades(base_url, pair, start_time, end_time)
    btc_df     = clean(btc_trades)
    btc_df.to_pickle(btc_fp)

    # build eth trades
    pair       = 'XETHZUSD'
    eth_fp     = '../1-data/raw/trades-kraken-eth.pkl'
    eth_trades = pullTrades(base_url, pair, start_time, end_time)
    eth_df     = clean(eth_trades)
    eth_df.to_pickle(eth_fp)
