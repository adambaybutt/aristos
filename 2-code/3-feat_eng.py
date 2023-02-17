import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA

def createPanelFeatures(df: pd.DataFrame, freq: int) -> pd.DataFrame:
    ''' Form all features at panel level.
    
    Args:
        df (pd.DataFrame): raw minute-level data of: date, ethbtc price, trade volume and counts. 
        freq (int): specific frequency to work at--in minutes; passed in to ensure we create correct RHS.
    
    Returns:
        df (pd.DataFrame): minute level data with features and without raw columns.
    '''
    # simple returns
    for t in [1, 5, 15, 30, 60, 120, 360, 1440, 4320, 8640, 20160, 40320, 86400, freq]:
        df['covar_r_tm'+str(t)] = df['price'].pct_change(periods=t)

    # moments of 5 minute returns
    for t in [30, 60, 720, 4320, 8640, 20160]:
        df['covar_r_5min_ma_tm'+str(t)]   = df['covar_r_tm5'].transform(lambda x: x.rolling(t).mean())
        df['covar_r_5min_min_tm'+str(t)]  = df['covar_r_tm5'].transform(lambda x: x.rolling(t).min())
        df['covar_r_5min_max_tm'+str(t)]  = df['covar_r_tm5'].transform(lambda x: x.rolling(t).max())
        df['covar_r_5min_vol_tm'+str(t)]  = df['covar_r_tm5'].transform(lambda x: x.rolling(t).std())
        df['covar_r_5min_skew_tm'+str(t)] = df['covar_r_tm5'].transform(lambda x: x.rolling(t).skew())
        df['covar_r_5min_kurt_tm'+str(t)] = df['covar_r_tm5'].transform(lambda x: x.rolling(t).kurt())

    # moments of 120 minute returns
    for t in [720, 4320, 8640, 20160, 40320, 86400]:
        df['covar_r_120min_ma_tm'+str(t)]   = df['covar_r_tm120'].transform(lambda x: x.rolling(t).mean())
        df['covar_r_120min_min_tm'+str(t)]  = df['covar_r_tm120'].transform(lambda x: x.rolling(t).min())
        df['covar_r_120min_max_tm'+str(t)]  = df['covar_r_tm120'].transform(lambda x: x.rolling(t).max())
        df['covar_r_120min_vol_tm'+str(t)]  = df['covar_r_tm120'].transform(lambda x: x.rolling(t).std())
        df['covar_r_120min_skew_tm'+str(t)] = df['covar_r_tm120'].transform(lambda x: x.rolling(t).skew())
        df['covar_r_120min_kurt_tm'+str(t)] = df['covar_r_tm120'].transform(lambda x: x.rolling(t).kurt())

    # moments of 1 day returns
    for t in [4320, 8640, 20160, 40320, 86400]:
        df['covar_r_1440min_ma_tm'+str(t)]   = df['covar_r_tm1440'].transform(lambda x: x.rolling(t).mean())
        df['covar_r_1440min_min_tm'+str(t)]  = df['covar_r_tm1440'].transform(lambda x: x.rolling(t).min())
        df['covar_r_1440min_max_tm'+str(t)]  = df['covar_r_tm1440'].transform(lambda x: x.rolling(t).max())
        df['covar_r_1440min_vol_tm'+str(t)]  = df['covar_r_tm1440'].transform(lambda x: x.rolling(t).std())
        df['covar_r_1440min_skew_tm'+str(t)] = df['covar_r_tm1440'].transform(lambda x: x.rolling(t).skew())
        df['covar_r_1440min_kurt_tm'+str(t)] = df['covar_r_tm1440'].transform(lambda x: x.rolling(t).kurt())

    # form price variables
    df = df.rename(columns = {'price': 'covar_p_t'})
    df['covar_p_log_t'] = np.log(df.covar_p_t)

    # current volume
    df = df.rename(columns = {'volume': 'covar_volume_t', 'trades': 'covar_trades_t'})

    # form functions of volume
    for col in ['covar_volume_t', 'covar_trades_t']:
        for t in [5, 30, 60, 360, 720, 4320, 8640, 20160]:
            df['covar_'+col+'_ma_tm'+str(t)]  = df[col].transform(lambda x: x.rolling(t).mean())
            df['covar_'+col+'_sum_tm'+str(t)] = df[col].transform(lambda x: x.rolling(t).sum())                                                                        
            df['covar_'+col+'_min_tm'+str(t)] = df[col].transform(lambda x: x.rolling(t).min()) 
            df['covar_'+col+'_max_tm'+str(t)] = df[col].transform(lambda x: x.rolling(t).max()) 
            df['covar_'+col+'_vol_tm'+str(t)] = df[col].transform(lambda x: x.rolling(t).std()) 
    
    # form returns from cum max and min prices
    df['covar_r_cummax_t'] = ((df.covar_p_t - df.covar_p_t.cummax()) 
                                    / df.covar_p_t.cummax())
    df['covar_r_cummin_t'] = ((df.covar_p_t - df.covar_p_t.cummin()) 
                                    / df.covar_p_t.cummin())

    return df

def createLHSVariables(df: pd.DataFrame, lhs_col: str, freq: int) -> pd.DataFrame:
    ''' Create LHS target variables of absolute return difference between btc and eth.
        Window is from every two hours (starting midnight) plus five minutes (to give time to 
        pull data and predict) to the subsequent twelve hours plus 10 minutes (to repeat the process 
        and give time to place/update trades).
    
    Args:
        df (pd.DataFrame): time bar level data with only RHS features.
        lhs_col (str): specific LHS name to create.
        freq (int): specific frequency that we are working at, in minutes.
    
    Returns:
        df (pd.DataFrame): time bar level data with LHS and RHS features.
    '''
    df[lhs_col] = df['covar_r_tm'+str(freq)].shift(-(freq+5))
    return df

def createRSIFeatures(df: pd.DataFrame) -> pd.DataFrame:
    ''' Form relative strength index features at one hour frequency over various windows.

    Args: 
        df (pd.DataFrame): time bar level data with necessary RHS features.
    
    Returns:
        df (pd.DataFrame): same time bar level data frame with RSI features added.    

    FUTURE TODO:
        -pass in freq to work at, noting units
        -pass in windows to create, noting units
        -make generic function in my relevant class
    '''
    for window in [360, 720, 4320, 8640, 20160]:
        two_hr_p_delta_col = 'covar_p_delta_tm120'
        df[two_hr_p_delta_col] = df['covar_p_t'].diff(periods=120)
        df['temp_neg_p_delta_2hr'] = df[two_hr_p_delta_col].clip(upper=0)
        df['temp_pos_p_delta_2hr'] = -1*df[two_hr_p_delta_col].clip(lower=0)
        df['temp_avg_neg_p_delta_2hr_tm'+str(window)] = df['temp_neg_p_delta_2hr'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df['temp_avg_pos_p_delta_2hr_tm'+str(window)] = df['temp_pos_p_delta_2hr'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df['covar_rsi_tm'+str(window)] = (100 - 100/(1 + df['temp_avg_pos_p_delta_2hr_tm'+str(window)]/df['temp_avg_neg_p_delta_2hr_tm'+str(window)]))
        df = df.drop([two_hr_p_delta_col,
                        'temp_neg_p_delta_2hr', 
                        'temp_pos_p_delta_2hr',
                        'temp_avg_neg_p_delta_2hr_tm'+str(window),
                        'temp_avg_pos_p_delta_2hr_tm'+str(window)], axis=1)
    
    return df

def createFeaturesAndLHS(df: pd.DataFrame, freq: int, lhs_col: str) -> pd.DataFrame:
    ''' Transform raw panel to time bars with clean LHS and RHS features.
    
    Args:
        df (pd.DataFrame): raw minute-level panel of BTC and ETH:
                           usd per token prices, token volume, and trade count.
        freq (int): specific frequency to work at, in minutes.
        lhs_col (str): specific LHS name to create.
    
    Returns:
        df (pd.DataFrame): time bars of LHS and RHS features.
    '''
    # form features
    df = createPanelFeatures(df, freq)

    # ensure no missing time bars
    min_date = np.min(df.date.values)
    max_date = np.max(df.date.values)
    number_of_bars = 1+int(max_date - min_date)/1e9/60 # ns to seconds to minutes plus one minute
    assert(df.shape[0] == number_of_bars)

    # form LHS
    df = createLHSVariables(df, lhs_col, freq)
    
    # form rsi features
    df = createRSIFeatures(df)

    # resample to target frequency
    if freq == 120:
        df = df[df.date.dt.hour.isin([0,2,4,6,8,10,12,14,16,18,20,22]) & (df.date.dt.minute==0)].reset_index(drop=True)
    if freq == 2880:
        df = df[(df.date.dt.hour == 14) & (df.date.dt.minute==0)].reset_index(drop=True)
        df = df[(df.index%2)==0]

    return df

def finalClean(df: pd.DataFrame, lhs_col: str, freq: int) -> pd.DataFrame:
    ''' Final checks and clean of the time bar data.

    Args:
        df (pd.DataFrame): time bar level data with LHS and RHS features.
        lhs_col (str): specific name for LHS column.
        freq (int): specific frequency that we are working at, in minutes.

    Returns:
        df (pd.DataFrame): clean time bar level data with sorted columns and rows.    
    '''
    # interpolate missing values in skew, kurt, and rsi columns with trailing mean
    if freq == 120:
        periods = 48 # four days
    if freq == 2880:
        periods = 10 # 20 days
    cols = list(df.columns.values)
    interpol_cols = [col for col in cols if ('rsi' in col) | ('kurt' in col) | ('skew' in col)]
    for col in interpol_cols:
        df['rolling_average'] = df[col].rolling(periods, min_periods=1).mean()
        df[col] = df[col].fillna(df['rolling_average'])
        df = df.drop('rolling_average', axis=1)

    # drop rows pre 2017 and post 2022
    df = df[(df.date.dt.year >= 2017) & (df.date.dt.year <= 2022)]

    # order columns
    cols = list(df.columns.values)
    cols.remove('date')
    cols.remove(lhs_col)
    sorted_cols = sorted(cols)
    df = df[['date', lhs_col]+sorted_cols]

    # ensure rows are sorted
    df = df.sort_values(by='date', ignore_index=True)

    # ensure there are the correct number of rows
    min_date = np.min(df.date.values)
    max_date = np.max(df.date.values)
    if freq == 120:
        number_of_bars = 1+int(max_date - min_date)/1e9/60/60/24*12 
    elif freq == 2880:
        number_of_bars = 1+int(max_date - min_date)/1e9/60/60/24/2
    assert(df.shape[0] == number_of_bars)

    # drop the first and last row
    df = df[:-1].reset_index(drop=True)

    # drop columns missing any data
    num_cols_pre = df.shape[1]
    df = df.dropna(axis=1)
    print('dropped '+str(int(num_cols_pre-df.shape[1]))+' columns that were still missing data.')

    # ensure no missing data
    assert(0==df.isnull().sum().sum()),('there is missing data to be fixed in the time bar data.')

    return df

def calcCorrTable(temp_df: pd.DataFrame, 
                  lhs_col: str, feat_col: str, 
                  method: str='pearson') -> pd.DataFrame:
    ''' calculate given correlation between given LHS and feat columns, stratified by given years.

    Args:
        temp_df (pd.DataFrame): time bar level data frame.
        lhs_col (str): specific LHS to use to calculate correlation.
        feat_col (str): specific feat column to use to calc corr.
        method (str): correlation method to use to pass to Pandas corr() function.

    Returns:
        corr_df (pd.DataFrame): correlation results with columns for `feature` name, 
                                time `window`, and `corr_`+method value.
    
    '''
    # determine the different years in the data
    years = list(np.unique(temp_df.date.dt.year))

    # calc corr by year and overall
    year_df = temp_df.groupby(['year'])[[lhs_col, feat_col]].corr(method=method)
    ovrl_df = temp_df[[lhs_col, feat_col]].corr(method=method)

    # set up dataframe to output the data and format it 
    num_stats     = int(len(years)+1)
    corr_col_name = 'corr_' + method
    corr_df       = pd.DataFrame(data={'feature': np.repeat(feat_col, num_stats),
                                 'window':        years + ['all'],
                                 corr_col_name:   np.zeros(num_stats)})
    for i in range(len(years)):
        year = years[i]
        corr_df.loc[corr_df.window == year, 
                    corr_col_name] = year_df[year_df.index.get_level_values(None) 
                                             == feat_col][lhs_col].values[i]
    corr_df.loc[corr_df.window == 'all', corr_col_name] = ovrl_df[ovrl_df.index 
                                                                  == feat_col][lhs_col].values[0]

    return corr_df

def calc_MI(temp_df: pd.DataFrame, lhs_col: str, feat_col: str):
    ''' Calculate mutual information between two given columns in given df.
    
    Args:
        temp_df (pd.DataFrame): dataframe to obtain x and y columns from.
        lhs_col (str): name of first column to use to calc MI.
        feat_col (str): name of second col to use to calc MI.

    Returns:
        mi (float): mutual information between two given vectors.
    '''
    # obtain the two data series
    x    = temp_df[lhs_col].values
    y    = temp_df[feat_col].values

    # calc mutual information
    bins = int(np.floor(np.sqrt(temp_df.shape[0]/5)))
    c_xy = np.histogram2d(x, y, bins)[0]
    mi   = mutual_info_score(None, None, contingency=c_xy)

    return mi

def calcMiTable(temp_df: pd.DataFrame, lhs_col: str, feat_col: str) -> pd.DataFrame:
    ''' calculate mutual information between given LHS and feat columns, stratified by given years.

    Args:
        temp_df (pd.DataFrame): time bar level data frame.
        lhs_col (str): specific LHS to use to calculate MI.
        feat_col (str): specific feat column to use to calc MI.

    Returns:
        mi_df (pd.DataFrame): MI results with columns for `feature` name, 
                              time `window`, and `mi` value.
    
    '''
    # determine the different years in the data
    years = list(np.unique(temp_df.date.dt.year))

    # calc MI overall and by year
    mi_all  = calc_MI(temp_df, lhs_col, feat_col)
    year_df = temp_df.groupby('year').apply(lambda x: calc_MI(x, lhs_col, feat_col))

    # set up dataframe to output the data and format it 
    num_stats = int(len(years)+1)
    mi_df = pd.DataFrame(data={'feature': np.repeat(feat_col, num_stats),
                               'window':  years+['all'],
                               'mi':      np.zeros(num_stats)})
    for i in range(len(years)):
        year = years[i]
        mi_df.loc[mi_df.window == year, 'mi']  = year_df[year_df.index == year].values[0]
    mi_df.loc[mi_df.window == 'all', 'mi'] = mi_all 

    return mi_df

def calcIndependenceTable(temp_df: pd.DataFrame, lhs_col: str, feat_col: str) -> pd.DataFrame:
    ''' calc a custom measure of independence of the given feature column with the first PCA
        of the remaining features.
    
    Args:
        temp_df (pd.DataFrame): time bar level data frame.
        lhs_col (str): LHS col that will be removed from the given data frame.
        feat_col (str): specific feat column to use to calculate this measure of independence.
    
    Returns:
        indep_df (pd.DataFrame): independence measure results with columns for the `feature` name, 
                                 time `window`, and value of the `indep`endence measure.
    '''
    # determine the different years in the data
    years = list(np.unique(temp_df.date.dt.year))
    
    # form column of interest
    feat_col_array = np.array(list(temp_df[feat_col].values))
    
    # form RHS matrix
    rhs_mat   = temp_df.drop(['year', 'date', lhs_col, feat_col], axis=1).values

    # apply PCA
    first_pc_array = PCA(n_components=1).fit_transform(rhs_mat).reshape(-1)

    # build the dfs containing the results
    year_df      = pd.DataFrame(data={'year': temp_df.year,
                                      feat_col: feat_col_array,
                                      'first_pc': first_pc_array})
    year_df      = year_df.groupby(['year'])[[feat_col, 'first_pc']].corr()
    yearly_corr  = year_df[year_df.index.get_level_values(None) == feat_col]['first_pc'].values
    yearly_indep = (1/np.abs(yearly_corr) - 1)
    ovrl_corr    = np.corrcoef(feat_col_array, first_pc_array)[1,0]
    ovrl_indep   = 1/np.abs(ovrl_corr) - 1  

    # set up dataframe to output the data and format it 
    num_stats = int(len(years)+1)
    indep_df = pd.DataFrame(data={'feature': np.repeat(feat_col, num_stats),
                                  'window':  years+['all'],
                                  'indep':   np.zeros(num_stats)})
    for i in range(len(years)):
        year = years[i]
        indep_df.loc[indep_df.window == year, 'indep']  = yearly_indep[i]
    indep_df.loc[indep_df.window == 'all', 'indep'] = ovrl_indep

    return indep_df

def calcCorrelationStatistics(df: pd.DataFrame, lhs_col: str) -> pd.DataFrame:
    ''' for all the features, calculate various measures of correlation and return results df.

    Args:
        df (pd.DataFrame): time bar level data.
        lhs_col (str): LHS col for studying feature correlation with.

    Returns:
        results_df (pd.DataFrame): 
    '''
    # copy to new df, remove 2022 data, add year col, remove non lhs col
    stats_df = df.copy()
    stats_df['year'] = stats_df.date.dt.year
    stats_df = stats_df[stats_df.year < 2022]

    # initialize a results data frame
    results_df = pd.DataFrame()

    # determine feat columns
    cols = list(stats_df.columns.values)
    feat_cols = [col for col in cols if 'covar'==col[:5]]

    # calc corr stats for each feat col
    for feat_col in feat_cols: 
        print(feat_col) # TODO REMOVE
        corr_pearson_df  = calcCorrTable(stats_df, lhs_col, feat_col, method='pearson')
        corr_spearman_df = calcCorrTable(stats_df, lhs_col, feat_col, method='spearman')
        temp_df          = corr_pearson_df.merge(corr_spearman_df,
                                                on=['feature', 'window'],
                                                how='inner',
                                                validate='one_to_one')
        mi_df            = calcMiTable(stats_df, lhs_col, feat_col)
        temp_df          = temp_df.merge(mi_df, 
                                        on=['feature', 'window'],
                                        how='inner',
                                        validate='one_to_one')
        indep_df         = calcIndependenceTable(stats_df, lhs_col, feat_col)
        temp_df          = temp_df.merge(indep_df, 
                                        on=['feature', 'window'],
                                        how='inner',
                                        validate='one_to_one')
        results_df       = pd.concat((results_df, temp_df))
    results_df = results_df.reset_index(drop=True)

    # normalize all the results as i will work on a quantile basis
    results_df['corr_pearson']  = (np.abs(results_df['corr_pearson'].values)
                                   / np.nanmax(np.abs(results_df['corr_pearson'].values)))
    results_df['corr_spearman'] = (np.abs(results_df['corr_spearman'].values)
                                   / np.nanmax(np.abs(results_df['corr_spearman'].values)))
    results_df['mi']            = results_df.mi/np.max(results_df.mi.values)
    results_df['indep']         = results_df.indep/np.nanmax(results_df.indep.values)

    return results_df

def selectFeaturesFromResults(results_df: pd.DataFrame) -> set:
    ''' Execute arbitrary algo to select features from correlation statistics results table.
    
    Args:
        results_df (pd.DataFrame): normalized correlation measures for each feature for each year.
    
    Returns:
        included_feats (set): features to include.
    '''
    # initialize set of included features
    included_feats = set([])

    # determine features that are useful across the entire data set
    limit1 = 4
    all_df = results_df[results_df.window == 'all']
    for stat in ['corr_pearson', 'corr_spearman', 'mi', 'indep']:
        included_feats = included_feats.union(set(np.unique(all_df.sort_values(by=stat, ascending=False).feature.values[:limit1])))

    # determine features that are useful in recent years
    limit2 = 3
    for year in [2018, 2019, 2020, 2021]:
        year_df = results_df[results_df.window == year]
        for stat in ['corr_pearson', 'corr_spearman', 'mi']: # note ignoring indep
            included_feats = included_feats.union(set(np.unique(year_df.sort_values(by=stat, ascending=False).feature.values[:limit2])))

    # manually fix
    if 'covar_p_t' in included_feats:
        included_feats.remove('covar_p_t')
    if 'covar_p_log_t' in included_feats:
        included_feats.remove('covar_p_log_t')
        
    return included_feats

if __name__ == "__main__":
    # set args
    in_fp   = '../1-data/clean/panel_ethbtc_1min.pkl'
    out_fp  = '../1-data/clean/bars_ethbtc_2d.pkl'
    freq    = 2880
    lhs_col = 'y_ethbtc_r_tp5_tp' + str(freq+5)

    # read in data
    df = pd.read_pickle(in_fp)

    # engineer features
    df = createFeaturesAndLHS(df, freq, lhs_col)
    
    # ensure time bar data is clean
    df = finalClean(df, lhs_col, freq)

    # calculate correlation statistics
    # TODO for the future: for feature selection, bootstrap like 90% of the data without replacement, 
    #      calc the corr stat, and average across these so results are not driven by outliers
    results_df = calcCorrelationStatistics(df, lhs_col=lhs_col)
    included_feats = selectFeaturesFromResults(results_df)
    
    # subset to selected features and save
    df = df[['date', lhs_col]+list(included_feats)]
    df.to_pickle(out_fp)
