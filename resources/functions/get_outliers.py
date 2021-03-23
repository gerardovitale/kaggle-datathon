from pandas import DataFrame, Series
from numpy import abs, where
from pandas.io.formats.format import return_docstring
from scipy.stats import zscore

def get_outliers_quantile_per_feature(feature:Series, tail=0.05):
    '''
    Given a feature and a percentage (0 < tail < 1), equivalent to the tail in a 
    normal distribution, the function returns the index list of outliers values in the given feature.
    '''
    filt = feature.between(feature.quantile(tail), feature.quantile(1 - tail))
    print(filt.value_counts())
    return feature[~filt].index

def get_outlier_zscore_per_feature(feature:Series, threshold=2):
    '''
    Given a feature (column) and a number that represent the Z-score.
        "The Z-score is the signed number of standard 
        deviations by which the value of an observation 
        or data point is above the mean value of what 
        is being observed or measured"
    The function returns basic info about those outliers found.
        result = {
            'threshold': threshold,
            'number_of_outliers': len(z[outliers]),
            'outliers': z[outliers],
            'outliers_index': outliers
        }
    '''
    z = abs(zscore(feature))
    outliers = where(z > threshold)
    return {
        'threshold': threshold,
        'number_of_outliers': len(z[outliers]),
        'outliers': z[outliers],
        'outlier_indexes': outliers[0].tolist()
    }

def get_outlier_df(outlier_function:function, df:DataFrame):
    '''
    Given an outlier_funtion (get_outlier_zscore_per_feature or get_outliers_quantile_per_feature) and a DataFrame
    returns a dictionary that would have feature name as key and its outliers values and their indexes as values.
    '''
    return {col:outlier_function(df[col]) for col in df.columns}

def get_outlier_index_list(outlier_function:function, df:DataFrame):
    '''
    Given an outlier_funtion (get_outlier_zscore_per_feature or get_outliers_quantile_per_feature) and a DataFrame
    returns a sorted list of outliers indexes.
    '''
    index_list = []
    for col in df.columns:
        index = list(outlier_function(df[col])['outlier_indexes'])
        for i in index:
            if i not in index_list:
                index_list.append(i)
    return sorted(index_list)

