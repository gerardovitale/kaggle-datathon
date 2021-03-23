from pandas import DataFrame
from pandas.api.types import is_numeric_dtype

def explore_feature(df:DataFrame):
    '''
    Given a DataFrame, returns a dictionary with the following structure:
    result = {
        <column name 1>: <some basic info depending on the data type of the column 1>,
        <column name 2>: <some basic info depending on the data type of the column 2>, ...
    }
    '''
    columns = df.columns
    result = {}
    for col in columns:  
        if is_numeric_dtype(df[col]):
            result[col] = {
                'min': df[col].min(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'max': df[col].max(),
            }
        else:
            result[col] = df[col].unique()
    return result
