from pandas import DataFrame, concat
from sklearn.feature_selection import SelectKBest, f_regression

def filter_numeric_feature(X:DataFrame, y:DataFrame, score_func, k=10):
    '''
    Given X and y, returns a DataFrame with the features selected and their score.
        - k: represent the number of top features to select, the "all" option 
        by passes selection, for use in a parameter search.
        - score_func:
            * f_regression: Used only for numeric targets and based on linear regression performance.
            * f_classif: Used only for categorical targets and based on the Analysis of Variance (ANOVA) statistical test.
            * chi2: Performs the chi-square statistic for categorical targets, which is less sensible to the nonlinear relationship between the predictive variable and its target.
    '''
    # feature selection using an score function
    fs = SelectKBest(score_func=score_func, k=k)
    fit = fs.fit(X,y)
    # create df for scores
    dfscores = DataFrame(fit.scores_)
    # create df for column names
    dfcolumns = DataFrame(X.columns)
    # concat two dataframes for better visualization 
    feature_scores = concat([dfcolumns,dfscores],axis=1)
    # naming the dataframe columns
    feature_scores.columns = ['selected_columns','score_pearsons']
    return feature_scores.sort_values(by='score_pearsons', ascending=False)
