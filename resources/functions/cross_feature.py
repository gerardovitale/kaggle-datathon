from pandas import DataFrame

def cross_feature(X:DataFrame):
    '''
    Given a serie of features as DataFrame (X),
    Performs x_k = x_i * x_j, where x_i and x_j are features/columns of X, and
    Returns X with these synthetic feautres calculated.
    '''
    # Feature Cross
    columns = X.columns
    for i in range(len(columns)):
        for j in range(i,len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            X[f"{col1}*{col2}"] = X[col1] * X[col2]
    return X