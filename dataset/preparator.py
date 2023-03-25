import numpy as np
import pandas as pd
import os


def prepare(path, prefix, seq_len, pca=False, task='reg', save=True):

    df_train = pd.read_csv(f'data/processed/{path}/{prefix}_train.csv', index_col='Date')
    df_val = pd.read_csv(f'data/processed/{path}/{prefix}_val.csv', index_col='Date')
    df_test = pd.read_csv(f'data/processed/{path}/{prefix}_test.csv', index_col='Date')

    X_col = [col for col in df_train.columns if 'Y_' not in col and '!' not in col and 'pca' not in col]
    # X_col = ['open','high','low','close','volume']
    X_col = ['pca_0','pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7','pca_8','pca_9']
    print(X_col)
    Y_col = f'Y_{task}'

    # Training data
    X_train, y_train = [], []
    for i in range(seq_len, df_train.shape[0]+1):
        X_train.append(df_train.iloc[i-seq_len:i, :][X_col].values) 
        y_train.append(df_train.iloc[i-1, :][Y_col])
    X_train, y_train = np.array(X_train), np.array(y_train)

    ###############################################################################

    # Validation data
    X_val, y_val = [], []
    for i in range(seq_len, df_val.shape[0]+1):
        X_val.append(df_val.iloc[i-seq_len:i, :][X_col].values) 
        y_val.append(df_val.iloc[i-1, :][Y_col])
    X_val, y_val = np.array(X_val), np.array(y_val)

    ###############################################################################

    # Test data
    X_test, y_test = [], []
    for i in range(seq_len, df_test.shape[0]+1):
        X_test.append(df_test.iloc[i-seq_len:i, :][X_col].values) 
        y_test.append(df_test.iloc[i-1, :][Y_col])
    X_test, y_test = np.array(X_test), np.array(y_test)

    print('Training set shape', X_train.shape, y_train.shape)
    print('Validation set shape', X_val.shape, y_val.shape)
    print('Testing set shape' ,X_test.shape, y_test.shape)

    if save:
        np.save(f'data/processed/{path}/X_train', X_train)
        np.save(f'data/processed/{path}/X_val', X_val)
        np.save(f'data/processed/{path}/X_test', X_test)
        np.save(f'data/processed/{path}/Y_train', y_train)
        np.save(f'data/processed/{path}/Y_val', y_val)
        np.save(f'data/processed/{path}/Y_test', y_test)
     
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__=='__main__':
    prepare('macro','macro', 40, pca=False, task='reg')