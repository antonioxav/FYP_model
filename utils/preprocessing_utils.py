from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def standardize_and_scale(df_train, df_val, df_test, scaler_type=None, outliers = None):
    """
    Scales the data using the scaling type specified

    Args:
        df_train (pd.DataFrame): Training Data. Pass df_train_fixed from get_outliers if scalar_type='gaussian'
        df_val (pd.DataFrame): Validation Data.
        df_test (pd.DataFrame): Test Data.
        outliers (pd.Series): Boolean Series indicating whether a row is an outlier or not. Required if scalar_type='gaussian'. Defaults to None.
        scaler_type (str, optional): type of scaling. Options: 'max', 'robust', 'gaussian'. Defaults to None.

    Returns:
        Tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame, dict): returns scaled dataframes along with a list of scalers used for each col
    """
    scalers = {}
    df_train_norm = pd.DataFrame()
    df_val_norm = pd.DataFrame()
    df_test_norm = pd.DataFrame()
    for i, col in enumerate(df_train.columns):
        
        if 'Y_' not in col and '!' not in col:

            if scaler_type == 'robust':
                scaler = RobustScaler()
                scaler = scaler.fit(np.array(df_train[col]).reshape(-1,1))
            elif scaler_type == 'gaussian' and outliers is not None:
                scaler = StandardScaler()
                scaler = scaler.fit(np.array(df_train[~outliers][col]).reshape(-1,1))
            else:
                scaler = MaxAbsScaler() if df_train[col].min() < 0 else MinMaxScaler()
                scaler = scaler.fit(np.array(df_train[col]).reshape(-1,1))
            
            df_train_norm[col] = pd.Series(np.squeeze(scaler.transform(np.array(df_train[col]).reshape(-1,1))), index=df_train.index)
            df_val_norm[col] = pd.Series(np.squeeze(scaler.transform(np.array(df_val[col]).reshape(-1,1))), index=df_val.index)
            df_test_norm[col] = pd.Series(np.squeeze(scaler.transform(np.array(df_test[col]).reshape(-1,1))), index=df_test.index)
            scalers[col] = scaler
    
        else:
            df_train_norm[col] = df_train[col]
            df_val_norm[col] = df_val[col]
            df_test_norm[col] = df_test[col]

    print(scalers)
    return df_train_norm, df_val_norm, df_test_norm, scalers



def add_PCA(PCA_col, df_train, df_val, df_test, n_components = None, prefix = '', append = False):
    """
    Performs Principal Components Analyis on the given column of the dataframe. Appends the columns to the given dataframes if append = True.

    Args:
        PCA_col (List): columns to perfrom PCA on.
        df_train (pd.DataFrame): Training dataset
        df_val (pd.DataFrame): Validation dataset
        df_test (pd.DataFrame): Test dataset
        n_components (bool, optional): number of components. Defaults to len(PCA_col).
        prefix (str, optional): prefix for column name. Defaults to ''.
        append (bool, optional): Appends the PCA columns to the dataframes. Marks important columns with '*'. Defaults to False.

    Returns:
        Tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame): _description_
    """
    df_train_pca, df_val_pca, df_test_pca = df_train.copy(), df_val.copy(), df_test.copy()
    
    PCA_df = df_train_pca[PCA_col]
    print(df_train_pca.shape)
    print(PCA_df.shape, PCA_df.columns)

    if n_components is None:
        n_components = len(PCA_col)
    pca = PCA(n_components=n_components).fit(PCA_df)
    PCA_df = pca.transform(PCA_df)
    print(PCA_df.shape)
    print(pca.explained_variance_ratio_)
    plt.bar(range(0,len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, alpha=0.5, align='center', label='Individual explained variance')
    plt.show()

    if append:
        for c in range(n_components):
            imp = '*' if pca.explained_variance_ratio_[c] > 0.01 else ''
            df_train_pca[f'{prefix}pca{imp}_{c}'] = pca.transform(df_train_pca[PCA_col])[:,c]
            df_val_pca[f'{prefix}pca{imp}_{c}'] = pca.transform(df_val_pca[PCA_col])[:,c]
            df_test_pca[f'{prefix}pca{imp}_{c}'] = pca.transform(df_test_pca[PCA_col])[:,c]
        return df_train_pca, df_val_pca, df_test_pca
    
def get_outliers(df, gaus_cols, iqr_mult = 1.5, replace = False):
    df = df.copy()
    outlier = df.iloc[:,0].apply(lambda x: False)
    print(df.shape)
    for col in df.columns:
        if col in gaus_cols:
            q1, q3 = np.percentile(df[col],[25,75])
            upper_bound = q3 + iqr_mult*(q3-q1)
            lower_bound = q1 - iqr_mult*(q3-q1)

            high_out = df[col] > upper_bound
            low_out = df[col] < lower_bound
            col_out = high_out | low_out
            print(f'{col}: {(lower_bound, upper_bound)} High Outliers = {high_out.sum()}, Low Outliers = {low_out.sum()}, Total Outliers = {col_out.sum()}')

            if replace:
                df[col] = np.where(high_out, upper_bound, df[col])
                df[col] = np.where(low_out, lower_bound, df[col])
            
            outlier |= col_out
    print(f'Total Outliers: {outlier.sum()}')
    print(df[~outlier].shape)
    return df, outlier