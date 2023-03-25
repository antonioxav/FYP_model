import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def plot_cols(df, ticker):
    fig, axs = plt.subplots(df.shape[-1], 1, sharex=True, figsize=(40, 10*df.shape[-1]))
    st = fig.suptitle(f"{ticker.upper()} Columns", fontsize=20)
    st.set_y(0.92)

    for i in range(df.shape[-1]):
        axs[i].set_title(df.columns[i])
        axs[i].plot(df.index, df.iloc[:,i])

def plot_col_histogram(df, ticker, plot_gauss = True):
    fig, axs = plt.subplots(df.shape[-1], 1, figsize=(40, 20*df.shape[-1]))
    st = fig.suptitle(f"{ticker.upper()} Column Histograms", fontsize=20)
    st.set_y(0.92)

    for i, col in enumerate(df.columns):
        axs[i].set_title(col)
        sns.histplot(data=df, x=col, kde=True, ax=axs[i], stat='density')
        if plot_gauss:
            mean = df[col].mean()
            std = df[col].std()
            x = np.linspace(mean - 3*std, mean + 3*std, df.shape[0])
            axs[i].plot(x, norm.pdf(x, mean, std), color='red')
            # sns.kdeplot(np.random.normal(df[col].mean(), df[col].std(), size = df.shape[0]), color='red', ax=axs[i])
    