import numpy as np
import pandas as pd
import random
import math

close = [1,2,3,4,5,6,7,8,9,10]

close = []
for i in range(0,20000):
  x = random.randint(0,1000000000000000000000000000000000000000000000000000)
  x = float(x)
  close.append(np.sin(x))

close_df = pd.DataFrame(close)

df_train = close_df.iloc[:int(0.8*close_df.shape[0]),:]
df_val = close_df.iloc[int(0.8*close_df.shape[0]):int(0.9*close_df.shape[0]),:]
df_test = close_df.iloc[int(0.9*close_df.shape[0]):,:]

df_train.to_csv('data/processed/test/dataset2_test_train.csv')
df_val.to_csv('data/processed/test/dataset2_test_val.csv')
df_test.to_csv('data/processed/test/dataset2_test_test.csv')