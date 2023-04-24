import numpy as np
import pandas as pd

close = [1,2,3,4,5,6,7,8,9,10]

for i in range(10,20001):
  close.append((1.001*close[i-1])-(1.0009*close[i-2])+(1.0008*close[i-3])-(1.0007*close[i-4])+(1.0006*close[i-5])-(1.0005*close[i-6])+(1.0004*close[i-7])-(1.0003*close[i-8])+(1.0002*close[i-9])-(1.0001*close[i-10]))

close_df = pd.DataFrame(close)

df_train = close_df.iloc[:int(0.8*close_df.shape[0]),:]
df_val = close_df.iloc[int(0.8*close_df.shape[0]):int(0.9*close_df.shape[0]),:]
df_test = close_df.iloc[int(0.9*close_df.shape[0]):,:]

df_train.to_csv('data/processed/test/self_generated_data_set_1_train.csv')
df_val.to_csv('data/processed/test/self_generated_data_set_1_val.csv')
df_test.to_csv('data/processed/test/self_generated_data_set_1_test.csv')