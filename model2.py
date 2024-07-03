# %% [markdown]
# # Time Series - Moving Average Example (Electricity Consumption)

# %%


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# initialize a Dataframe 'df' with some dummy values
# Dummy values are in the  csv file, the string in the read_csv() represents where our csv file is located
# you can change the string path in the read_csv() to import the csv file from different folder

df = pd.read_csv('Electricity_Consumption.csv', index_col=0,
     parse_dates=True, infer_datetime_format=True)

df.head()

# %%
print(df.dtypes)

# %%
# get first row
first_date = df.index[0]
first_date

# %%
df.Electricity_Consumed.head()

# %% [markdown]
# # Visualize

# %%
df.plot()

# %% [markdown]
# # Average of Recent Period
# Perform some analysis of the data 

# %%
RECENT_PERIOD = 3  # set the recent period
df.Electricity_Consumed[-RECENT_PERIOD:]

# %%
df_recent_period = df[-RECENT_PERIOD:] 

# %%
df_recent_period.head()

# %%
# average value of recent period
df_recent_period.Electricity_Consumed.mean()  

# %%
# total value of recent period
df_recent_period.Electricity_Consumed.sum()

# %% [markdown]
# # Moving Average
# Moving Average (or rolling aveage) is used to analyze data points by creating a series of averages of different subsets of the full data set.  A moving average is commonly used with time series data to smooth out short-term fluctuations and highlight longer-term trends or cycles.  The estimate of the trend-cycle at time t is obtained by averaging values of the time series within k periods of t.

# %%
rolling_mean_3 = df.Electricity_Consumed.rolling(window=3).mean().shift(1)  
# moving average of past 3 months, shift one row
# Using the method rolling(window), we need to shfit 1 row [rolling(window=3).mean().shift(1)] 
#    to indicate the moving average as
#    Yt = computed based on Y(t-1) + Y(t-2) + Y(t-3)
rolling_mean_12 = df.Electricity_Consumed.rolling(window=12).mean().shift(1) # moving average of past 12 months, shift one row

print(df.Electricity_Consumed.head())
print("--  rolling mean 3")
print(rolling_mean_3.head())
print("--  rolling mean 12")
print(rolling_mean_12.head())

# %%
# plot the graphs with different moving average
plt.plot(df.index, df.Electricity_Consumed, label='Electricity Consumed')
plt.plot(df.index, rolling_mean_3, label='3 Months SMA', color='orange')
plt.plot(df.index, rolling_mean_12, label='12 Months SMA', color='magenta')
plt.legend(loc='upper left')
plt.show()

# %% [markdown]
# To know more about the concepts on Time Series: https://otexts.org/fpp2/intro.html
# 
#         

# %%

# %%
print("3")

# %%
print(rolling_mean_3[10])

# %%
#rolling_mean_3.to_pickle("moving_average.pkl")
rolling_mean_3.to_pickle("model2.pkl")




