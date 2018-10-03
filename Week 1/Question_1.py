import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

gs_5years = pd.read_csv('GS 5 Years.csv')
spy_5years = pd.read_csv('SPY 5 Years.csv')
dates = pd.to_datetime(gs_5years['Date'])
stock_price_gs = gs_5years.values[:, 5]
stock_price_spy = spy_5years.values[:, 5]

stock_price = np.array([stock_price_gs, stock_price_spy], dtype=float).transpose()

print('GS & SPY Stock Price Statistics: ')
print('mean: ',  np.mean(stock_price, axis=0))
print('median: ', np.median(stock_price, axis=0))
print('variance: ', np.var(stock_price, axis=0))
print('standard deviation', np.std(stock_price, axis=0))
print('skewness: ', skew(stock_price, axis=0))
print('kurtosis: ', kurtosis(stock_price, axis=0))
print('')

plt.figure(figsize=(10, 6))
plt.plot(dates, stock_price[:, 0], label='Goldman Sachs')
plt.plot(dates, stock_price[:, 1], label='S&P 500 ETF')
plt.grid(True) 
plt.legend(loc=0) 
plt.xlabel('Time') 
plt.ylabel('Stock Price') 
plt.title('Time Series of Stock Price')
plt.show()

#Calculate percentage returns statistics
percentage_returns = np.diff(stock_price, axis=0)/stock_price[:-1, :]

print('GS & SPY Percentage Returns Statistics: ')
print('mean: ',  np.mean(percentage_returns, axis=0))
print('median: ', np.median(percentage_returns, axis=0))
print('variance: ', np.var(percentage_returns, axis=0))
print('standard deviation', np.std(percentage_returns, axis=0))
print('skewness: ', skew(percentage_returns, axis=0))
print('kurtosis: ', kurtosis(percentage_returns, axis=0))
print('')

plt.figure(figsize=(10, 6))
plt.plot(dates[1:], percentage_returns[:, 0], label='Goldman Sachs')
plt.plot(dates[1:], percentage_returns[:, 1], label='S&P 500 ETF')
plt.grid(True) 
plt.legend(loc=0) 
plt.xlabel('Time') 
plt.ylabel('Percentage Returns') 
plt.title('Time Series of Percentage Return')
plt.show()
