# Question1_4.py

from ComputationalFinanceClass import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# initialization

stock_price = np.arange(0, 100, 1)
strike_price = np.arange(0, 100, 1)

stock_price, strike_price = np.meshgrid(stock_price, strike_price)

# Create an instance frin the class
option_example = ComputationalFinance(stock_price, strike_price)

# Calculate payoff
european_call_payoff = option_example.European_Call_Option_Payoff()
european_put_payoff = option_example.European_Put_Option_Payoff()

# Plot 2D
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(stock_price[1, :], european_call_payoff[25, :], label='Strike Price = '+str(strike_price[25, 0]))
plt.plot(stock_price[1, :], european_call_payoff[50, :], label='Strike Price = '+str(strike_price[50, 0]))
plt.plot(stock_price[1, :], european_call_payoff[75, :], label='Strike Price = '+str(strike_price[75, 0]))
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('Stock Price, $S_T$')
plt.ylabel('max($S_T - K, 0$)')
plt.title('Payoff of European Call Option')

plt.subplot(122)
plt.plot(stock_price[1, :], european_put_payoff[25, :], label='Strike Price = '+str(strike_price[25, 0]))
plt.plot(stock_price[1, :], european_put_payoff[50, :], label='Strike Price = '+str(strike_price[50, 0]))
plt.plot(stock_price[1, :], european_put_payoff[75, :], label='Strike Price = '+str(strike_price[75, 0]))
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('Stock Price, $S_T$')
plt.ylabel('max($K - S_T, 0$)')
plt.title('Payoff of European Put Option')

# plot 3D
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(strike_price, stock_price, european_call_payoff, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)
ax.view_init(azim=30)
ax.set_xlabel('Strike Price, K')
ax.set_ylabel('Stock Price, $S_T$')
ax.set_zlabel('max($S_T - K, 0$)')
ax.set_title('European Call Option Payoff')
ax.grid(True)
fig.colorbar(surf, shrink=0.5, aspect=5)


ax = fig.add_subplot(122, projection='3d')
surf = ax.plot_surface(strike_price, stock_price, european_put_payoff, rstride=2, cstride=2, cmap=plt.cm.hot, linewidth=0.5, antialiased=True)
ax.view_init(azim=50)
ax.set_xlabel('Strike Price, K')
ax.set_ylabel('Stock Price, $S_T$')
ax.set_zlabel('max($K - S_T, 0$)')
ax.set_title('European Put Option Payoff')
ax.grid(True)
fig.colorbar(surf, shrink=0.5, aspect=5)

