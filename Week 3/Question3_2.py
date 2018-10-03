# Question3_2.py

import numpy as np
import matplotlib.pyplot as plt
from ComputationalFinanceClass import *
from mpl_toolkits.mplot3d import Axes3D

''' 
european option bs pricing model parameter description:
t: time
T: maturity date of the option
St: the spot price of the underlying asset at time t
K: the strike price of the option 
r: riskfree rate
d: the dividend yield of the underlying asset
vol: the volatility of returns of the underlying asset
'''

t = [0.0, 0.4, 0.8]
T =1.0
St = np.arange(0.1, 100, 1)
K = 50.0
r = 0.05
d = 0.02
vol = 0.3

euro_opt = ComputationalFinance(St)

# Plot 2D European Call Option Black-Scholes Price
fig = plt.figure(figsize=(12,6))
fig.add_subplot(121)
for i in range(3):
    bs_european_call_price = euro_opt.Black_Scholes_European_Call(t[i], T, St, K, r, d, vol)[0]
    plt.plot(St, bs_european_call_price, label='K = 50.0, t = '+str(t[i])+', T = 1.0')

euro_call_payoff = euro_opt.European_Call_Payoff(K)
plt.plot(St, euro_call_payoff, label='Payoff, K = 50.0, T = 1.0')

plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('Stock Price')
plt.ylabel('Price')
plt.title('European Call Option Black-Scholes Price')

# Plot 2D European Put Option Black-Scholes Price
fig.add_subplot(122)
for i in range(3):
    bs_european_put_price = euro_opt.Black_Scholes_European_Put(t[i], T, St, K, r, d, vol)[0]
    plt.plot(St, bs_european_put_price, label='K = 50.0, t = '+str(t[i])+', T = 1.0')

euro_put_payoff = euro_opt.European_Put_Payoff(K)
plt.plot(St, euro_put_payoff, label='Payoff, K = 50.0, T = 1.0')

plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('Stock Price')
plt.ylabel('Price')
plt.title('European Put Option Black-Scholes Price')

# Plot 3D European Call Option Black-Scholes Price and Greeks
t = 0.0
T = np.arange(0.1, 1, 0.1)
St, T = np.meshgrid(St, T)

bs_european_call_price, bs_european_call_delta, bs_european_call_theta, \
bs_european_call_vega, bs_european_call_gamma, bs_european_call_rho = \
euro_opt.Black_Scholes_European_Call(t, T, St, K, r, d, vol)

fig = plt.figure(figsize=(15,11))
plt.title('Black-Scholes Price and Greeks of European Call Option')
ax = fig.add_subplot(3, 3, 1, projection='3d')
surf = ax.plot_surface(St, T, bs_european_call_price, rstride=1, cstride=1, cmap=plt.cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('Price')

ax = fig.add_subplot(3, 3, 2, projection='3d')
surf = ax.plot_surface(St, T, bs_european_call_delta, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('Delta')

ax = fig.add_subplot(3, 3, 3, projection='3d')
surf = ax.plot_surface(St, T, bs_european_call_theta, rstride=1, cstride=1, cmap=plt.cm.plasma, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('Theta')

ax = fig.add_subplot(3, 3, 4, projection='3d')
surf = ax.plot_surface(St, T, bs_european_call_gamma, rstride=1, cstride=1, cmap=plt.cm.inferno, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('Gamma')

ax = fig.add_subplot(3, 3, 5, projection='3d')
surf = ax.plot_surface(St, T, bs_european_call_rho, rstride=1, cstride=1, cmap=plt.cm.magma, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('Rho')

ax = fig.add_subplot(3, 3, 6, projection='3d')
surf = ax.plot_surface(St, T, bs_european_call_vega, rstride=1, cstride=1, cmap=plt.cm.cividis, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('Vega')

# Plot 3D European Put Option Black-Scholes Price and Greeks
bs_european_put_price, bs_european_put_delta, bs_european_put_theta, \
bs_european_put_vega, bs_european_put_gamma, bs_european_put_rho = \
euro_opt.Black_Scholes_European_Put(t, T, St, K, r, d, vol)

fig = plt.figure(figsize=(15,11))
plt.title('Black-Scholes Price and Greeks of European Put Option')
ax = fig.add_subplot(3, 3, 1, projection='3d')
surf = ax.plot_surface(St, T, bs_european_put_price, rstride=1, cstride=1, cmap=plt.cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('Price')

ax = fig.add_subplot(3, 3, 2, projection='3d')
surf = ax.plot_surface(St, T, bs_european_put_delta, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('Delta')

ax = fig.add_subplot(3, 3, 3, projection='3d')
surf = ax.plot_surface(St, T, bs_european_put_theta, rstride=1, cstride=1, cmap=plt.cm.plasma, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('Theta')

ax = fig.add_subplot(3, 3, 4, projection='3d')
surf = ax.plot_surface(St, T, bs_european_call_gamma, rstride=1, cstride=1, cmap=plt.cm.inferno, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('Gamma')

ax = fig.add_subplot(3, 3, 5, projection='3d')
surf = ax.plot_surface(St, T, bs_european_put_rho, rstride=1, cstride=1, cmap=plt.cm.magma, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('Rho')

ax = fig.add_subplot(3, 3, 6, projection='3d')
surf = ax.plot_surface(St, T, bs_european_put_vega, rstride=1, cstride=1, cmap=plt.cm.cividis, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('Vega')








