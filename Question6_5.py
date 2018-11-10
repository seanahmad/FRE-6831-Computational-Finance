# Question6_5.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import stats
from ComputationalFinanceClass import *

''' 
    parameter description:
    Smin: minimum value of stock price
    Smax: maximum value of stock price
    K: strike price
    t: time
    T: maturity date of the option
    r: interest rate
    d: the dividend yield of the underlying asset
    sigma: the volatility of returns of the underlying asset
    N: number of sub-intervals in s-direction
    M: number of sub-intervals in tao-direction
'''

Smax = 150
Smin = 5
K = 100
t = 0
T = 1
N = 40
M = 40
epsilon = 2
sigma = 0.2
r = 0.03
d = 0.02

# initialize
euro_opt = ComputationalFinance(0, 0)

# calculate european option price by RBF
euro_call_rbf = euro_opt.Black_Scholes_RBF_FD_EO(Smax, Smin, K, r, d, sigma, t, T, M, N, epsilon, 'ga_rbf', 'ic_call', 'dirichlet_bc')
euro_put_rbf = euro_opt.Black_Scholes_RBF_FD_EO(Smax, Smin, K, r, d, sigma, t, T, M, N, epsilon, 'ga_rbf', 'ic_put', 'dirichlet_bc')

St = np.linspace(Smin, Smax, N+1)
tao = np.linspace(t, T, M+1)
St, tao = np.meshgrid(St, tao)

# calculate european option price analytically by BS model
T = tao
euro_call_BS = euro_opt.Black_Scholes_European_Call( t, T, St, K, r, d, sigma )[0]
euro_put_BS = euro_opt.Black_Scholes_European_Put( t, T, St, K, r, d, sigma )[0]

# Question 6.5(2)
# Plot 2D European Option Price
fig = plt.figure(figsize=(15, 7))
fig.add_subplot(111)
plt.plot(St[0, :], euro_call_rbf[M, :], 'r*', label='Call RBF')
plt.plot(St[0, :], euro_call_BS[M, :], label='Call BS')
plt.plot(St[0, :], euro_put_rbf[M, :], 'r.', label='Put RBF')
plt.plot(St[0, :], euro_put_BS[M, :], label='Put BS')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('Stock Price')
plt.ylabel('European Option Price')
plt.title('European Option Price by RBF Finite Difference Method')

# Plot 3D European Option Price
fig = plt.figure(figsize=(15, 7))
plt.title('European Option Price by RBF Finite Difference Method')
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(St, tao, euro_call_rbf, rstride=1, cstride=1, cmap=plt.cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('European Call Price by RBF Finite Difference Method')
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(St, tao, euro_put_rbf, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('European Put Price by RBF Finite Difference Method')

fig = plt.figure(figsize=(15, 7))
plt.title('European Option Price by BS Model')
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(St, tao, euro_call_BS, rstride=1, cstride=1, cmap=plt.cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('European Call Price by BS Model')
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(St, tao, euro_put_BS, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('European Put Price by BS Model')

# Question 6.5(3)
# calculate error call option
S_call_out = Smin - 10
S_call_in = Smax - 10
S_call_at = K

S_put_out = Smax - 10
S_put_in = Smin - 10
S_put_at = K

# calculate the price of out-of-money situation
for i in range(10, 100, 20):
    M = i
    N = i
# calculate european option price by rbf finite difference   
    t = 0.1
    T = 1.1
    
    euro_call_rbf_diri = euro_opt.Black_Scholes_RBF_FD_EO(Smax, Smin, K, r, d, sigma, t, T, M, N, epsilon, 'ga_rbf', 'ic_call', 'dirichlet_bc')
# calculate european option price analytically by BS model
    St = np.linspace(Smin, Smax, N+1)
    index_out = int((S_call_out-Smin)/(St[1]-St[0]))
    
    tao = np.linspace(t, T, M+1)
    St, tao = np.meshgrid(St, tao)
    T = tao
    euro_call_BS = euro_opt.Black_Scholes_European_Call( t, T, St, K, r, d, sigma )[0]
    
    V_call_BS_out = euro_call_BS[M, index_out]
    V_call_diri_out = euro_call_rbf_diri[M, index_out]
    error_call_diri_out = (V_call_diri_out - V_call_BS_out) / V_call_BS_out
    print('Call Option Out-of-money Situation:')
    print('M = N = '+str(i)+':')
    print('Value Call BS: '+str(V_call_BS_out))
    print('Value Call RBF dirichlet: '+str(V_call_diri_out))
    print('Error Call RBF dirichlet: '+str(error_call_diri_out))

# calculate european put option price by RBF finite difference   
    t = 0.1
    T = 1.1

    euro_put_rbf_diri = euro_opt.Black_Scholes_RBF_FD_EO(Smax, Smin, K, r, d, sigma, t, T, M, N, epsilon, 'ga_rbf', 'ic_put', 'dirichlet_bc')
# calculate european put option price analytically by BS model
    St = np.linspace(Smin, Smax, N+1)
    index_out = int((S_put_out-Smin)/(St[1]-St[0]))
    
    tao = np.linspace(t, T, M+1)
    St, tao = np.meshgrid(St, tao)
    T = tao
    euro_put_BS = euro_opt.Black_Scholes_European_Put( t, T, St, K, r, d, sigma )[0]
    
    V_put_BS_out = euro_put_BS[M, index_out]
    V_put_diri_out = euro_put_rbf_diri[M, index_out]
    error_put_diri_out = (V_put_diri_out - V_put_BS_out) / V_put_BS_out
    print('Put Option Out-of-money Situation:')
    print('M = N = '+str(i)+':')
    print('Value Put BS: '+str(V_put_BS_out))
    print('Value Put RBF dirichlet: '+str(V_put_diri_out))
    print('Error Put RBF dirichlet: '+str(error_put_diri_out))


# calculate the price of in-the-money situation
for i in range(10, 100, 20):
    M = i
    N = i
# calculate european option price by rbf finite difference   
    t = 0.1
    T = 1.1
    
    euro_call_rbf_diri = euro_opt.Black_Scholes_RBF_FD_EO(Smax, Smin, K, r, d, sigma, t, T, M, N, epsilon, 'ga_rbf', 'ic_call', 'dirichlet_bc')
# calculate european option price analytically by BS model
    St = np.linspace(Smin, Smax, N+1)
    index_in = int((S_call_in-Smin)/(St[1]-St[0]))
    
    tao = np.linspace(t, T, M+1)
    St, tao = np.meshgrid(St, tao)
    T = tao
    euro_call_BS = euro_opt.Black_Scholes_European_Call( t, T, St, K, r, d, sigma )[0]
    
    V_call_BS_in = euro_call_BS[M, index_in]
    V_call_diri_in = euro_call_rbf_diri[M, index_in]
    error_call_diri_in = (V_call_diri_in - V_call_BS_in) / V_call_BS_in
    print('Call Option In-the-money Situation:')
    print('M = N = '+str(i)+':')
    print('Value Call BS: '+str(V_call_BS_in))
    print('Value Call RBF dirichlet: '+str(V_call_diri_in))
    print('Error Call RBF dirichlet: '+str(error_call_diri_in))

# calculate european put option price by RBF finite difference   
    t = 0.1
    T = 1.1

    euro_put_rbf_diri = euro_opt.Black_Scholes_RBF_FD_EO(Smax, Smin, K, r, d, sigma, t, T, M, N, epsilon, 'ga_rbf', 'ic_put', 'dirichlet_bc')
# calculate european put option price analytically by BS model
    St = np.linspace(Smin, Smax, N+1)
    index_in = int((S_put_in-Smin)/(St[1]-St[0]))
    
    tao = np.linspace(t, T, M+1)
    St, tao = np.meshgrid(St, tao)
    T = tao
    euro_put_BS = euro_opt.Black_Scholes_European_Put( t, T, St, K, r, d, sigma )[0]
    
    V_put_BS_in = euro_put_BS[M, index_in]
    V_put_diri_in = euro_put_rbf_diri[M, index_in]
    error_put_diri_in = (V_put_diri_in - V_put_BS_in) / V_put_BS_in
    print('Put Option In-of-money Situation:')
    print('M = N = '+str(i)+':')
    print('Value Put BS: '+str(V_put_BS_in))
    print('Value Put RBF dirichlet: '+str(V_put_diri_in))
    print('Error Put RBF dirichlet: '+str(error_put_diri_in))


# calculate the price of at-the-money situation
for i in range(10, 100, 20):
    M = i
    N = i
# calculate european option price by rbf finite difference   
    t = 0.1
    T = 1.1
    
    euro_call_rbf_diri = euro_opt.Black_Scholes_RBF_FD_EO(Smax, Smin, K, r, d, sigma, t, T, M, N, epsilon, 'ga_rbf', 'ic_call', 'dirichlet_bc')
# calculate european option price analytically by BS model
    St = np.linspace(Smin, Smax, N+1)
    index_at = int((S_call_at-Smin)/(St[1]-St[0]))
    
    tao = np.linspace(t, T, M+1)
    St, tao = np.meshgrid(St, tao)
    T = tao
    euro_call_BS = euro_opt.Black_Scholes_European_Call( t, T, St, K, r, d, sigma )[0]
    
    V_call_BS_at = euro_call_BS[M, index_at]
    V_call_diri_at = euro_call_rbf_diri[M, index_at]
    error_call_diri_at = (V_call_diri_at - V_call_BS_at) / V_call_BS_at
    print('Call Option At-the-money Situation:')
    print('M = N = '+str(i)+':')
    print('Value Call BS: '+str(V_call_BS_at))
    print('Value Call RBF dirichlet: '+str(V_call_diri_at))
    print('Error Call RBF dirichlet: '+str(error_call_diri_at))

# calculate european put option price by RBF finite difference   
    t = 0.1
    T = 1.1

    euro_put_rbf_diri = euro_opt.Black_Scholes_RBF_FD_EO(Smax, Smin, K, r, d, sigma, t, T, M, N, epsilon, 'ga_rbf', 'ic_put', 'dirichlet_bc')
# calculate european put option price analytically by BS model
    St = np.linspace(Smin, Smax, N+1)
    index_at = int((S_put_at-Smin)/(St[1]-St[0]))
    
    tao = np.linspace(t, T, M+1)
    St, tao = np.meshgrid(St, tao)
    T = tao
    euro_put_BS = euro_opt.Black_Scholes_European_Put( t, T, St, K, r, d, sigma )[0]
    
    V_put_BS_at = euro_put_BS[M, index_at]
    V_put_diri_at = euro_put_rbf_diri[M, index_at]
    error_put_diri_at = (V_put_diri_at - V_put_BS_at) / V_put_BS_at
    print('Put Option In-of-money Situation:')
    print('M = N = '+str(i)+':')
    print('Value Put BS: '+str(V_put_BS_at))
    print('Value Put RBF dirichlet: '+str(V_put_diri_at))
    print('Error Put RBF dirichlet: '+str(error_put_diri_at))



































