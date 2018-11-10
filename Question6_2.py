# Question6_2 (3).py

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
    vol: the volatility of returns of the underlying asset
    N: number of sub-intervals in s-direction
    M: number of sub-intervals in tao-direction
'''

Smin = 50.0
Smax = 150.0
t = 0.1
T = 1.1
K = 100.0
r = 0.05
d = 0.02
vol = 0.3
N = 50
M = 50

# initialize
euro_opt = ComputationalFinance(0, 0)
 
# calculate european option price by implicit finite difference
euro_call_implicit = euro_opt.Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M, 'ic_call', 'dirichlet_bc' )
euro_put_implicit = euro_opt.Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M, 'ic_put', 'dirichlet_bc' )

St = np.linspace(Smin, Smax, N+1)
tao = np.linspace(t, T, M+1)
St, tao = np.meshgrid(St, tao)

# calculate european option price analytically by BS model
T = tao
euro_call_BS = euro_opt.Black_Scholes_European_Call( t, T, St, K, r, d, vol )[0]
euro_put_BS = euro_opt.Black_Scholes_European_Put( t, T, St, K, r, d, vol )[0]

# Plot 2D European Option Price
fig = plt.figure(figsize=(15, 7))
fig.add_subplot(111)
plt.plot(St[0, :], euro_call_implicit[M, :], 'r*', label='Call FDM')
plt.plot(St[0, :], euro_call_BS[M, :], label='Call BS')
plt.plot(St[0, :], euro_put_implicit[M, :], 'r.', label='Put FDM')
plt.plot(St[0, :], euro_put_BS[M, :], label='Put BS')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('Stock Price')
plt.ylabel('European Option Price')
plt.title('European Option Price by Implicit Finite Difference Method')

# Plot 3D European Option Price
fig = plt.figure(figsize=(15, 7))
plt.title('European Option Price by Implicit Finite Difference Method')
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(St, tao, euro_call_implicit, rstride=1, cstride=1, cmap=plt.cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('European Call Price by Implicit Finite Difference Method')
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(St, tao, euro_put_implicit, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Maturity')
ax.set_title('European Put Price by Implicit Finite Difference Method')

# Question 6_2 (4)
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
# calculate european option price by implicit finite difference   
    t = 0.1
    T = 1.1
    euro_call_implicit_diri = euro_opt.Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M, 'ic_call', 'dirichlet_bc' )
    euro_call_implicit_neum = euro_opt.Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M, 'ic_call', 'neumann_bc' )
# calculate european option price analytically by BS model
    St = np.linspace(Smin, Smax, N+1)
    index_out = int((S_call_out-Smin)/(St[1]-St[0]))
    
    tao = np.linspace(t, T, M+1)
    St, tao = np.meshgrid(St, tao)
    T = tao
    euro_call_BS = euro_opt.Black_Scholes_European_Call( t, T, St, K, r, d, vol )[0]
    
    V_call_BS_out = euro_call_BS[M, index_out]
    V_call_diri_out = euro_call_implicit_diri[M, index_out]
    V_call_neum_out = euro_call_implicit_neum[M, index_out]
    error_call_diri_out = (V_call_diri_out - V_call_BS_out) / V_call_BS_out
    error_call_neum_out = (V_call_neum_out - V_call_BS_out) / V_call_BS_out
    print('Call Option Out-of-money Situation:')
    print('M = N = '+str(i)+':')
    print('Value Call BS: '+str(V_call_BS_out))
    print('Value Call Implicit dirichlet: '+str(V_call_diri_out))
    print('Value Call Implicit neumman: '+str(V_call_neum_out))
    print('Error Call Implicit dirichlet: '+str(error_call_neum_out))
    print('Error Call Implicit neumman: '+str(error_call_neum_out))

# calculate european put option price by implicit finite difference   
    t = 0.1
    T = 1.1
    euro_put_implicit_diri = euro_opt.Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M, 'ic_put', 'dirichlet_bc' )
    euro_put_implicit_neum = euro_opt.Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M, 'ic_put', 'neumann_bc' )
# calculate european put option price analytically by BS model
    St = np.linspace(Smin, Smax, N+1)
    index_out = int((S_put_out-Smin)/(St[1]-St[0]))
    
    tao = np.linspace(t, T, M+1)
    St, tao = np.meshgrid(St, tao)
    T = tao
    euro_put_BS = euro_opt.Black_Scholes_European_Put( t, T, St, K, r, d, vol )[0]
    
    V_put_BS_out = euro_put_BS[M, index_out]
    V_put_diri_out = euro_put_implicit_diri[M, index_out]
    V_put_neum_out = euro_put_implicit_neum[M, index_out]
    error_put_diri_out = (V_put_diri_out - V_put_BS_out) / V_put_BS_out
    error_put_neum_out = (V_put_neum_out - V_put_BS_out) / V_put_BS_out
    print('Put Option Out-of-money Situation:')
    print('M = N = '+str(i)+':')
    print('Value Put BS: '+str(V_put_BS_out))
    print('Value Put Implicit dirichlet: '+str(V_put_diri_out))
    print('Value Put Implicit neumman: '+str(V_put_neum_out))
    print('Error Put Implicit dirichlet: '+str(error_put_neum_out))
    print('Error Put Implicit neumman: '+str(error_put_neum_out))


# calculate the price of in-the-money situation
for i in range(10, 100, 20):
    M = i
    N = i
# calculate european option price by implicit finite difference   
    t = 0.1
    T = 1.1
    euro_call_implicit_diri = euro_opt.Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M, 'ic_call', 'dirichlet_bc' )
    euro_call_implicit_neum = euro_opt.Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M, 'ic_call', 'neumann_bc' )
# calculate european option price analytically by BS model
    St = np.linspace(Smin, Smax, N+1)
    index_in = int((S_call_in-Smin)/(St[1]-St[0]))
    
    tao = np.linspace(t, T, M+1)
    St, tao = np.meshgrid(St, tao)
    T = tao
    euro_call_BS = euro_opt.Black_Scholes_European_Call( t, T, St, K, r, d, vol )[0]
    
    V_call_BS_in = euro_call_BS[M, index_in]
    V_call_diri_in = euro_call_implicit_diri[M, index_in]
    V_call_neum_in = euro_call_implicit_neum[M, index_in]
    error_call_diri_in = (V_call_diri_in - V_call_BS_in) / V_call_BS_in
    error_call_neum_in = (V_call_neum_in - V_call_BS_in) / V_call_BS_in
    print('Call Option in-the-money Situation:')
    print('M = N = '+str(i)+':')
    print('Value Call BS: '+str(V_call_BS_in))
    print('Value Call Implicit dirichlet: '+str(V_call_diri_in))
    print('Value Call Implicit neumman: '+str(V_call_neum_in))
    print('Error Call Implicit dirichlet: '+str(error_call_neum_in))
    print('Error Call Implicit neumman: '+str(error_call_neum_in))

# calculate european put option price by implicit finite difference   
    t = 0.1
    T = 1.1
    euro_put_implicit_diri = euro_opt.Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M, 'ic_put', 'dirichlet_bc' )
    euro_put_implicit_neum = euro_opt.Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M, 'ic_put', 'neumann_bc' )
# calculate european put option price analytically by BS model
    St = np.linspace(Smin, Smax, N+1)
    index_in = int((S_put_in-Smin)/(St[1]-St[0]))
    
    tao = np.linspace(t, T, M+1)
    St, tao = np.meshgrid(St, tao)
    T = tao
    euro_put_BS = euro_opt.Black_Scholes_European_Put( t, T, St, K, r, d, vol )[0]
    
    V_put_BS_in = euro_put_BS[M, index_in]
    V_put_diri_in = euro_put_implicit_diri[M, index_in]
    V_put_neum_in = euro_put_implicit_neum[M, index_in]
    error_put_diri_in = (V_put_diri_in - V_put_BS_in) / V_put_BS_in
    error_put_neum_in = (V_put_neum_in - V_put_BS_in) / V_put_BS_in
    print('Put Option in-the-money Situation:')
    print('M = N = '+str(i)+':')
    print('Value Put BS: '+str(V_put_BS_out))
    print('Value Put Implicit dirichlet: '+str(V_put_diri_in))
    print('Value Put Implicit neumman: '+str(V_put_neum_in))
    print('Error Put Implicit dirichlet: '+str(error_put_neum_in))
    print('Error Put Implicit neumman: '+str(error_put_neum_in))


# calculate the price of at-the-money situation
for i in range(10, 100, 20):
    M = i
    N = i
# calculate european option price by implicit finite difference   
    t = 0.1
    T = 1.1
    euro_call_implicit_diri = euro_opt.Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M, 'ic_call', 'dirichlet_bc' )
    euro_call_implicit_neum = euro_opt.Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M, 'ic_call', 'neumann_bc' )
# calculate european option price analytically by BS model
    St = np.linspace(Smin, Smax, N+1)
    index_at = int((S_call_at-Smin)/(St[1]-St[0]))
    
    tao = np.linspace(t, T, M+1)
    St, tao = np.meshgrid(St, tao)
    T = tao
    euro_call_BS = euro_opt.Black_Scholes_European_Call( t, T, St, K, r, d, vol )[0]
    
    V_call_BS_at = euro_call_BS[M, index_at]
    V_call_diri_at = euro_call_implicit_diri[M, index_at]
    V_call_neum_at = euro_call_implicit_neum[M, index_at]
    error_call_diri_at = (V_call_diri_at - V_call_BS_at) / V_call_BS_at
    error_call_neum_at = (V_call_neum_at - V_call_BS_at) / V_call_BS_at
    print('Call Option at-the-money Situation:')
    print('M = N = '+str(i)+':')
    print('Value Call BS: '+str(V_call_BS_at))
    print('Value Call Implicit dirichlet: '+str(V_call_diri_at))
    print('Value Call Implicit neumman: '+str(V_call_neum_at))
    print('Error Call Implicit dirichlet: '+str(error_call_neum_at))
    print('Error Call Implicit neumman: '+str(error_call_neum_at))

# calculate european put option price by implicit finite difference   
    t = 0.1
    T = 1.1
    euro_put_implicit_diri = euro_opt.Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M, 'ic_put', 'dirichlet_bc' )
    euro_put_implicit_neum = euro_opt.Black_Scholes_Implicit_FD_EO( K, r, d, vol, Smin, Smax, t, T, N, M, 'ic_put', 'neumann_bc' )
# calculate european put option price analytically by BS model
    St = np.linspace(Smin, Smax, N+1)
    index_at = int((S_put_at-Smin)/(St[1]-St[0]))
    
    tao = np.linspace(t, T, M+1)
    St, tao = np.meshgrid(St, tao)
    T = tao
    euro_put_BS = euro_opt.Black_Scholes_European_Put( t, T, St, K, r, d, vol )[0]
    
    V_put_BS_at = euro_put_BS[M, index_at]
    V_put_diri_at = euro_put_implicit_diri[M, index_at]
    V_put_neum_at = euro_put_implicit_neum[M, index_at]
    error_put_diri_at = (V_put_diri_at - V_put_BS_at) / V_put_BS_at
    error_put_neum_at = (V_put_neum_at - V_put_BS_at) / V_put_BS_at
    print('Put Option at-the-money Situation:')
    print('M = N = '+str(i)+':')
    print('Value Put BS: '+str(V_put_BS_at))
    print('Value Put Implicit dirichlet: '+str(V_put_diri_at))
    print('Value Put Implicit neumman: '+str(V_put_neum_at))
    print('Error Put Implicit dirichlet: '+str(error_put_neum_at))
    print('Error Put Implicit neumman: '+str(error_put_neum_at))

















