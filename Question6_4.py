# Question6_4 (3).py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import stats
from ComputationalFinanceClass import *

Smax = 150
Smin = 5
K = 100
t = 0
N = 40
M = 40
epsilon = 4
sigma = 0.2
r = 0.03
q = 0
T =  np.linspace(0.1, 1, 20)
Stock_price = np.exp(np.linspace(np.log(Smin),np.log(Smax),N))
c = ComputationalFinance(10,10)

Call = []
Put = []

for i in range(len(T)):
    GRBF_call_price = c.Black_Scholes_Global_RBF_EO(Smax,Smin,K,r,sigma,t,T[i],N,M,epsilon,"ga_rbf","ic_call","dirichlet_bc")[:,-1]
    Call.append(GRBF_call_price)
    GRBF_put_price = c.Black_Scholes_Global_RBF_EO(Smax,Smin,K,r,sigma,t,T[i],N,M,epsilon,"ga_rbf","ic_put","dirichlet_bc")[:,-1]
    Put.append(GRBF_put_price)

Stock, T = np.meshgrid(Stock_price, T)
fig = plt.figure( figsize=(15, 7) );

# calculate call price by RBF
ax = fig.add_subplot( 1, 2, 1, projection = '3d' )
surf = ax.plot_surface( np.array(Stock_price), np.array(T), np.array(Call), cmap = plt.cm.viridis)
ax.set_title("European Call Price by RBF")
ax.set_xlabel("Stock Price")
ax.set_ylabel("Time to Maturity")
ax.set_zlabel("Price")

# calculate put price by RBF
ax = fig.add_subplot( 1, 2, 2, projection = '3d' )
surf = ax.plot_surface( np.array(Stock_price), np.array(T), np.array(Put), cmap = plt.cm.coolwarm)
ax.set_title("European Put Price by RBF")
ax.set_xlabel("Stock Price")
ax.set_ylabel("Time to Maturity")
ax.set_zlabel("Price")
plt.show()

