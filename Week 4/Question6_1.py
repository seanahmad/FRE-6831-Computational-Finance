import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import stats
from ComputationalFinanceClass import *

variable = ComputationalFinance(0,0)
strike_price = 100
interest_rate = 0.05
dividend_yield = 0.02
volatility = 0.3
stock_price_min = 0
stock_price_max = 200
tall_min = 0
tall_max = 1
stock_price_div = 100
tall_div = 2000

tall_delta = (tall_max - tall_min) / tall_div
stock_price_delta = (stock_price_max - stock_price_min) / stock_price_div
stock_price1_tempo = list(np.arange(stock_price_min, stock_price_max, stock_price_delta))
stock_price1_tempo.append(stock_price_max)
stock_price1 = np.array(stock_price1_tempo)
ttm1_tempo = list(np.arange(tall_min, tall_max, tall_delta))
ttm1_tempo.append(tall_max)
ttm1 = np.array(ttm1_tempo)
stock_price1_initial = stock_price1.copy()
ttm1_initial = ttm1.copy()
stock_price1, ttm1 = np.meshgrid(stock_price1, ttm1)

stock_price2_tempo = list(np.arange(stock_price_min, stock_price_max, (stock_price_max - stock_price_min) / stock_price_div))
stock_price2_tempo.append(stock_price_max)
stock_price2 = np.array(stock_price2_tempo)
ttm2_tempo = list(np.arange(tall_max, tall_min, -(tall_max - tall_min) / tall_div ))
ttm2_tempo.append(tall_min)
ttm2 = np.array(ttm2_tempo)
maturity_date = 1
stock_price2_initial = stock_price2.copy()
ttm2_initial = ttm2.copy()
stock_price2, ttm2 = np.meshgrid(stock_price2, ttm2)



call_price_numerical_method_dataset = variable.Black_Scholes_Explicit_FD_EO(strike_price, interest_rate, dividend_yield, volatility, \
                                     stock_price_min, stock_price_max, tall_min, tall_max, stock_price_div, \
                                     tall_div, 'ic_call', 'dirichlet_bc')

put_price_numerical_method_dataset = variable.Black_Scholes_Explicit_FD_EO(strike_price, interest_rate, dividend_yield, volatility, \
                                     stock_price_min, stock_price_max, tall_min, tall_max, stock_price_div, \
                                     tall_div, 'ic_put', 'dirichlet_bc')

call_price_BS_dataset = variable.Black_Scholes_European_Call(ttm2, maturity_date, stock_price2, strike_price, interest_rate, \
                                    dividend_yield, volatility)

put_price_BS_dataset = variable.Black_Scholes_European_Put(ttm2, maturity_date, stock_price2, strike_price, interest_rate, \
                                    dividend_yield, volatility)


# Question 6(3)
fig = plt.figure(figsize=(16,10))

ax1 = plt.subplot(2, 2, 1, projection = '3d')
data_tempo1 = np.array(call_price_numerical_method_dataset)
surf = ax1.plot_surface(stock_price1, ttm1, data_tempo1.T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax1.set_zlim(0, 105)
ax1.set_xlabel('Stock Price')
ax1.set_ylabel('Time to Maturity')
ax1.set_title('Call Price Using Numerical Method')
ax1.zaxis.set_major_locator(LinearLocator(10))
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)


ax2 = plt.subplot(2, 2, 2, projection = '3d')
data_tempo2 = np.array(put_price_numerical_method_dataset)
surf = ax2.plot_surface(stock_price1, ttm1, data_tempo2.T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax2.set_zlim(0, 105)
ax2.set_xlabel('Stock Price')
ax2.set_ylabel('Time to Maturity')
ax2.set_title('Put Price Using Numerical Method')
ax2.zaxis.set_major_locator(LinearLocator(10))
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)


ax3 = plt.subplot(2, 2, 3, projection = '3d')
surf = ax3.plot_surface(stock_price2, 1 - ttm2, call_price_BS_dataset[0], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax3.set_zlim(0, 105)
ax3.set_xlabel('Stock Price')
ax3.set_ylabel('Time to Maturity')
ax3.set_title('Call Price Using BS Method')
ax3.zaxis.set_major_locator(LinearLocator(10))
ax3.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)


ax4 = plt.subplot(2, 2, 4, projection = '3d')
surf = ax4.plot_surface(stock_price2, 1 - ttm2, put_price_BS_dataset[0], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax4.set_zlim(0, 105)
ax4.set_xlabel('Stock Price')
ax4.set_ylabel('Time to Maturity')
ax4.set_title('Put Price Using BS Method')
ax4.zaxis.set_major_locator(LinearLocator(10))
ax4.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)

# Question 6(4)
error_sout_dataset = pd.DataFrame(index = ['V_call_sout_bs', 'V_call_sout_explicit_dirichlet', 'V_call_sout_explicit_neumann',\
                                           'error_call_sout_explicit_dirichlet', 'error_call_sout_explicit_neumann', \
                                           'V_put_sout_bs', 'V_put_sout_explicit_dirichlet', 'V_put_sout_explicit_neumann',\
                                           'error_put_sout_explicit_dirichlet', 'error_put_sout_explicit_neumann'])

error_sin_dataset = pd.DataFrame(index = ['V_call_sin_bs', 'V_call_sin_explicit_dirichlet', 'V_call_sin_explicit_neumann',\
                                           'error_call_sin_explicit_dirichlet', 'error_call_sin_explicit_neumann', \
                                           'V_put_sin_bs', 'V_put_sin_explicit_dirichlet', 'V_put_sin_explicit_neumann',\
                                           'error_put_sin_explicit_dirichlet', 'error_put_sin_explicit_neumann'])

error_sat_dataset = pd.DataFrame(index = ['V_call_sat_bs', 'V_call_sat_explicit_dirichlet', 'V_call_sat_explicit_neumann',\
                                           'error_call_sat_explicit_dirichlet', 'error_call_sat_explicit_neumann', \
                                           'V_put_sat_bs', 'V_put_sat_explicit_dirichlet', 'V_put_sat_explicit_neumann',\
                                           'error_put_sat_explicit_dirichlet', 'error_put_sat_explicit_neumann'])

for i in range(5, 70, 5):
    M = i * 100
    N = 100

    stock_price_div = N
    tall_div = M

    tall_delta = (tall_max - tall_min) / tall_div
    stock_price_delta = (stock_price_max - stock_price_min) / stock_price_div
    stock_price1_tempo = list(np.arange(stock_price_min, stock_price_max, stock_price_delta))
    stock_price1_tempo.append(stock_price_max)
    stock_price1 = np.array(stock_price1_tempo)
    ttm1_tempo = list(np.arange(tall_min, tall_max, tall_delta))
    ttm1_tempo.append(tall_max)
    ttm1 = np.array(ttm1_tempo)
    stock_price1_initial = stock_price1.copy()
    ttm1_initial = ttm1.copy()
    stock_price1, ttm1 = np.meshgrid(stock_price1, ttm1)

    stock_price2_tempo = list(np.arange(stock_price_min, stock_price_max, (stock_price_max - stock_price_min) / stock_price_div))
    stock_price2_tempo.append(stock_price_max)
    stock_price2 = np.array(stock_price2_tempo)
    ttm2_tempo = list(np.arange(tall_max, tall_min, -(tall_max - tall_min) / tall_div ))
    ttm2_tempo.append(tall_min)
    ttm2 = np.array(ttm2_tempo)
    maturity_date = 1
    stock_price2_initial = stock_price2.copy()
    ttm2_initial = ttm2.copy()
    stock_price2, ttm2 = np.meshgrid(stock_price2, ttm2)



    call_price_numerical_method_dirichlet_dataset = variable.Black_Scholes_Explicit_FD_EO(strike_price, interest_rate, dividend_yield, volatility, \
                                         stock_price_min, stock_price_max, tall_min, tall_max, stock_price_div, \
                                         tall_div, 'ic_call', 'dirichlet_bc')

    put_price_numerical_method_dirichlet_dataset = variable.Black_Scholes_Explicit_FD_EO(strike_price, interest_rate, dividend_yield, volatility, \
                                         stock_price_min, stock_price_max, tall_min, tall_max, stock_price_div, \
                                         tall_div, 'ic_put', 'dirichlet_bc')

    call_price_numerical_method_neumann_dataset = variable.Black_Scholes_Explicit_FD_EO(strike_price, interest_rate, dividend_yield, volatility, \
                                         stock_price_min, stock_price_max, tall_min, tall_max, stock_price_div, \
                                         tall_div, 'ic_call', 'neumann_bc')

    put_price_numerical_method_neumann_dataset = variable.Black_Scholes_Explicit_FD_EO(strike_price, interest_rate, dividend_yield, volatility, \
                                         stock_price_min, stock_price_max, tall_min, tall_max, stock_price_div, \
                                         tall_div, 'ic_put', 'neumann_bc')

    call_price_BS_dataset = variable.Black_Scholes_European_Call(ttm2, maturity_date, stock_price2, strike_price, interest_rate, \
                                        dividend_yield, volatility)

    put_price_BS_dataset = variable.Black_Scholes_European_Put(ttm2, maturity_date, stock_price2, strike_price, interest_rate, \
                                        dividend_yield, volatility)

    error_call_explicit_dirichlet = call_price_numerical_method_dirichlet_dataset / call_price_BS_dataset[0].T - 1

    error_call_explicit_neumann = call_price_numerical_method_neumann_dataset / call_price_BS_dataset[0].T - 1

    error_put_explicit_dirichlet = put_price_numerical_method_dirichlet_dataset / put_price_BS_dataset[0].T - 1

    error_put_explicit_neumann = put_price_numerical_method_neumann_dataset / put_price_BS_dataset[0].T - 1


    error_sout_list = [call_price_BS_dataset[0].T[1][M], np.array(call_price_numerical_method_dirichlet_dataset)[1][M], np.array(call_price_numerical_method_neumann_dataset)[1][M], \
                       np.array(error_call_explicit_dirichlet)[1][M], np.array(error_call_explicit_neumann)[1][M], \
                       put_price_BS_dataset[0].T[N-1][M], np.array(put_price_numerical_method_dirichlet_dataset)[N-1][M], np.array(put_price_numerical_method_neumann_dataset)[N-1][M], \
                       np.array(error_put_explicit_dirichlet)[N-1][M], np.array(error_put_explicit_neumann)[N-1][M],]


    error_sin_list = [call_price_BS_dataset[0].T[N-1][M], np.array(call_price_numerical_method_dirichlet_dataset)[N-1][M], np.array(call_price_numerical_method_neumann_dataset)[N-1][M], \
                       np.array(error_call_explicit_dirichlet)[N-1][M], np.array(error_call_explicit_neumann)[N-1][M], \
                       put_price_BS_dataset[0].T[1][M], np.array(put_price_numerical_method_dirichlet_dataset)[1][M], np.array(put_price_numerical_method_neumann_dataset)[1][M], \
                       np.array(error_put_explicit_dirichlet)[1][M], np.array(error_put_explicit_neumann)[1][M],]

    n = int(N / 2)
    error_sat_list = [call_price_BS_dataset[0].T[n][M], np.array(call_price_numerical_method_dirichlet_dataset)[n][M], np.array(call_price_numerical_method_neumann_dataset)[n][M], \
                       np.array(error_call_explicit_dirichlet)[n][M], np.array(error_call_explicit_neumann)[n][M], \
                       put_price_BS_dataset[0].T[n][M], np.array(put_price_numerical_method_dirichlet_dataset)[n][M], np.array(put_price_numerical_method_neumann_dataset)[n][M], \
                       np.array(error_put_explicit_dirichlet)[n][M], np.array(error_put_explicit_neumann)[n][M],]

    error_sout_dataset['M = N = %.1f' %i] = error_sout_list
    error_sin_dataset['M = N = %.1f' %i] = error_sin_list
    error_sat_dataset['M = N = %.1f' %i] = error_sat_list

error_sin_dataset
