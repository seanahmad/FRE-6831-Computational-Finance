import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import stats


class ComputationalFinance(object):
    def __init__(self, stock_price, strike_price):
        self.stock_price = stock_price
        self.strike_price = strike_price
        
    def European_Call_Payoff(self):
        european_call_payoff = np.maximum(self.stock_price - self.strike_price, 0)
        return european_call_payoff
    
    
    def European_Put_Payoff(self):
        european_put_payoff = np.maximum(self.strike_price - self.stock_price, 0)
        return european_put_payoff


    def Black_Scholes_European_Call(self, t, maturity_date, stock_price, strike_price, interest_rate, \
                                    dividend_yield, volatility):
        
        d1 = 1 / volatility / ((maturity_date - t) ** 0.5) * (np.log(stock_price / strike_price) + \
             (interest_rate - dividend_yield + 0.5 * volatility ** 2) * (maturity_date - t))
        d2 = d1 - volatility * ((maturity_date - t) ** 0.5)
        bs_european_call_price = np.exp(-dividend_yield * (maturity_date - t)) * stock_price * \
                                 stats.norm.cdf(d1) - np.exp(-interest_rate * (maturity_date - t)) * \
                                 strike_price * stats.norm.cdf(d2)
        
        bs_european_call_delta = np.exp(-dividend_yield * (maturity_date - t)) * stats.norm.cdf(d1)
        
        bs_european_call_theta = dividend_yield * np.exp(-dividend_yield * (maturity_date - t)) * stock_price * \
                                stats.norm.cdf(d1) - volatility * np.exp(-dividend_yield * (maturity_date - t)) \
                                * stock_price * stats.norm.cdf(d1) / 2 / ((maturity_date - t) ** 0.5) - \
                                interest_rate * np.exp(-interest_rate * (maturity_date - t)) * \
                                strike_price * stats.norm.cdf(d2)
        
        bs_european_call_rho = (maturity_date - t) * np.exp(-interest_rate * (maturity_date - t)) * \
                                 strike_price * stats.norm.cdf(d2)
        
        bs_european_call_gamma = np.exp(-interest_rate * (maturity_date - t)) * \
                                 strike_price * stats.norm.cdf(d2) / (stock_price ** 2) \
                                 / (volatility * (maturity_date - t) ** 0.5)
        
        bs_european_call_vega = ((maturity_date - t) ** 0.5) * np.exp(-interest_rate * \
                                (maturity_date - t)) * strike_price * stats.norm.cdf(d2)
        
        return [bs_european_call_price, bs_european_call_delta, bs_european_call_theta, bs_european_call_gamma, \
                bs_european_call_rho, bs_european_call_vega]
        
        
    def Black_Scholes_European_Put(self, t, maturity_date, stock_price, strike_price, interest_rate, \
                                    dividend_yield, volatility):
        
        d1 = 1 / volatility / ((maturity_date - t) ** 0.5) * (np.log(stock_price / strike_price) + \
             (interest_rate - dividend_yield + 0.5 * volatility ** 2) * (maturity_date - t))
        d2 = d1 - volatility * ((maturity_date - t) ** 0.5)
        
        bs_european_put_price = -np.exp(-dividend_yield * (maturity_date - t)) * stock_price * \
                                 stats.norm.cdf(-d1) + np.exp(-interest_rate * (maturity_date - t)) * \
                                 strike_price * stats.norm.cdf(-d2)
                
        bs_european_put_delta = -np.exp(-dividend_yield * (maturity_date - t)) * stats.norm.cdf(-d1)
        
        bs_european_put_theta = -dividend_yield * np.exp(-dividend_yield * (maturity_date - t)) * stock_price * \
                               stats.norm.cdf(-d1) - volatility * np.exp(-dividend_yield * (maturity_date - t)) \
                               * stock_price * stats.norm.cdf(d1) / 2 / ((maturity_date - t) ** 0.5) + \
                               interest_rate * np.exp(-interest_rate * (maturity_date - t)) * \
                               strike_price * stats.norm.cdf(-d2)
        
        bs_european_put_rho = -(maturity_date - t) * np.exp(-interest_rate * (maturity_date - t)) * \
                                 strike_price * stats.norm.cdf(-d2)
        
        bs_european_put_gamma = np.exp(-interest_rate * (maturity_date - t)) * \
                                 strike_price * stats.norm.cdf(d2) / (stock_price ** 2) \
                                 / (volatility * (maturity_date - t) ** 0.5)
        
        bs_european_put_vega = ((maturity_date - t) ** 0.5) * np.exp(-interest_rate * \
                                (maturity_date - t)) * strike_price * stats.norm.cdf(d2)
        
        return [bs_european_put_price, bs_european_put_delta, bs_european_put_theta, bs_european_put_gamma, \
                bs_european_put_rho, bs_european_put_vega]
    
    
    def Black_Scholes_Explicit_FD_EO(self, strike_price, interest_rate, dividend_yield, volatility, \
                                     stock_price_min, stock_price_max, tall_min, tall_max, stock_price_div, \
                                     tall_div, initial_condition, boundary_condition):
        tall_delta = (tall_max - tall_min) / tall_div
        stock_price_delta = (stock_price_max - stock_price_min) / stock_price_div
        if tall_delta / stock_price_delta ** 2 >= 0.5:
            print('stock price and ttm division not proper')
            quit()
        else:
            ti_tempo = list(np.arange(tall_min, tall_max, tall_delta))
            ti_tempo.append(tall_max)
            ti = np.array(ti_tempo)

            si_tempo = list(np.arange(stock_price_min, stock_price_max, stock_price_delta))
            si_tempo.append(stock_price_max)
            si = np.array(si_tempo)

            price_dataset = pd.DataFrame(index = list(si))
            if initial_condition == 'ic_call':
                price_dataset['price at ttm %.5f' %ti[0]] = np.maximum(si - strike_price, 0)
            elif initial_condition == 'ic_put':
                price_dataset['price at ttm %.5f' %ti[0]] = np.maximum(strike_price - si, 0)
            else:
                print('initial_condition should be ic_call or ic_put')
            
            l = ((volatility * si) ** 2 / 2) * tall_delta / stock_price_delta ** 2 - \
                (interest_rate - dividend_yield) * si / 2 * tall_delta / stock_price_delta
            d = 1 - interest_rate * tall_delta - (volatility * si) ** 2 * tall_delta / stock_price_delta ** 2
            u = ((volatility * si) ** 2 / 2) * tall_delta / stock_price_delta ** 2 + \
                (interest_rate - dividend_yield) * si / 2 * tall_delta / stock_price_delta

            l1 = l[0].copy()
            d1 = d[0].copy()
            u1 = u[0].copy()
            l_end = l[l.size-1].copy()
            d_end = d[d.size-1].copy()
            u_end = u[u.size-1].copy()

            d[0] = 2 * l1 + d1
            d[d.size-1] = d_end + 2 * u_end
            u[0] = u1 - l1
            l[l.size-1] = l_end - u_end

            l_tempo = np.diag(l[1:l.size], k=-1)
            d_tempo = np.diag(d)
            u_tempo = np.diag(u[0:u.size-1], k=1)

            transition = l_tempo + d_tempo + u_tempo

            K = ti.size - 1
            for i in range(K):
                if boundary_condition == 'dirichlet_bc':
                    value_tempo = np.dot(transition, price_dataset['price at ttm %.5f' %ti[i]])
                    value_tempo[0] = price_dataset['price at ttm %.5f' %ti[i]][si[0]]
                    value_tempo[value_tempo.size-1] = price_dataset['price at ttm %.5f' %ti[i]][si[si.size-1]]
                    price_dataset['price at ttm %.5f' %ti[i+1]] = value_tempo
                    
                elif boundary_condition == 'neumann_bc':
                    value_tempo = np.dot(transition, price_dataset['price at ttm %.5f' %ti[i]])
                    price_dataset['price at ttm %.5f' %ti[i+1]] = value_tempo
   
                else:
                    print('boundary condition should be dirichlet_bc or neumann_bc')
            
        return price_dataset
