# ComputationalFinanceClass.py

import numpy as np
from scipy import stats
from numpy import power, sqrt, log, exp
import scipy.sparse
from scipy.stats.mstats import gmean

class ComputationalFinance( object ):
    def __init__( self, stock_price, strike_price):
          self.stock_price = stock_price
          self.strike_price = strike_price
        
    def European_Call_Payoff( self, strike_price ): # compute the payoff
        boolean_values = self.stock_price > strike_price
        european_call_payoff = (boolean_values + 0.0) * (self.stock_price - strike_price)
        return european_call_payoff

    def European_Put_Payoff( self, strike_price ): # compute the payoff
        boolean_values = self.stock_price < strike_price
        european_put_payoff = (boolean_values + 0.0) * (strike_price - self.stock_price)
        return european_put_payoff

    '''
        parameter description:
        S0: initial price
        n: number of steps
        t: starting time
        T: terminating time
        St: trajectory of price
        a, b: the first and second moment of log Y
        d: dividend yield
    '''

    def Geometric_Brownian_Motion_Trajectory(self, mu, sigma, S0, n, t, T): 
        time = np.linspace(t, T, n + 1) 
        delta_time = time[1] - time[0] 
        St = np.zeros(n + 1)
        St[0] = S0
        z = np.random.standard_normal(n) 
        for i in range(n):
            St[i + 1] = St[i] * np.exp((mu - 1 / 2 * sigma ** 2) * delta_time + sigma * delta_time ** (1 / 2) * z[i])
        return St

    def Geometric_Brownian_Motion_Jump(self, mu, sigma, d, S0, n, t, T, a, b, lam):
        
        delta_t = (T - t) / n
        St = np.zeros(n + 1)
        X = np.zeros(n + 1)
        z = np.random.normal(size=(n + 1, 1))
        X[0] = np.log(S0)
        for i in range(1, n + 1):
            n = np.random.poisson(lam * delta_t)
            if n == 0:
                m =0 
            else:
                m = a * n + b * n ** 0.5 * np.random.normal()
            X[i] = X[i - 1] + (mu - d - 0.5 * sigma ** 2) * delta_t + sigma * delta_t ** 0.5 * z[i] + m
        St = np.exp(X)
        return St
    
    def Arithmetic_Average_Price_Asian_Call(self, T, t, S0, sigma, lam, a, b, mu, d, K, n, M):

        prices = []
        for i in range(M):
            S = self.Geometric_Brownian_Motion_Jump(mu, sigma, d, S0, n, t, T, a, b, lam)
            Sm = np.mean(S)
            Payoff = np.maximum(Sm - K, 0)
            price = np.exp(-mu * T) * Payoff
            prices.append(price)
        AAPAC = np.mean(prices)
        volatility = np.std(prices)
        
        return AAPAC, volatility
    
    def Geometric_Average_Price_Asian_Call(self, T, t, S0, sigma, lam, a, b, mu, d, K, n, M):

        prices = []
        for i in range(M):
            S = self.Geometric_Brownian_Motion_Jump(mu, sigma, d, S0, n, t, T, a, b, lam)
            log_S = np.log(S)
            Sm = np.exp(np.mean(log_S))
            Payoff = np.maximum(Sm - K, 0)
            price = np.exp(-mu * T) * Payoff
            prices.append(price)
        GAPAC = np.mean(prices)
        
        return GAPAC
    
    def BS_Geometric_Average_Price_Asian_Call(self, T, t, mu, d, sigma, S0, K):
        b = 0.5 * (mu + d + (1 / 6) * sigma * sigma)
        sigma = sigma / np.sqrt(3)
        d1 = (np.log(S0 / K) + (mu - b + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        BSGAPAC = S0 * np.exp(-b * T) * stats.norm.cdf(d1, 0, 1) - K * np.exp(-mu * T) * stats.norm.cdf(d2, 0, 1)
        
        return BSGAPAC
    
    def Control_Variates_Arithmetic_Average_Asian_Call(self, T, t, S0, sigma, lam, a, b, mu, d, K, n, M):
        
        ari_price = []
        geo_price = []
        
        for i in range(M):
            S = self.Geometric_Brownian_Motion_Jump(mu, sigma, d, S0, n, t, T, a, b, lam)
            S_ari_mean = np.mean(S)
            S_geo_mean = np.exp(np.mean(np.log(S)))
            ariprice = np.exp(-mu * T) * np.maximum(S_ari_mean - K, 0)
            ari_price.append(ariprice)
            geoprice = np.exp(-mu * T) * np.maximum(S_geo_mean - K, 0)
            geo_price.append(geoprice)
            
        b = (np.cov(ari_price, geo_price)[0][1]) / (np.var(geo_price))
        EX = self.BS_Geometric_Average_Price_Asian_Call(T, t, mu, d, sigma, S0, K)
        Price = ari_price - b * (geo_price - EX)
        
        return np.mean(Price), np.std(Price), b, np.mean(ari_price), np.mean(geo_price)





    
    
    
    
    
    
    
    
