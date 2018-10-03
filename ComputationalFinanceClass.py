# ComputationalFinanceClass.py

from scipy.stats import norm
import numpy as np

class ComputationalFinance( object ):
    def __init__( self, stock_price):
          self.stock_price = stock_price
          #self.strike_price = strike_price
        
    def European_Call_Payoff( self, strike_price ): # compute the payoff
        boolean_values = self.stock_price > strike_price
        european_call_payoff = (boolean_values + 0.0) * (self.stock_price - strike_price)
        return european_call_payoff

    def European_Put_Payoff( self, strike_price ): # compute the payoff
        boolean_values = self.stock_price < strike_price
        european_put_payoff = (boolean_values + 0.0) * (strike_price - self.stock_price)
        return european_put_payoff

# European Option Trading Strategy
    def Bull_Call_Spread( self, strike_price_1, strike_price_2 ):
        if strike_price_1 < strike_price_2:
            bull_call_spread = self.European_Call_Payoff(strike_price_1) - self.European_Call_Payoff(strike_price_2)
        else:
            print('Warning: inputs are incorrect!', '\n')
            print('strike_price_1 < strike_price_2 does not hold!')
            quit()
        return bull_call_spread
    
    def Bull_Put_Spread( self, strike_price_1, strike_price_2 ):
        if strike_price_1 > strike_price_2:
            bull_put_spread = self.European_Put_Payoff(strike_price_1) - self.European_Put_Payoff(strike_price_2)
        else:
            print('Warning: inputs are incorrect!', '\n')
            print('strike_price_1 > strike_price_2 does not hold!')
            quit()
        return bull_put_spread
    
    def Bear_Call_Spread( self, strike_price_1, strike_price_2 ):
        if strike_price_1 > strike_price_2:
            bear_call_spread = self.European_Call_Payoff(strike_price_1) - self.European_Call_Payoff(strike_price_2)
        else:
            print('Warning: inputs are incorrect!', '\n')
            print('strike_price_1 > strike_price_2 does not hold!')
            quit()
        return bear_call_spread
    
    def Collar( self, strike_price_1, strike_price_2 ):
        if strike_price_1 < strike_price_2:
            collar = self.European_Put_Payoff(strike_price_1) - self.European_Call_Payoff(strike_price_2)
        else:
            print('Warning: inputs are incorrect!', '\n')
            print('strike_price_1 < strike_price_2 does not hold!')
            quit()
        return collar
    
    def Straddle( self, strike_price ):
        straddle = self.European_Call_Payoff(strike_price) + self.European_Put_Payoff(strike_price)
        return straddle
    
    def Strangle( self, strike_price_1, strike_price_2 ):
        if strike_price_1 != strike_price_2:
            strangle = self.European_Call_Payoff(strike_price_1) + self.European_Put_Payoff(strike_price_2)
        else:
            print('Warning: inputs are incorrect!', '\n')
            print('strike_price_1 != strike_price_2 does not hold!')
            quit()
        return strangle

    def Butterfly_Spread( self, strike_price_1, strike_price_2, strike_price_3 ):
        if strike_price_1 < strike_price_2 and strike_price_2 < strike_price_3:
            para_lambda = (strike_price_3 - strike_price_2) / (strike_price_3 - strike_price_1)
            butterfly_spread = para_lambda * self.European_Call_Payoff(strike_price_1) + \
            (1 - para_lambda) * self.European_Call_Payoff(strike_price_3) - self.European_Call_Payoff(strike_price_2)
        else:        
            print('Warning: inputs are incorrect!', '\n')
            print('strike_price_1 < strike_price_2 < strike_price_3 does not hold!')
            quit()
        return butterfly_spread

# Balck Scholes European Option Price and Greeks
    ''' 
    parameter description:
    t: time
    T: maturity date of the option
    St: the spot price of the underlying asset at time t
    K: the strike price of the option 
    r: riskfree rate
    d: the dividend yield of the underlying asset
    vol: the volatility of returns of the underlying asset
    '''
    
    def Black_Scholes_European_Call( self, t, T, St, K, r, d, vol ): 
        
        d1 = 1 / (vol * (T-t)**(1/2)) * (np.log(St/K) + (r - d + 1/2 * vol**2) * (T-t))
        d2 = d1 - vol * (T-t)**(1/2)
        norm1 = norm.cdf(d1)
        norm2 = norm.cdf(d2)
        
        bs_european_call_price = np.exp(-d*(T-t)) * St * norm1 - np.exp(-r*(T-t)) * K * norm2
        bs_european_call_delta = np.exp(-d*(T-t)) * norm1
        bs_european_call_theta = d * np.exp(-d*(T-t)) * St * norm1 - (vol * np.exp(-d*(T-t)) * St * norm.pdf(d1)) / (2 * (T-t)**(1/2)) - \
        r * np.exp(-r*(T-t)) * K * norm2
        bs_european_call_vega = (T-t)**(1/2) * np.exp(-r*(T-t)) * K * norm.pdf(d2)
        bs_european_call_gamma = (np.exp(-r*(T-t)) * K * norm.pdf(d2)) / (St**2 * vol * (T-t)**(1/2))
        bs_european_call_rho = (T-t)*np.exp(-r*(T-t)) * K * norm2
        
        
        return bs_european_call_price, bs_european_call_delta, bs_european_call_theta, \
                bs_european_call_vega, bs_european_call_gamma, bs_european_call_rho
    
    
    
    
    def Black_Scholes_European_Put( self, t, T, St, K, r, d, vol): 

        d1 = 1 / (vol * (T-t)**(1/2)) * (np.log(St/K) + (r - d + 1/2 * vol**2) * (T-t))
        d2 = d1 - vol * (T-t)**(1/2)
        norm1 = norm.cdf(-d1)
        norm2 = norm.cdf(-d2)
        
        bs_european_put_price = -np.exp(-d*(T-t)) * St * norm1 + np.exp(-r*(T-t)) * K * norm2
        bs_european_put_delta = -np.exp(-d*(T-t)) * norm1
        bs_european_put_theta = -d * np.exp(-d*(T-t)) * St * norm1 - (vol * np.exp(-d*(T-t)) * St * norm.pdf(d1)) / (2 * (T-t)**(1/2)) + \
        r * np.exp(-r*(T-t)) * K * norm2
        bs_european_put_vega = (T-t)**(1/2) * np.exp(-r*(T-t)) * K * norm.pdf(d2)
        bs_european_put_gamma = (np.exp(-r*(T-t)) * K * norm.pdf(d2)) / (St**2 * vol * (T-t)**(1/2))
        bs_european_put_rho = -(T-t)*np.exp(-r*(T-t)) * K * norm2
        
        return bs_european_put_price, bs_european_put_delta, bs_european_put_theta, \
                bs_european_put_vega, bs_european_put_gamma, bs_european_put_rho































    
    
    
    
    
    
    
    