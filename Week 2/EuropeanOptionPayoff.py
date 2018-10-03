# EuropeanOptionPayoff.py
'''
There are two functions, which calculate payoffs for European Call Option 
and European Put Option
Input parameters: stock_price, strike_price
Out parameters: european_call_payoff, european_put_payoff
'''

import numpy as np

def European_Call_Option_Payoff( stock_price, strike_price ): # compute the payoff
    boolean_values = stock_price > strike_price
    european_call_payoff = (boolean_values + 0.0) * (stock_price - strike_price)
    return european_call_payoff
    
def European_Put_Option_Payoff( stock_price, strike_price ): # compute the payoff
    boolean_values = stock_price < strike_price
    european_put_payoff = (boolean_values + 0.0) * (strike_price - stock_price)
    return european_put_payoff


    
