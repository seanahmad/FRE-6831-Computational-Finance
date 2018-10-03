# ComputationalFinanceClass.py

class ComputationalFinance( object ):
    def __init__( self, a, b ):
          self.stock_price = a
          self.strike_price = b
        
    def European_Call_Option_Payoff( self ): # compute the payoff
        boolean_values = self.stock_price > self.strike_price
        european_call_payoff = (boolean_values + 0.0) * (self.stock_price - self.strike_price)
        return european_call_payoff

    def European_Put_Option_Payoff( self ): # compute the payoff
        boolean_values = self.stock_price < self.strike_price
        european_put_payoff = (boolean_values + 0.0) * (self.strike_price - self.stock_price)
        return european_put_payoff