# Question3_1.py

import numpy as np
import matplotlib.pyplot as plt
from ComputationalFinanceClass import *

stock_price = np.arange(0, 100, 1)

option_example = ComputationalFinance(stock_price)

bull_call_spread = option_example.Bull_Call_Spread(30, 50)
bull_put_spread = option_example.Bull_Put_Spread(50, 30)
bear_call_spread = option_example.Bear_Call_Spread(55, 45)
collar = option_example.Collar(40, 60)
straddle = option_example.Straddle(55)
strangle = option_example.Strangle(25, 75)
butterfly_spread = option_example.Butterfly_Spread(15, 50, 85)

plt.figure(figsize=(12, 6))
plt.plot(stock_price, bull_call_spread, label='Bull Call Spread')
plt.plot(stock_price, bull_put_spread, label='Bull Put Spread')
plt.plot(stock_price, bear_call_spread, label='Bear Call Spread')
plt.plot(stock_price, collar, label='Collar')
plt.plot(stock_price, straddle, label='Straddle')
plt.plot(stock_price, strangle, label='Strangle')
plt.plot(stock_price, butterfly_spread, label='Butterfly Spread')

plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('Stock Price')
plt.ylabel('Payoff of Trading Strategy')
plt.title('European Option Trading Strategy')