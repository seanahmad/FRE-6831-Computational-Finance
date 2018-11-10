# Question5_3.py

import numpy as np
import matplotlib.pyplot as plt
from ComputationalFinanceClass import *

case = ComputationalFinance(0,0)

S0 = 10
n = 250
t = 0
T = 1
mu = 0
sigma = 0.2
d = 0.05
a = 0.0
b = 0.1
lam = 10

terminating_value = []

fig = plt.figure(figsize=(15, 7))
for i in range(5):
    GeoBroJump = case.Geometric_Brownian_Motion_Jump( mu, sigma, d, S0, n, t, T, a, b, lam )
    plt.plot(GeoBroJump)
    terminating_value.append(GeoBroJump[-1])

plt.grid(True)
plt.axis('tight')
plt.ylabel('Stock Price')
plt.xlabel('time')
plt.title('Monte Carlo Simulation for Stock Price by Jump Diffusion Process')


fig = plt.figure(figsize=(15, 7))
for i in range(1000):
    GeoBroJump = case.Geometric_Brownian_Motion_Jump( mu, sigma, d, S0, n, t, T, a, b, lam )
    terminating_value.append(GeoBroJump[-1])

fig = plt.figure(figsize=(15, 7))
plt.hist(terminating_value, bins=30)
plt.grid(True)
plt.axis('tight')
plt.title('Histogram with Lognormal Distribution Fit for Price at Terminal Time')







