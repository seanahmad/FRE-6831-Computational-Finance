# Question5_2.py

import numpy as np
import matplotlib.pyplot as plt
from ComputationalFinanceClass import *

case = ComputationalFinance(0,0)

S0 = 10
n = 250
t = 0
T = 1
mu = 0
sigma = 0.1

terminating_value = []

fig = plt.figure(figsize=(15, 7))
for i in range(1000):
    GeoBro = case.Geometric_Brownian_Motion_Trajectory( mu, sigma, S0, n, t, T )
    plt.plot(GeoBro)
    terminating_value.append(GeoBro[-1])

plt.grid(True)
plt.axis('tight')
plt.title('Sample Trajectory of Geometric Brownian Motion')

fig = plt.figure(figsize=(15, 7))
plt.hist(terminating_value, bins=30)
plt.grid(True)
plt.axis('tight')
plt.title('Histogram of Price at terminating time')


