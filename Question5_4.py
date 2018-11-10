# Question5_4.py

import numpy as np
import matplotlib.pyplot as plt
from ComputationalFinanceClass import *
import pandas as pd

asianoption = ComputationalFinance(0, 0) 

T = 2
t = 0
S0 = 10
mu = 0 
sigma = 0.2 
lam = 10
a =0
b = 0.1
d = 0.05
K = 10
n = 250
M = 5000

p1 = asianoption.Arithmetic_Average_Price_Asian_Call(T, t, S0, sigma, lam, a, b, mu, d, K, n, M)[0]

p2 = asianoption.Geometric_Average_Price_Asian_Call(T, t, S0, sigma, lam, a, b, mu, d, K, n, M)

p3 = asianoption.BS_Geometric_Average_Price_Asian_Call(T, t, mu, d, sigma, S0, K)

p4 = asianoption.Control_Variates_Arithmetic_Average_Asian_Call(T, t, S0, sigma, lam, a, b, mu, d, K, n, M)[0]

std1 = asianoption.Arithmetic_Average_Price_Asian_Call(T, t, S0, sigma, lam, a, b, mu, d, K, n, M)[1]

std2 = asianoption.Control_Variates_Arithmetic_Average_Asian_Call(T, t, S0, sigma, lam, a, b, mu, d, K, n, M)[1]


columns = ['Step Numbers of Each Trajectory', 'Trajectory Numbers', 'Price', 'Standard Deviation of Price', 'Confident Interval of Price', 'Confident Interval error']
index = ['Without Control Variates', 'With Control Variates']
df = pd.DataFrame(index=index, columns=columns)
df['Step Numbers of Each Trajectory'] = [n, n]
df['Trajectory Numbers'] = [M, M]
df['Price'] = [p1, p4]
df['Standard Deviation of Price'] = [std1, std2]
ci1_lower = p1 - 1.96 * std1 / np.sqrt(M)
ci1_upper = p1 + 1.96 * std1 / np.sqrt(M)
ci2_lower = p4 - 1.96 * std2 / np.sqrt(M)
ci2_upper = p4 + 1.96 * std2 / np.sqrt(M)
ci1 = [ci1_lower, ci1_upper]
ci2 = [ci2_lower, ci2_upper]
df['Confident Interval of Price'] = [ci1, ci2]
df['Confident Interval error'] = [p1-ci1_lower, p4-ci2_lower]
print(df)


Smean_ari = []
Smean_geo = []
for K in np.arange(5, 25, 0.01):
    S = asianoption.Geometric_Brownian_Motion_Jump(mu, sigma, d, S0, n, t, T, a, b, lam)
    p1 = np.maximum(np.mean(S)-K, 0)
    Smean_ari.append(p1)
    S_geo_mean = np.exp(np.mean(np.log(S)))
    p2 = np.maximum(S_geo_mean-K, 0)
    Smean_geo.append(p2)
    
fig = plt.figure(figsize=(15, 7))
plt.plot(Smean_ari, Smean_geo, 'o')
plt.grid(True)
plt.xlabel('Payoff of Arithmetic Average Asian Call')
plt.ylabel('Payoff of Geometric Average Asian Call')
plt.title('Correlation of Geometric and Arithmetic Average Asian Call')
plt.show()

