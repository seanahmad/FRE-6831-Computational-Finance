# ComputationalFinanceClass.py

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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



    # Explicit Finite Difference Method
    ''' 
    parameter description:
    Smin: minimum value of stock price
    Smax: maximum value of stock price
    K: strike price
    t: time
    T: maturity date of the option
    r: interest rate
    d: the dividend yield of the underlying asset
    vol: the volatility of returns of the underlying asset
    N: number of sub-intervals in s-direction
    M: number of sub-intervals in tao-direction
    initial_condition: ic_call, ic_put
    boundary_condition: dirichlet_bc, neumann_bc
    '''
    def Black_Scholes_Explicit_FD_EO( self, K, r, d, vol, Smin, Smax, t, T, N, M, initial_condition, boundary_condition ):
        
        delta_s = (Smax-Smin)/N
        delta_tao = (T-t)/M
        
        if delta_tao / (delta_s)**2 < 0 or delta_tao / (delta_s)**2 >= 1/2:
            print( 'stability condition does not hold.' )
            quit()
            
        else:
            St = np.linspace(Smin, Smax, N+1)
            tao = np.linspace(t, T, M+1)
            v = np.zeros((N+1, M+1))  # option value array
            
            # Calculating the weighting matix
            lE = (St**2 * vol**2 * delta_tao) / (2 * (delta_s)**2) - ((r-d) * St * delta_tao) / (2 * delta_s)
            dE = 1 - r*delta_tao - vol**2 * St**2 * delta_tao / (delta_s)**2
            uE = vol**2 * St**2 * delta_tao / (2 * (delta_s)**2) + (r-d) * St * delta_tao / (2 * delta_s) 
            
            wm = np.zeros((N+1, N+1))
            for i in range(1, N):
                wm[i, i-1] = lE[i]
                wm[i, i] = dE[i]
                wm[i, i+1] = uE[i]
            
            if boundary_condition == 'dirichlet_bc':
                if initial_condition == 'ic_call':
                    v[:, 0] = ((St > K) + 0.0) * (St - K)
                    for k in range(1, M+1):
                        v[1:N, k] = wm[1:N, :] @ v[:, k-1]
                    v[0, :] = 0
                    v[N, :] = np.exp(-d*tao)*Smax - np.exp(-r*tao)*K
                elif initial_condition == 'ic_put':
                    v[:, 0] = ((St < K) + 0.0) * (K - St)
                    for k in range(1, M+1):
                        v[1:N, k] = wm[1:N, :] @ v[:, k-1]
                    v[0, :] = np.exp(-r*tao)*K - np.exp(-d*tao)*Smin
                    v[N, :] = 0
                else:
                    print( 'initial condition is not appropriate!' )
                    
            elif boundary_condition == 'neumann_bc':
                wm[0, 0] = 2*lE[0] + dE[0]
                wm[0, 1] = uE[0] - lE[0]
                wm[N, N-1] = lE[N] - uE[N]
                wm[N, N] = dE[N] + 2*uE[N]
                
                if initial_condition == 'ic_call':
                    v[:, 0] = ((St > K) + 0.0) * (St - K)
                    for i in range(1, M+1):
                        v[:, i] = wm @ v[:, i-1]
                        
                elif initial_condition == 'ic_put':
                    v[:, 0] = ((St < K) + 0.0) * (K - St)
                    for i in range(1, M+1):
                        v[:, i] = wm @ v[:, i-1]
                else:
                    print( 'initial condition is not appropriate!' )
                    
            else:
                print( 'boundary condition is not appropriate!' )
                
        bs_explicit_fd_eo_price = v
        return bs_explicit_fd_eo_price
    
    
    # Implicit Finite Difference Method
    def Black_Scholes_Implicit_FD_EO( self, K, r, d, vol, Smin, Smax, t, T, N, M, initial_condition, boundary_condition ):
        
        delta_s = (Smax-Smin)/N
        delta_tao = (T-t)/M
        
        if delta_tao / (delta_s)**2 < 0 or delta_tao / (delta_s)**2 >= 1/2:
            print( 'stability condition does not hold.' )
            quit()
            
        else:
            St = np.linspace(Smin, Smax, N+1)
            tao = np.linspace(t, T, M+1)
            v = np.zeros((N+1, M+1))  # option value array
            
            # Calculating the weighting matix
            lI = -(St**2 * vol**2 * delta_tao) / (2 * (delta_s)**2) + ((r-d) * St * delta_tao) / (2 * delta_s)
            dI = 1 + r*delta_tao + vol**2 * St**2 * delta_tao / (delta_s)**2
            uI = -vol**2 * St**2 * delta_tao / (2 * (delta_s)**2) - (r-d) * St * delta_tao / (2 * delta_s) 
            
            wm = np.zeros((N+1, N+1))
            for i in range(1, N):
                wm[i, i-1] = lI[i]
                wm[i, i] = dI[i]
                wm[i, i+1] = uI[i]
            
            wm[0, 0] = 2*lI[0] + dI[0]
            wm[0, 1] = uI[0] - lI[0]
            wm[N, N-1] = lI[N] - uI[N]
            wm[N, N] = dI[N] + 2*uI[N]
            
            # calculate the price
            if initial_condition == 'ic_call':
                v[:, 0] = ((St > K) + 0.0) * (St - K)
            elif initial_condition == 'ic_put':
                v[:, 0] = ((St < K) + 0.0) * (K - St)
            else:
                print( 'initial condition is not appropriate!' )
            
            if boundary_condition == 'dirichlet_bc':
                for k in range(1, M+1):
                    v[:, k] = np.linalg.inv(wm) @ (v[:, k-1])
                    v[0, k] = v[0, k-1]
                    v[N, k] = v[N, k-1]
            elif boundary_condition == 'neumann_bc':
                for k in range(1, M+1):
                    v[:, k] = np.linalg.inv(wm) @ v[:, k-1]      
            else:
                print( 'boundary condition is not appropriate!' )

        bs_implicit_fd_eo_price = np.transpose(v)
        return bs_implicit_fd_eo_price
    

# Gaussian Radial Basis Function
    def Gaussian_RBF(self, epsilon, x, center_node):
        r = (x - center_node) ** 2
        phi_ga_rbf = np.exp(-epsilon ** 2 * r)
        phi_x_ga_rbf = -2 * (epsilon ** 2) * (x - center_node) * phi_ga_rbf
        phi_xx_ga_rbf = 4 * (epsilon ** 4) * (x - center_node) ** 2 * phi_ga_rbf - 2 * epsilon ** 2 * phi_ga_rbf
        return phi_ga_rbf, phi_x_ga_rbf, phi_xx_ga_rbf

    def Multiquadric_RBF(self, epsilon, x, center_node):
        r = (x - center_node) ** 2
        phi_ga_rbf = np.sqrt(1 + epsilon ** 2 * r)
        phi_x_ga_rbf = (epsilon ** 2 * (x - center_node)) / phi_ga_rbf
        phi_xx_ga_rbf = (epsilon ** 2) / phi_ga_rbf - (epsilon ** 4 * r) / (phi_ga_rbf ** 3)
        return phi_ga_rbf, phi_x_ga_rbf, phi_xx_ga_rbf

    def Inverse_Multiquadric_RBF(self, epsilon, x, center_node):
        r = (x - center_node) ** 2
        phi_ga_rbf = 1 / np.sqrt(1 + epsilon ** 2 * r)
        phi_x_ga_rbf = - (epsilon ** 2 * (x - center_node)) * (phi_ga_rbf ** 3)
        phi_xx_ga_rbf = - (epsilon ** 2 * (-2 * epsilon ** 2 * r + 1)) * (phi_ga_rbf ** 5)
        return phi_ga_rbf, phi_x_ga_rbf, phi_xx_ga_rbf

    def Inverse_Quadric_RBF(self, epsilon, x, center_node):
        r = (x - center_node) ** 2
        phi_ga_rbf = 1 / (1 + epsilon ** 2 * r)
        phi_x_ga_rbf = -(2 * epsilon ** 2 * (x - center_node)) * (phi_ga_rbf ** 2)
        phi_xx_ga_rbf = -(2 * epsilon ** 2 * (-3 * epsilon ** 2 * r + 1)) * (phi_ga_rbf ** 3)
        return phi_ga_rbf, phi_x_ga_rbf, phi_xx_ga_rbf


# calculate PDE of BS model by Global RBF
    ''' 
    parameter description:
    rbf_function: ga_rbf, mq_rbf, imq_rbf, iq_rbf
    initial_condition: ic_call, ic_put
    boundary_condition: dirichlet_bc
    assume d = 0
    '''    

    def Black_Scholes_Global_RBF_EO(self, Smax, Smin, K, r, sigma, t, T, M, N, epsilon, rbf_function, initial_condition, boundary_condition):
        xi = np.linspace(np.log(Smin), np.log(Smax), N)
        t = np.linspace(0, T, M+1)
        dt = 1 / M
        bs_global_rbf_eo_price = np.zeros((N, M + 1))

        # calculate L, Lx, Lxx
        x = xi.reshape((N, 1))
        if rbf_function == "ga_rbf":
            L, Lx, Lxx = self.Gaussian_RBF(epsilon, x, xi)
        elif rbf_function == "mq_rbf":
            L, Lx, Lxx = self.Multiquadric_RBF(epsilon, x, xi)
        elif rbf_function == "imq_rbf":
            L, Lx, Lxx = self.Inverse_Multiquadric_RBF(epsilon, x, xi)
        elif rbf_function == "iq_rbf":
            L, Lx, Lxx = self.Inverse_Quadric_RBF(epsilon, x, xi)
        else:
            print("RBF condition is not appropriate!")
            quit()

        P = r * np.identity(N) - (r - 0.5 * sigma ** 2) * np.linalg.inv(L).dot(Lx) - 0.5 * sigma ** 2 * np.linalg.inv(
            L).dot(Lxx)

        if initial_condition == "ic_call":
            if boundary_condition == "dirichlet_bc":
                bs_global_rbf_eo_price[:, 0] = np.maximum(np.exp(xi) - K, 0)
                lamda = np.linalg.inv(L).dot(bs_global_rbf_eo_price[:, 0])
                for i in range(M):
                    lamda = (np.linalg.inv(np.identity(N) + 0.5 * dt * P).dot(np.identity(N) - 0.5 * dt * P)).dot(lamda)
                    bs_global_rbf_eo_price[:, i + 1] = L.dot(lamda)
                    bs_global_rbf_eo_price[0, i + 1] = 0
                    bs_global_rbf_eo_price[-1, i + 1] = Smax - np.exp(-r * t[i + 1]) * K
                    lamda = np.linalg.inv(L).dot(bs_global_rbf_eo_price[:, i + 1])
            else:
                print("The boundary condition is not appropriate!")
                quit()

        if initial_condition == "ic_put":
            if boundary_condition == "dirichlet_bc":
                bs_global_rbf_eo_price[:, 0] = np.maximum(K - np.exp(xi), 0)
                lamda = np.linalg.inv(L).dot(bs_global_rbf_eo_price[:, 0])
                for i in range(M):
                    lamda = (np.linalg.inv(np.identity(N) + 0.5 * dt * P).dot(np.identity(N) - 0.5 * dt * P)).dot(lamda)
                    bs_global_rbf_eo_price[:, i + 1] = L.dot(lamda)
                    bs_global_rbf_eo_price[0, i + 1] = np.exp(-r * t[i + 1]) * K - Smin
                    bs_global_rbf_eo_price[-1, i + 1] = 0
                    lamda = np.linalg.inv(L).dot(bs_global_rbf_eo_price[:, i + 1])
            else:
                print("The boundary condition is not appropriate!")
                quit()

        return bs_global_rbf_eo_price


    # calculat PDE by RBF
    def Black_Scholes_RBF_FD_EO(self, Smax, Smin, K, r, d, sigma, t, T, M, N, epsilon, rbf_function, initial_condition, boundary_condition):
        
        delta_tao = (T - t) / M
        delta_S = (Smax - Smin) / N

        St = np.log(np.linspace(Smin, Smax, N+1))
        tao = np.linspace(t, T, M+1)
        v = np.zeros((N+1, M+1))  # option value array
  
        # initial condition  
        if initial_condition == 'ic_call':
            v[:, 0] = np.maximum(np.exp(St) - K, 0)
        elif initial_condition == 'ic_put':
            v[:, 0] = np.maximum(K - np.exp(St), 0)
        else:
            print('initial_condition is not appropriate!')
            quit()
        
        # calculate L, Lx, Lxx
        x = St.reshape((N+1, 1))
        if rbf_function == "ga_rbf":
            L, Lx, Lxx = self.Gaussian_RBF(epsilon, x, St)
        elif rbf_function == "mq_rbf":
            L, Lx, Lxx = self.Multiquadric_RBF(epsilon, x, St)
        elif rbf_function == "imq_rbf":
            L, Lx, Lxx = self.Inverse_Multiquadric_RBF(epsilon, x, St)
        elif rbf_function == "iq_rbf":
            L, Lx, Lxx = self.Inverse_Quadric_RBF(epsilon, x, St)
        else:
            print('RBF condition is not appropriate!')
            quit()
        
        I = np.diag(np.ones(N-1))
        L_prime = L[1:-1,1:-1]
        L_inverse = np.linalg.solve(L_prime, I)
        L = (r - 0.5 * sigma ** 2) * Lx[1:-1,1:-1] + 0.5 * sigma ** 2 * Lxx[1:-1,1:-1] - r * L[1:-1,1:-1]
        W = (L_inverse @ L).T

        # calculate value at different ttm
        tempo1 = I - 0.5 * delta_tao * W
        tempo2 = I + 0.5 * delta_tao * W

        for i in range(1, M+1):
            value_tempo = v[:, i-1]
            v[1:-1, i] = np.linalg.solve(tempo1, I) @ tempo2 @ v[1:-1, i-1]
            if initial_condition == 'ic_call':
                v[0, i] = v[0, i-1]
                v[N, i] = Smax - K * np.exp(-r * (tao[i]-tao[0]))
            elif initial_condition == 'ic_put':
                v[N, i] = v[N, i-1]
                v[0, i] = K * np.exp( -r * (tao[i]-tao[0])) - Smin

        bs_rbf_fd_eo_price = v
        
        return bs_rbf_fd_eo_price














    
    
    
    
    
    
    
    