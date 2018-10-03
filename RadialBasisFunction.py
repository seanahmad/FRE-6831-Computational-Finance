# RadialBasisFunction.py

import numpy as np

def Gaussian_RBF(x, y, eps):
    rbf = np.exp(- (eps**2) * (x - y)**2)
    return rbf

def Multiquadric_RBF(x, y, eps):
    rbf = np.sqrt(1 + (eps**2) * (x - y)**2)
    return rbf

def Inverse_Multiquadric_RBF(x, y, eps):
    rbf = 1 / np.sqrt(1 + (eps**2) * (x - y)**2)
    return rbf

def Inverse_Quadratic_RBF(x, y, eps):
    rbf = 1 / (1 + (eps**2) * (x - y)**2)
    return rbf