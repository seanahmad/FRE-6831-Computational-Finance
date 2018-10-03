# Question1_3
# plot surface of RBF

from RadialBasisFunction import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# initialization

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
eps = [0.2, 1.0, 5.0]

x, y = np.meshgrid(x, y)

# plot 3D
# Gaussian RBF plot
fig = plt.figure(figsize=plt.figaspect(1/3))
for k in np.arange(3):  
    RBF_values = Gaussian_RBF(x, y, eps[k])
    ax = fig.add_subplot(1, 3, k+1, projection='3d')
    surf = ax.plot_surface(x, y, RBF_values, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.view_init(azim=30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$\exp(-\epsilon^2||x-y||_2^2$)')
    ax.set_title('Gaussian RBF, $\epsilon$ = '+str(eps[k]))
    
# Multiquadric RBF plot
fig = plt.figure(figsize=plt.figaspect(1/3))
for k in np.arange(3):  
    RBF_values = Multiquadric_RBF(x, y, eps[k])
    ax = fig.add_subplot(1, 3, k+1, projection='3d')
    surf = ax.plot_surface(x, y, RBF_values, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.view_init(azim=30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$\sqrt{1+\epsilon^2||x-y||_2^2}$')
    ax.set_title('Multiquadric RBF, $\epsilon$ = '+str(eps[k]))
    
# Inverse Multiquadric RBF plot
fig = plt.figure(figsize=plt.figaspect(1/3))
for k in np.arange(3):  
    RBF_values = Inverse_Multiquadric_RBF(x, y, eps[k])
    ax = fig.add_subplot(1, 3, k+1, projection='3d')
    surf = ax.plot_surface(x, y, RBF_values, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.view_init(azim=30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$1/(\sqrt{1+\epsilon^2||x-y||_2^2})$')
    ax.set_title('Inverse Multiquadric RBF, $\epsilon$ = '+str(eps[k]))
    
# Inverse Quadratic RBF plot
fig = plt.figure(figsize=plt.figaspect(1/3))
for k in np.arange(3):  
    RBF_values = Inverse_Quadratic_RBF(x, y, eps[k])
    ax = fig.add_subplot(1, 3, k+1, projection='3d')
    surf = ax.plot_surface(x, y, RBF_values, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.view_init(azim=30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$1/(1+\epsilon^2||x-y||_2^2)$')
    ax.set_title('Inverse Quadratic RBF, $\epsilon$ = '+str(eps[k]))