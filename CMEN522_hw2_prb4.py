import numpy as np
from scipy.integrate import quad,simpson


#constanats & parameters
pi=3.1416
NA=6.02e+23
K=1.38e-23
T=247.5

B=0.001

sigma=[8.443e-9, 9.416e-9, 8.9295e-9]

energy=[3.4155e-21, 2.6531e-21, 3.0102e-21]

def integral_function(r,a,b):
    return (1-np.exp(-4*b*((a/r)*12-(a/r)**6)/(K*T)))*r*2

for i in range(len(sigma)):
    a =sigma[i]
    b =energy[i]
    B=[a, b, quad(integral_function,0,np.infty, args=(a,b))[0]]
    print('the result is:', B)