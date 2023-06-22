# -*- coding: utf-8 -*-
"""
Gauss Quadrature Rules

Created on Fri Apr 12 15:06:19 2019
@author: Ehsan
"""

import numpy as np
from scipy.special import gamma
from scipy.special import jacobi
from scipy.special import roots_jacobi
#from scipy.special import legendre
import matplotlib.pyplot as plt

##################################################################
# Recursive generation of the Jacobi polynomial of order n
def Jacobi(n,a,b,x):
    '''
    This function returns the Jacobi polynomial of order n with parameters a and b
    at the points x.
    input:
        - n: order of the Jacobi polynomial
        - a: parameter a of the Jacobi polynomial, (a>-1)
        - b: parameter b of the Jacobi polynomial, (b>-1)
        - x: points at which the Jacobi polynomial is evaluated
    output:
        - Jacobi polynomial of order n with parameters a and b at the points x
    '''
    x=np.array(x)

    # fig, ax = plt.subplots()
    # ax.plot(x, jacobi(n,a,b)(x))
    # # ax.set_ylim(-2.0, 2.0)
    # ax.set_title("Jacobi polynomial of order n = {} with parameters a = {} and b = {}".format(n,a,b))
    # fig.set_size_inches(w=11,h=11)
    # plt.savefig('Jacobi.png')

    return (jacobi(n,a,b)(x))
    
##################################################################
# Derivative of the Jacobi polynomials
def DJacobi(n,a,b,x,k: int):
    '''
    This function returns the k-th derivative of the Jacobi polynomial of order n with parameters a and b
    at the points x.
    input:
        - n: order of the Jacobi polynomial
        - a: parameter a of the Jacobi polynomial, (a>-1)
        - b: parameter b of the Jacobi polynomial, (b>-1)
        - x: points at which the Jacobi polynomial is evaluated
        - k: order of the derivative
    output:
        - k-th derivative of the Jacobi polynomial of order n with parameters a and b at the points x
    '''
    x=np.array(x)
    ctemp = gamma(a+b+n+1+k)/(2**k)/gamma(a+b+n+1)
    return (ctemp*Jacobi(n-k,a+k,b+k,x))

    
##################################################################
# Weight coefficients
def GaussJacobiWeights(Q: int,a,b):
    '''
    This function returns the weights and nodes of Gauss-Jacobi quadrature rule
    of order Q for the Jacobi polynomial of order n with parameters a and b.
    input:
        - Q: order of the Gauss-Jacobi quadrature rule
        - a: parameter a of the Jacobi polynomial, (a>-1)
        - b: parameter b of the Jacobi polynomial, (b>-1)
    output:
        - X: nodes of the Gauss-Jacobi quadrature rule
        - W: weights of the Gauss-Jacobi quadrature rule
    '''
    [X , W] = roots_jacobi(Q,a,b)
    return [X, W]
	


##################################################################
# Weight coefficients
def GaussLobattoJacobiWeights(Q: int,a,b):
    '''
    This function returns the weights and nodes of Gauss-Lobatto-Jacobi quadrature rule
    of order Q for the Jacobi polynomial of order n with parameters a and b.
    input:  
        - Q: order of the Gauss-Lobatto-Jacobi quadrature rule
        - a: parameter a of the Jacobi polynomial, (a>-1)
        - b: parameter b of the Jacobi polynomial, (b>-1)
    output: 
        - X: nodes of the Gauss-Lobatto-Jacobi quadrature rule
        - W: weights of the Gauss-Lobatto-Jacobi quadrature rule
    '''
    W = []
    X = roots_jacobi(Q-2,a+1,b+1)[0]
    if a == 0 and b==0:
        W = 2/( (Q-1)*(Q)*(Jacobi(Q-1,0,0,X)**2) )
        Wl = 2/( (Q-1)*(Q)*(Jacobi(Q-1,0,0,-1)**2) )
        Wr = 2/( (Q-1)*(Q)*(Jacobi(Q-1,0,0,1)**2) )
    else:
        W = 2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/( (Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1,a,b,X)**2) )
        Wl = (b+1)*2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/( (Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1,a,b,-1)**2) )
        Wr = (a+1)*2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/( (Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1,a,b,1)**2) )
    W = np.append(W , Wr)
    W = np.append(Wl , W)
    X = np.append(X , 1)
    X = np.append(-1 , X)    
    return [X, W]
##################################################################


    
