# Commencement de HMM
"""
probabilité d'emission sert a savoir si c'est dans alpha0 ou alpha 1 ou 2 
"""
from math import comb # comb sert a calculer le coefficient binomiale 
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln

from math import comb

def binomial_probability(x, n, p):
    return (
        comb(n, x)
        * p**x
        * (1-p)**(n-x)
    )