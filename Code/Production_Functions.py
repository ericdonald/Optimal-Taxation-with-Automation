"""""""""""
Functions

Notes: Functions that describe the production block of the economy.
    
"""""""""""

import numpy as np
from numba import njit



@njit
def Lamba_l(x_j, x_bar, ζ, ν, σ):
    "Labor Task Integrals"
    
    Λ_l = (x_bar**(ζ*ν*(σ-1) + 1) - x_j**(ζ*ν*(σ-1) + 1)) / (ζ*ν*(σ-1) + 1)
    
    return Λ_l
    


@njit
def Lamba_k(x_j, x_bar, ζ, ν, σ, ε=0.0001):
    "Capital Task Integral"
    
    Λ_k = np.sum((x_j**(ζ*(ν-1)*(σ-1) + 1) - ε**(ζ*(ν-1)*(σ-1) + 1)) / (ζ*(ν-1)*(σ-1) + 1))
    
    return Λ_k
    
    
 
@njit
def relα(x_j, x_bar, ζ, ν, σ, cap):
    "Relative Task Productivity"
    
    if cap==0:
        Λ_l = Lamba_l(x_j, x_bar, ζ, ν, σ)
        return ( x_j**(ζ*ν*(σ-1)) ) / Λ_l
    
    if cap==1:
        Λ_k = Lamba_k(x_j, x_bar, ζ, ν, σ)
        return ( x_j**(ζ*(ν-1)*(σ-1)) ) / Λ_k
    
    
    
@njit
def zeta(Γ, χ):
    "Comparative Advantage"
    
    ζ = np.exp(Γ * χ)
    
    return ζ



@njit
def Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ):
    "Output"
        
    Λ_j = Lamba_l(x, x_bar, ζ, ν, σ)
    Λ_k = Lamba_k(x, x_bar, ζ, ν, σ)
    
    CES_inner = Λ_k**(1/σ) * (A_k * K)**((σ-1)/σ) + np.sum(Λ_j**(1/σ) * (A_j * L)**((σ-1)/σ))
    
    Y = CES_inner**(σ / (σ-1))
    
    return Y



@njit
def Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ):
    "Marginal Product of Labor"
    
    Λ_j = Lamba_l(x, x_bar, ζ, ν, σ)
    Y = Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    
    w = Λ_j**(1/σ) * A_j**((σ-1)/σ) * (Y / L)**(1/σ)
    
    return w
    


@njit
def Rents(x, L, K, A_j, A_k, x_bar, ζ, ν, σ):
    "Marginal Product of Capital"
    
    Λ_k = Lamba_k(x, x_bar, ζ, ν, σ)
    Y = Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    
    r = Λ_k**(1/σ) * A_k**((σ-1)/σ) * (Y / K)**(1/σ)
    
    return r



@njit
def V(c_0, c_1, l, β, var_θ, φ, ε):
    "Household Utility"
    
    if var_θ == 1:
        Val = np.log(c_0) + (β / (1-β)) * (np.log(c_1) - φ * l**(1 + 1/ε) / (1 + 1/ε))
    else:
        Val = (c_0**(1-var_θ) - 1) / (1-var_θ) + (β / (1-β)) * ((c_1**(1-var_θ) - 1) / (1-var_θ) - φ * l**(1 + 1/ε) / (1 + 1/ε))
        
    return Val

    
