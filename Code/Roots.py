"""""""""""
Roots

Last Modified: Eric Donald 5/25

Notes:
    
Output:
"""""""""""

import numpy as np
import scipy as sp
from numba import njit
import Functions as fn
import Perturbations as pr



def WealthShapeRoot(Θ, WS, YS):
    "Wealth Convexity Root"
    
    WS_hat = ( YS**Θ ) / ( sum(YS**Θ) )
    
    error = WS - WS_hat
    
    return sum(error**2)



def ConCalRoot(CSQ, J, Y, K, G, n, w_j, l_j, y_j0, r, δ, g, τ_k, Ψ, ψ, var_θ):
    "Consumption, Tax Scale, and Investment Price Root"
    
    'Unpack Variables'
    c_j0 = CSQ[:J]
    β = CSQ[-1]
    
    R = (1-τ_k) * (r - δ) - g
    D_1 = G * Y
    
    'Find c_j1 with Euler'
    c_j1 = c_j0 * (R * β / (1-β))**(1 / var_θ)
    
    'Consumption Root'
    κ_j1 = (c_j1 - Ψ * (w_j * l_j)**(1-ψ) - D_1) / R
    
    RHS_c0 = y_j0 - κ_j1
    Root_c0 = c_j0 - RHS_c0
    
    'Investment Root'
    RHS_β = np.sum(n * (y_j0 - c_j0))
    Root_β = np.array([K - RHS_β])
    
    return np.concatenate((Root_c0, Root_β))



def GammaRoot(Γ, var_κ, Σ_k, χ, x_bar, σ, S_k, S_j, J):
    "Automation Exposure Root"
    
    ζ = fn.zeta(Γ, χ)
    
    nu_g = np.ones(J) * var_κ
    
    nu = sp.optimize.root(νRoot, nu_g,
                  args=(Γ, χ, var_κ, x_bar, σ, S_j, S_k),
                  method='lm')
    
    ν = nu.x
    x_j = var_κ * x_bar
    
    z_j = fn.relα(x_j, x_bar, ζ, ν, σ, 0) * x_j
    
    z_jk = fn.relα(x_j, x_bar, ζ, ν, σ, 1) * x_j
    
    Σ_j = σ + (z_j + z_jk) / ζ
    
    ES_weight = S_j * (Σ_k - σ) * ζ / z_jk
    unit = np.sum(ES_weight)
    
    RHS = np.sum(ES_weight * Σ_j) / unit
        
    return np.log(Σ_k) - np.log(RHS)



@njit
def νRoot(ν, Γ, χ, var_κ, x_bar, σ, S_j, S_k):
    "Absolute Advantage Root"
    
    ζ = fn.zeta(Γ, χ)
    
    x_j = var_κ * x_bar
    
    Λ_l = fn.Lamba_l(x_j, x_bar, ζ, ν, σ)
    Λ_k = fn.Lamba_k(x_j, x_bar, ζ, ν, σ)
    
    RHS = (S_j / S_k) * Λ_k * x_j**(ζ * (σ-1))
    
    return np.log(Λ_l) - np.log(RHS)



@njit
def Eqbm_Root(E, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, Ψ, ψ, β, var_θ, ε, τ_k, δ, g, φ):
    "Equilibrium Root"
    
    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    x = E[3*J:]
    
    L = n * l
    K = np.sum(n * (y_0 - c_0))
    
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    r = fn.Rents(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    R = (1-τ_k)*(r-δ) - g
    
    D = np.sum(n * (w * l - Ψ * (w * l)**(1-ψ))) + τ_k * (r - δ) * K
    
    
    'Log Labor Supply'
    ln_LS = np.log(φ) + (ψ + 1/ε) * np.log(l) + var_θ * np.log(c_1) - (np.log(Ψ) + np.log(1-ψ) + (1-ψ) * np.log(w))
    
    'Log Euler Equation'
    ln_EE = var_θ * np.log(c_1) - var_θ * np.log(c_0) - (np.log(R) + np.log(β/(1-β)))
    
    'Budget Constraint'
    BC = R * c_0 + c_1 - (R * y_0 + Ψ * (w*l)**(1-ψ) + D)
    
    'Log Automation Threshold'
    ln_AT = ζ * np.log(x) - ((np.log(w) - np.log(A_j)) - (np.log(1+θ) + np.log(r) - np.log(A_k)))


    return np.concatenate((ln_LS, ln_EE, BC, ln_AT))



def Optimalθ_Root(θ, E_sq, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, Ψ, ψ, β, var_θ, ε, τ_k, δ, g, φ):
    "Root for Optimal Threshold Rule with Status Quo Taxes"
    
    'Solve for Equilibrium Allocation'
    Eqbm = sp.optimize.root(Eqbm_Root, E_sq,
                  args=(θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, Ψ, ψ, β, var_θ, ε, τ_k, δ, g, φ),
                  jac=pr.δH_δclx)
    
    E = Eqbm.x
    
    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    x = E[3*J:]
    
    'Compute Threshold Rule Perturbations'
    ΔH_ΔΕ = pr.δH_δclx(E, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, Ψ, ψ, β, var_θ, ε, τ_k, δ, g, φ)
    ΔH_Δθ = pr.δH_δθ(θ, J)

    dE = - np.linalg.inv(ΔH_ΔΕ) @ ΔH_Δθ
    
    dc_0 = dE[:J,0]
    dl = dE[2*J:3*J,0]
    dx = dE[3*J:,0]
    
    L = n * l
    κ = y_0 - c_0
    K = np.sum(n * κ)
    
    δlnl = dl / l
    δlnκ = - dc_0 / κ
    
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    r = fn.Rents(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    R = (1-τ_k)*(r-δ) - g
    Y = fn.Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    
    δlnw = pr.dlnw(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0)
    δlnr = pr.dlnr(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0)
    δlnR = (1-τ_k) * r * δlnr / R
    
    α_k = x**(ζ*(ν-1))
    
    if σ == 1:
        θ_wedge = np.log(1+θ)
    else:
        θ_wedge = ((1+θ)**(1-σ) - 1) / (1 - σ)
    
    δY_δX = Y * (A_k * α_k / r)**(σ-1) * θ_wedge
    
    λ = c_1**(-var_θ) / np.sum(n * c_1**(-var_θ))
    
    'Optimality Condition'
    MC = -np.sum(δY_δX * dx)
    cov = np.sum(n * (λ-1) * (Ψ * (1-ψ) * (w*l)**(1-ψ) * δlnw + (1-τ_k) * R * κ * δlnR))
    Expect = np.sum(n * ((w*l - Ψ * (1-ψ) * (w*l)**(1-ψ)) * δlnl + τ_k * R * κ * δlnκ))
    
    δW = MC - cov - Expect
    
    return δW



@njit
def CERoot(CE, c_0prime, c_1prime, lprime, c_0, c_1, l, n, β, var_θ, φ, ε):
    "Consumption Equivalence Root"
    
    Val_CE = fn.V(CE * c_0, CE * c_1, l, β, var_θ, φ, ε)
    Val_prime = fn.V(c_0prime, c_1prime, lprime, β, var_θ, φ, ε)
    
    W_CE = np.sum(n * Val_CE)
    W_prime = np.sum(n * Val_prime)
    
    return W_prime - W_CE



@njit
def Mir_obj(X, J, n, Y_bar, δ, g, A_j, A_k, x_bar, ζ, ν, σ, β, var_θ, φ, ε, IC_Comp_J, IC_count, TR):
    "Mirrlees Objective"
    
    c_0 = X[:J]
    c_1 = X[J:2*J]
    l = X[2*J:3*J]
    
    Val = fn.V(c_0, c_1, l, β, var_θ, φ, ε)
    
    W = np.sum(n * Val)
    
    return -W
    
    
  
@njit
def Equal_Constr(X, J, n, Y_bar, δ, g, A_j, A_k, x_bar, ζ, ν, σ, β, var_θ, φ, ε, IC_Comp_J, IC_count, TR):
    "Equality Constraints"
    
    c_0 = X[:J]
    c_1 = X[J:2*J]
    l = X[2*J:3*J]
    x = X[3*J:4*J]
    K = X[4*J]
    
    if TR==1:
        θ = X[-1]
    else:
        θ = 0
    
    δ_hat = 1 + δ + g
    
    L = n * l
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    r = fn.Rents(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    Y = fn.Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    
    'Initial Period Resource Constraint'
    RC_0 = np.sum(n * c_0) + K - Y_bar
    
    'Second Period Resource Constraint'
    RC_1 = np.sum(n * c_1) - Y - (1-δ_hat) * K
    
    'Log Automation Threshold'
    ln_AT = ζ * np.log(x) - ((np.log(w) - np.log(A_j)) - (np.log(1+θ) + np.log(r) - np.log(A_k)))
    
    return np.concatenate((np.array([RC_0, RC_1]), ln_AT)) 



@njit
def Inequal_Constr(X, J, n, Y_bar, δ, g, A_j, A_k, x_bar, ζ, ν, σ, β, var_θ, φ, ε, IC_Comp_J, IC_count, TR):
    "Local Incentive Compatibility Constraints"
    
    c_0 = X[:J]
    c_1 = X[J:2*J]
    l = X[2*J:3*J]
    x = X[3*J:4*J]
    K = X[4*J]
    
    L = n * l
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    Val = fn.V(c_0, c_1, l, β, var_θ, φ, ε)
    
    Val_con = Val + (β / (1-β)) * φ * l**(1 + 1/ε) / (1 + 1/ε)
    
    IC = np.zeros(IC_count)
    k=0
    
    for i in range(J):
        for j in IC_Comp_J[i]:
            IC[k] = Val[i] - (Val_con[j] - (β / (1-β)) * φ[i] * (w[j] * l[j] / w[i])**(1 + 1/ε) / (1 + 1/ε))
            k += 1
        
    return IC 



@njit
def IC_Full(X, J, n, A_j, A_k, x_bar, ζ, ν, σ, β, var_θ, φ, ε):
    "All Incentive Compatibility Constraints"
    
    c_0 = X[:J]
    c_1 = X[J:2*J]
    l = X[2*J:3*J]
    x = X[3*J:4*J]
    K = X[4*J]
    
    L = n * l
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    Val = fn.V(c_0, c_1, l, β, var_θ, φ, ε)
    
    Val_con = Val + (β / (1-β)) * φ * l**(1 + 1/ε) / (1 + 1/ε)
    
    IC = np.zeros((J,J))
    
    for i in range(J):
        for j in range(J):
           IC[i,j] = Val[i] - (Val_con[j] - (β / (1-β)) * φ[i] * (w[j] * l[j] / w[i])**(1 + 1/ε) / (1 + 1/ε))
    
    return IC 
















