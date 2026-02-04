"""""""""""
Roots

Notes: Functions that define the various roots of the economy.

"""""""""""

import numpy as np
import scipy as sp
from numba import njit
import Production_Functions as fn
import Perturbations as pr



def WealthShapeRoot(Θ, WS, YS):
    "Wealth Convexity Root"
    
    WS_hat = ( YS**Θ ) / ( sum(YS**Θ) )
    
    error = WS - WS_hat
    
    return sum(error**2)



def ConCalRoot(CSQ, J, Y, K, G, n, w_j, l_j, y_j0, r, δ, g, τ_k, Ψ, ψ, var_θ):
    "Consumption and Discount Factor Root"
    
    # ---------------- #
    # Unpack Variables #
    # ---------------- #
    c_j0 = CSQ[:J]
    β = CSQ[-1]
    
    R_tilde = (1-τ_k) * (r - δ) - g
    beta_tilde = β / (1 - β * (1+g)**(1-var_θ))
    D_1 = G * Y
    
    
    # -------------------- #
    # Find c_j1 with Euler #
    # -------------------- #
    c_j1 = c_j0 * (R_tilde * beta_tilde)**(1 / var_θ)
    
    
    # ---------------- #
    # Consumption Root #
    # ---------------- #
    κ_j1 = (c_j1 - Ψ * (w_j * l_j)**(1-ψ) - D_1) / R_tilde
    
    RHS_c0 = y_j0 - κ_j1
    Root_c0 = c_j0 - RHS_c0
    
    
    # -------------------- #
    # Discount Factor Root #
    # -------------------- #
    RHS_β = np.sum(n * (y_j0 - c_j0))
    Root_β = np.array([K - RHS_β])
    
    return np.concatenate((Root_c0, Root_β))



def GammaRoot(Γ, E_sq, w_j_sq, r_sq, COR, var_κ, Σ_k, χ, x_bar, σ, S_k, S_j, J, n, y_0, Ψ, ψ, β, var_θ, ε, τ_k, δ, g, φ):
    "Automation Exposure Root"
    
    ζ = fn.zeta(Γ, χ)
    
    
    # ------------------------------------------- #
    # Compute Productivity and Absolute Advantage #
    # ------------------------------------------- #
    nu_g = np.ones(J) * var_κ
    
    nu = sp.optimize.root(νRoot, nu_g,
                  args=(Γ, χ, var_κ, x_bar, σ, S_j, S_k),
                  method='lm')
    
    ν = nu.x
    x_j = var_κ * x_bar
    
    Λ_k = fn.Lamba_k(x_j, x_bar, ζ, ν, σ)
    A_k = r_sq**(σ / (σ-1)) * (COR / Λ_k)**(1 / (σ-1))
    
    A_j = (w_j_sq / x_j**(ζ)) / (r_sq / A_k)
    
    
    # ---------------------------------- #
    # Compute Elasticity of Substitution #
    # ---------------------------------- #
    c_0 = E_sq[:J]
    l = E_sq[2*J:3*J]
    
    ΔH_ΔΕ = pr.δH_δclx(E_sq, 0, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, Ψ, ψ, var_θ, ε, τ_k, δ, g, φ)
    ΔH_ΔA_k = pr.δH_δA_k(E_sq, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, Ψ, ψ, var_θ, ε, τ_k, δ, g, φ)

    dE = - np.linalg.inv(ΔH_ΔΕ) @ ΔH_ΔA_k
    
    dc_0 = dE[:J,0]
    dl = dE[2*J:3*J,0]
    dx = dE[3*J:,0]
    
    κ = y_0 - c_0
    K = np.sum(n * κ)
    dK = - np.sum(n * dc_0)
    
    δlnK = dK / K
    
    #Add direct effects
    δlnY = pr.dlnY(dc_0, dl, dx, c_0, l, x_j, 0, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0) + S_k
    δlnr = pr.dlnr(dc_0, dl, dx, c_0, l, x_j, 0, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0) + (σ-1)/σ + S_k / σ
    
    δlnS_k = δlnr + δlnK / δlnY
    
    RHS = δlnS_k / δlnr
        
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
    R_tilde = (1-τ_k)*(r-δ) - g
    
    D = np.sum(n * (w * l - Ψ * (w * l)**(1-ψ))) + τ_k * (r - δ) * K
    
    MRS_l = fn.lab_MRS(c_1, l, β, var_θ, φ, ε, g)
    MRS_c = fn.cap_MRS(c_0, c_1, β, var_θ, ε, g)
    keep_l = fn.Heath_keep(w, l, Ψ, ψ)
    
    
    # ---------------- #
    # Log Labor Supply #
    # ---------------- #
    ln_LS = np.log(MRS_l) - np.log(keep_l) - np.log(w)
    
    
    # ------------------ #
    # Log Euler Equation #
    # ------------------ #
    ln_EE = np.log(MRS_c) - np.log(R_tilde)
    
    
    # ----------------- #
    # Budget Constraint #
    # ----------------- #
    BC = R_tilde * c_0 + c_1 - (R_tilde * y_0 + Ψ * (w*l)**(1-ψ) + D)
    
    
    # ------------------------ #
    # Log Automation Threshold #
    # ------------------------ #
    ln_AT = ζ * np.log(x) - ((np.log(w) - np.log(A_j)) - (np.log(1+θ) + np.log(r) - np.log(A_k)))


    return np.concatenate((ln_LS, ln_EE, BC, ln_AT))



def Optimalθ_SQ_Root(θ, E_sq, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, Ψ, ψ, β, var_θ, ε, τ_k, δ, g, φ):
    "Root for Optimal Threshold Rule with Status Quo Taxes"
    
    # -------------------------------- #
    # Solve for Equilibrium Allocation #
    # -------------------------------- #
    Eqbm = sp.optimize.root(Eqbm_Root, E_sq,
                  args=(θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, Ψ, ψ, var_θ, ε, τ_k, δ, g, φ),
                  jac=pr.δH_δclx)
    
    E = Eqbm.x
    
    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    x = E[3*J:]
    
    
    # ------------------------------------ #
    # Compute Threshold Rule Perturbations #
    # ------------------------------------ #
    ΔH_ΔΕ = pr.δH_δclx(E, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, Ψ, ψ, var_θ, ε, τ_k, δ, g, φ)
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
    Y = fn.Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    
    δ_tilde = 1 + δ + g
    R = 1 + r - δ_tilde
    
    δlnw = pr.dlnw(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0)
    δlnr = pr.dlnr(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0)
    δlnR = r * δlnr / R
    
    α_k = x**(ζ*(ν-1))
    
    if σ == 1:
        θ_wedge = np.log(1+θ)
    else:
        θ_wedge = ((1+θ)**(1-σ) - 1) / (1 - σ)
    
    δY_δX = Y * (A_k * α_k / r)**(σ-1) * θ_wedge
    
    MRS_l = fn.lab_MRS(c_1, l, β, var_θ, φ, ε, g)
    MRS_c = fn.cap_MRS(c_0, c_1, β, var_θ, ε, g)
    λ = c_1**(-var_θ) / np.sum(n * c_1**(-var_θ))
    
    
    # -------------------- #
    # Optimality Condition #
    # -------------------- #
    MC = -np.sum(δY_δX * dx)
    cov = np.sum(n * (λ-1) * (MRS_l * l * δlnw + MRS_c * κ * δlnR))
    Expect = np.sum(n * ((w - MRS_l) * l * δlnl + (R - MRS_c) * κ * δlnκ))
    
    δW = MC - cov - Expect
    
    return δW



@njit
def CERoot(CE, c_0prime, c_1prime, lprime, c_0, c_1, l, n, β, var_θ, φ, ε, g):
    "Consumption Equivalence Root"
    
    Val_CE = fn.V(CE * c_0, CE * c_1, l, β, var_θ, φ, ε, g)
    Val_prime = fn.V(c_0prime, c_1prime, lprime, β, var_θ, φ, ε, g)
    
    W_CE = np.sum(n * Val_CE)
    W_prime = np.sum(n * Val_prime)
    
    return W_prime - W_CE



@njit
def Mir_obj(E, n, Y_bar, δ, g, A_j, A_k, β, var_θ, φ, ε, J):
    "Mirrlees Objective"
    
    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    
    Val = fn.V(c_0, c_1, l, β, var_θ, φ, ε, g)
    
    W = np.sum(n * Val)
    
    return W
    
    
  
@njit
def Equal_Constr(E, w, r, n, Y_bar, δ, g, A_j, A_k, β, var_θ, φ, ε, J):
    "Equality Constraints"
    
    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    
    δ_tilde = 1 + δ + g
    
    
    # ---------------------------------- #
    # Initial Period Resource Constraint #
    # ---------------------------------- #
    K = Y_bar - np.sum(n * c_0)
    L = n * l
    Y = r * K + np.sum(w * L)
    
    
    # --------------------------------- #
    # Second Period Resource Constraint #
    # --------------------------------- #
    RC_1 = np.sum(n * c_1) - Y - (1-δ_tilde) * K
    
    
    return np.array([RC_1]) 



@njit
def Inequal_Constr(E, w, n, Y_bar, δ, g, A_j, A_k, β, var_θ, φ, ε, J):
    "Incentive Compatibility Constraints"
    
    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    
    Val = fn.V(c_0, c_1, l, β, var_θ, φ, ε, g)
    
    Val_con = Val + (β / (1-β)) * φ * l**(1 + 1/ε) / (1 + 1/ε)
    
    IC = np.zeros((J,J))
    
    for i in range(J):
        for j in range(J):
           IC[i,j] = Val[i] - (Val_con[j] - (β / (1-β)) * φ[i] * (w[j] * l[j] / w[i])**(1 + 1/ε) / (1 + 1/ε))
    
    return IC 



@njit
def obj_fun(E, w, Δ, args):
    "Penalized Objective Function"
    
    W = Mir_obj(E, *args)
    IC_mat = Inequal_Constr(E, w, *args)
    
    viol = np.minimum(IC_mat, 0.0)
    pen = np.sum(viol**2)

    return -(W - Δ * pen)



def Optimalθ_NL_Root(θ, E, x, w, r, n, Y_bar, δ, g, A_j, A_k, β, var_θ, φ, ε, J, x_bar, ζ, ν, σ):
    "Root for Optimal Threshold Rule with Nonlinear Taxes"
    
    # ----------------- #
    # Unpack Allocation #
    # ----------------- #
    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    
    L = n * l
    K = Y_bar - np.sum(n * c_0)
    κ = Y_bar - c_0
    
    
    # ------------------------------------ #
    # Compute Threshold Rule Perturbations #
    # ------------------------------------ #
    ΔH_ΔΕ = pr.δH_δclx_NL(θ, E, x, w, r, n, Y_bar, δ, g, A_j, A_k, β, var_θ, φ, ε, J, x_bar, ζ, ν, σ)
    ΔH_Δθ = pr.δH_δθ(θ, J)

    dE = - np.linalg.inv(ΔH_ΔΕ) @ ΔH_Δθ
    
    dc_0 = dE[:J,0]
    dl = dE[2*J:3*J,0]
    dx = dE[3*J:,0]
    
    δlnl = dl / l
    δlnκ = - dc_0 / κ
    
    δ_tilde = 1 + δ + g
    R = 1 + r - δ_tilde
    Y = fn.Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    
    δlnw = pr.dlnw(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, Y_bar)
    δlnr = pr.dlnr(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, Y_bar)
    δlnR = r * δlnr / R
    
    α_k = x**(ζ*(ν-1))
    
    if σ == 1:
        θ_wedge = np.log(1+θ)
    else:
        θ_wedge = ((1+θ)**(1-σ) - 1) / (1 - σ)
    
    δY_δX = Y * (A_k * α_k / r)**(σ-1) * θ_wedge
    
    MRS_l = fn.lab_MRS(c_1, l, β, var_θ, φ, ε, g)
    MRS_c = fn.cap_MRS(c_0, c_1, β, var_θ, ε, g)
    λ = c_1**(-var_θ) / np.sum(n * c_1**(-var_θ))
    
    
    # -------------------- #
    # Optimality Condition #
    # -------------------- #
    MC = -np.sum(δY_δX * dx)
    cov = np.sum(n * (λ-1) * (MRS_l * l * δlnw + MRS_c * κ * δlnR))
    Expect = np.sum(n * ((w - MRS_l) * l * δlnl + (R - MRS_c) * κ * δlnκ))
    
    δW = MC - cov - Expect
    
    return δW








