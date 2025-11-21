"""""""""""
Perturbations

Notes: Functions that describe the perturbations of the economy.
    
"""""""""""

import numpy as np
from numba import njit
import Production_Functions as fn
import Processing_Functions as gpf



@njit
def δH_δclx(E, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, Ψ, ψ, β, var_θ, ε, τ_k, δ, g, φ):
    "Jacobian of H wrt Equilibrium Allocation"
    
    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    x = E[3*J:]
    
    L = n * l
    K = np.sum(n * (y_0 - c_0))
    
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    r = fn.Rents(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    R = (1-τ_k)*(r-δ) - g
        
    Y = fn.Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    S_k = r * K / Y
    S_j = w * L / Y
    
    # ------------------ #
    # Output Derivatives #
    # ------------------ #
    α_k = x**(ζ*(ν-1))
    
    if σ == 1:
        θ_wedge = np.log(1+θ)
    else:
        θ_wedge = ((1+θ)**(1-σ) - 1) / (1 - σ)
    
    δlnY_δX = (A_k * α_k / r)**(σ-1) * θ_wedge
    δlnY_δlnK = S_k
    δlnY_δlnL = S_j
    
    # ---------------- #
    # Wage Derivatives #
    # ---------------- #
    rel_l = fn.relα(x, x_bar, ζ, ν, σ, 0)
    
    δlnw_δX = - (1/σ) * gpf.make_diag(rel_l) + (1/σ) * gpf.broadcast_row_to_matrix(δlnY_δX)
    δlnw_δlnK = (1/σ) * δlnY_δlnK * np.ones(J)
    δlnw_δlnL = (1/σ) * gpf.broadcast_row_to_matrix(δlnY_δlnL) - np.eye(J) / σ
    
    # ---------------- #
    # Rent Derivatives #
    # ---------------- #
    rel_k = fn.relα(x, x_bar, ζ, ν, σ, 1)
    
    δlnr_δX = (1/σ) * rel_k + (1/σ) * δlnY_δX
    δlnr_δlnK = (1/σ) * (δlnY_δlnK - 1)
    δlnr_δlnL = (1/σ) * δlnY_δlnL
    
    # --------- #
    # Lump-Sum  #
    # --------- #
    δD_δX = (n * (w * l - Ψ * (1-ψ) * (w * l)**(1-ψ))).reshape((1,J)) @ δlnw_δX + τ_k * r * K * δlnr_δX
    δD_δK = np.sum(n * (w * l - Ψ * (1-ψ) * (w * l)**(1-ψ)) * δlnw_δlnK) / K + τ_k * r * δlnr_δlnK + τ_k * (r - δ)
    δD_δL = (w * l - Ψ * (1-ψ) * (w * l)**(1-ψ)) / L + (((n * (w * l - Ψ * (1-ψ) * (w * l)**(1-ψ))).reshape((1,J)) @ δlnw_δlnL) / L
                                                            + τ_k * r * K * δlnr_δlnL / L)
    δD_δX = δD_δX.reshape(J)
    δD_δL = δD_δL.reshape(J)
    
    
    # ----------------------------------------- #
    # Derivatives of Log Labor Supply Condition #
    # ----------------------------------------- #
    #   wrt to c_0
    δlnLS_δc_0 = - (1-ψ) * np.outer(δlnw_δlnK, -n/K)
    
    #   wrt to c_1
    δlnLS_δc_1 = gpf.make_diag(1/c_1) * var_θ
    
    #   wrt to l
    δlnLS_δl = gpf.make_diag(1/l) * (1/ε + ψ) - (1-ψ) * δlnw_δlnL * gpf.broadcast_row_to_matrix(n / L)
    
    #   wrt to x
    δlnLS_δx =  - (1-ψ) * δlnw_δX
    
    δlnLS = np.hstack((δlnLS_δc_0, δlnLS_δc_1, δlnLS_δl, δlnLS_δx))
    
    
    # --------------------------------- #
    # Derivatives of Log Euler Equation #
    # --------------------------------- #
    #   wrt to c_0
    δlnEE_δc_0 = - gpf.make_diag(1/c_0) * var_θ - ((1-τ_k) * r / R) * δlnr_δlnK / K * gpf.broadcast_row_to_matrix(-n)
    
    #   wrt to c_1
    δlnEE_δc_1 = gpf.make_diag(1/c_1) * var_θ
    
    #   wrt to l
    δlnEE_δl = - ((1-τ_k) * r / R) * gpf.broadcast_row_to_matrix(δlnr_δlnL * n / L)
    
    #   wrt to x
    δlnEE_δx = - ((1-τ_k) * r / R) * gpf.broadcast_row_to_matrix(δlnr_δX)
    
    δlnEE = np.hstack((δlnEE_δc_0, δlnEE_δc_1, δlnEE_δl, δlnEE_δx))
    
    
    # ------------------------------- #
    # Derivatives of Household Budget #
    # ------------------------------- #
    #   wrt to c_0
    δHB_δc_0 = np.eye(J)*R + (gpf.broadcast_col_to_matrix(c_0 - y_0) * (1-τ_k) * r * δlnr_δlnK / K 
                              - gpf.broadcast_col_to_matrix(Ψ * (1-ψ) * (w * l)**(1-ψ) * δlnw_δlnK) / K
                              - δD_δK) * gpf.broadcast_row_to_matrix(-n)
    
    #   wrt to c_1
    δHB_δc_1 = np.eye(J)
    
    #   wrt to l
    δHB_δl = -gpf.make_diag(Ψ * (1-ψ) * (w * l)**(1-ψ) / l) + (gpf.broadcast_col_to_matrix(c_0 - y_0) * (1-τ_k) * r * gpf.broadcast_row_to_matrix(δlnr_δlnL / L)
                                                           - gpf.broadcast_col_to_matrix(Ψ * (1-ψ) * (w * l)**(1-ψ)) * δlnw_δlnL * gpf.broadcast_row_to_matrix(1 / L)
                                                           - gpf.broadcast_row_to_matrix(δD_δL) * gpf.broadcast_row_to_matrix(n))
    
    #   wrt to x
    δHB_δx = (gpf.broadcast_col_to_matrix(c_0 - y_0) * (1-τ_k) * r * gpf.broadcast_row_to_matrix(δlnr_δX)
                  - gpf.broadcast_col_to_matrix(Ψ * (1-ψ) * (w * l)**(1-ψ)) * δlnw_δX
                  - gpf.broadcast_row_to_matrix(δD_δX))
    
    δHB = np.hstack((δHB_δc_0, δHB_δc_1, δHB_δl, δHB_δx))
    
    # ---------------------------------------- #
    # Derivatives of Log Automation Thresholds #
    # ---------------------------------------- #
    #   wrt to c_0
    δlnAT_δc_0 = (- gpf.broadcast_col_to_matrix(δlnw_δlnK) / K + δlnr_δlnK / K) * gpf.broadcast_row_to_matrix(-n)
    
    #   wrt to c_1
    δlnAT_δc_1 = np.zeros((J,J))
    
    #   wrt to l
    δlnAT_δl = - δlnw_δlnL * gpf.broadcast_row_to_matrix(n / L) + gpf.broadcast_row_to_matrix(δlnr_δlnL * n / L)
    
    #   wrt to x
    δlnAT_δx = gpf.make_diag(ζ/x) - δlnw_δX + gpf.broadcast_row_to_matrix(δlnr_δX)
    
    δlnAT = np.hstack((δlnAT_δc_0, δlnAT_δc_1, δlnAT_δl, δlnAT_δx))
    
    
    return np.vstack((δlnLS, δlnEE, δHB, δlnAT))



@njit
def δH_δθ(θ, J):
    "Jacobian of H wrt Threshold Rule"
    
    # ---------------------------------------- #
    # Derivative of Log Labor Supply Condition #
    # ---------------------------------------- #
    δlnLS = np.zeros((J,1))
    
    
    # -------------------------------- #
    # Derivative of Log Euler Equation #
    # -------------------------------- #
    δlnEE = np.zeros((J,1))
    
    
    # ------------------------------ #
    # Derivative of Household Budget #
    # ------------------------------ #
    δHB = np.zeros((J,1))
    
    
    # --------------------------------------- #
    # Derivative of Log Automation Thresholds #
    # --------------------------------------- #
    δlnAT = np.ones((J,1)) / (1+θ)
    
    return np.vstack((δlnLS, δlnEE, δHB, δlnAT))



@njit
def dlnY(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0):
    "Total Derivative of Log Output"
    
    L = n * l
    K = np.sum(n * (y_0 - c_0))
    
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    r = fn.Rents(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
        
    Y = fn.Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    S_k = r * K / Y
    S_j = w * L / Y

    δK = - np.sum(n * dc_0)
    δL = n * dl
        
    # ----------------- #
    # Output Derivative #
    # ----------------- #
    α_k = x**(ζ*(ν-1))
    
    if σ == 1:
        θ_wedge = np.log(1+θ)
    else:
        θ_wedge = ((1+θ)**(1-σ) - 1) / (1 - σ)
    
    δlnY_δX = (A_k * α_k / r)**(σ-1) * θ_wedge
    
    δlnY = np.sum(δlnY_δX * dx) + S_k * δK / K + np.sum(S_j * δL / L)

    return δlnY



@njit
def dlnw(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0):
    "Total Derivative of Log Wages"

    L = n * l
    δL = n * dl
    
    # ----------------- #
    # Output Derivative #
    # ----------------- #
    δlnY = dlnY(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0)

    # ---------------- #
    # Wage Derivatives #
    # ---------------- #
    rel_l = fn.relα(x, x_bar, ζ, ν, σ, 0)
    
    δlnw = (1/σ) * (- rel_l * dx + δlnY - δL / L)
    
    return δlnw


    
@njit    
def dlnr(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0):
    "Total Derivative of Log Rent"

    K = np.sum(n * (y_0 - c_0))
    δK = - np.sum(n * dc_0)
        
    # ----------------- #
    # Output Derivative #
    # ----------------- #
    δlnY = dlnY(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0)

    # --------------- #
    # Rent Derivative #
    # --------------- #
    rel_k = fn.relα(x, x_bar, ζ, ν, σ, 1)
    
    δlnr = (1/σ) * (np.sum(rel_k * dx) + δlnY - δK / K)
        
    return δlnr  



@njit
def δObj_δX(X, n, Y_bar, δ, g, A_j, A_k, β, var_θ, φ, ε, J):
    "Jacobian of Mirrlees Objective"
    
    c_0 = X[:J]
    c_1 = X[J:2*J]
    l = X[2*J:3*J]
    
    # ---------------------- #
    # Derivatives of Welfare #
    # ---------------------- #
    #   wrt to c_0
    δW_δc_0 = n * c_0**(-var_θ)
    
    #   wrt to c_1
    δW_δc_1 = n * (β / (1-β * (1+g)**(1-var_θ))) *  c_1**(-var_θ)
    
    #   wrt to l
    δW_δl = - n * (β / (1-β)) * φ * l**(1/ε)
    
    #   wrt to K
    δW_δK = np.zeros(1)
    
    δW = np.concatenate((δW_δc_0, δW_δc_1, δW_δl, δW_δK))
    
    return - δW
    


@njit
def δEC_δX(X, w, r, n, Y_bar, δ, g, A_j, A_k, β, var_θ, φ, ε, J):
    "Jacobian of Equality Constraints"
        
    δ_hat = 1 + δ + g
    
    
    # ------------------------------------------------- #
    # Derivatives of Initial Period Resource Constraint #
    # ------------------------------------------------- #
    #   wrt to c_0
    δRC_0_δc_0 = n.reshape((1,J))
    
    #   wrt to c_1
    δRC_0_δc_1 = np.zeros((1,J))
    
    #   wrt to l
    δRC_0_δl = np.zeros((1,J))
    
    #   wrt to K
    δRC_0_δK = np.ones((1,1))
    
    δRC_0 = np.hstack((δRC_0_δc_0, δRC_0_δc_1, δRC_0_δl, δRC_0_δK))
        
    
    # ------------------------------------------------ #
    # Derivatives of Second Period Resource Constraint #
    # ------------------------------------------------ #
    #   wrt to c_0
    δRC_1_δc_0 = np.zeros((1,J))
    
    #   wrt to c_1
    δRC_1_δc_1 = n.reshape((1,J))
    
    #   wrt to l
    δRC_1_δl = - (w * n).reshape((1,J))
    
    #   wrt to K
    δRC_1_δK = - np.ones((1,1)) * (1 + r - δ_hat)
    
    δRC_1 = np.hstack((δRC_1_δc_0, δRC_1_δc_1, δRC_1_δl, δRC_1_δK))
        
        
    return np.vstack((δRC_0, δRC_1))
    
    
 
@njit
def δIC_δX(X, w, IC_act, n, Y_bar, δ, g, A_j, A_k, β, var_θ, φ, ε, J):
    "Jacobian of Inequality Constraints"
    
    c_0 = X[:J]
    c_1 = X[J:2*J]
    l = X[2*J:3*J]
    
    IC_Comp_J = [np.where(IC_act[i] == 1)[0] for i in range(J)]
    IC_count = int(IC_act.sum()) 
    
    
    # ----------------- #
    # Derivatives of IC #
    # ----------------- #
    #   wrt to c_0
    δIC_δc_0 = np.zeros((IC_count,J))
    k = 0
    
    for i in range(J):
        for j in IC_Comp_J[i]:
            δIC_δc_0[k, i] = c_0[i]**(-var_θ)
            δIC_δc_0[k, j] = -c_0[j]**(-var_θ)
            k += 1
   
    #   wrt to c_1
    δIC_δc_1 = np.zeros((IC_count,J))
    k = 0
    
    for i in range(J):
        for j in IC_Comp_J[i]:
            δIC_δc_1[k, i] = (β/(1-β*(1+g)**(1-var_θ))) * c_1[i]**(-var_θ)
            δIC_δc_1[k, j] = -(β/(1-β*(1+g)**(1-var_θ))) * c_1[j]**(-var_θ)
            k += 1
    
    #   wrt to l
    δIC_δl = np.zeros((IC_count,J))
    k = 0
    
    for i in range(J):
        for j in IC_Comp_J[i]:
            δIC_δl[k, i] = - (β/(1-β)) * φ[i] * l[i]**(1/ε)
            δIC_δl[k, j] = (β/(1-β)) * φ[i] * l[j]**(1/ε) * (w[j]/w[i])**(1+1/ε)
            k += 1
            
    #   wrt to K
    δIC_δK = np.zeros((IC_count,1))
    
    δIC = np.hstack((δIC_δc_0, δIC_δc_1, δIC_δl, δIC_δK))
    
    return δIC
    
    



