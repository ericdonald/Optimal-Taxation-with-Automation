"""""""""""
Perturbations

Notes: Functions that describe the perturbations of the economy.
    
"""""""""""

import numpy as np
from numba import njit
import Production_Functions as fn
import Processing_Functions as gpf



@njit
def δH_δclx(E, θ, τ_k, Ψ, ψ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, β, var_θ, ε, δ, g, φ):
    "Jacobian of H wrt Equilibrium Allocation"
    
    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    x = E[3*J:]
    
    L = n * l
    Y_bar = np.sum(n * y_0)
    K = Y_bar - np.sum(n * c_0)
    
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    r = fn.Rents(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    R_tilde = (1-τ_k)*(r-δ) - g
        
    Y = fn.Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    S_k = r * K / Y
    S_j = w * L / Y
    
    
    # ------------------ #
    # Output Derivatives #
    # ------------------ #
    δlnY_δX = dlnY_dx(c_0, l, x, A_j, A_k, x_bar, ζ, ν, σ, J, n, Y_bar)
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
    
    
    # ------------------------ #
    # Tax Function Derivatives #
    # ------------------------ #
    keep_l = fn.Heath_keep(w, l, Ψ, ψ)
    τ_l = 1 - keep_l
    y_l = w * l
    
    δD_δl = n * τ_l * w + ((n * τ_l * y_l).reshape((1,J)) @ δlnw_δlnL) / l + τ_k * r * K * δlnr_δlnL / l
    δD_δK = np.sum(n * τ_l * y_l * δlnw_δlnK) / K + τ_k * r * δlnr_δlnK + τ_k * (r - δ)
    δD_δX = (n * τ_l * y_l).reshape((1,J)) @ δlnw_δX + τ_k * r * K * δlnr_δX
    
    δD_δl = δD_δl.reshape(J)
    δD_δX = δD_δX.reshape(J)
    
    
    # ----------------------------------------- #
    # Derivatives of Log Labor Supply Condition #
    # ----------------------------------------- #
    #   wrt to c_0
    δlnLS_δc_0 = - (1-ψ) * np.outer(δlnw_δlnK, -n/K)
    
    #   wrt to c_1
    δlnLS_δc_1 = gpf.make_diag(1/c_1) * var_θ
    
    #   wrt to l
    δlnLS_δl = gpf.make_diag(1/l) * (1/ε + ψ) - (1-ψ) * δlnw_δlnL * gpf.broadcast_row_to_matrix(1 / l)
    
    #   wrt to x
    δlnLS_δx = - (1-ψ) * δlnw_δX
    
    δlnLS = np.hstack((δlnLS_δc_0, δlnLS_δc_1, δlnLS_δl, δlnLS_δx))
    
    
    # --------------------------------- #
    # Derivatives of Log Euler Equation #
    # --------------------------------- #
    #   wrt to c_0
    δlnEE_δc_0 = - gpf.make_diag(1/c_0) * var_θ - ((1-τ_k) * r / R_tilde) * δlnr_δlnK / K * gpf.broadcast_row_to_matrix(-n)
    
    #   wrt to c_1
    δlnEE_δc_1 = gpf.make_diag(1/c_1) * var_θ
    
    #   wrt to l
    δlnEE_δl = - ((1-τ_k) * r / R_tilde) * gpf.broadcast_row_to_matrix(δlnr_δlnL / l)
    
    #   wrt to x
    δlnEE_δx = - ((1-τ_k) * r / R_tilde) * gpf.broadcast_row_to_matrix(δlnr_δX)
    
    δlnEE = np.hstack((δlnEE_δc_0, δlnEE_δc_1, δlnEE_δl, δlnEE_δx))
    
    
    # ------------------------------- #
    # Derivatives of Household Budget #
    # ------------------------------- #
    #   wrt to c_0
    δHB_δc_0 = np.eye(J)*R_tilde + (gpf.broadcast_col_to_matrix(c_0 - y_0) * (1-τ_k) * r * δlnr_δlnK / K 
                              - gpf.broadcast_col_to_matrix(keep_l * y_l * δlnw_δlnK) / K
                              - δD_δK) * gpf.broadcast_row_to_matrix(-n)
    
    #   wrt to c_1
    δHB_δc_1 = np.eye(J)
    
    #   wrt to l
    δHB_δl = -gpf.make_diag(keep_l * w) + (gpf.broadcast_col_to_matrix(c_0 - y_0) * (1-τ_k) * r * gpf.broadcast_row_to_matrix(δlnr_δlnL / l)
                                                           - gpf.broadcast_col_to_matrix(keep_l * y_l) * δlnw_δlnL * gpf.broadcast_row_to_matrix(1 / l)
                                                           - gpf.broadcast_row_to_matrix(δD_δl))
    
    #   wrt to x
    δHB_δx = (gpf.broadcast_col_to_matrix(c_0 - y_0) * (1-τ_k) * r * gpf.broadcast_row_to_matrix(δlnr_δX)
                  - gpf.broadcast_col_to_matrix(keep_l * y_l) * δlnw_δX
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
    δlnAT_δl = - δlnw_δlnL * gpf.broadcast_row_to_matrix(1 / l) + gpf.broadcast_row_to_matrix(δlnr_δlnL / l)
    
    #   wrt to x
    δlnAT_δx = gpf.make_diag(ζ/x) - δlnw_δX + gpf.broadcast_row_to_matrix(δlnr_δX)
    
    δlnAT = np.hstack((δlnAT_δc_0, δlnAT_δc_1, δlnAT_δl, δlnAT_δx))
    
    
    return np.vstack((δlnLS, δlnEE, δHB, δlnAT))



@njit
def δH_δIST(E, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, κ_0, Ψ, ψ, var_θ, ε, τ_k, δ, g, φ):
    "Jacobian of H wrt Investment Price"
    
    # ------------------- #
    # Equilibrium Changes #
    # ------------------- #
    c_0 = E[:J]
    l = E[2*J:3*J]
    x = E[3*J:]
    
    L = n * l
    K = np.sum(n * (y_0 - c_0))
    
    r = fn.Rents(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    R_tilde = (1-τ_k)*(r-δ) - g
    
    δD_0 = τ_k * δ * K / (1+g)
    δD_1 = τ_k * δ * K
    
    δy_0 = δD_0 - (1 - (1-τ_k)*δ) * κ_0.reshape((J,1))
    
    
    # ---------------------------------------- #
    # Derivative of Log Labor Supply Condition #
    # ---------------------------------------- #
    δlnLS = np.zeros((J,1))
    
    
    # -------------------------------- #
    # Derivative of Log Euler Equation #
    # -------------------------------- #
    δlnEE = - np.ones((J,1)) * ((1-τ_k) * r / R_tilde)
    
    
    # ------------------------------ #
    # Derivative of Household Budget #
    # ------------------------------ #
    δHB = ((c_0 - y_0).reshape((J,1)) * (1-τ_k) * r
                  - R_tilde * δy_0
                  - δD_1)
    
    
    # --------------------------------------- #
    # Derivative of Log Automation Thresholds #
    # --------------------------------------- #
    δlnAT = np.zeros((J,1))
    
    
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
def δH_δτ(E, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, τ_k, δ, g):
    "Jacobian of H wrt Capital Tax"
    
    c_0 = E[:J]
    l = E[2*J:3*J]
    x = E[3*J:]
    
    L = n * l
    κ = y_0 - c_0
    K = np.sum(n * κ)
    
    r = fn.Rents(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    R_tilde = (1-τ_k)*(r-δ) - g
    
    dD = (r - δ) * K
    
    
    # ---------------------------------------- #
    # Derivative of Log Labor Supply Condition #
    # ---------------------------------------- #
    δlnLS = np.zeros((J,1))
    
    
    # -------------------------------- #
    # Derivative of Log Euler Equation #
    # -------------------------------- #
    δlnEE = np.ones((J,1)) * (r-δ) / R_tilde
    
    
    # ------------------------------ #
    # Derivative of Household Budget #
    # ------------------------------ #
    δHB = (r-δ) * κ.reshape((J,1)) - dD
    
    
    # --------------------------------------- #
    # Derivative of Log Automation Thresholds #
    # --------------------------------------- #
    δlnAT = np.zeros((J,1))
    
    
    return np.vstack((δlnLS, δlnEE, δHB, δlnAT))



@njit
def δH_δΨ(E, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, Ψ, ψ):
    "Jacobian of H wrt Labor Tax Scale"
    
    c_0 = E[:J]
    l = E[2*J:3*J]
    x = E[3*J:]
    
    L = n * l
    K = np.sum(n * (y_0 - c_0))
    
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    keep_l = fn.Heath_keep(w, l, Ψ, ψ).reshape((J,1))
    
    dD = - np.sum(n * (w * l)**(1-ψ))
    y = (w * l).reshape((J,1))
    
    
    # ---------------------------------------- #
    # Derivative of Log Labor Supply Condition #
    # ---------------------------------------- #
    δlnLS = - (1-ψ) * y**(-ψ) / keep_l
    
    
    # -------------------------------- #
    # Derivative of Log Euler Equation #
    # -------------------------------- #
    δlnEE = np.zeros((J,1))
    
    
    # ------------------------------ #
    # Derivative of Household Budget #
    # ------------------------------ #
    δHB = - (y**(1-ψ) + dD)
    
    
    # --------------------------------------- #
    # Derivative of Log Automation Thresholds #
    # --------------------------------------- #
    δlnAT = np.zeros((J,1))
    
    
    return np.vstack((δlnLS, δlnEE, δHB, δlnAT))



@njit
def δH_δψ(E, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0, Ψ, ψ):
    "Jacobian of H wrt Labor Tax Curvature"
    
    c_0 = E[:J]
    l = E[2*J:3*J]
    x = E[3*J:]
    
    L = n * l
    K = np.sum(n * (y_0 - c_0))
    
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    keep_l = fn.Heath_keep(w, l, Ψ, ψ).reshape((J,1))
        
    dD = np.sum(n * Ψ * (w * l)**(1-ψ) * np.log(w * l))
    y = (w * l).reshape((J,1))
    
    
    # ---------------------------------------- #
    # Derivative of Log Labor Supply Condition #
    # ---------------------------------------- #
    δlnLS = (1 + (1-ψ) * np.log(y)) * Ψ * y**(-ψ) / keep_l
    
    
    # -------------------------------- #
    # Derivative of Log Euler Equation #
    # -------------------------------- #
    δlnEE = np.zeros((J,1))
    
    
    # ------------------------------ #
    # Derivative of Household Budget #
    # ------------------------------ #
    δHB = - (- Ψ * (y)**(1-ψ) * np.log(y) + dD)
    
    
    # --------------------------------------- #
    # Derivative of Log Automation Thresholds #
    # --------------------------------------- #
    δlnAT = np.zeros((J,1))
    
    
    return np.vstack((δlnLS, δlnEE, δHB, δlnAT))



@njit
def dlnY_dx(c_0, l, x, A_j, A_k, x_bar, ζ, ν, σ, J, n, Y_bar):
    "Derivative of Log Output wrt Automation"
    
    L = n * l
    K = Y_bar - np.sum(n * c_0)
    
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    r = fn.Rents(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
        
    Y = fn.Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    S_k = r * K / Y
    S_j = w * L / Y
    
    rel_k = fn.relα(x, x_bar, ζ, ν, σ, 1)
    α_k = x**(ζ*(ν-1))
    rel_l = fn.relα(x, x_bar, ζ, ν, σ, 0)
    α_l = x**(ζ*ν)
    
    if σ == 1:
        δlnY_δX = np.log(α_k * A_k / r) - np.log(α_l * A_j / w)
    else:
        δlnY_δX = (S_k * rel_k - S_j * rel_l) / (σ-1)
        
    return δlnY_δX
    
    

@njit
def dlnY(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0):
    "Total Derivative of Log Output"
    
    L = n * l
    Y_bar = np.sum(n * y_0)
    K = Y_bar - np.sum(n * c_0)
    
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
    δlnY_δX = dlnY_dx(c_0, l, x, A_j, A_k, x_bar, ζ, ν, σ, J, n, Y_bar)
    
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
def δObj_δX(E, θ, n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, β, var_θ, φ, ε, x_bar, J):
    "Jacobian of Mirrlees Objective"
    
    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    
    beta_tilde = β / (1 - β * (1+g)**(1-var_θ))
    beta_tilde_l = (β / (1-β))
    
    
    # ---------------------- #
    # Derivatives of Welfare #
    # ---------------------- #
    #   wrt to c_0
    δW_δc_0 = n * c_0**(-var_θ)
    
    #   wrt to c_1
    δW_δc_1 = n * beta_tilde * c_1**(-var_θ)
    
    #   wrt to l
    δW_δl = - n * beta_tilde_l * φ * l**(1/ε)
    
    #   wrt to x
    δW_δx = np.zeros(J)
    
    δW = np.concatenate((δW_δc_0, δW_δc_1, δW_δl, δW_δx))
        
        
    return δW


@njit
def δ2Obj_δX_δX(E, θ, n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, β, var_θ, φ, ε, x_bar, J):
    "Hessian of Mirrlees Objective"
    
    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    
    beta_tilde = β / (1 - β * (1+g)**(1-var_θ))
    beta_tilde_l = (β / (1-β))
    
    
    # ---------------------- #
    # Derivatives of Welfare #
    # ---------------------- #
    #   wrt to c_0
    δ2W_δc_0_δc_0 = (-var_θ) * n * c_0**(-var_θ- 1)
    
    #   wrt to c_1
    δ2W_δc_1_δc_1 = (-var_θ) * n * beta_tilde * c_1**(-var_θ-1)
    
    #   wrt to l
    δ2W_δl_δl = - (1/ε) * n * beta_tilde_l * φ * l**(1/ε - 1)
    
    #   wrt to x
    δ2W_δx_δx = np.zeros(J)
    
    δ2W = np.concatenate((δ2W_δc_0_δc_0, δ2W_δc_1_δc_1, δ2W_δl_δl, δ2W_δx_δx))
        
        
    return gpf.make_diag(δ2W)
    


@njit
def δEC_δX(E, θ, n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, β, var_θ, φ, ε, x_bar, J):
    "Jacobian of Equality Constraints"
    
    c_0 = E[:J]; l = E[2*J:3*J]; x = E[3*J:4*J]
    
    K = Y_bar - np.sum(n * c_0)
    L = n * l
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    r = fn.Rents(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    
    δ_tilde = 1 + δ + g
    Y = fn.Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    S_k = r * K / Y
    S_j = w * L / Y
    
    
    # ------------------ #
    # Output Derivatives #
    # ------------------ #
    δlnY_δX = dlnY_dx(c_0, l, x, A_j, A_k, x_bar, ζ, ν, σ, J, n, Y_bar)
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
    
    
    # ------------------------------------------------ #
    # Derivatives of Second Period Resource Constraint #
    # ------------------------------------------------ #
    #   wrt to c_0
    δRC_δc_0 = (1 + r - δ_tilde) * n.reshape((1,J))
    
    #   wrt to c_1
    δRC_δc_1 = n.reshape((1,J))
    
    #   wrt to l
    δRC_δl = - (w * n).reshape((1,J))
    
    #   wrt to x
    δRC_δx = - Y * δlnY_δX.reshape((1,J))
    
    δRC = np.hstack((δRC_δc_0, δRC_δc_1, δRC_δl, δRC_δx))
    
    
    # --------------------------------------- #
    # Derivative of Log Automation Thresholds #
    # --------------------------------------- #
    #   wrt to c_0
    δlnAT_δc_0 = (- gpf.broadcast_col_to_matrix(δlnw_δlnK) / K + δlnr_δlnK / K) * gpf.broadcast_row_to_matrix(-n)
    
    #   wrt to c_1
    δlnAT_δc_1 = np.zeros((J,J))
    
    #   wrt to l
    δlnAT_δl = - δlnw_δlnL * gpf.broadcast_row_to_matrix(1 / l) + gpf.broadcast_row_to_matrix(δlnr_δlnL / l)
    
    #   wrt to x
    δlnAT_δx = gpf.make_diag(ζ/x) - δlnw_δX + gpf.broadcast_row_to_matrix(δlnr_δX)
    
    δlnAT = np.hstack((δlnAT_δc_0, δlnAT_δc_1, δlnAT_δl, δlnAT_δx))
 
    
    return np.vstack((δRC, δlnAT))



@njit
def δ2EC_δX_δX(E, λ_eq, θ, n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, β, var_θ, φ, ε, x_bar, J):
    "Hessian of Equality Constraints"
    
    c_0 = E[:J]; l = E[2*J:3*J]; x = E[3*J:4*J]
    
    K = Y_bar - np.sum(n * c_0)
    L = n * l
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    r = fn.Rents(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    
    Y = fn.Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    S_k = r * K / Y
    S_j = w * L / Y
    
    λ_RC = λ_eq[0]
    λ_AT = λ_eq[1:]
    H = np.zeros((4*J, 4*J))
    
    
    # ------------------ #
    # Output Derivatives #
    # ------------------ #
    δlnY_δX = dlnY_dx(c_0, l, x, A_j, A_k, x_bar, ζ, ν, σ, J, n, Y_bar)
    δlnY_δlnK = S_k
    δlnY_δlnL = S_j
    
    
    # ---------------- #
    # Wage Derivatives #
    # ---------------- #
    rel_l = fn.relα(x, x_bar, ζ, ν, σ, 0)
    
    δlnw_δX = - (1/σ) * gpf.make_diag(rel_l) + (1/σ) * gpf.broadcast_row_to_matrix(δlnY_δX)
    δlnw_δlnL = (1/σ) * gpf.broadcast_row_to_matrix(δlnY_δlnL) - np.eye(J) / σ
    
    
    # ---------------- #
    # Rent Derivatives #
    # ---------------- #
    rel_k = fn.relα(x, x_bar, ζ, ν, σ, 1)
    
    δlnr_δX = (1/σ) * rel_k + (1/σ) * δlnY_δX
    δlnr_δlnK = (1/σ) * (δlnY_δlnK - 1)
    δlnr_δlnL = (1/σ) * δlnY_δlnL
    
    
    # ------------------------------------------------------- #
    # Second Derivatives of Second Period Resource Constraint #
    # ------------------------------------------------------- #
    c0 = slice(0, J); l_ = slice(2*J, 3*J); x_ = slice(3*J, 4*J)
    
    #   wrt to c_0
    #       wrt to c_0
    δ2RC_δc_0_δc_0 = - (δlnr_δlnK * r / K) * (n.reshape((J,1)) @ n.reshape((1,J)))
    
    #       wrt to l
    δ2RC_δc_0_δl = n.reshape((J,1)) @ (δlnr_δlnL * r / L * n).reshape((1,J))
    
    #       wrt to x
    δ2RC_δc_0_δx = n.reshape((J,1)) @ (δlnr_δX * r).reshape((1,J))
        
    
    #   wrt to l
    #       wrt to l
    δ2RC_δl_δl = - δlnw_δlnL * ((w * n).reshape((J,1)) @ (n/L).reshape((1,J)))
    
    #       wrt to x
    δ2RC_δl_δx = - δlnw_δX * gpf.broadcast_col_to_matrix(w * n)
        
    
    #   wrt to x
    #       wrt to x
    δ2RC_δx_δx = (- (δlnY_δX.reshape((J,1)) @ δlnY_δX.reshape((1,J))) + S_k * (rel_k.reshape((J,1)) @ δlnr_δX.reshape((1,J))) / (σ-1) 
                      - gpf.broadcast_col_to_matrix(S_j * rel_l) * δlnw_δX / (σ-1) - S_k * (rel_k.reshape((J,1)) @ rel_k.reshape((1,J))) / (σ-1))
    δ2RC_δx_δx += gpf.make_diag(S_k * rel_k * ζ * (ν-1) / x - S_j * rel_l * (ζ * ν * (σ-1) / x + rel_l) / (σ-1))
    
    H[c0, c0] += λ_RC * δ2RC_δc_0_δc_0
    H[c0, l_] += λ_RC * δ2RC_δc_0_δl ;  H[l_, c0] += λ_RC * δ2RC_δc_0_δl.T
    H[c0, x_] += λ_RC * δ2RC_δc_0_δx ;  H[x_, c0] += λ_RC * δ2RC_δc_0_δx.T
    H[l_, l_] += λ_RC * δ2RC_δl_δl
    H[l_, x_] += λ_RC * δ2RC_δl_δx ;  H[x_, l_] += λ_RC * δ2RC_δl_δx.T
    H[x_, x_] += λ_RC * δ2RC_δx_δx
    
    
    # ---------------------------------------------- #
    # Second Derivative of Log Automation Thresholds #
    # ---------------------------------------------- #
    sumλ = np.sum(λ_AT)
    
    for j in range(J):
        #   wrt to c_0
        #       wrt to c_0
        H[c0, c0] += sumλ * (n.reshape((J,1)) @ n.reshape((1,J))) / σ / (K**2)
        
        
        #   wrt to l
        #       wrt to l
        diag_ll = - (n**2) / (L**2) / σ * λ_AT
        H[l_, l_] += gpf.make_diag(diag_ll)
        
        
        #   wrt to x
        #       wrt to x
        H[x_, x_] += sumλ * (- (rel_k.reshape((J,1)) @ rel_k.reshape((1,J))))
        diag_xx = ( - ζ / (x**2)
                + rel_l * (ζ * ν * (σ-1) / x + rel_l)
                + rel_k * ζ * (ν-1) * (σ-1) / x ) * λ_AT
        H[x_, x_] += gpf.make_diag(diag_xx)
 
    
    return H
    
    
 
@njit
def δIC_δX(E, WS, θ, n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, β, var_θ, φ, ε, x_bar, J):
    "Jacobian of Inequality Constraints"
    
    c_0 = E[:J]; c_1 = E[J:2*J]; l = E[2*J:3*J]; x = E[3*J:4*J]
        
    K = Y_bar - np.sum(n * c_0)
    L = n * l
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    rel_l = fn.relα(x, x_bar, ζ, ν, σ, 0)
        
    beta_tilde = β / (1 - β * (1+g)**(1-var_θ))
    beta_tilde_l = β / (1-β)
    
    
    # ----------------- #
    # Derivatives of IC #
    # ----------------- #    
    P = WS.shape[0]
    Q = 8
    rows = np.empty(Q*P, np.int64)
    cols = np.empty(Q*P, np.int64)
    vals = np.empty(Q*P)
    
    for p in range(P):
        i = WS[p, 0]; j = WS[p, 1]
        b = Q*p
        im  = (w[j]*l[j]/w[i])
        hh = beta_tilde_l*φ[i]*(im**(1+1/ε))
        #   wrt to c_0
        rows[b]   = p; cols[b]   = i;       vals[b]   =  c_0[i]**(-var_θ)
        rows[b+1] = p; cols[b+1] = j;       vals[b+1] = -c_0[j]**(-var_θ)
        
        #   wrt to c_1
        rows[b+2] = p; cols[b+2] = J+i;     vals[b+2] =  beta_tilde*c_1[i]**(-var_θ)
        rows[b+3] = p; cols[b+3] = J+j;     vals[b+3] = -beta_tilde*c_1[j]**(-var_θ)
        
        #   wrt to l
        rows[b+4] = p; cols[b+4] = 2*J+i;   vals[b+4] = beta_tilde_l*φ[i]*(-l[i]**(1+1/ε) + im**(1+1/ε)/σ)/l[i]
        rows[b+5] = p; cols[b+5] = 2*J+j;   vals[b+5] = hh*(1 - 1/σ)/l[j]
        
        #   wrt to x
        rows[b+6] = p; cols[b+6] = 3*J+i;   vals[b+6] =  hh*rel_l[i]/σ
        rows[b+7] = p; cols[b+7] = 3*J+j;   vals[b+7] = -hh*rel_l[j]/σ
        
    
    return rows, cols, vals



@njit
def δ2IC_δX_δX(E, WS, θ, n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, β, var_θ, φ, ε, x_bar, J):
    "Hessian of Inequality Constraints"
    
    c_0 = E[:J]; c_1 = E[J:2*J]; l = E[2*J:3*J]; x = E[3*J:4*J]
        
    K = Y_bar - np.sum(n * c_0)
    L = n * l
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    rel_l = fn.relα(x, x_bar, ζ, ν, σ, 0)
        
    beta_tilde = β / (1 - β * (1+g)**(1-var_θ))
    beta_tilde_l = β / (1-β)
    
    
    # ------------------------ #
    # Second Derivatives of IC #
    # ------------------------ #    
    P = WS.shape[0]
    Q = 8
    blocks = np.zeros((P, Q, Q))
    idx    = np.empty((P, Q), np.int64)
    
    for p in range(P):
        i = WS[p,0]; j = WS[p,1]
        idx[p,0]=i;     idx[p,1]=j
        idx[p,2]=J+i;   idx[p,3]=J+j
        idx[p,4]=2*J+i; idx[p,5]=2*J+j
        idx[p,6]=3*J+i; idx[p,7]=3*J+j
        
        im  = (w[j]*l[j]/w[i])
        hh = beta_tilde_l*φ[i]*(im**(1+1/ε))
        
        #   wrt to c_0
        #       wrt to c_0
        blocks[p,0,0] = (-var_θ)*c_0[i]**(-var_θ-1)
        blocks[p,1,1] = -(-var_θ)*c_0[j]**(-var_θ-1)
        
        
        #   wrt to c_1
        #       wrt to c_1
        blocks[p,2,2] =  beta_tilde*(-var_θ)*c_1[i]**(-var_θ-1)
        blocks[p,3,3] = -beta_tilde*(-var_θ)*c_1[j]**(-var_θ-1)
        
        
        #   wrt to l
        #       wrt to l
        blocks[p,4,4] = beta_tilde_l*φ[i]*(-(1/ε) * l[i]**(1/ε) + im**(1+1/ε) * ((1+1/ε)/σ - 1) / σ / l[i]) / l[i]
        blocks[p,4,5] = blocks[p,5,4] = (1+1/ε) * hh / l[i] * (1 - 1/σ) / σ / l[j]
        blocks[p,5,5] = (1/ε - (1+1/ε)/σ) * hh * (1 - 1/σ) / l[j]**2
        
        #       wrt to x
        blocks[p,4,6] = (1+1/ε) * hh * rel_l[i] / (σ**2) / l[i]
        blocks[p,4,7] = - (1+1/ε) * hh * rel_l[j] / (σ**2) / l[i]
        
        blocks[p,5,6] = (1+1/ε) * hh * rel_l[i] * (1 - 1/σ) / σ / l[j]
        blocks[p,5,7] = - (1+1/ε) * hh * rel_l[j] * (1 - 1/σ) / σ / l[j]
        
        
        #   wrt to x
        #       wrt to l
        blocks[p,6,4] = blocks[p,4,6]
        blocks[p,6,5] = blocks[p,5,6]
        
        blocks[p,7,4] = blocks[p,4,7]
        blocks[p,7,5] = blocks[p,5,7]
        
        #       wrt to x
        blocks[p,6,6] = hh * rel_l[i] * ((1+1/ε) * rel_l[i] / σ + ζ[i] * ν[i] * (σ-1) / x[i] + rel_l[i]) /σ
        blocks[p,6,7] = blocks[p,7,6] = - (1+1/ε) * hh * rel_l[j] * rel_l[i] / σ**2
        blocks[p,7,7] = hh * rel_l[j] * ((1+1/ε) * rel_l[j] / σ - ζ[j] * ν[j] * (σ-1) / x[j] - rel_l[j]) /σ
        
    
    return blocks, idx



def δH_δclx_NL(E, θ, w, r, n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, β, var_θ, φ, ε, x_bar, J):
    "Jacobian of H wrt Mirrlees Allocation"
    
    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    x = E[3*J:4*J]
    
    L = n * l
    K = Y_bar - np.sum(n * c_0)
    
    δ_tilde = 1 + δ + g
    R = 1 + r - δ_tilde
    Y = fn.Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    S_k = r * K / Y
    S_j = w * L / Y
    
    
    # ------------------ #
    # Output Derivatives #
    # ------------------ #
    δlnY_δX = dlnY_dx(c_0, l, x, A_j, A_k, x_bar, ζ, ν, σ, J, n, Y_bar)
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
    
    
    # ------------------------ #
    # Tax Function Derivatives #
    # ------------------------ #
    MRS_l = fn.lab_MRS(c_1, l, β, var_θ, φ, ε, g)
    keep_l = MRS_l / w
    τ_l = 1 - keep_l
    y_l = w * l
    
    coefs = np.polyfit(y_l, τ_l, 4)
    P_prime = np.polyder(np.poly1d(coefs))
    dτ_dy = P_prime(y_l)
    
    δD_δl = - n * τ_l * w - ((n * τ_l * y_l).reshape((1,J)) @ δlnw_δlnL) / l
    δD_δK = - np.sum(n * τ_l * y_l * δlnw_δlnK) / K
    δD_δX = - (n * τ_l * y_l).reshape((1,J)) @ δlnw_δX
    
    δD_δl = δD_δl.reshape(J)
    δD_δX = δD_δX.reshape(J)
    
    
    # ----------------------------------------- #
    # Derivatives of Log Labor Supply Condition #
    # ----------------------------------------- #
    #   wrt to c_0
    δlnLS_δc_0 = (dτ_dy * y_l / keep_l - 1) * np.outer(δlnw_δlnK, -n/K)
    
    #   wrt to c_1
    δlnLS_δc_1 = gpf.make_diag(1/c_1) * var_θ
    
    #   wrt to l
    δlnLS_δl = gpf.make_diag(1/l) * (1/ε + dτ_dy * y_l / keep_l) + (dτ_dy * y_l / keep_l - 1) * δlnw_δlnL * gpf.broadcast_row_to_matrix(1 / l)
    
    #   wrt to x
    δlnLS_δx = (dτ_dy * y_l / keep_l - 1) * δlnw_δX
    
    δlnLS = np.hstack((δlnLS_δc_0, δlnLS_δc_1, δlnLS_δl, δlnLS_δx))
    
    
    # --------------------------------- #
    # Derivatives of Log Euler Equation #
    # --------------------------------- #
    #   wrt to c_0
    δlnEE_δc_0 = - gpf.make_diag(1/c_0) * var_θ - (r / R) * δlnr_δlnK / K * gpf.broadcast_row_to_matrix(-n)
    
    #   wrt to c_1
    δlnEE_δc_1 = gpf.make_diag(1/c_1) * var_θ
    
    #   wrt to l
    δlnEE_δl = - (r / R) * gpf.broadcast_row_to_matrix(δlnr_δlnL / l)
    
    #   wrt to x
    δlnEE_δx = - (r / R) * gpf.broadcast_row_to_matrix(δlnr_δX)
    
    δlnEE = np.hstack((δlnEE_δc_0, δlnEE_δc_1, δlnEE_δl, δlnEE_δx))
    
    
    # ------------------------------- #
    # Derivatives of Household Budget #
    # ------------------------------- #
    #   wrt to c_0
    δHB_δc_0 = np.eye(J)*R + (gpf.broadcast_col_to_matrix(c_0 - Y_bar) * r * δlnr_δlnK / K 
                              - gpf.broadcast_col_to_matrix(keep_l * y_l * δlnw_δlnK) / K
                              + δD_δK) * gpf.broadcast_row_to_matrix(-n)
    
    #   wrt to c_1
    δHB_δc_1 = np.eye(J)
    
    #   wrt to l
    δHB_δl = - gpf.make_diag(keep_l * w) + (gpf.broadcast_col_to_matrix(c_0 - Y_bar) * r * gpf.broadcast_row_to_matrix(δlnr_δlnL / l)
                                                           - gpf.broadcast_col_to_matrix(keep_l * y_l) * δlnw_δlnL * gpf.broadcast_row_to_matrix(1 / l)
                                                           + gpf.broadcast_row_to_matrix(δD_δl))
    
    #   wrt to x
    δHB_δx = (gpf.broadcast_col_to_matrix(c_0 - Y_bar) * r * gpf.broadcast_row_to_matrix(δlnr_δX)
                  - gpf.broadcast_col_to_matrix(keep_l * y_l) * δlnw_δX
                  + gpf.broadcast_row_to_matrix(δD_δX))
    
    δHB = np.hstack((δHB_δc_0, δHB_δc_1, δHB_δl, δHB_δx))
    
    
    # ---------------------------------------- #
    # Derivatives of Log Automation Thresholds #
    # ---------------------------------------- #
    #   wrt to c_0
    δlnAT_δc_0 = (- gpf.broadcast_col_to_matrix(δlnw_δlnK) / K + δlnr_δlnK / K) * gpf.broadcast_row_to_matrix(-n)
    
    #   wrt to c_1
    δlnAT_δc_1 = np.zeros((J,J))
    
    #   wrt to l
    δlnAT_δl = - δlnw_δlnL * gpf.broadcast_row_to_matrix(1 / l) + gpf.broadcast_row_to_matrix(δlnr_δlnL / l)
    
    #   wrt to x
    δlnAT_δx = gpf.make_diag(ζ/x) - δlnw_δX + gpf.broadcast_row_to_matrix(δlnr_δX)
    
    δlnAT = np.hstack((δlnAT_δc_0, δlnAT_δc_1, δlnAT_δl, δlnAT_δx))
    
    
    return np.vstack((δlnLS, δlnEE, δHB, δlnAT))













