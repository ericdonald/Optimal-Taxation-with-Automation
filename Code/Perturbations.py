"""""""""""
Perturbations

Notes: Functions that describe the perturbations of the economy.
    
"""""""""""

import numpy as np
from numba import njit
import Production_Functions as fn
import Processing_Functions as gpf
import Roots as rt



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
    R_tilde = (1-τ_k)*(r-δ) - g
        
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
def δObj_δX(E, n, Y_bar, δ, g, A_j, A_k, β, var_θ, φ, ε, J):
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
    
    δW = np.concatenate((δW_δc_0, δW_δc_1, δW_δl))
    
    return δW
    


@njit
def δEC_δX(E, w, r, n, Y_bar, δ, g, A_j, A_k, β, var_θ, φ, ε, J):
    "Jacobian of Equality Constraints"
        
    δ_tilde = 1 + δ + g
    
    
    # ------------------------------------------------ #
    # Derivatives of Second Period Resource Constraint #
    # ------------------------------------------------ #
    #   wrt to c_0
    δRC_1_δc_0 = (1 + r - δ_tilde) * n.reshape((1,J))
    
    #   wrt to c_1
    δRC_1_δc_1 = n.reshape((1,J))
    
    #   wrt to l
    δRC_1_δl = - (w * n).reshape((1,J))
    
    δRC_1 = np.hstack((δRC_1_δc_0, δRC_1_δc_1, δRC_1_δl))
        
        
    return δRC_1
    
    
 
@njit
def δIC_δX(E, w, n, Y_bar, δ, g, A_j, A_k, β, var_θ, φ, ε, J):
    "Jacobian of Penalty Inequality Constraints"
    
    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    
    IC_mat = rt.Inequal_Constr(E, w, n, Y_bar, δ, g, A_j, A_k, β, var_θ, φ, ε, J)
    viols = (IC_mat <= 0.0)
   
    # ----------------- #
    # Derivatives of IC #
    # ----------------- #
    grad = np.zeros(3*J)
    
    for i in range(J):
        for j in range(J):
            if viols[i,j] == True:
                
                #   wrt to c_0
                grad[i] += 2 * IC_mat[i,j] * c_0[i]**(-var_θ)
                grad[j] += 2 * IC_mat[i,j] * (-c_0[j]**(-var_θ))
                
                #   wrt to c_1
                grad[J+i] += 2 * IC_mat[i,j] * (β/(1-β*(1+g)**(1-var_θ))) * c_1[i]**(-var_θ)
                grad[J+j] += 2 * IC_mat[i,j] * (-β/(1-β*(1+g)**(1-var_θ))) * c_1[j]**(-var_θ)
    
                #   wrt to l
                grad[2*J+i] += 2 * IC_mat[i,j] * (-β/(1-β)) * φ[i] * l[i]**(1/ε)
                grad[2*J+j] += 2 * IC_mat[i,j] * (β/(1-β)) * φ[i] * l[j]**(1/ε) * (w[j]/w[i])**(1+1/ε)
                
    
    return grad

    

@njit
def obj_jac(E, w, Δ, args):
    "Jacobian of Penalized Objective"
    
    W_jac = δObj_δX(E, *args)
    grad_pen_IC = δIC_δX(E, w, *args)

    return -(W_jac - Δ * grad_pen_IC)



def δH_δclx_NL(θ, E, x, w, r, n, Y_bar, δ, g, A_j, A_k, β, var_θ, φ, ε, J, x_bar, ζ, ν, σ):
    "Jacobian of H wrt Mirrlees Allocation"
    
    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    
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











