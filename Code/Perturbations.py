"""""""""""
Perturbations

Notes:
    
Output:
"""""""""""

import numpy as np
from numba import njit
import Functions as fn



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
    
    'Output Derivatives'
    α_k = x**(ζ*(ν-1))
    
    if σ == 1:
        θ_wedge = np.log(1+θ)
    else:
        θ_wedge = ((1+θ)**(1-σ) - 1) / (1 - σ)
    
    δlnY_δX = (A_k * α_k / r)**(σ-1) * θ_wedge
    δlnY_δlnK = S_k
    δlnY_δlnL = S_j
    
    'Wage Derivatives'
    rel_l = fn.relα(x, x_bar, ζ, ν, σ, 0)
    
    δlnw_δX = - (1/σ) * fn.make_diag(rel_l) + (1/σ) * fn.broadcast_row_to_matrix(δlnY_δX)
    δlnw_δlnK = (1/σ) * δlnY_δlnK * np.ones(J)
    δlnw_δlnL = (1/σ) * fn.broadcast_row_to_matrix(δlnY_δlnL) - np.eye(J) / σ
    
    'Rent Derivatives'
    rel_k = fn.relα(x, x_bar, ζ, ν, σ, 1)
    
    δlnr_δX = (1/σ) * rel_k + (1/σ) * δlnY_δX
    δlnr_δlnK = (1/σ) * (δlnY_δlnK - 1)
    δlnr_δlnL = (1/σ) * δlnY_δlnL
    
    'Lump-Sum'    
    δD_δX = (n * (w * l - Ψ * (1-ψ) * (w * l)**(1-ψ))).reshape((1,J)) @ δlnw_δX + τ_k * r * K * δlnr_δX
    δD_δK = np.sum(n * (w * l - Ψ * (1-ψ) * (w * l)**(1-ψ)) * δlnw_δlnK) / K + τ_k * r * δlnr_δlnK + τ_k * (r - δ)
    δD_δL = (w * l - Ψ * (1-ψ) * (w * l)**(1-ψ)) / L + (((n * (w * l - Ψ * (1-ψ) * (w * l)**(1-ψ))).reshape((1,J)) @ δlnw_δlnL) / L
                                                            + τ_k * r * K * δlnr_δlnL / L)
    δD_δX = δD_δX.reshape(J)
    δD_δL = δD_δL.reshape(J)
    
    
    'Derivatives of Log Labor Supply Condition'
    '   wrt to c_0'
    δlnLS_δc_0 = - (1-ψ) * np.outer(δlnw_δlnK, -n/K)
    
    '   wrt to c_1'
    δlnLS_δc_1 = fn.make_diag(1/c_1) * var_θ
    
    '   wrt to l'
    δlnLS_δl = fn.make_diag(1/l) * (1/ε + ψ) - (1-ψ) * δlnw_δlnL * fn.broadcast_row_to_matrix(n / L)
    
    '   wrt to x'
    δlnLS_δx =  - (1-ψ) * δlnw_δX
    
    δlnLS = np.hstack((δlnLS_δc_0, δlnLS_δc_1, δlnLS_δl, δlnLS_δx))
    
    
    'Derivatives of Log Euler Equation'
    '   wrt to c_0'
    δlnEE_δc_0 = - fn.make_diag(1/c_0) * var_θ - ((1-τ_k) * r / R) * δlnr_δlnK / K * fn.broadcast_row_to_matrix(-n)
    
    '   wrt to c_1'
    δlnEE_δc_1 = fn.make_diag(1/c_1) * var_θ
    
    '   wrt to l'
    δlnEE_δl = - ((1-τ_k) * r / R) * fn.broadcast_row_to_matrix(δlnr_δlnL * n / L)
    
    '   wrt to x'
    δlnEE_δx = - ((1-τ_k) * r / R) * fn.broadcast_row_to_matrix(δlnr_δX)
    
    δlnEE = np.hstack((δlnEE_δc_0, δlnEE_δc_1, δlnEE_δl, δlnEE_δx))
    
    
    'Derivatives of Household Budget'
    '   wrt to c_0'
    δHB_δc_0 = np.eye(J)*R + (fn.broadcast_col_to_matrix(c_0 - y_0) * (1-τ_k) * r * δlnr_δlnK / K 
                              - fn.broadcast_col_to_matrix(Ψ * (1-ψ) * (w * l)**(1-ψ) * δlnw_δlnK) / K
                              - δD_δK) * fn.broadcast_row_to_matrix(-n)
    
    '   wrt to c_1'
    δHB_δc_1 = np.eye(J)
    
    '   wrt to l'
    δHB_δl = -fn.make_diag(Ψ * (1-ψ) * (w * l)**(1-ψ) / l) + (fn.broadcast_col_to_matrix(c_0 - y_0) * (1-τ_k) * r * fn.broadcast_row_to_matrix(δlnr_δlnL / L)
                                                           - fn.broadcast_col_to_matrix(Ψ * (1-ψ) * (w * l)**(1-ψ)) * δlnw_δlnL * fn.broadcast_row_to_matrix(1 / L)
                                                           - fn.broadcast_row_to_matrix(δD_δL) * fn.broadcast_row_to_matrix(n))
    
    '   wrt to x'
    δHB_δx = (fn.broadcast_col_to_matrix(c_0 - y_0) * (1-τ_k) * r * fn.broadcast_row_to_matrix(δlnr_δX)
                  - fn.broadcast_col_to_matrix(Ψ * (1-ψ) * (w * l)**(1-ψ)) * δlnw_δX
                  - fn.broadcast_row_to_matrix(δD_δX))
    
    δHB = np.hstack((δHB_δc_0, δHB_δc_1, δHB_δl, δHB_δx))
    
    'Derivatives of Log Automation Thresholds'
    '   wrt to c_0'
    δlnAT_δc_0 = (- fn.broadcast_col_to_matrix(δlnw_δlnK) / K + δlnr_δlnK / K) * fn.broadcast_row_to_matrix(-n)
    
    '   wrt to c_1'
    δlnAT_δc_1 = np.zeros((J,J))
    
    '   wrt to l'
    δlnAT_δl = - δlnw_δlnL * fn.broadcast_row_to_matrix(n / L) + fn.broadcast_row_to_matrix(δlnr_δlnL * n / L)
    
    '   wrt to x'
    δlnAT_δx = fn.make_diag(ζ/x) - δlnw_δX + fn.broadcast_row_to_matrix(δlnr_δX)
    
    δlnAT = np.hstack((δlnAT_δc_0, δlnAT_δc_1, δlnAT_δl, δlnAT_δx))
    
    
    return np.vstack((δlnLS, δlnEE, δHB, δlnAT))



@njit
def δH_δθ(θ, J):
    "Jacobian of H wrt Threshold Rule"
    
    'Derivative of Log Labor Supply Condition'
    δlnLS = np.zeros((J,1))
    
    
    'Derivative of Log Euler Equation'
    δlnEE = np.zeros((J,1))
    
    
    'Derivative of Household Budget'
    δHB = np.zeros((J,1))
    
    
    'Derivative of Log Automation Thresholds'
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
        
    'Output Derivative'
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
    
    'Output Derivative'
    δlnY = dlnY(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0)

    'Wage Derivatives'
    rel_l = fn.relα(x, x_bar, ζ, ν, σ, 0)
    
    δlnw = (1/σ) * (- rel_l * dx + δlnY - δL / L)
    
    return δlnw


    
@njit    
def dlnr(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0):
    "Total Derivative of Log Rent"

    K = np.sum(n * (y_0 - c_0))
    δK = - np.sum(n * dc_0)
        
    'Output Derivative'
    δlnY = dlnY(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, J, n, y_0)

    'Rent Derivative'
    rel_k = fn.relα(x, x_bar, ζ, ν, σ, 1)
    
    δlnr = (1/σ) * (np.sum(rel_k * dx) + δlnY - δK / K)
        
    return δlnr  



@njit
def δObj_δX(X, J, n, Y_bar, δ, g, A_j, A_k, x_bar, ζ, ν, σ, β, var_θ, φ, ε, IC_Comp_J, IC_count, TR):
    "Jacobian of Mirrlees Objective"
    
    c_0 = X[:J]
    c_1 = X[J:2*J]
    l = X[2*J:3*J]
    
    'Derivatives of Welfare'
    '   wrt to c_0'
    δW_δc_0 = n * c_0**(-var_θ)
    
    '   wrt to c_1'
    δW_δc_1 = n * (β / (1-β)) *  c_1**(-var_θ)
    
    '   wrt to l'
    δW_δl = - n * (β / (1-β)) * φ * l**(1/ε)
    
    '   wrt to x'
    δW_δx = np.zeros(J)
    
    '   wrt to K'
    δW_δK = np.zeros(1)
    
    '   wrt to θ'
    δW_δθ = np.zeros(1)
    
    if TR==1:
        δW = np.concatenate((δW_δc_0, δW_δc_1, δW_δl, δW_δx, δW_δK, δW_δθ))
    else:
        δW = np.concatenate((δW_δc_0, δW_δc_1, δW_δl, δW_δx, δW_δK))
    
    return - δW
    


@njit
def δEC_δX(X, J, n, Y_bar, δ, g, A_j, A_k, x_bar, ζ, ν, σ, β, var_θ, φ, ε, IC_Comp_J, IC_count, TR):
    "Jacobian of Equality Constraints"
    
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
    
    S_k = r * K / Y
    S_j = w * L / Y
    α_k = x**(ζ*(ν-1))
    
    if σ == 1:
        θ_wedge = np.log(1+θ)
    else:
        θ_wedge = ((1+θ)**(1-σ) - 1) / (1 - σ)
    
    δY_δX = Y * (A_k * α_k / r)**(σ-1) * θ_wedge
    δlnY_δX = δY_δX / Y
    δlnY_δlnK = S_k
    δlnY_δlnL = S_j
    
    rel_l = fn.relα(x, x_bar, ζ, ν, σ, 0)
    
    δlnw_δX = - (1/σ) * fn.make_diag(rel_l) + (1/σ) * fn.broadcast_row_to_matrix(δlnY_δX)
    δlnw_δlnK = (1/σ) * δlnY_δlnK * np.ones(J)
    δlnw_δlnL = (1/σ) * fn.broadcast_row_to_matrix(δlnY_δlnL) - np.eye(J) / σ
    
    rel_k = fn.relα(x, x_bar, ζ, ν, σ, 1)
    
    δlnr_δX = (1/σ) * rel_k + (1/σ) * δlnY_δX
    δlnr_δlnK = (1/σ) * (δlnY_δlnK - 1)
    δlnr_δlnL = (1/σ) * δlnY_δlnL
    
    'Derivatives of Initial Period Resource Constraint'
    '   wrt to c_0'
    δRC_0_δc_0 = n.reshape((1,J))
    
    '   wrt to c_1'
    δRC_0_δc_1 = np.zeros((1,J))
    
    '   wrt to l'
    δRC_0_δl = np.zeros((1,J))
    
    '   wrt to x'
    δRC_0_δx = np.zeros((1,J))
    
    '   wrt to K'
    δRC_0_δK = np.ones((1,1))
    
    '   wrt to θ'
    δRC_0_δθ = np.zeros((1,1))
    
    if TR==1:
        δRC_0 = np.hstack((δRC_0_δc_0, δRC_0_δc_1, δRC_0_δl, δRC_0_δx, δRC_0_δK, δRC_0_δθ))
    else:
        δRC_0 = np.hstack((δRC_0_δc_0, δRC_0_δc_1, δRC_0_δl, δRC_0_δx, δRC_0_δK))
        
    
    'Derivatives of Second Period Resource Constraint'
    '   wrt to c_0'
    δRC_1_δc_0 = np.zeros((1,J))
    
    '   wrt to c_1'
    δRC_1_δc_1 = n.reshape((1,J))
    
    '   wrt to l'
    δRC_1_δl = - (w * n).reshape((1,J))
    
    '   wrt to x'
    δRC_1_δx = - δY_δX.reshape((1,J))
    
    '   wrt to K'
    δRC_1_δK = - np.ones((1,1)) * (1 + r - δ_hat)
    
    '   wrt to θ'
    δRC_1_δθ = np.zeros((1,1))
    
    if TR==1:
        δRC_1 = np.hstack((δRC_1_δc_0, δRC_1_δc_1, δRC_1_δl, δRC_1_δx, δRC_1_δK, δRC_1_δθ))
    else:
        δRC_1 = np.hstack((δRC_1_δc_0, δRC_1_δc_1, δRC_1_δl, δRC_1_δx, δRC_1_δK))
        
    
    'Derivatives of Log Automation Thresholds'
    '   wrt to c_0'
    δlnAT_δc_0 = np.zeros((J,J))
    
    '   wrt to c_1'
    δlnAT_δc_1 = np.zeros((J,J))
    
    '   wrt to l'
    δlnAT_δl = - δlnw_δlnL * fn.broadcast_row_to_matrix(n / L) + fn.broadcast_row_to_matrix(δlnr_δlnL * n / L)
    
    '   wrt to x'
    δlnAT_δx = fn.make_diag(ζ/x) - δlnw_δX + fn.broadcast_row_to_matrix(δlnr_δX)
    
    '   wrt to K'
    δlnAT_δK = - (δlnw_δlnK).reshape((J,1)) / K + δlnr_δlnK / K * np.ones((J,1))
    
    '   wrt to θ'
    δlnAT_δθ = np.ones((J,1)) / (1+θ)
    
    if TR==1:
        δlnAT = np.hstack((δlnAT_δc_0, δlnAT_δc_1, δlnAT_δl, δlnAT_δx, δlnAT_δK, δlnAT_δθ))
    else:
        δlnAT = np.hstack((δlnAT_δc_0, δlnAT_δc_1, δlnAT_δl, δlnAT_δx, δlnAT_δK))
        
        
    return np.vstack((δRC_0, δRC_1, δlnAT))
    
    
 
@njit
def δIC_δX(X, J, n, Y_bar, δ, g, A_j, A_k, x_bar, ζ, ν, σ, β, var_θ, φ, ε, IC_Comp_J, IC_count, TR):
    "Jacobian of Inequality Constraints"
    
    c_0 = X[:J]
    c_1 = X[J:2*J]
    l = X[2*J:3*J]
    x = X[3*J:4*J]
    K = X[4*J]
    
    L = n * l
    w = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    rel_l = fn.relα(x, x_bar, ζ, ν, σ, 0)
    
    
    'Derivatives of IC'
    '   wrt to c_0'
    δIC_δc_0 = np.zeros((IC_count,J))
    k = 0
    
    for i in range(J):
        for j in IC_Comp_J[i]:
            δIC_δc_0[k, i] = c_0[i]**(-var_θ)
            δIC_δc_0[k, j] = -c_0[j]**(-var_θ)
            k += 1
   
    '   wrt to c_1'
    δIC_δc_1 = np.zeros((IC_count,J))
    k = 0
    
    for i in range(J):
        for j in IC_Comp_J[i]:
            δIC_δc_1[k, i] = (β/(1-β)) * c_1[i]**(-var_θ)
            δIC_δc_1[k, j] = -(β/(1-β)) * c_1[j]**(-var_θ)
            k += 1
    
    '   wrt to l'
    δIC_δl = np.zeros((IC_count,J))
    k = 0
    
    for i in range(J):
        for j in IC_Comp_J[i]:
            δIC_δl[k, i] = (β/(1-β)) * φ[i] * l[i]**(1/ε) * ((1/σ) * ((w[j]*l[j])/(w[i]*l[i]))**(1+1/ε) - 1)
            δIC_δl[k, j] = (β/(1-β)) * ((σ-1)/σ) * φ[i] * l[j]**(1/ε) * (w[j]/w[i])**(1+1/ε)
            k += 1
    
    '   wrt to x'
    δIC_δx = np.zeros((IC_count,J))
    k = 0
    
    for i in range(J):
        for j in IC_Comp_J[i]:
            δIC_δx[k, i] = (β/(1-β)) * (φ[i]/σ) * (w[j]*l[j]/w[i])**(1+1/ε) * rel_l[i]
            δIC_δx[k, j] = -(β/(1-β)) * (φ[i]/σ) * (w[j]*l[j]/w[i])**(1+1/ε) * rel_l[j]
            k += 1
    
    '   wrt to K'
    δIC_δK = np.zeros((IC_count,1))
    
    '   wrt to θ'
    δIC_δθ = np.zeros((IC_count,1))
    
    if TR==1:
        δIC = np.hstack((δIC_δc_0, δIC_δc_1, δIC_δl, δIC_δx, δIC_δK, δIC_δθ))
    else:
        δIC = np.hstack((δIC_δc_0, δIC_δc_1, δIC_δl, δIC_δx, δIC_δK))
    
    return δIC
    
    



