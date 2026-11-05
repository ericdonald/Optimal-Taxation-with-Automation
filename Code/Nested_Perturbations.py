"""""""""""
Nested Perturbations

Notes: Perturbations of the economy under the nested-CES production block (see Nested_Production.py).
    Same structure as Perturbations.py: only the production derivatives change, everything built on
    top of them (tax, labor supply, Euler, budget blocks) is copied unchanged. The extra argument ρ
    follows σ in every signature.

"""""""""""

import numpy as np
import Production_Functions as fn
import Processing_Functions as gpf
import Nested_Production as npf



def _Production(E, J, n, y_0, A_j, A_k, x_bar, ζ, ν, σ, ρ):
    "Prices, output and comparative statics at an allocation"

    c_0 = E[:J]
    l = E[2*J:3*J]
    x = E[3*J:]

    L = n * l
    K = np.sum(n * (y_0 - c_0))

    u = npf.Prices(x, L, K, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    w, r, Y, K_j, P = npf.Unpack(u, x, L, K, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    cs = npf.CompStat(u, x, L, K, A_j, A_k, x_bar, ζ, ν, σ, ρ)

    return w, r, Y, K, L, cs



def δH_δclx(E, θ, τ_k, Ψ, ψ, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0, β, var_θ, ε, δ, g, φ):
    "Jacobian of H wrt Equilibrium Allocation"

    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    x = E[3*J:]

    w, r, Y, K, L, cs = _Production(E, J, n, y_0, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    R_tilde = (1-τ_k)*(r-δ) - g


    # ----------------------------- #
    # Production Derivatives (only #
    # block that differs from the   #
    # one-CES Jacobian)             #
    # ----------------------------- #
    δlnw_δX, δlnw_δlnK, δlnw_δlnL = cs['dlnw_dX'], cs['dlnw_dlnK'], cs['dlnw_dlnL']
    δlnr_δX, δlnr_δlnK, δlnr_δlnL = cs['dlnr_dX'], cs['dlnr_dlnK'], cs['dlnr_dlnL']


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
    δlnLS_δc_0 = - (1-ψ) * np.outer(δlnw_δlnK, -n/K)
    δlnLS_δc_1 = gpf.make_diag(1/c_1) * var_θ
    δlnLS_δl = gpf.make_diag(1/l) * (1/ε + ψ) - (1-ψ) * δlnw_δlnL * gpf.broadcast_row_to_matrix(1 / l)
    δlnLS_δx = - (1-ψ) * δlnw_δX

    δlnLS = np.hstack((δlnLS_δc_0, δlnLS_δc_1, δlnLS_δl, δlnLS_δx))


    # --------------------------------- #
    # Derivatives of Log Euler Equation #
    # --------------------------------- #
    δlnEE_δc_0 = - gpf.make_diag(1/c_0) * var_θ - ((1-τ_k) * r / R_tilde) * δlnr_δlnK / K * gpf.broadcast_row_to_matrix(-n)
    δlnEE_δc_1 = gpf.make_diag(1/c_1) * var_θ
    δlnEE_δl = - ((1-τ_k) * r / R_tilde) * gpf.broadcast_row_to_matrix(δlnr_δlnL / l)
    δlnEE_δx = - ((1-τ_k) * r / R_tilde) * gpf.broadcast_row_to_matrix(δlnr_δX)

    δlnEE = np.hstack((δlnEE_δc_0, δlnEE_δc_1, δlnEE_δl, δlnEE_δx))


    # ------------------------------- #
    # Derivatives of Household Budget #
    # ------------------------------- #
    δHB_δc_0 = np.eye(J)*R_tilde + (gpf.broadcast_col_to_matrix(c_0 - y_0) * (1-τ_k) * r * δlnr_δlnK / K
                              - gpf.broadcast_col_to_matrix(keep_l * y_l * δlnw_δlnK) / K
                              - δD_δK) * gpf.broadcast_row_to_matrix(-n)
    δHB_δc_1 = np.eye(J)
    δHB_δl = -gpf.make_diag(keep_l * w) + (gpf.broadcast_col_to_matrix(c_0 - y_0) * (1-τ_k) * r * gpf.broadcast_row_to_matrix(δlnr_δlnL / l)
                                                           - gpf.broadcast_col_to_matrix(keep_l * y_l) * δlnw_δlnL * gpf.broadcast_row_to_matrix(1 / l)
                                                           - gpf.broadcast_row_to_matrix(δD_δl))
    δHB_δx = (gpf.broadcast_col_to_matrix(c_0 - y_0) * (1-τ_k) * r * gpf.broadcast_row_to_matrix(δlnr_δX)
                  - gpf.broadcast_col_to_matrix(keep_l * y_l) * δlnw_δX
                  - gpf.broadcast_row_to_matrix(δD_δX))

    δHB = np.hstack((δHB_δc_0, δHB_δc_1, δHB_δl, δHB_δx))


    # ---------------------------------------- #
    # Derivatives of Log Automation Thresholds #
    # ---------------------------------------- #
    #   ln AT = ζ ln x_j - (ln w_j - ln r) + ln(1+θ) + const, so every block is minus the
    #   derivative of (ln w_j - ln r); this is the one-CES block with 1/σ terms replaced.
    δlnAT_δc_0 = np.outer(δlnw_δlnK - δlnr_δlnK, n) / K
    δlnAT_δc_1 = np.zeros((J,J))
    δlnAT_δl = - (δlnw_δlnL - gpf.broadcast_row_to_matrix(δlnr_δlnL)) / gpf.broadcast_row_to_matrix(l)
    δlnAT_δx = gpf.make_diag(ζ/x) - (δlnw_δX - gpf.broadcast_row_to_matrix(δlnr_δX))

    δlnAT = np.hstack((δlnAT_δc_0, δlnAT_δc_1, δlnAT_δl, δlnAT_δx))


    return np.vstack((δlnLS, δlnEE, δHB, δlnAT))



def δH_δIST(E, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0, κ_0, Ψ, ψ, var_θ, ε, τ_k, δ, g, φ):
    "Jacobian of H wrt Investment Price"

    c_0 = E[:J]

    w, r, Y, K, L, cs = _Production(E, J, n, y_0, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    R_tilde = (1-τ_k)*(r-δ) - g

    δD_0 = τ_k * δ * K / (1+g)
    δD_1 = τ_k * δ * K

    δy_0 = δD_0 - (1 - (1-τ_k)*δ) * κ_0.reshape((J,1))

    δlnLS = np.zeros((J,1))
    δlnEE = - np.ones((J,1)) * ((1-τ_k) * r / R_tilde)
    δHB = ((c_0 - y_0).reshape((J,1)) * (1-τ_k) * r
                  - R_tilde * δy_0
                  - δD_1)
    δlnAT = np.zeros((J,1))

    return np.vstack((δlnLS, δlnEE, δHB, δlnAT))



def dlnY(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0):
    "Total Derivative of Log Output"

    E = np.concatenate((c_0, np.zeros(J), l, x))
    w, r, Y, K, L, cs = _Production(E, J, n, y_0, A_j, A_k, x_bar, ζ, ν, σ, ρ)

    δK = - np.sum(n * dc_0)
    δL = n * dl

    return np.sum(cs['dlnY_dX'] * dx) + cs['dlnY_dlnK'] * δK / K + np.sum(cs['dlnY_dlnL'] * δL / L)



def dlnr(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0):
    "Total Derivative of Log Rent"

    E = np.concatenate((c_0, np.zeros(J), l, x))
    w, r, Y, K, L, cs = _Production(E, J, n, y_0, A_j, A_k, x_bar, ζ, ν, σ, ρ)

    δK = - np.sum(n * dc_0)
    δL = n * dl

    return np.sum(cs['dlnr_dX'] * dx) + cs['dlnr_dlnK'] * δK / K + np.sum(cs['dlnr_dlnL'] * δL / L)



def δH_δτ(E, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0, τ_k, δ, g):
    "Jacobian of H wrt Capital Tax"

    c_0 = E[:J]

    w, r, Y, K, L, cs = _Production(E, J, n, y_0, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    κ = y_0 - c_0
    R_tilde = (1-τ_k)*(r-δ) - g

    dD = (r - δ) * K

    δlnLS = np.zeros((J,1))
    δlnEE = np.ones((J,1)) * (r-δ) / R_tilde
    δHB = (r-δ) * κ.reshape((J,1)) - dD
    δlnAT = np.zeros((J,1))

    return np.vstack((δlnLS, δlnEE, δHB, δlnAT))



def dlnw(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0):
    "Total Derivative of Log Wages"

    E = np.concatenate((c_0, np.zeros(J), l, x))
    w, r, Y, K, L, cs = _Production(E, J, n, y_0, A_j, A_k, x_bar, ζ, ν, σ, ρ)

    δK = - np.sum(n * dc_0)
    δL = n * dl

    return cs['dlnw_dX'] @ dx + cs['dlnw_dlnK'] * δK / K + cs['dlnw_dlnL'] @ (δL / L)
