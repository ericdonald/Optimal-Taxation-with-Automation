"""""""""""
Nested Production Functions

Notes: Production block for the nested-CES robustness check. Type goods Y_j are CES over tasks
    (elasticity σ) and final output is CES over type goods (elasticity ρ). ρ = σ recovers
    Production_Functions.py. Final good is the numeraire.

    Unknowns u = (ln w_j, ln r, ln Y); equations (in logs):
        (c)  L_j = Y P_j^(σ-ρ) Λ_j w_j^(-σ) A_j^(σ-1)                  [labor demand, J eqs]
        (d)  K   = Y r^(-σ) A_k^(σ-1) Σ_j P_j^(σ-ρ) Λ_kj                [capital market]
        (b)  Σ_j P_j^(1-ρ) = 1                                          [ideal price index]
    with P_j^(1-σ) = Λ_kj (r/A_k)^(1-σ) + Λ_j (w_j/A_j)^(1-σ)           [(a), type unit cost]

"""""""""""

import numpy as np
import Production_Functions as fn



def Lambda_kj(x_j, ζ, ν, σ, x_underbar=1e-7):
    "Type-level Capital Task Integral (summand of fn.Lambda_k)"

    e = ζ*(ν-1)*(σ-1) + 1

    return (x_j**e - x_underbar**e) / e



def _blocks(u, x, A_j, A_k, x_bar, ζ, ν, σ, ρ):
    "Prices and shares implied by u"

    J = len(x)
    lnw = u[:J]
    lnr = u[J]

    Λ_j = fn.Lambda_l(x, x_bar, ζ, ν, σ)
    Λ_kj = Lambda_kj(x, ζ, ν, σ)
    dΛ_j = -x**(ζ*ν*(σ-1))                  #dΛ_j/dx_j
    dΛ_kj = x**(ζ*(ν-1)*(σ-1))              #dΛ_kj/dx_j

    cap = Λ_kj * np.exp((1-σ)*(lnr - np.log(A_k)))      #Capital cost term of P_j^(1-σ)
    lab = Λ_j * np.exp((1-σ)*(lnw - np.log(A_j)))       #Labor cost term of P_j^(1-σ)
    P1σ = cap + lab
    lnP = np.log(P1σ) / (1-σ)

    a_k = cap / P1σ                         #Capital cost share in type j
    a_l = lab / P1σ

    #dlnP_j/dx_j at fixed factor prices
    c_x = (dΛ_kj * np.exp((1-σ)*(lnr - np.log(A_k))) + dΛ_j * np.exp((1-σ)*(lnw - np.log(A_j)))) / (P1σ * (1-σ))

    return Λ_j, Λ_kj, dΛ_j, dΛ_kj, lnP, a_k, a_l, c_x



def _resid_jac(u, lnL, lnK, x, A_j, A_k, x_bar, ζ, ν, σ, ρ):
    "Residuals of (c), (d), (b) in logs and Jacobian wrt u"

    J = len(x)
    lnw = u[:J]
    lnY = u[J+1]

    Λ_j, Λ_kj, dΛ_j, dΛ_kj, lnP, a_k, a_l, c_x = _blocks(u, x, A_j, A_k, x_bar, ζ, ν, σ, ρ)

    ω_raw = np.exp((σ-ρ)*lnP) * Λ_kj
    ω = ω_raw / np.sum(ω_raw)               #K_j / K
    π = np.exp((1-ρ)*lnP)                   #P_j^(1-ρ), sums to 1

    F = np.empty(J+2)
    F[:J] = lnY + (σ-ρ)*lnP + np.log(Λ_j) - σ*lnw + (σ-1)*np.log(A_j) - lnL
    F[J] = lnY + np.log(np.sum(ω_raw)) - σ*u[J] + (σ-1)*np.log(A_k) - lnK
    F[J+1] = np.log(np.sum(π))

    Ju = np.zeros((J+2, J+2))
    Ju[:J, :J] = np.diag((σ-ρ)*a_l - σ)
    Ju[:J, J] = (σ-ρ)*a_k
    Ju[:J, J+1] = 1
    Ju[J, :J] = ω * (σ-ρ) * a_l
    Ju[J, J] = (σ-ρ) * np.sum(ω*a_k) - σ
    Ju[J, J+1] = 1
    Ju[J+1, :J] = (1-ρ) * π * a_l / np.sum(π)
    Ju[J+1, J] = (1-ρ) * np.sum(π*a_k) / np.sum(π)

    return F, Ju



def Prices(x, L, K, A_j, A_k, x_bar, ζ, ν, σ, ρ, u0=None, tol=1e-12, maxit=100):
    "Solve for (w, r, Y) given (x, L, K). Returns u = (ln w, ln r, ln Y) and K_j, P_j"

    J = len(x)
    if u0 is None: #One-CES prices as starting guess
        w0 = fn.Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
        r0 = fn.Rents(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
        Y0 = fn.Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
        u = np.concatenate((np.log(w0), [np.log(r0)], [np.log(Y0)]))
    else:
        u = u0.copy()

    lnL, lnK = np.log(L), np.log(K)
    for it in range(maxit):
        F, Ju = _resid_jac(u, lnL, lnK, x, A_j, A_k, x_bar, ζ, ν, σ, ρ)
        if np.max(np.abs(F)) < tol:
            break
        step = np.linalg.solve(Ju, -F)
        t = 1.0
        while t > 1e-4: #Backtracking
            Fn, _ = _resid_jac(u + t*step, lnL, lnK, x, A_j, A_k, x_bar, ζ, ν, σ, ρ)
            if np.linalg.norm(Fn) < np.linalg.norm(F): break
            t /= 2
        u = u + t*step
    else:
        raise RuntimeError('Nested price solve did not converge')

    return u



def Unpack(u, x, L, K, A_j, A_k, x_bar, ζ, ν, σ, ρ):
    "Prices, output, type-level capital and shares from u"

    J = len(x)
    w, r, Y = np.exp(u[:J]), np.exp(u[J]), np.exp(u[J+1])
    Λ_j, Λ_kj, dΛ_j, dΛ_kj, lnP, a_k, a_l, c_x = _blocks(u, x, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    P = np.exp(lnP)
    K_j = Y * P**(σ-ρ) * Λ_kj * r**(-σ) * A_k**(σ-1)

    return w, r, Y, K_j, P



def CompStat(u, x, L, K, A_j, A_k, x_bar, ζ, ν, σ, ρ):
    """Comparative statics of (ln w, ln r) wrt (ln L, ln K, x) by the implicit function theorem.
    Matrix convention follows Perturbations.py: entry [i, j] = ∂ln w_i / ∂(·)_j."""

    J = len(x)
    lnL, lnK = np.log(L), np.log(K)
    F, Ju = _resid_jac(u, lnL, lnK, x, A_j, A_k, x_bar, ζ, ν, σ, ρ)

    Λ_j, Λ_kj, dΛ_j, dΛ_kj, lnP, a_k, a_l, c_x = _blocks(u, x, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    ω_raw = np.exp((σ-ρ)*lnP) * Λ_kj
    ω = ω_raw / np.sum(ω_raw)
    π = np.exp((1-ρ)*lnP) / np.sum(np.exp((1-ρ)*lnP))

    #∂F/∂lnL (only labor-demand rows, -1 on diagonal)
    FL = np.zeros((J+2, J)); FL[:J, :] = -np.eye(J)
    #∂F/∂lnK
    FK = np.zeros(J+2); FK[J] = -1
    #∂F/∂x
    Fx = np.zeros((J+2, J))
    Fx[:J, :] = np.diag((σ-ρ)*c_x + dΛ_j/Λ_j)
    Fx[J, :] = ω * ((σ-ρ)*c_x + dΛ_kj/Λ_kj)
    Fx[J+1, :] = (1-ρ) * π * c_x

    du_dL = -np.linalg.solve(Ju, FL)        #(J+2) x J
    du_dK = -np.linalg.solve(Ju, FK)        #(J+2)
    du_dx = -np.linalg.solve(Ju, Fx)        #(J+2) x J

    return {'dlnw_dlnL': du_dL[:J], 'dlnw_dlnK': du_dK[:J], 'dlnw_dX': du_dx[:J],
            'dlnr_dlnL': du_dL[J], 'dlnr_dlnK': du_dK[J], 'dlnr_dX': du_dx[J],
            'dlnY_dlnL': du_dL[J+1], 'dlnY_dlnK': du_dK[J+1], 'dlnY_dX': du_dx[J+1]}



def Sigma_j(u, x, A_j, A_k, x_bar, ζ, ν, σ, ρ):
    """Type-j elasticity of substitution with capital (Appendix A):
        Σ_j = σ + ∂ ln( Λ_j P_j^(σ-ρ) / Σ_i Λ_ki P_i^(σ-ρ) ) / ∂ ln(r / w_j).
    The perturbation moves w_j alone, so w_j/r is the only relative factor price that changes. Three
    things respond: type j's threshold (ζ_j dln x_j = dln w_j - dln r, other thresholds fixed),
    and through it Λ_j and Λ_kj, and the type price P_j. All derivatives are per unit dln w_j, and
    ∂/∂ln(r/w_j) = - d/dln w_j. At ρ = σ this equals σ + (z_j + z_jk)/ζ."""

    J = len(x)
    Λ_j, Λ_kj, dΛ_j, dΛ_kj, lnP, a_k, a_l, c_x = _blocks(u, x, A_j, A_k, x_bar, ζ, ν, σ, ρ)

    dlnx = 1 / ζ                                        #dln x_j per unit dln w_j
    dlnΛ_j = dΛ_j * x * dlnx / Λ_j
    dlnΛ_kj = dΛ_kj * x * dlnx / Λ_kj
    dlnP_j = a_l + c_x * x * dlnx                       #Direct cost-share effect plus threshold effect

    wgt = Λ_kj * np.exp((σ-ρ)*lnP)
    ω = wgt / np.sum(wgt)                               #Capital weight of type j in the denominator (K_j/K)

    dln_num = dlnΛ_j + (σ-ρ) * dlnP_j                   #ln(Λ_j P_j^(σ-ρ))
    dln_den = ω * (dlnΛ_kj + (σ-ρ) * dlnP_j)            #ln Σ_i Λ_ki P_i^(σ-ρ); only the i = j term moves

    return σ - (dln_num - dln_den)
