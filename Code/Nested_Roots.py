"""""""""""
Nested Roots

Notes: Calibration roots for the nested-CES production block. Same structure as the calibration
    pieces of Roots.py (νRoot, GammaRoot) and Economy.Calibrate, with ρ following σ.

    At the status quo threshold x_j = var_κ * x_bar (θ = 0), with M_j ≡ Λ_kj + Λ_j x_j^(-ζ_j(σ-1)):
        P_j = (r/A_k) M_j^(1/(1-σ))
        A_k = r [Σ_j M_j^((1-ρ)/(1-σ))]^(1/(1-ρ))                      [from Σ_j P_j^(1-ρ) = 1]
        A_j = (w_j / x_j^ζ_j) / (r / A_k)                              [threshold rule]
        S_j = M_j^((σ-ρ)/(1-σ)) Λ_j x_j^(-ζ_j(σ-1)) / Σ_i M_i^((1-ρ)/(1-σ))

"""""""""""

import numpy as np
import scipy as sp
import Production_Functions as fn
import Nested_Production as npf
import Nested_Perturbations as npr
from Perturbations import δH_δθ as pr_δH_δθ #Production-free



def _M(x_j, x_bar, ζ, ν, σ):
    "Type Cost Aggregator"

    Λ_j = fn.Lambda_l(x_j, x_bar, ζ, ν, σ)
    Λ_kj = npf.Lambda_kj(x_j, ζ, ν, σ)

    return Λ_j, Λ_kj + Λ_j * x_j**(-ζ*(σ-1))



def νRoot(ν, Γ, χ, var_κ, x_bar, σ, ρ, S_j):
    "Absolute Advantage Root"

    ζ = fn.zeta(Γ, χ)
    x_j = var_κ * x_bar

    Λ_j, M = _M(x_j, x_bar, ζ, ν, σ)

    S_model = M**((σ-ρ)/(1-σ)) * Λ_j * x_j**(-ζ*(σ-1)) / np.sum(M**((1-ρ)/(1-σ)))

    return np.log(S_model) - np.log(S_j)



def TechRoots(ν, ζ, var_κ, x_bar, σ, ρ, w_j_sq, r_sq):
    "Capital- and Labor-Augmenting Technology"

    x_j = var_κ * x_bar
    Λ_j, M = _M(x_j, x_bar, ζ, ν, σ)

    A_k = r_sq * np.sum(M**((1-ρ)/(1-σ)))**(1/(1-ρ))
    A_j = (w_j_sq / x_j**ζ) / (r_sq / A_k)

    return A_k, A_j



def GammaRoot(Γ, E_sq, w_j_sq, r_sq, var_κ, Σ_k, χ, x_bar, σ, ρ, S_j, J, n, y_0, κ_0, Ψ, ψ, β, var_θ, ε, τ_k, δ, g, φ):
    "Automation Exposure Root"

    ζ = fn.zeta(Γ, χ)


    # ------------------------------------------- #
    # Compute Productivity and Absolute Advantage #
    # ------------------------------------------- #
    nu_g = np.ones(J) * var_κ

    nu = sp.optimize.root(νRoot, nu_g,
                  args=(Γ, χ, var_κ, x_bar, σ, ρ, S_j),
                  method='lm')

    ν = nu.x
    x_j = var_κ * x_bar

    A_k, A_j = TechRoots(ν, ζ, var_κ, x_bar, σ, ρ, w_j_sq, r_sq)


    # ---------------------------------- #
    # Compute Elasticity of Substitution #
    # ---------------------------------- #
    c_0 = E_sq[:J]
    l = E_sq[2*J:3*J]

    ΔH_ΔΕ = npr.δH_δclx(E_sq, 0, τ_k, Ψ, ψ, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0, β, var_θ, ε, δ, g, φ)
    ΔH_ΔIST = npr.δH_δIST(E_sq, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0, κ_0, Ψ, ψ, var_θ, ε, τ_k, δ, g, φ)

    dE = - np.linalg.solve(ΔH_ΔΕ, ΔH_ΔIST)

    dc_0 = dE[:J,0]
    dl = dE[2*J:3*J,0]
    dx = dE[3*J:,0]

    κ = y_0 - c_0
    K = np.sum(n * κ)
    dK = - np.sum(n * dc_0)

    δlnK = dK / K
    δlnY = npr.dlnY(dc_0, dl, dx, c_0, l, x_j, 0, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0)
    δlnr = npr.dlnr(dc_0, dl, dx, c_0, l, x_j, 0, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0)

    δlnS_k = δlnr + δlnK - δlnY

    RHS = 1 - δlnS_k / δlnr

    return np.log(Σ_k) - np.log(RHS)



def Eqbm_Root(E, θ, τ_k, Ψ, ψ, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0, β, var_θ, ε, δ, g, φ):
    "Equilibrium Root (Roots.Eqbm_Root with nested-CES factor prices)"

    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    x = E[3*J:]

    L = n * l
    K = np.sum(n * (y_0 - c_0))

    u = npf.Prices(x, L, K, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    w, r, Y, K_j, P = npf.Unpack(u, x, L, K, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    R_tilde = (1-τ_k)*(r-δ) - g

    D = np.sum(n * (w * l - Ψ * (w * l)**(1-ψ))) + τ_k * (r - δ) * K

    MRS_l = fn.lab_MRS(c_1, l, β, var_θ, φ, ε, g)
    MRS_c = fn.cap_MRS(c_0, c_1, β, var_θ, g)
    keep_l = fn.Heath_keep(w, l, Ψ, ψ)

    ln_LS = np.log(MRS_l) - np.log(keep_l) - np.log(w)
    ln_EE = np.log(MRS_c) - np.log(R_tilde)
    BC = R_tilde * c_0 + c_1 - (R_tilde * y_0 + Ψ * (w*l)**(1-ψ) + D)
    ln_AT = ζ * np.log(x) - ((np.log(w) - np.log(A_j)) - (np.log(1+θ) + np.log(r) - np.log(A_k)))

    return np.concatenate((ln_LS, ln_EE, BC, ln_AT))


def Eqbm_Solve(E0, args, tol=1e-10, maxit=50):
    """Damped Newton for Eqbm_Root with the analytic Jacobian (replaces sp.optimize.root 'hybr', which
    refactorizes the full 4J x 4J Jacobian by QR at every step). Returns an object with .x and .success."""

    def _F(E):
        try:
            return Eqbm_Root(E, *args)
        except RuntimeError: #Price solve failed at a trial point
            return None

    E = E0.copy()
    F = _F(E)
    for _ in range(maxit):
        if np.max(np.abs(F)) < tol:
            return sp.optimize.OptimizeResult(x=E, fun=F, success=True)
        step = np.linalg.solve(npr.δH_δclx(E, *args), -F)
        t = 1.0
        while t > 1e-6:
            Fn = _F(E + t*step)
            if Fn is not None and np.all(np.isfinite(Fn)) and np.linalg.norm(Fn) < np.linalg.norm(F):
                break
            t /= 2
        else:
            break
        E = E + t*step
        F = Fn

    return sp.optimize.OptimizeResult(x=E, fun=F, success=bool(np.max(np.abs(F)) < tol))



def Optimal_Para_Root(θ, τ_k, Ψ, ψ, digamma, E_sq, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0, β, var_θ, ε, δ, g, φ, _cache=None):
    "Root for Parametric Policy Tools ('theta' and 'tau' only; Roots.Optimal_Para_Root with nested-CES production)"

    warm = E_sq if (_cache is None or _cache[0] is None) else _cache[0]
    eargs = (θ, τ_k, Ψ, ψ, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0, β, var_θ, ε, δ, g, φ)

    Eqbm = Eqbm_Solve(warm, eargs)
    if not Eqbm.success and warm is not E_sq:
        Eqbm = Eqbm_Solve(E_sq, eargs)

    E = Eqbm.x
    if _cache is not None:
        _cache[0] = E

    c_0 = E[:J]
    c_1 = E[J:2*J]
    l = E[2*J:3*J]
    x = E[3*J:]

    ΔH_ΔΕ = npr.δH_δclx(E, *eargs)

    if digamma == 'theta':
        dE = - np.linalg.solve(ΔH_ΔΕ, pr_δH_δθ(θ, J))
    if digamma == 'tau':
        dE = - np.linalg.solve(ΔH_ΔΕ, npr.δH_δτ(E, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0, τ_k, δ, g))

    dc_0 = dE[:J,0]
    dl = dE[2*J:3*J,0]
    dx = dE[3*J:,0]

    L = n * l
    κ = y_0 - c_0
    K = np.sum(n * κ)

    δlnl = dl / l
    δlnκ = - dc_0 / κ

    u = npf.Prices(x, L, K, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    w, r, Y, K_j, P = npf.Unpack(u, x, L, K, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    cs = npf.CompStat(u, x, L, K, A_j, A_k, x_bar, ζ, ν, σ, ρ)

    δ_tilde = 1 + δ + g
    R = 1 + r - δ_tilde
    R_tilde = (1-τ_k)*(r-δ) - g

    δlnw = npr.dlnw(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0)
    δlnr = npr.dlnr(dc_0, dl, dx, c_0, l, x, θ, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0)
    δlnR_tilde = (1-τ_k) * r * δlnr / R_tilde

    δY_δX = Y * cs['dlnY_dX']   #Marginal product of the thresholds at the allocation (zero at θ = 0)

    MRS_l = fn.lab_MRS(c_1, l, β, var_θ, φ, ε, g)
    MRS_c = fn.cap_MRS(c_0, c_1, β, var_θ, g)
    λ = c_1**(-var_θ) / np.sum(n * c_1**(-var_θ))

    dT = 0 if digamma == 'theta' else np.sum(n * (λ-1) * (r - δ) * κ)

    MC = -np.sum(δY_δX * dx)
    cov = np.sum(n * (λ-1) * (MRS_l * l * δlnw + MRS_c * κ * δlnR_tilde))
    Expect = np.sum(n * ((w - MRS_l) * l * δlnl + (R - MRS_c) * κ * δlnκ))

    return dT + MC - cov - Expect
