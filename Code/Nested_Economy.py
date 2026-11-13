"""""""""""
Nested Economy Module

Notes: Economy with nested-CES production (type goods CES over tasks with elasticity σ, final output
    CES over type goods with elasticity ρ). Everything other than the production calibration is
    inherited from Economy.

"""""""""""

import numpy as np
import scipy as sp
import quantecon as qe
import Economy as ec
import Production_Functions as fn
import Nested_Production as npf
import Processing_Functions as gpf
import Nested_Roots as nrt



class Nested_Economy(ec.Economy):

    def __init__(self, J, ρ, groups=None):
        "Initialize Nested Economy Object"

        super().__init__(J, groups)
        self.ρ = ρ #Cross-Type Elasticity of Substitution



    def Calibrate(self):
        "Calibrate Parameters and Status Quo Allocation"

        super().Calibrate() #Allocation block is unchanged; the one-CES production block it ends with is overwritten below


        # ---------------------------------- #
        # Task-Level Productivity Parameters #
        # ---------------------------------- #
        K_0 = self.K_sq / (1+self.g)
        κ_0 = self.S_jk_sq * K_0 / self.n

        E_sq = np.concatenate((self.c_0_sq, self.c_1_sq, self.l_j_sq, self.var_κ * self.x_bar))
        args = (E_sq, self.w_j_sq, self.r_sq, self.var_κ, self.Σ_k, self.χ, self.x_bar, self.σ, self.ρ, self.S_j_sq, self.J, self.n, self.y_0, κ_0, self.Ψ, self.ψ, self.β, self.var_θ, self.ε, self.τ_k, self.δ, self.g, self.φ)

        a = 0
        b = -np.log(1000) / 100 #Lower bound for ζ of 1/1000

        Gamma_Σ = lambda x: nrt.GammaRoot(x, *args)
        Γ = gpf.secant_scalar(Gamma_Σ, a, b)

        self.Γ = Γ
        self.ζ = np.exp(self.Γ * self.χ)

        nu_g = np.ones(self.J) * self.var_κ

        nu = sp.optimize.root(nrt.νRoot, nu_g,
                      args=(self.Γ, self.χ, self.var_κ, self.x_bar, self.σ, self.ρ, self.S_j_sq),
                      method='lm')

        self.ν = nu.x

        self.A_k, self.A_j = nrt.TechRoots(self.ν, self.ζ, self.var_κ, self.x_bar, self.σ, self.ρ, self.w_j_sq, self.r_sq)




    def Para_Solver(self, θ_on, τ_on, y_0=None, E=None, θ_init=0.0, τ_k_init=None, damp=2/3, tol=1e-4, max_iter=10_000):
        "Solve Parametric Policy Problem (Economy.Para_Solver with nested-CES roots; labor tax tools not included)"

        if y_0 is None:
            y_0 = self.y_0
        args = (self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ, self.ρ, self.J, self.n, y_0, self.β, self.var_θ, self.ε, self.δ, self.g, self.φ)

        if E is None:
            E = np.concatenate((self.c_0_sq, self.c_1_sq, self.l_j_sq, self.var_κ * self.x_bar))
        θ = θ_init
        τ_k = self.τ_k if τ_k_init is None else τ_k_init
        Ψ = self.Ψ
        ψ = self.ψ

        qe.tic()
        for _ in range(max_iter):

            cache_θ = [E.copy()]
            cache_τ = [E.copy()]

            if θ_on == 1:
                Optimal_θ_Root = lambda x: nrt.Optimal_Para_Root(x, τ_k, Ψ, ψ, 'theta', E, *args, _cache=cache_θ)
                θ_lower = θ/2
                θ_upper = np.maximum(θ * 1.5, 1/3)
                θ_new = gpf.secant_scalar(Optimal_θ_Root, θ_lower, θ_upper, lb=-0.999, ub=2, expansion='additive')
            else:
                θ_new = 0

            if τ_on == 1:
                Optimal_τ_Root = lambda x: nrt.Optimal_Para_Root(θ, x, Ψ, ψ, 'tau', E, *args, _cache=cache_τ)
                τ_new = gpf.secant_scalar(Optimal_τ_Root, τ_k/2, τ_k * 1.5, ub=0.5, expansion='additive') #Equilibrium (capital) collapses for τ_k above about 0.5, where the price solve fails
            else:
                τ_new = self.τ_k

            if θ_on + τ_on == 1:
                θ = θ_new
                τ_k = τ_new
                break

            error_θ = np.abs(θ - θ_new)
            error_τ = np.abs(τ_k - τ_new)

            if error_θ < tol and error_τ < tol:
                break

            θ = θ * (1-damp) + θ_new * damp
            τ_k = τ_k * (1-damp) + τ_new * damp

            Eqbm = nrt.Eqbm_Solve(E, (θ, τ_k, Ψ, ψ, *args))

            E = Eqbm.x

        qe.toc()
        print("Parametric Policy Found")

        Out = ()
        if θ_on == 1:
            Out += (θ,)
        if τ_on == 1:
            Out += (τ_k,)

        return Out



def solve_eqbm(θ, τ_k, Ψ, ψ, y_0, E_init, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, β, var_θ, ε, δ, g, φ):
    "Equilibrium allocation (gpf.solve_eqbm with nested-CES roots)"

    sol = nrt.Eqbm_Solve(E_init, (θ, τ_k, Ψ, ψ, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, y_0, β, var_θ, ε, δ, g, φ))
    E = sol.x
    return E[:J], E[J:2*J], E[2*J:3*J], E[3*J:]



def alloc_stats(c_0, c_1, l, x, y_0, A_j, A_k, x_bar, ζ, ν, σ, ρ, J, n, β, var_θ, ε, δ, g, φ):
    "Table 2 statistics (gpf.alloc_stats with nested-CES wages and Σ_j)"

    λ     = c_1**(-var_θ) / np.sum(n * c_1**(-var_θ))
    var_λ = np.sum(n * λ**2) - 1

    E_ln_Λ = (np.sum(n * np.log(fn.Lambda_l(x, x_bar, ζ, ν, σ))) / σ)

    L = n * l
    K = np.sum(n * (y_0 - c_0))
    u = npf.Prices(x, L, K, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    w, r, Y, K_j, P = npf.Unpack(u, x, L, K, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    ln_w = np.log(w)
    Σ_j = npf.Sigma_j(u, x, A_j, A_k, x_bar, ζ, ν, σ, ρ)

    E_ln_w = np.sum(n * ln_w)
    E_Σ_j  = np.sum(n * Σ_j)
    cov    = np.sum(n * Σ_j * ln_w) - E_Σ_j * E_ln_w

    return dict(var_λ=var_λ, E_ln_Λ=E_ln_Λ, ln_w=ln_w, Σ_j=Σ_j, cov=cov)
