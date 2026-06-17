"""""""""""
Processing Functions

Notes: Functions that accomplish basic processing for the project.
    
"""""""""""

import numpy as np
import pandas as pd
from numba import njit
import scipy as sp
import cyipopt as cp
import Roots as rt
import Perturbations as pr
import Production_Functions as fn



class ResultsTable:
    "Object for Saving Results in CSV File"
    
    def __init__(self):
        "Initialize Results Table Object"
        
        self.rows = []


    def add(self, variable, value):
        "Add Row to Table"
        
        self.rows.append({"Variable": variable, "Value": value})


    def to_csv(self, path):
        "Save Table to CSV"
        
        pd.DataFrame(self.rows).to_csv(path, index=False)
        
        
        
def clean_round(number, decimals):
    "Cut a Hanging Zero"
    
    rounded_number = np.round(number, decimals)
    if decimals == 0:
        if rounded_number == int(rounded_number):
            return int(rounded_number)
    elif decimals > 0:
        if rounded_number == np.round(rounded_number, decimals-1):
            return np.round(rounded_number, decimals-1)
    return rounded_number
    


def compute_decile_shares(df, value_col, weight_col='Weight', n_groups=10):
    "Returns Shares by Decile"
    
    df_sorted = df.sort_values(by=value_col, ascending=True).reset_index(drop=True)
    
    
    # ------------------------------------ #
    # Compute cumulative weights and total #
    # ------------------------------------ #
    df_sorted['cum_weight'] = df_sorted[weight_col].cumsum()
    total_weight = df_sorted[weight_col].sum()
    
    
    # -------------------------------------------- #
    # Compute population thresholds for each group #
    # -------------------------------------------- #
    cut_points = np.linspace(0, total_weight, n_groups + 1)
    
    df_sorted['decile'] = pd.cut(
        df_sorted['cum_weight'],
        bins=cut_points,
        labels=False,  
        include_lowest=True
    )
    df_sorted.at[df_sorted.index[-1], 'decile'] = 9
    
    # ---------------------- #
    # Compute weighted value #
    # ---------------------- #
    df_sorted['weighted_value'] = df_sorted[value_col] * df_sorted[weight_col]
    decile_sums = df_sorted.groupby('decile')['weighted_value'].sum()
    
    total_value = decile_sums.sum()
    decile_shares = decile_sums / total_value
    
    return decile_shares.values



def secant_scalar(func, x0, x1, lb=None, ub=None, tol=1e-7, max_iter=50, expansion='multiplicative'):
    "Secant Root Finder with Bisection Fallback"

    x0_orig, x1_orig = x0, x1

    f0 = func(x0)
    f1 = func(x1)
    for _ in range(max_iter):
        if abs(f1 - f0) < 1e-14:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

        if lb is not None:
            x2 = max(x2, lb)
        if ub is not None:
            x2 = min(x2, ub)

        x0, f0, x1 = x1, f1, x2
        f1 = func(x1)

        if abs(f1) < tol:
            return x1
    
    return bisect_scalar(func, x0_orig, x1_orig, expansion=expansion)



def bisect_scalar(func, a, b, args=(), expansion='multiplicative'):
    "Scalar Bisection"
    
    for _ in range(20):
        fa, fb = func(a, *args), func(b, *args)
        if fa * fb < 0:
            break
        if expansion == 'multiplicative':   # (0, inf)
            a *= 0.5
            b *= 2.0
        elif expansion == 'additive':       # (-inf, inf)
            mid = 0.5 * (a + b)
            step = b - a
            a = mid - step
            b = mid + step

    if fa * fb > 0:
        ε = 0.01
        grid = np.arange(0.0, 1.0 - ε, 0.1)
        f_grid = np.array([func(pt, *args) for pt in grid])
        bracket_found = False
        for i in range(len(grid) - 1):
            if f_grid[i] * f_grid[i+1] < 0:
                a, b = grid[i], grid[i+1]
                bracket_found = True
                break
        if not bracket_found:
            return grid[np.argmin(np.abs(f_grid))]

    return sp.optimize.brentq(func, a, b, args=args)



@njit
def make_diag(v):
    "Numba Friendly diag Function"
    
    I = np.eye(v.size)
    D = I * v
    
    return D



@njit
def broadcast_row_to_matrix(row):
    "Numba Friendly tile Function: row -> matrix"
    
    J = row.shape[0]
    M = np.empty((J,J))
    for i in range(J):
        for j in range(J):
            M[i,j] = row[j]
            
    return M



@njit
def broadcast_col_to_matrix(col):
    "Numba Friendly tile Function: column -> matrix"
    
    J = col.shape[0]
    M = np.empty((J, J))
    for i in range(J):
        for j in range(J):
            M[i, j] = col[i]
            
    return M



def inner_solve(w, r, E_0, args, error, max_inner_iter=1000):
    "Solve Inner Loop"
    
    Pen_N = 1
    Pen_0 = np.sum(np.minimum(rt.Inequal_Constr(E_0, w, *args), 0)**2)
    
    for _ in range(max_inner_iter):

        # ----- #
        # Solve #
        # ----- #
        alloc = solve_planner(w, r, E_0, args, Pen_N, Pen_0, error)


        # --------------------- #
        # Scan for IC Violation #
        # --------------------- #
        J = args[-1]
        var_θ = args[-4]
        
        IC_full = -np.minimum(rt.Inequal_Constr(alloc, w, *args), 0.0)
        IC_max = IC_full.max(axis=1)
        Pen_0 = np.sum(IC_full**2)
        c_0 = alloc[:J]
        MU_0 = c_0**(-var_θ)
        
        deviat = IC_max / (MU_0 * c_0)

        viols = (deviat > np.minimum(error, 1.0))
        #print(f'IC Violation: {np.max(deviat)}')

        if not viols.any():
            break
        

        # ------ #
        # Update #
        # ------ #
        E_0 = alloc.copy()
        Pen_N += 1
        

    return alloc



def solve_planner(w, r, E_0, args, Pen_N, Pen_0, error):
    "Solve Mirrlees with Normalized Penalty"
    
    J = args[-1]
    W_0 = rt.Mir_obj(E_0, *args)
    Δ = Pen_N * np.abs(W_0) / (Pen_0 + 1e-12)
    maxiter = 20 if error > 1e-2 else 100
    
    
    # ------------------ #
    # Define Constraints #
    # ------------------ #
    eq_fun = lambda x: rt.Equal_Constr(x, w, r, *args)
    eq_jac = lambda x: pr.δEC_δX(x, w, r, *args)
    
    obj_pen_fun = lambda x: rt.obj_fun(x, w, Δ, args)
    obj_pen_jac = lambda x: pr.obj_jac(x, w, Δ, args)
    
    eq_cons = sp.optimize.NonlinearConstraint(eq_fun, lb=0, ub=0, jac=eq_jac)
    
    bounds = sp.optimize.Bounds(np.ones(3 * J), np.ones(3 * J)*1e4)
    
    
    # ----- #
    # Solve #
    # ----- #

    opt = cp.minimize_ipopt(obj_pen_fun, E_0, jac=obj_pen_jac,
                            bounds=bounds, constraints=[eq_cons],
                            options={'maxiter': maxiter,
                                 'hessian_approximation': 'limited-memory',
                                 'print_level': 0,
                                 'sb': 'yes'})
    
    return opt.x



def solve_eqbm(θ, τ_k, Ψ, ψ, y_0, E_init, A_j, A_k, x_bar, ζ, ν, σ, J, n, β, var_θ, ε, δ, g, φ):
    sol = sp.optimize.root(
        rt.Eqbm_Root, E_init,
        args=(θ, τ_k, Ψ, ψ, A_j, A_k,
              x_bar, ζ, ν, σ,
              J, n, y_0, β, var_θ,
              ε, δ, g, φ),
        jac=pr.δH_δclx)
    E = sol.x
    return E[:J], E[J:2*J], E[2*J:3*J], E[3*J:]



def alloc_stats(c_0, c_1, l, x, y_0, A_j, A_k, x_bar, ζ, ν, σ, J, n, β, var_θ, ε, δ, g, φ):
    λ     = c_1**(-var_θ) / np.sum(n * c_1**(-var_θ))
    var_λ = np.sum(n * λ**2) - 1

    E_ln_Λ = (np.sum(n * np.log(fn.Lamba_l(x, x_bar, ζ, ν, σ))) / σ)

    L    = n * l
    K    = np.sum(n * (y_0 - c_0))
    ln_w = np.log(fn.Wages(x, L, K, A_j, A_k,
                           x_bar, ζ, ν, σ))
    z_j  = fn.relα(x, x_bar, ζ, ν, σ, 0) * x
    z_jk = fn.relα(x, x_bar, ζ, ν, σ, 1) * x
    Σ_j  = σ + (z_j + z_jk) / ζ

    E_ln_w = np.sum(n * ln_w)
    E_Σ_j  = np.sum(n * Σ_j)
    cov    = np.sum(n * Σ_j * ln_w) - E_Σ_j * E_ln_w

    return dict(var_λ=var_λ, E_ln_Λ=E_ln_Λ, ln_w=ln_w, Σ_j=Σ_j, cov=cov)



def consumption_equiv(c_0_new, c_1_new, l_new, c_0_base, c_1_base, l_base, A_j, A_k, x_bar, ζ, ν, σ, J, n, β, var_θ, ε, δ, g, φ):
    CE = sp.optimize.root(
        rt.CERoot, 1,
        args=(c_0_new, c_1_new, l_new,
              c_0_base, c_1_base, l_base,
              n, β, var_θ, φ, ε, g),
        method='lm')
    return (CE.x[0] - 1) * 100



def deltas(stats_new, stats_base):
    Δ_ln_Λ  = (stats_new['E_ln_Λ'] - stats_base['E_ln_Λ']) * 100
    Δ_var_λ = (stats_new['var_λ']  - stats_base['var_λ'])  * 100 / stats_base['var_λ']
    Δ_cov   = (stats_new['cov']    - stats_base['cov'])    * 100 / stats_base['cov']
    return Δ_ln_Λ, Δ_var_λ, Δ_cov



def cov_dataframe(stats_A, label_A, stats_B, label_B, n, J):

    def _ols_fit(Σ, ln_w):
        X = np.hstack((np.ones((J, 1)), Σ.reshape((-1, 1))))
        W = np.diag(n)
        β = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ ln_w.reshape((-1, 1))
        return (X @ β).ravel()

    df = pd.DataFrame(np.hstack((
        n.reshape((-1, 1)),
        stats_A['Σ_j'].reshape((-1, 1)),
        stats_A['ln_w'].reshape((-1, 1)),
        _ols_fit(stats_A['Σ_j'], stats_A['ln_w']).reshape((-1, 1)),
        stats_B['Σ_j'].reshape((-1, 1)),
        stats_B['ln_w'].reshape((-1, 1)),
        _ols_fit(stats_B['Σ_j'], stats_B['ln_w']).reshape((-1, 1)),
    )), columns=[
        'Weight',
        f'ES {label_A}',         f'Log Wages {label_A}', f'Log Wages_hat {label_A}',
        f'ES {label_B}',         f'Log Wages {label_B}', f'Log Wages_hat {label_B}',
    ])
    return df.sort_values('Weight', ascending=False)









