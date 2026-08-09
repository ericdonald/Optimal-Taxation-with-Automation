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
    
    return bisect_scalar(func, x0_orig, x1_orig, expansion=expansion, lb=lb, ub=ub)



def bisect_scalar(func, a, b, args=(), expansion='multiplicative', lb=None, ub=None):
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
        if lb is not None:
            a = max(a, lb)
        if ub is not None:
            b = min(b, ub)

    if fa * fb > 0:
        ε = 0.01
        g_lo = 0.0 if lb is None else lb
        g_hi = (1.0 - ε) if ub is None else ub
        grid = np.arange(g_lo, g_hi, 0.1)
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



def _expand(z, J):
    c_0 = z[:J]
    l   = z[J:2*J]
    x   = z[2*J:3*J]
    m   = z[3*J]
    c_1 = c_0 * np.exp(m)
    if z.size > 3*J+1:
        θ = z[-1]
        return np.concatenate((c_0, c_1, l, x, np.array([θ]))), m
    else:
        return np.concatenate((c_0, c_1, l, x)), m



def _reduce_grad(g_full, c_1, m, J):
    g_c0, g_c1, g_l, g_x = g_full[:J], g_full[J:2*J], g_full[2*J:3*J], g_full[3*J:4*J]
    g_c0p = g_c0 + np.exp(m) * g_c1
    g_m = np.array([np.sum(g_c1 * c_1)])
    if g_full.size > 4*J:
        g_θ = np.array([g_full[-1]])
        return np.concatenate((g_c0p, g_l, g_x, g_m, g_θ))
    else:
        return np.concatenate((g_c0p, g_l, g_x, g_m))
    



def _reduce_jac(Jac_full, c_1, m, J):
    Jc0, Jc1, Jl, Jx = Jac_full[:, :J], Jac_full[:, J:2*J], Jac_full[:, 2*J:3*J], Jac_full[:,3*J:4*J]
    Jc0p = Jc0 + np.exp(m) * Jc1
    Jm = (Jc1 * c_1).sum(axis=1, keepdims=True)
    if Jac_full.shape[1] > 4*J:
        Jθ = Jac_full[:,-1:]
        return np.hstack((Jc0p, Jl, Jx, Jm, Jθ))
    else:
        return np.hstack((Jc0p, Jl, Jx, Jm))



def obj_reduced(z, args):
    J = args[-1]
    E, _ = _expand(z, J)
    return -rt.Mir_obj(E, *args)



def obj_jac_reduced(z, args):
    J = args[-1]
    E, m = _expand(z, J)
    return _reduce_grad(-pr.δObj_δX(E, *args), E[J:2*J], m, J)



def eq_reduced(z, args):
    J = args[-1]
    E, _ = _expand(z, J)
    return rt.Equal_Constr(E, *args)



def eq_jac_reduced(z, args):
    J = args[-1]
    E, m = _expand(z, J)
    return _reduce_jac(pr.δEC_δX(E, *args), E[J:2*J], m, J)



def ic_reduced(z, WS, args):
    J = args[-1]
    E, _ = _expand(z, J)
    return rt.IC_on_set(E, WS, *args)



def ic_jac_reduced(z, WS, args):
    J = args[-1]
    E, m = _expand(z, J)
    rows, cols, vals = pr.δIC_δX(E, m, WS, *args)
    return sp.sparse.csr_matrix((vals, (rows, cols)), shape=(WS.shape[0], z.size))



class _MirrleesNLP:
    "Cyipopt Problem Object"
    def __init__(self, WS, θ_on, args):
        self.WS, self.args = WS, args
        J = args[-1]; self.J = J
        P = WS.shape[0]; self.P = P
        self.nz  = 3*J + 1 + (1 if θ_on else 0)
        self.meq = J + 1                           
        i = WS[:, 0]; j = WS[:, 1]
        eq_rows = np.repeat(np.arange(self.meq), self.nz)
        eq_cols = np.tile(np.arange(self.nz), self.meq)
        ic_rows = np.repeat(np.arange(self.meq, self.meq + P), 7)
        ic_cols = np.empty(7*P, np.int64)
        ic_cols[0::7] = i;        ic_cols[1::7] = j
        ic_cols[2::7] = J + i;    ic_cols[3::7] = J + j
        ic_cols[4::7] = 2*J + i;  ic_cols[5::7] = 2*J + j
        ic_cols[6::7] = 3*J
        self._rows = np.concatenate([eq_rows, ic_rows])
        self._cols = np.concatenate([eq_cols, ic_cols])

    def objective(self, z):
        return obj_reduced(z, self.args)

    def gradient(self, z):
        return obj_jac_reduced(z, self.args)

    def constraints(self, z):
        eq = eq_reduced(z, self.args)     
        ic = ic_reduced(z, self.WS, self.args)    
        return np.concatenate([eq, ic])

    def jacobian(self, z):
        eq_jac = eq_jac_reduced(z, self.args).ravel()
        E, m = _expand(z, self.J)
        _, _, vals = pr.δIC_δX(E, m, self.WS, *self.args)
        return np.concatenate([eq_jac, vals])

    def jacobianstructure(self):
        return self._rows, self._cols



def solve_planner(E_0, θ_on, args, WS):
    "Solve Mirrlees with IC Subset"
    
    J = args[-1]
    x_bar = args[-2]
    σ = args[8]
    
    
    # -------------------- #
    # Build Reduced Vector #
    # -------------------- #
    c_0_0 = E_0[:J]
    l_0 = E_0[2*J:3*J]
    x_0 = E_0[3*J:4*J]
    m_0 = np.median(np.log(E_0[J:2*J] / c_0_0))
    if θ_on==1:
        θ_0 = E_0[-1]
        z_0 = np.concatenate((c_0_0, l_0, x_0, np.array([m_0, θ_0])))
        lb  = np.concatenate((np.ones(3*J)*1e-2, np.ones(2)*(-10.0)))
        ub  = np.concatenate((np.ones(2*J)*1e4, np.ones(J)*x_bar, np.ones(2)*10.0))
    else:
        z_0 = np.concatenate((c_0_0, l_0, x_0, np.array([m_0])))
        lb  = np.concatenate((np.ones(3*J)*1e-2, np.ones(1)*(-10.0)))
        ub  = np.concatenate((np.ones(2*J)*1e4, np.ones(J)*x_bar, np.ones(1)*10.0))
    
    P  = WS.shape[0]
    cl = np.zeros(J+1+P)
    cu = np.concatenate((np.zeros(J+1), np.full(P, 2.0e19)))
    
    
    # ----- #
    # Solve #
    # ----- #
    if θ_on==1:
        n = 3*J+2
    else:
        n = 3*J+1
        
    nlp = cp.Problem(n=n, m=J+1+P,
                     problem_obj=_MirrleesNLP(WS, θ_on, args),
                     lb=lb, ub=ub, cl=cl, cu=cu)
    
    for k, v in {'hessian_approximation': 'limited-memory',
                 'limited_memory_max_history': 100, 'mu_strategy': 'adaptive',
                 'print_level': 0, 'sb': 'yes'}.items():
        nlp.add_option(k, v)
    if σ<0.5:
        nlp.add_option('acceptable_tol', 1e-4)

    z_opt, info = nlp.solve(z_0)
    E, _ = _expand(z_opt, J)
    status = info['status']
    
    _IPOPT_STATUS = {0: 'solved', 1: 'solved to acceptable tolerance',
                     2: 'infeasible problem detected', -1: 'maximum iterations exceeded',
                    -2: 'restoration failed', -3: 'error in step computation'}
    print(f"IPOPT: {_IPOPT_STATUS.get(status, 'status ' + str(status))} ({status})")
    
    return E




def build_working_set(E, w, args, IC_k, IC_slack):
    
    J = args[-1]; var_θ = args[-5]
    c_0 = E[:J]; l = E[2*J:3*J]
    y = w * l
    order = np.argsort(y)
    
    keep = np.zeros((J, J), dtype=bool)
    
    
    # --------- #
    # Neighbors #
    # --------- #
    for p in range(J):
        i  = order[p]
        lo = max(0, p - IC_k); hi = min(J, p + IC_k + 1)
        for q in range(lo, hi):
            if q != p:
                keep[i, order[q]] = True
    
    
    # ------------------ #
    # Near-Binding Pairs #
    # ------------------ #
    IC    = rt.Inequal_Constr(E, *args)
    denom = c_0**(-var_θ) * c_0
    dev   = IC / denom[:, None]
    keep |= (dev < IC_slack)
    np.fill_diagonal(keep, False)
    
    return np.argwhere(keep).astype(np.int64)



def verify_working_set(E, w, WS, args, tol):
    
    J = args[-1]; var_θ = args[-5]
    c_0 = E[:J]
    denom = c_0**(-var_θ) * c_0
    dev = -np.minimum(rt.Inequal_Constr(E, *args), 0.0) / denom[:, None]
    
    in_set = np.zeros((J, J), dtype=bool)
    in_set[WS[:, 0], WS[:, 1]] = True
    add = np.argwhere((dev > tol) & (~in_set)).astype(np.int64)
    
    if add.shape[0] == 0:
        return WS, 0
    else:
        return np.unique(np.vstack((WS, add)), axis=0), add.shape[0]



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









