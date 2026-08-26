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
    return np.concatenate((c_0, c_1, l, x)), m



def _reduce_grad(g_full, c_1, m, J):
    g_c0, g_c1, g_l, g_x = g_full[:J], g_full[J:2*J], g_full[2*J:3*J], g_full[3*J:4*J]
    g_c0p = g_c0 + np.exp(m) * g_c1
    g_m = np.array([np.sum(g_c1 * c_1)])
    return np.concatenate((g_c0p, g_l, g_x, g_m))
    


def _reduce_jac(Jac_full, c_1, m, J):
    Jc0, Jc1, Jl, Jx = Jac_full[:, :J], Jac_full[:, J:2*J], Jac_full[:, 2*J:3*J], Jac_full[:,3*J:4*J]
    Jc0p = Jc0 + np.exp(m) * Jc1
    Jm = (Jc1 * c_1).sum(axis=1, keepdims=True)
    return np.hstack((Jc0p, Jl, Jx, Jm))



def _reduce_hess(H_full, g_full, c_1, m, J):
    em = np.exp(m)
    Hc0, Hc1 = H_full[:, :J], H_full[:, J:2*J]
    Hl,  Hx  = H_full[:, 2*J:3*J], H_full[:, 3*J:4*J]
    H_col = np.hstack((Hc0 + em*Hc1, Hl, Hx,
                       (Hc1 * c_1).sum(1, keepdims=True)))
    Rc0, Rc1 = H_col[:J], H_col[J:2*J]
    Rl,  Rx  = H_col[2*J:3*J], H_col[3*J:4*J]
    H_z = np.vstack((Rc0 + em*Rc1, Rl, Rx,
                     (Rc1 * c_1[:, None]).sum(0, keepdims=True)))
    g_c1 = g_full[J:2*J]
    ix = np.arange(J)
    H_z[ix, 3*J] += g_c1 * em
    H_z[3*J, ix] += g_c1 * em
    H_z[3*J, 3*J] += np.sum(g_c1 * c_1)
    return H_z



@njit
def _scatter_ic(blocks, idx, λ_ic, J):
    H = np.zeros((4*J, 4*J))
    P = blocks.shape[0]
    for p in range(P):
        w = λ_ic[p]
        for a in range(8):
            ia = idx[p, a]
            for b in range(8):
                H[ia, idx[p, b]] += w * blocks[p, a, b]
    return H



def _ic_full(E, WS, λ_ic, args):
    J = args[-1]
    blocks, idx = pr.δ2IC_δX_δX(E, WS, *args)
    H = _scatter_ic(blocks, idx, λ_ic, J)

    rows, cols, vals = pr.δIC_δX(E, WS, *args)
    g = np.zeros(4*J)
    for k in range(vals.size):
        g[cols[k]] += λ_ic[rows[k]] * vals[k]
    return H, g



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



def lagr_hess_reduced(z, lagrange, obj_factor, WS, args):
    J = args[-1]
    E, m = _expand(z, J); c_1 = E[J:2*J]
    meq = J + 1
    λ_eq, λ_ic = lagrange[:meq], lagrange[meq:]

    # Objective
    H_full = obj_factor * (-pr.δ2Obj_δX_δX(E, *args))
    g_full = obj_factor * (-pr.δObj_δX(E, *args))

    # Equality Constraints
    H_full += pr.δ2EC_δX_δX(E, λ_eq, *args)
    g_full += λ_eq @ pr.δEC_δX(E, *args)

    # IC Constraints
    H_ic, g_ic = _ic_full(E, WS, λ_ic, args)
    H_full += H_ic
    g_full += g_ic

    H_z = _reduce_hess(H_full, g_full, c_1, m, J)
    return H_z



class _MirrleesNLP:
    "Cyipopt Problem Object"
    def __init__(self, WS, θ, args):
        self.WS, self.args = WS, (θ,) + args
        J = args[-1]; self.J = J
        P = WS.shape[0]; self.P = P
        x_bar = args[-2]
        self.nz  = 3*J + 1
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
        z_probe = np.concatenate((np.ones(3*J)*0.5, np.full(J, 0.5*x_bar),
                                  np.array([0.0])))
        Hp = lagr_hess_reduced(z_probe, np.ones(self.meq + P), 1.0, WS, self.args)
        tril_mask = np.tril(np.abs(Hp) > 1e-300)
        self._hrows, self._hcols = np.nonzero(tril_mask)
        self._htril = (self._hrows, self._hcols)

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
        J = self.J; P = self.P
        E, m = _expand(z, J)
        em = np.exp(m); c_1 = E[J:2*J]
        i = self.WS[:, 0]; j = self.WS[:, 1]
        _, _, vals = pr.δIC_δX(E, self.WS, *self.args)   
        V = vals.reshape(P, 8)   
        ic = np.empty((P, 7))
        ic[:, 0] = V[:, 0] + em * V[:, 2]              
        ic[:, 1] = V[:, 1] + em * V[:, 3]             
        ic[:, 2] = V[:, 4]                               
        ic[:, 3] = V[:, 5]                               
        ic[:, 4] = V[:, 6]                               
        ic[:, 5] = V[:, 7]                               
        ic[:, 6] = c_1[i] * V[:, 2] + c_1[j] * V[:, 3]
        return np.concatenate([eq_jac, ic.ravel()])

    def jacobianstructure(self):
        return self._rows, self._cols
    
    def hessian(self, z, lagrange, obj_factor):
        H_z = lagr_hess_reduced(z, lagrange, obj_factor, self.WS, self.args)
        return H_z[self._htril]

    def hessianstructure(self):
        return self._hrows, self._hcols



def solve_planner(E_0, θ, args, WS, exact):
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
    
    z_0 = np.concatenate((c_0_0, l_0, x_0, np.array([m_0])))
    lb  = np.concatenate((np.ones(3*J)*1e-2, np.ones(1)*(-10.0)))
    ub  = np.concatenate((np.ones(2*J)*1e4, np.ones(J)*x_bar, np.ones(1)*10.0))
    
    P  = WS.shape[0]
    cl = np.zeros(J+1+P)
    cu = np.concatenate((np.zeros(J+1), np.full(P, 2.0e19)))
    
    
    # ----- #
    # Solve #
    # ----- #
    n = 3*J+1
        
    nlp = cp.Problem(n=n, m=J+1+P,
                     problem_obj=_MirrleesNLP(WS, θ, args),
                     lb=lb, ub=ub, cl=cl, cu=cu)
    
    for k, v in {'mu_strategy': 'adaptive',
                 'print_level': 0, 'sb': 'yes'}.items():
        nlp.add_option(k, v)
    #nlp.add_option('output_file', 'ipopt.log')
    
    if exact == True:
        nlp.add_option('hessian_approximation', 'exact')
    else:
        nlp.add_option('hessian_approximation', 'limited-memory')
        nlp.add_option('limited_memory_max_history', 50)
    
    if σ<0.5 and σ>0.25:
        nlp.add_option('max_iter', 1500)
        nlp.add_option('acceptable_tol', 1e-5)

    z_opt, info = nlp.solve(z_0)
    E, _ = _expand(z_opt, J)
    status = info['status']
    
    _IPOPT_STATUS = {0: 'solved', 1: 'solved to acceptable tolerance',
                     2: 'infeasible problem detected', -1: 'maximum iterations exceeded',
                    -2: 'restoration failed', -3: 'error in step computation'}
    print(f"IPOPT: {_IPOPT_STATUS.get(status, 'status ' + str(status))} ({status})")
    
    return E, status




def build_working_set(E, w, args, IC_k, IC_slack):
    
    J = args[-1]; ε = args[-3]; φ = args[-4]; var_θ = args[-5]; β = args[-6]; g = args[4]
    beta_tilde = β / (1 - β * (1+g)**(1-var_θ))
    beta_tilde_l = β / (1-β)
    c_0 = E[:J]; c_1 = E[J:2*J]; l = E[2*J:3*J]
    
    MRS_order = (beta_tilde_l / beta_tilde) * (c_1**(var_θ)) * φ * (l**(1/ε)) / w
    order = np.argsort(MRS_order)
    
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



def verify_working_set(E, w, WS, args, tol, max_add=None):
    
    J = args[-1]; var_θ = args[-5]
    c_0 = E[:J]
    denom = c_0**(-var_θ) * c_0
    dev = -np.minimum(rt.Inequal_Constr(E, *args), 0.0) / denom[:, None]
    
    in_set = np.zeros((J, J), dtype=bool)
    in_set[WS[:, 0], WS[:, 1]] = True
    cand = np.argwhere((dev > tol) & (~in_set))
    if cand.shape[0] == 0:
        return WS, 0

    viol = dev[cand[:, 0], cand[:, 1]]
    order = np.argsort(-viol)
    if max_add is not None:
        order = order[:max_add]
    add = cand[order].astype(np.int64)
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



def _fd_grad_scalar(f, E, h=1e-6):
    n = E.size; g = np.zeros(n)
    for k in range(n):
        Ep = E.copy(); Ep[k] += h
        Em = E.copy(); Em[k] -= h
        g[k] = (f(Ep) - f(Em)) / (2*h)
    return g



def _fd_jac_vec(f, E, m_out, h=1e-6):
    n = E.size; Jf = np.zeros((m_out, n))
    for k in range(n):
        Ep = E.copy(); Ep[k] += h
        Em = E.copy(); Em[k] -= h
        Jf[:, k] = (f(Ep) - f(Em)) / (2*h)
    return Jf



def check_obj_grad(E, args, h=1e-6):
    g_an = pr.δObj_δX(E, *args)
    g_fd = _fd_grad_scalar(lambda e: rt.Mir_obj(e, *args), E, h)
    err = np.max(np.abs(g_an - g_fd))
    print(f"[obj grad]   max|an-fd| = {err:.3e}")
    return err



def check_obj_hess(E, args, h=1e-5):
    H_an = pr.δ2Obj_δX_δX(E, *args)
    H_fd = _fd_jac_vec(lambda e: pr.δObj_δX(e, *args), E, E.size, h)
    H_fd = 0.5*(H_fd + H_fd.T)
    err = np.max(np.abs(H_an - H_fd))
    print(f"[obj hess]   max|an-fd| = {err:.3e}")
    return err



def check_eq_jac(E, args, h=1e-6):
    J = args[-1]
    f = lambda e: rt.Equal_Constr(e, *args)
    m_out = J + 1
    Jf = _fd_jac_vec(f, E, m_out, h)
    J_an = pr.δEC_δX(E, *args)
    err = np.max(np.abs(J_an - Jf))
    print(f"[eq  jac]    max|an-fd| = {err:.3e}")
    a, b = np.unravel_index(np.argmax(np.abs(J_an - Jf)), J_an.shape)
    print(f"             worst entry ({a},{b}): an={J_an[a,b]:.4e} fd={Jf[a,b]:.4e}")
    return err



def check_eq_hess(E, λ_eq, args, h=1e-5):
    H_an = pr.δ2EC_δX_δX(E, λ_eq, *args)
    def wg(e):
        return λ_eq @ pr.δEC_δX(e, *args)
    H_fd = _fd_jac_vec(wg, E, E.size, h)
    H_fd = 0.5*(H_fd + H_fd.T)
    err = np.max(np.abs(H_an - H_fd))
    print(f"[eq  hess]   max|an-fd| = {err:.3e}")
    a, b = np.unravel_index(np.argmax(np.abs(H_an - H_fd)), H_an.shape)
    print(f"             worst entry ({a},{b}): an={H_an[a,b]:.4e} fd={H_fd[a,b]:.4e}")
    return err



def check_ic_jac(E, WS, args, h=1e-6):
    J = args[-1]; P = WS.shape[0]
    rows, cols, vals = pr.δIC_δX(E, WS, *args)
    G_an = np.zeros((P, 4*J))
    G_an[rows, cols] = vals
    G_fd = _fd_jac_vec(lambda e: rt.IC_on_set(e, WS, *args), E, P, h)
    err = np.max(np.abs(G_an - G_fd))
    print(f"[ic  jac]    max|an-fd| = {err:.3e}")
    off = G_fd.copy(); off[rows, cols] = 0.0
    print(f"             off-support FD energy = {np.max(np.abs(off)):.3e}")
    return err



def check_ic_hess(E, WS, args, pair_idx=0, h=1e-5):
    blocks, idx = pr.δ2IC_δX_δX(E, WS, *args)
    B_an = blocks[pair_idx]
    coords = idx[pair_idx]
    WSp = WS[pair_idx:pair_idx+1]
    B_fd = np.zeros((8, 8))
    for a in range(8):
        for b in range(8):
            ka, kb = coords[a], coords[b]
            Epp = E.copy(); Epp[ka]+=h; Epp[kb]+=h
            Epm = E.copy(); Epm[ka]+=h; Epm[kb]-=h
            Emp = E.copy(); Emp[ka]-=h; Emp[kb]+=h
            Emm = E.copy(); Emm[ka]-=h; Emm[kb]-=h
            B_fd[a,b] = (rt.IC_on_set(Epp,WSp,*args)
                        - rt.IC_on_set(Epm,WSp,*args)
                        - rt.IC_on_set(Emp,WSp,*args)
                        + rt.IC_on_set(Emm,WSp,*args))[0] / (4*h*h)
    err = np.max(np.abs(B_an - B_fd))
    print(f"[ic  hess p={pair_idx}] max|an-fd| = {err:.3e}")
    labs = ['c0i','c0j','c1i','c1j','li','lj','xi','xj']
    a, b = np.unravel_index(np.argmax(np.abs(B_an - B_fd)), (8,8))
    print(f"             worst [{labs[a]},{labs[b]}]: an={B_an[a,b]:.4e} fd={B_fd[a,b]:.4e}")
    return err



def check_all(E, WS, args_θ, h=1e-6):
    J = args_θ[-1]
    λ_eq = np.ones(J + 1)
    check_obj_grad(E, args_θ, h)
    check_obj_hess(E, args_θ, max(h,1e-5))
    check_eq_jac(E, args_θ, h)
    check_eq_hess(E, λ_eq, args_θ, max(h,1e-5))
    check_ic_jac(E, WS, args_θ, h)
    for p in (0, WS.shape[0]//2, WS.shape[0]-1):
        check_ic_hess(E, WS, args_θ, p, max(h,1e-5))





