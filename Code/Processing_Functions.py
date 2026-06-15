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



def secant_scalar(func, x0, x1, lb=None, ub=None, tol=1e-7, max_iter=50):
    "Secant Root Finder with Bisection Fallback"

    try:
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

            if abs(f1) < tol or abs(x1 - x0) < tol:
                return x1

        if abs(f1) < tol * 100:
            return x1
        raise ValueError("Secant did not converge")

    except (ValueError, FloatingPointError):
    
        return bisect_scalar(func, x0, x1)



def bisect_scalar(func, a=0, b=0.1, args=(), expansion='multiplicative'):
    "Scalar Bisection"
    
    if expansion == 'unit':           # [0, 1)
        for _ in range(10):
            fa, fb = func(a, *args), func(b, *args)
            if fa * fb < 0:
                break
            
            a += 0.1
            b += 0.1
            
    else:
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
        raise ValueError("Bisection interval does not bracket a root.")

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



def inner_solve(w, r, E_0, args, viol_frac, max_inner_iter=1000):
    "Solve Inner Loop"
    
    Pen_N = 1
    
    for _ in range(max_inner_iter):

        # ----- #
        # Solve #
        # ----- #
        alloc = solve_planner(w, r, E_0, args, Pen_N)


        # --------------------- #
        # Scan for IC Violation #
        # --------------------- #
        J = args[-1]
        var_θ = args[-4]
        
        IC_full = -np.minimum(rt.Inequal_Constr(alloc, w, *args), 0.0)
        IC_max = IC_full.max(axis=1)
        c_0 = alloc[:J]
        MU_0 = c_0**(-var_θ)
        
        deviat = IC_max / (MU_0 * c_0)

        viols = (deviat > np.minimum(viol_frac, 1.0))
        #print(f'IC Violation: {np.max(deviat)}')

        if not viols.any():
            break
        

        # ------ #
        # Update #
        # ------ #
        E_0 = alloc.copy()
        Pen_N += 1
        

    return alloc



def solve_planner(w, r, E_0, args, Pen_N):
    "Solve Mirrlees with Normalized Penalty"
    
    J = args[-1]
    W_0 = rt.Mir_obj(E_0, *args)
    Pen_0 = np.sum(np.minimum(rt.Inequal_Constr(E_0, w, *args), 0)**2)
    Δ = Pen_N * np.abs(W_0) / (Pen_0 + 1e-12)
    
    
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
                            options={'maxiter':100})
    
    return opt.x













