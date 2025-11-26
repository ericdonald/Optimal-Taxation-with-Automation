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
        labels=False,  # 0 .. n_groups-1
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



def bisect_scalar(func, a, b, args, tol=1e-8):
    "Scalar Bisection"
    
    fa, fb = func(a, *args), func(b, *args)

    # -------- #
    # Fallback #
    # -------- #
    if fa * fb > 0:
        raise ValueError("Bisection interval does not bracket a root.")

    # ----------------- #
    # Classic Bisection #
    # ----------------- #
    while abs(fa) > tol:
        c  = 0.5 * (a + b)
        fc = func(c, *args)
        if fa * fc <= 0:          # root is in [a,c]
            b, fb = c, fc
        else:                     # root is in (c,b]
            a, fa = c, fc
    return 0.5 * (a + b)



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



def solve_planner(w, r, X_0, args):
    "Solve Mirrlees for Normalized Penalty"
    
    J = args[-1]
    W_0 = rt.Mir_obj(X_0, *args)
    Pen_0 = np.sum(np.minimum(rt.Inequal_Constr(X_0, w, *args), 0)**2)
    Δ = np.abs(W_0) / (Pen_0 + 1e-12)
    
    
    # ------------------ #
    # Define Constraints #
    # ------------------ #
    eq_fun = lambda x: rt.Equal_Constr(x, w, r, *args)
    eq_jac = lambda x: pr.δEC_δX(x, w, r, *args)
    
    @njit
    def obj_fun(x):
        W = rt.Mir_obj(x, *args)
        IC_mat = rt.Inequal_Constr(x, w, *args)
        viol = np.minimum(IC_mat, 0.0)
        pen = np.sum(viol**2)
    
        return -(W - Δ * pen)

    @njit
    def obj_jac(x):
        W_jac = pr.δObj_δX(x, *args)
        IC_vec = rt.Inequal_Constr(x, w, *args).flatten()
        viol = np.minimum(IC_vec, 0.0)
        IC_jac = pr.δIC_δX(x, w, *args)
    
        weights = 2.0 * viol
        grad_pen = weights @ IC_jac
    
        return -(W_jac - Δ * grad_pen)
    
    eq_cons = sp.optimize.NonlinearConstraint(eq_fun, lb=0, ub=0, jac=eq_jac)
    
    bounds = sp.optimize.Bounds(np.ones(3 * J), np.ones(3 * J)*1e4)
    
    
    # ----- #
    # Solve #
    # ----- #
    opt = cp.minimize_ipopt(obj_fun, X_0, jac=obj_jac,
                            bounds=bounds, constraints=[eq_cons],
                            options={'max_iter':100})
    
    return opt.x



def inner_solve(w, r, X_0, args, max_inner_iter=100):
    "Solve Inner Loop"
    
    
    for _ in range(max_inner_iter):

        
        # ----- #
        # Solve #
        # ----- #
        alloc = solve_planner(w, r, X_0, args)


        # --------------------- #
        # Scan for IC Violation #
        # --------------------- #
        W = rt.Mir_obj(alloc, *args)
        IC_full = rt.Inequal_Constr(alloc, w, *args)
        viol_tol = np.abs(W) / 1000

        viols = (IC_full < -viol_tol)
        print(W)
        print(np.min(IC_full))

        if not viols.any():
            break
        

        # ------ #
        # Update #
        # ------ #
        X_0 = alloc.copy()
        

    return alloc

