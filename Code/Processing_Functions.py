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



def solve_planner(w, r, IC_act, X_0, args, Δ):
    "Solve Mirrlees for Fixed IC Set"
    
    J = args[-1]
    
    
    # ------------------ #
    # Define Constraints #
    # ------------------ #
    eq_fun = lambda x: rt.Equal_Constr(x, w, r, *args)
    eq_jac = lambda x: pr.δEC_δX(x, w, r, *args)
    
    ineq_fun = lambda x: rt.Inequal_Constr(x, w, IC_act, *args) + Δ
    ineq_jac = lambda x: pr.δIC_δX(x, w, IC_act, *args)
    
    eq_cons = sp.optimize.NonlinearConstraint(eq_fun, lb=0, ub=0, jac=eq_jac)
    ineq_cons = sp.optimize.NonlinearConstraint(ineq_fun, lb=0, ub=np.inf, jac=ineq_jac)
    
    bounds = sp.optimize.Bounds(np.ones(3 * J)*1e-8, np.ones(3 * J)*np.inf)
    
    
    # ----- #
    # Solve #
    # ----- #
    opt = cp.minimize_ipopt(rt.Mir_obj, X_0, jac=pr.δObj_δX,
                            args=args, bounds=bounds,
                            constraints=[eq_cons, ineq_cons])
    
    print(opt.message)
    print(opt.fun)
    
    return opt.x



def inner_solve(w, r, IC_act, X_0, args, viol_tol=1e-8, bind_tol=1e-6, max_inner_iter=20, max_new_ic=10, Δ_damp=0.9):
    "Solve Inner Loop"
    
    Δ = 0
    
    for _ in range(max_inner_iter):

        # ----- #
        # Solve #
        # ----- #
        alloc = solve_planner(w, r, IC_act, X_0, args, Δ)


        # --------------------- #
        # Scan for IC Violation #
        # --------------------- #
        IC_full = rt.IC_Full(alloc, w, *args)

        mask_new = (IC_full < -viol_tol) & (~IC_act)

        if not mask_new.any():
            break

        viol_vals = IC_full[mask_new]
        i_idx, j_idx = np.where(mask_new)

        order = np.argsort(viol_vals)

        IC_N = min(max_new_ic, len(order))
        chosen = order[:IC_N]

        additions = np.zeros_like(IC_act, dtype=bool)
        additions[i_idx[chosen], j_idx[chosen]] = True

        IC_act = IC_act | additions
        Δ = np.minimum(np.min(IC_full) * Δ_damp, 0)
        X_0 = alloc.copy()


    # --------------------------- #
    # Save Binding IC at Solution #
    # --------------------------- #
    IC_full = rt.IC_Full(alloc, w, *args)
    IC_act = (IC_full <= bind_tol)

    return alloc, IC_act

