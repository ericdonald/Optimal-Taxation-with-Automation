"""""""""""
Processing Functions

Notes: Functions that accomplish basic processing for the project.
    
"""""""""""

import numpy as np
import pandas as pd
from numba import njit



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



def bisect_scalar(func, a, b, args, tol=1e-5):
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



def compute_comp_J(x, delta):
    "Find IC Comparison Indices"
    
    x = np.asarray(x)
    J = x.shape[0]
   
    sorted_idx  = np.argsort(x)
    pos_of_idx  = np.empty(J, dtype=int)
    pos_of_idx[sorted_idx] = np.arange(J)
    logx = np.log(x)

    comp_J = []
    for j in range(J):
        pos  = pos_of_idx[j]
        nbrs = set()

        # -------------------- #
        # immediate neighbours #
        # -------------------- #
        if pos > 0:
            nbrs.add(sorted_idx[pos - 1])
        if pos < J-1:
            nbrs.add(sorted_idx[pos + 1])

        # ------------------------- #
        # within‐delta log‐distance #
        # ------------------------- #
        mask = np.abs(logx - logx[j]) <= delta
        for i in np.nonzero(mask)[0]:
            if i != j:
                nbrs.add(i)

        comp_J.append(nbrs)

    return comp_J








