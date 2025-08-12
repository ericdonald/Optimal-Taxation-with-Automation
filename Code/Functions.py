"""""""""""
Functions

Notes:
    
Output:
"""""""""""

import numpy as np
import pandas as pd
from numba import njit



def compute_decile_shares(df, value_col, weight_col='Weight', n_groups=10):
    "Returns Shares by Decile"
    
    df_sorted = df.sort_values(by=value_col, ascending=True).reset_index(drop=True)
    
    # Compute cumulative weights and total
    df_sorted['cum_weight'] = df_sorted[weight_col].cumsum()
    total_weight = df_sorted[weight_col].sum()
    
    # Compute population thresholds for each group
    cut_points = np.linspace(0, total_weight, n_groups + 1)
    
    # Use pd.cut to assign each row to a bin (1 to n_groups)
    df_sorted['decile'] = pd.cut(
        df_sorted['cum_weight'],
        bins=cut_points,
        labels=False,  # 0 .. n_groups-1
        include_lowest=True
    )
    df_sorted.at[df_sorted.index[-1], 'decile'] = 9
    
    # Compute weighted value
    df_sorted['weighted_value'] = df_sorted[value_col] * df_sorted[weight_col]
    decile_sums = df_sorted.groupby('decile')['weighted_value'].sum()
    
    # Group and normalize
    total_value = decile_sums.sum()
    decile_shares = decile_sums / total_value
    
    return decile_shares.values



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



def bisect_scalar(func, a, b, args, tol=1e-5):
    "Scalar Bisection"
    
    fa, fb = func(a, *args), func(b, *args)

    # ----------  fallback --------------------------------------------
    if fa * fb > 0:
        raise ValueError("Bisection interval does not bracket a root.")

    # ----------  classical bisection ---------------------------------
    while abs(fa) > tol:
        c  = 0.5 * (a + b)
        fc = func(c, *args)
        if fa * fc <= 0:          # root is in [a,c]
            b, fb = c, fc
        else:                     # root is in (c,b]
            a, fa = c, fc
    return 0.5 * (a + b)



@njit
def Lamba_l(x_j, x_bar, ζ, ν, σ):
    "Labor Task Integrals"
    
    Λ_l = (x_bar**(ζ*ν*(σ-1) + 1) - x_j**(ζ*ν*(σ-1) + 1)) / (ζ*ν*(σ-1) + 1)
    
    return Λ_l
    


@njit
def Lamba_k(x_j, x_bar, ζ, ν, σ, ε=0.0001):
    "Capital Task Integral"
    
    Λ_k = np.sum((x_j**(ζ*(ν-1)*(σ-1) + 1) - ε**(ζ*(ν-1)*(σ-1) + 1)) / (ζ*(ν-1)*(σ-1) + 1))
    
    return Λ_k
    
    
 
@njit
def relα(x_j, x_bar, ζ, ν, σ, cap):
    "Relative Task Productivity"
    
    if cap==0:
        Λ_l = Lamba_l(x_j, x_bar, ζ, ν, σ)
        return ( x_j**(ζ*ν*(σ-1)) ) / Λ_l
    
    if cap==1:
        Λ_k = Lamba_k(x_j, x_bar, ζ, ν, σ)
        return ( x_j**(ζ*(ν-1)*(σ-1)) ) / Λ_k
    
    
    
@njit
def zeta(Γ, χ):
    "Comparative Advantage"
    
    ζ = np.exp(Γ * χ)
    
    return ζ



@njit
def Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ):
    "Output"
        
    Λ_j = Lamba_l(x, x_bar, ζ, ν, σ)
    Λ_k = Lamba_k(x, x_bar, ζ, ν, σ)
    
    CES_inner = Λ_k**(1/σ) * (A_k * K)**((σ-1)/σ) + np.sum(Λ_j**(1/σ) * (A_j * L)**((σ-1)/σ))
    
    Y = CES_inner**(σ / (σ-1))
    
    return Y


@njit
def Wages(x, L, K, A_j, A_k, x_bar, ζ, ν, σ):
    "Marginal Product of Labor"
    
    Λ_j = Lamba_l(x, x_bar, ζ, ν, σ)
    Y = Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    
    w = Λ_j**(1/σ) * A_j**((σ-1)/σ) * (Y / L)**(1/σ)
    
    return w
    


@njit
def Rents(x, L, K, A_j, A_k, x_bar, ζ, ν, σ):
    "Marginal Product of Capital"
    
    Λ_k = Lamba_k(x, x_bar, ζ, ν, σ)
    Y = Output(x, L, K, A_j, A_k, x_bar, ζ, ν, σ)
    
    r = Λ_k**(1/σ) * A_k**((σ-1)/σ) * (Y / K)**(1/σ)
    
    return r



@njit
def V(c_0, c_1, l, β, var_θ, φ, ε):
    "Household Utility"
    
    if var_θ == 1:
        Val = np.log(c_0) + (β / (1-β)) * (np.log(c_1) - φ * l**(1 + 1/ε) / (1 + 1/ε))
    else:
        Val = (c_0**(1-var_θ) - 1) / (1-var_θ) + (β / (1-β)) * ((c_1**(1-var_θ) - 1) / (1-var_θ) - φ * l**(1 + 1/ε) / (1 + 1/ε))
        
    return Val

    
    
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
    "Find IC Comparison indices"
    
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

        # immediate neighbours
        if pos > 0:
            nbrs.add(sorted_idx[pos - 1])
        if pos < J-1:
            nbrs.add(sorted_idx[pos + 1])

        # within‐delta log‐distance
        mask = np.abs(logx - logx[j]) <= delta
        for i in np.nonzero(mask)[0]:
            if i != j:
                nbrs.add(i)

        comp_J.append(nbrs)

    return comp_J

