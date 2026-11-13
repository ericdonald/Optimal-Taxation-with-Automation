"""""""""""
Nested Mirrlees Module

Notes: Mirrlees planner for the nested-CES economy (type goods CES over tasks with elasticity σ, final output CES
    over type goods with elasticity ρ). With ρ ≠ σ, wages and rents have no closed form in (x, L, K), so the
    equilibrium price system of Nested_Production is added to the planner's problem as equality constraints on
    explicit price variables (full-space formulation): every constraint is then an elementwise expression with
    short analytic first and second derivatives. ρ = σ reproduces the one-CES planner (Processing_Functions.py).

    Planner variables z = (c_0, l, x, m, a, b, y, p, q, k), with c_1 = c_0 exp(m) as in the one-CES planner and
        a_j = ln w_j      b = ln r      y = ln Y      p_j = ln P_j (type price)
        q = ln Σ_j P_j^(σ-ρ) Λ_kj     k = ln K

    Equality constraints (3J+5), in order:
        RC_1   Σ n c_1 - Y - (1-δ_tilde) K = 0
        AT_j   ζ_j ln x_j - ((a_j - ln A_j) - (ln(1+θ) + b - ln A_k)) = 0                    [J]
        P_j    ln(Λ_kj e^((1-σ)(b-ln A_k)) + Λ_j e^((1-σ)(a_j-ln A_j))) - (1-σ) p_j = 0       [J]  type unit cost
        C_j    y + (σ-ρ) p_j + ln Λ_j - σ a_j + (σ-1) ln A_j - ln(n_j l_j) = 0                 [J]  labor demand
        D      y + q - σ b + (σ-1) ln A_k - k = 0                                                  capital demand
        Q      1 - Σ_j exp((σ-ρ) p_j + ln Λ_kj - q) = 0                                            definition of q
        B      1 - Σ_j exp((1-ρ) p_j) = 0                                                          price index
        KR     Σ n c_0 + K - Y_bar = 0                                                             first-period resources

"""""""""""

import numpy as np
import pandas as pd
import cyipopt as cp
import Production_Functions as fn
import Processing_Functions as gpf
import Nested_Production as npf
import Nested_Economy as ne



# --------------------- #
# Indexing and Helpers  #
# --------------------- #

def _idx(J):
    "Positions of each block of z"
    return dict(c0=np.arange(0, J), l=np.arange(J, 2*J), x=np.arange(2*J, 3*J), m=3*J,
                a=np.arange(3*J+1, 4*J+1), b=4*J+1, y=4*J+2, p=np.arange(4*J+3, 5*J+3), q=5*J+3, k=5*J+4)


def _unpack(z, J):
    return (z[:J], z[J:2*J], z[2*J:3*J], z[3*J], z[3*J+1:4*J+1], z[4*J+1], z[4*J+2], z[4*J+3:5*J+3], z[5*J+3], z[5*J+4])


def _lam(x, x_bar, ζ, ν, σ):
    "ln Λ_j, ln Λ_kj and their first and second derivatives wrt x_j"
    e_l = ζ*ν*(σ-1) + 1
    e_k = ζ*(ν-1)*(σ-1) + 1
    Λl = fn.Lambda_l(x, x_bar, ζ, ν, σ)
    Λk = fn.Lambda_kj(x, ζ, ν, σ)
    dΛl = -x**(e_l-1);              d2Λl = -(e_l-1) * x**(e_l-2)
    dΛk = x**(e_k-1);               d2Λk = (e_k-1) * x**(e_k-2)
    κ_l = dΛl / Λl;                 κ2_l = d2Λl / Λl - κ_l**2
    κ_k = dΛk / Λk;                 κ2_k = d2Λk / Λk - κ_k**2
    return np.log(Λl), np.log(Λk), κ_l, κ2_l, κ_k, κ2_k


def _u(c0, m, var_θ, β_t):
    "Present-value consumption utility terms (up to constants) and derivatives, as functions of (c_0, m) with c_1 = c_0 e^m"
    em1 = np.exp((1-var_θ)*m)
    u    = c0**(1-var_θ) / (1-var_θ) * (1 + β_t*em1) if var_θ != 1 else np.log(c0) * (1 + β_t) + β_t*m
    u_c  = c0**(-var_θ) * (1 + β_t*em1)
    u_cc = -var_θ * c0**(-var_θ-1) * (1 + β_t*em1)
    u_cm = c0**(-var_θ) * β_t * (1-var_θ) * em1
    u_m  = β_t * c0**(1-var_θ) * em1
    u_mm = β_t * (1-var_θ) * c0**(1-var_θ) * em1
    return u, u_c, u_cc, u_cm, u_m, u_mm


def _consts(args):
    n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, ρ, β, var_θ, φ, ε, x_bar, J = args
    β_t = β / (1 - β * (1+g)**(1-var_θ))
    β_l = β / (1-β)
    return β_t, β_l



# ------------------------------- #
# Starting Prices from Allocation #
# ------------------------------- #

def Start_Vector(E, args):
    "Planner vector z (and price block) for an allocation E = (c_0, c_1, l, x), prices from the nested equilibrium price system"
    n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, ρ, β, var_θ, φ, ε, x_bar, J = args
    c_0 = E[:J]; l = E[2*J:3*J]; x = E[3*J:4*J]
    K = Y_bar - np.sum(n * c_0); L = n * l
    u = npf.Prices(x, L, K, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    Λ_j, Λ_kj, dΛ_j, dΛ_kj, lnP, a_k, a_l, c_x = npf._blocks(u, x, A_j, A_k, x_bar, ζ, ν, σ, ρ)
    m = np.median(np.log(E[J:2*J] / c_0))
    q = np.log(np.sum(np.exp((σ-ρ)*lnP) * Λ_kj))
    return np.concatenate((c_0, l, x, [m], u[:J], [u[J]], [u[J+1]], lnP, [q], [np.log(K)]))


def Alloc(z, J):
    "Allocation E = (c_0, c_1, l, x) and (w, r) from planner vector"
    c0, l, x, m, a, b, y, p, q, k = _unpack(z, J)
    return np.concatenate((c0, c0*np.exp(m), l, x)), np.exp(a), np.exp(b)



# ---------------------- #
# Objective and Constraints #
# ---------------------- #

def obj(z, args):
    "Negative welfare"
    n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, ρ, β, var_θ, φ, ε, x_bar, J = args
    c0, l, x, m, a, b, y, p, q, k = _unpack(z, J)
    return -np.sum(n * fn.V(c0, c0*np.exp(m), l, β, var_θ, φ, ε, g))


def obj_grad(z, args):
    n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, ρ, β, var_θ, φ, ε, x_bar, J = args
    c0, l, x, m, a, b, y, p, q, k = _unpack(z, J)
    β_t, β_l = _consts(args)
    _, u_c, _, _, u_m, _ = _u(c0, m, var_θ, β_t)
    grad = np.zeros(5*J+5)
    grad[:J] = -n * u_c
    grad[J:2*J] = n * β_l * φ * l**(1/ε)
    grad[3*J] = -np.sum(n * u_m)
    return grad


def eq(z, θ, args):
    n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, ρ, β, var_θ, φ, ε, x_bar, J = args
    c0, l, x, m, a, b, y, p, q, k = _unpack(z, J)
    δt = 1 + δ + g
    lnΛl, lnΛk, _, _, _, _ = _lam(x, x_bar, ζ, ν, σ)
    K = np.exp(k); Y = np.exp(y)
    RC = np.sum(n * c0 * np.exp(m)) - Y - (1-δt) * K
    AT = ζ*np.log(x) - ((a - np.log(A_j)) - (np.log(1+θ) + b - np.log(A_k)))
    cap = lnΛk + (1-σ)*(b - np.log(A_k))
    lab = lnΛl + (1-σ)*(a - np.log(A_j))
    P = np.logaddexp(cap, lab) - (1-σ)*p
    C = y + (σ-ρ)*p + lnΛl - σ*a + (σ-1)*np.log(A_j) - np.log(n*l)
    t = np.exp((σ-ρ)*p + lnΛk - q)
    D = y + q - σ*b + (σ-1)*np.log(A_k) - k
    Q = 1 - np.sum(t)
    B = 1 - np.sum(np.exp((1-ρ)*p))
    KR = np.sum(n*c0) + K - Y_bar
    return np.concatenate(([RC], AT, P, C, [D, Q, B, KR]))


def eq_jac_triplets(z, θ, args):
    "Jacobian of the equality constraints in triplet form (fixed pattern)"
    n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, ρ, β, var_θ, φ, ε, x_bar, J = args
    c0, l, x, m, a, b, y, p, q, k = _unpack(z, J)
    I = _idx(J); δt = 1 + δ + g; d = σ - ρ
    lnΛl, lnΛk, κ_l, κ2_l, κ_k, κ2_k = _lam(x, x_bar, ζ, ν, σ)
    K = np.exp(k); Y = np.exp(y)
    cap = lnΛk + (1-σ)*(b - np.log(A_k)); lab = lnΛl + (1-σ)*(a - np.log(A_j))
    α_l = np.exp(lab - np.logaddexp(cap, lab)); α_k = 1 - α_l
    t = np.exp(d*p + lnΛk - q)
    π = np.exp((1-ρ)*p)
    ar = np.arange(J)
    R0, RAT, RP, RC_, RD, RQ, RB, RK = 0, 1, J+1, 2*J+1, 3*J+1, 3*J+2, 3*J+3, 3*J+4

    rows = []; cols = []; vals = []
    def add(r, c, v):
        r = np.atleast_1d(r); c = np.atleast_1d(c); v = np.atleast_1d(v)
        shp = np.broadcast(r, c, v).shape
        rows.append(np.broadcast_to(r, shp).ravel()); cols.append(np.broadcast_to(c, shp).ravel()); vals.append(np.broadcast_to(v, shp).ravel())

    # RC_1
    add(np.full(J, R0), I['c0'], n*np.exp(m))
    add([R0], [I['m']], [np.sum(n*c0*np.exp(m))])
    add([R0], [I['y']], [-Y])
    add([R0], [I['k']], [(δ+g)*K])
    # AT_j
    add(RAT+ar, I['x'], ζ/x)
    add(RAT+ar, I['a'], -np.ones(J))
    add(RAT+ar, np.full(J, I['b']), np.ones(J))
    # P_j
    add(RP+ar, I['x'], α_k*κ_k + α_l*κ_l)
    add(RP+ar, I['a'], (1-σ)*α_l)
    add(RP+ar, np.full(J, I['b']), (1-σ)*α_k)
    add(RP+ar, I['p'], -(1-σ)*np.ones(J))
    # C_j
    add(RC_+ar, np.full(J, I['y']), np.ones(J))
    add(RC_+ar, I['p'], d*np.ones(J))
    add(RC_+ar, I['x'], κ_l)
    add(RC_+ar, I['a'], -σ*np.ones(J))
    add(RC_+ar, I['l'], -1/l)
    # D
    add([RD]*4, [I['y'], I['q'], I['b'], I['k']], [1.0, 1.0, -σ, -1.0])
    # Q
    add(np.full(J, RQ), I['p'], -d*t)
    add(np.full(J, RQ), I['x'], -t*κ_k)
    add([RQ], [I['q']], [np.sum(t)])
    # B
    add(np.full(J, RB), I['p'], -(1-ρ)*π)
    # KR
    add(np.full(J, RK), I['c0'], n)
    add([RK], [I['k']], [K])

    return np.concatenate(rows).astype(np.int64), np.concatenate(cols).astype(np.int64), np.concatenate(vals)


def ic_values(z, WS, args):
    "IC values on a working set"
    n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, ρ, β, var_θ, φ, ε, x_bar, J = args
    c0, l, x, m, a, b, y, p, q, k = _unpack(z, J)
    β_t, β_l = _consts(args)
    κ = 1 + 1/ε
    Val = fn.V(c0, c0*np.exp(m), l, β, var_θ, φ, ε, g)
    Val_con = Val + β_l * φ * l**κ / κ
    i = WS[:, 0]; j = WS[:, 1]
    im = np.exp(a[j] - a[i]) * l[j]
    return Val[i] - Val_con[j] + β_l * φ[i] * im**κ / κ


def ic_matrix(z, args):
    "Full J x J matrix of IC values (rows: truthful type i, columns: mimicked type j)"
    n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, ρ, β, var_θ, φ, ε, x_bar, J = args
    c0, l, x, m, a, b, y, p, q, k = _unpack(z, J)
    β_t, β_l = _consts(args)
    κ = 1 + 1/ε
    Val = fn.V(c0, c0*np.exp(m), l, β, var_θ, φ, ε, g)
    Val_con = Val + β_l * φ * l**κ / κ
    im = np.exp(a[None, :] - a[:, None]) * l[None, :]
    return Val[:, None] - Val_con[None, :] + β_l * φ[:, None] * im**κ / κ


# IC derivative columns, in order, as positions in z
def _ic_cols(WS, J):
    i = WS[:, 0]; j = WS[:, 1]
    return [i, j, J+i, J+j, np.full(i.shape, 3*J), 3*J+1+i, 3*J+1+j]     # c0_i, c0_j, l_i, l_j, m, a_i, a_j


def ic_jac_vals(z, WS, args):
    "Jacobian of ICs wrt (c0_i, c0_j, l_i, l_j, m, a_i, a_j), shape (P, 7)"
    n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, ρ, β, var_θ, φ, ε, x_bar, J = args
    c0, l, x, m, a, b, y, p, q, k = _unpack(z, J)
    β_t, β_l = _consts(args)
    κ = 1 + 1/ε
    i = WS[:, 0]; j = WS[:, 1]
    _, u_c, _, _, u_m, _ = _u(c0, m, var_θ, β_t)
    H = β_l * φ[i] * (np.exp(a[j] - a[i]) * l[j])**κ
    V = np.empty((WS.shape[0], 7))
    V[:, 0] = u_c[i]
    V[:, 1] = -u_c[j]
    V[:, 2] = -β_l * φ[i] * l[i]**(1/ε)
    V[:, 3] = H / l[j]
    V[:, 4] = u_m[i] - u_m[j]
    V[:, 5] = -H
    V[:, 6] = H
    return V


def hess_triplets(z, λ, f, WS, θ, args):
    "Lower-triangle Hessian of the Lagrangian f*obj + λ·(eq, ic) in triplet form (fixed pattern)"
    n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, ρ, β, var_θ, φ, ε, x_bar, J = args
    c0, l, x, m, a, b, y, p, q, k = _unpack(z, J)
    I = _idx(J); d = σ - ρ; κ = 1 + 1/ε
    β_t, β_l = _consts(args)
    ar = np.arange(J); meq = 3*J + 5
    λ_RC = λ[0]; λ_AT = λ[1:J+1]; λ_P = λ[J+1:2*J+1]; λ_C = λ[2*J+1:3*J+1]
    λ_D, λ_Q, λ_B, λ_K = λ[3*J+1:3*J+5]
    λ_ic = λ[meq:]
    lnΛl, lnΛk, κ_l, κ2_l, κ_k, κ2_k = _lam(x, x_bar, ζ, ν, σ)
    K = np.exp(k); Y = np.exp(y)
    cap = lnΛk + (1-σ)*(b - np.log(A_k)); lab = lnΛl + (1-σ)*(a - np.log(A_j))
    α_l = np.exp(lab - np.logaddexp(cap, lab)); α_k = 1 - α_l
    t = np.exp(d*p + lnΛk - q)
    π = np.exp((1-ρ)*p)

    rows = []; cols = []; vals = []
    def add(r, c, v):
        r = np.atleast_1d(r); c = np.atleast_1d(c); v = np.atleast_1d(v)
        shp = np.broadcast(r, c, v).shape
        rows.append(np.broadcast_to(r, shp).ravel()); cols.append(np.broadcast_to(c, shp).ravel()); vals.append(np.broadcast_to(v, shp).ravel())

    # Objective (−W): second derivatives of u-terms and labor
    _, u_c, u_cc, u_cm, u_m, u_mm = _u(c0, m, var_θ, β_t)
    add(I['c0'], I['c0'], -f * n * u_cc)
    add(I['c0'], I['m'], -f * n * u_cm)
    add(I['m'], I['m'], -f * np.sum(n * u_mm))
    add(I['l'], I['l'], f * n * β_l * φ * (1/ε) * l**(1/ε - 1))

    # RC_1: Σ n c0 e^m − Y + (δ+g) K
    em = np.exp(m)
    add(I['c0'], I['m'], λ_RC * n * em)
    add(I['m'], I['m'], λ_RC * np.sum(n * c0 * em))
    add(I['y'], I['y'], -λ_RC * Y)
    add(I['k'], I['k'], λ_RC * (δ+g) * K)

    # AT_j
    add(I['x'], I['x'], -λ_AT * ζ / x**2)

    # P_j: blocks in (x_j, a_j, b)
    g_x = α_k*κ_k + α_l*κ_l; g_a = (1-σ)*α_l; g_b = (1-σ)*α_k
    Hxx = α_k*κ2_k + α_l*κ2_l + α_k*κ_k**2 + α_l*κ_l**2 - g_x**2
    Hxa = α_l*κ_l*(1-σ) - g_x*g_a
    Hxb = α_k*κ_k*(1-σ) - g_x*g_b
    Haa = α_l*(1-σ)**2 - g_a**2
    Hbb = α_k*(1-σ)**2 - g_b**2
    Hab = -g_a*g_b
    add(I['x'], I['x'], λ_P * Hxx)
    add(I['a'], I['x'], λ_P * Hxa)
    add(np.full(J, I['b']), I['x'], λ_P * Hxb)
    add(I['a'], I['a'], λ_P * Haa)
    add(np.full(J, I['b']), I['a'], λ_P * Hab)
    add([I['b']], [I['b']], [np.sum(λ_P * Hbb)])

    # C_j
    add(I['x'], I['x'], λ_C * κ2_l)
    add(I['l'], I['l'], λ_C / l**2)

    # Q
    add(I['p'], I['p'], -λ_Q * d**2 * t)
    add(I['x'], I['p'], -λ_Q * d * t * κ_k)
    add(I['x'], I['x'], -λ_Q * t * (κ_k**2 + κ2_k))
    add(np.full(J, I['q']), I['p'], λ_Q * d * t)
    add(np.full(J, I['q']), I['x'], λ_Q * t * κ_k)
    add([I['q']], [I['q']], [-λ_Q * np.sum(t)])

    # B
    add(I['p'], I['p'], -λ_B * (1-ρ)**2 * π)

    # KR
    add([I['k']], [I['k']], [λ_K * K])

    # IC: 12 entries per pair
    P = WS.shape[0]
    if P > 0:
        i = WS[:, 0]; j = WS[:, 1]
        _, u_c, u_cc, u_cm, u_m, u_mm = _u(c0, m, var_θ, β_t)
        H = β_l * φ[i] * (np.exp(a[j] - a[i]) * l[j])**κ
        w_ = λ_ic
        ci, cj, li, lj, mm, ai, aj = i, j, J+i, J+j, np.full(P, 3*J), 3*J+1+i, 3*J+1+j
        add(ci, ci, w_ * u_cc[i]); add(cj, cj, -w_ * u_cc[j])
        add(ci, mm, w_ * u_cm[i]);  add(cj, mm, -w_ * u_cm[j])
        add(mm, mm, w_ * (u_mm[i] - u_mm[j]))
        add(li, li, -w_ * β_l * φ[i] * (1/ε) * l[i]**(1/ε - 1))
        add(lj, lj, w_ * (κ-1) * H / l[j]**2)
        add(ai, ai, w_ * κ * H); add(aj, aj, w_ * κ * H); add(aj, ai, -w_ * κ * H)
        add(lj, aj, w_ * κ * H / l[j]); add(lj, ai, -w_ * κ * H / l[j])

    r = np.concatenate(rows).astype(np.int64); c = np.concatenate(cols).astype(np.int64); v = np.concatenate(vals)
    lo = np.maximum(r, c); hi = np.minimum(r, c)      #Lower triangle
    return lo, hi, v



# ------------ #
# NLP Problem  #
# ------------ #

class _NestedNLP:
    "Cyipopt Problem Object"
    def __init__(self, WS, θ, args):
        self.WS, self.θ, self.args = WS, θ, args
        J = args[-1]; self.J = J
        self.P = WS.shape[0]
        self.nz = 5*J + 5
        self.meq = 3*J + 5
        z_p = np.concatenate((np.ones(3*J)*0.5, [0.0], np.zeros(J), [0.0, 0.0], np.zeros(J), [0.0, 0.0]))
        z_p[2*J:3*J] = 0.5 * args[-2]
        er, ec, _ = eq_jac_triplets(z_p, θ, args)
        P = self.P
        ic_rows = np.repeat(np.arange(self.meq, self.meq + P), 7)
        ic_cols = np.stack(_ic_cols(WS, J), axis=1).ravel()
        self._rows = np.concatenate((er, ic_rows)); self._cols = np.concatenate((ec, ic_cols))
        self._neq = er.size
        lo, hi, _ = hess_triplets(z_p, np.ones(self.meq + P), 1.0, WS, θ, args)
        lin = lo * self.nz + hi
        uniq, inv = np.unique(lin, return_inverse=True)
        self._hrows = uniq // self.nz; self._hcols = uniq % self.nz; self._inv = inv; self._nh = uniq.size

    def objective(self, z): return obj(z, self.args)
    def gradient(self, z): return obj_grad(z, self.args)
    def constraints(self, z):
        return np.concatenate((eq(z, self.θ, self.args), ic_values(z, self.WS, self.args)))
    def jacobian(self, z):
        _, _, ev = eq_jac_triplets(z, self.θ, self.args)
        return np.concatenate((ev, ic_jac_vals(z, self.WS, self.args).ravel()))
    def jacobianstructure(self): return self._rows, self._cols
    def hessian(self, z, lagrange, obj_factor):
        _, _, v = hess_triplets(z, lagrange, obj_factor, self.WS, self.θ, self.args)
        return np.bincount(self._inv, weights=v, minlength=self._nh)
    def hessianstructure(self): return self._hrows, self._hcols


def solve_planner(z_0, θ, args, WS, exact, options=None):
    "Solve nested Mirrlees with an IC subset; returns the planner vector and the Ipopt status"
    n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, ρ, β, var_θ, φ, ε, x_bar, J = args
    BIG = 50.0
    lb = np.concatenate((np.ones(3*J)*1e-2, [-10.0], -BIG*np.ones(J), [-BIG, -BIG], -BIG*np.ones(J), [-BIG, -BIG]))
    ub = np.concatenate((np.ones(2*J)*1e4, np.ones(J)*x_bar, [10.0], BIG*np.ones(J), [BIG, BIG], BIG*np.ones(J), [BIG, BIG]))
    P = WS.shape[0]; meq = 3*J + 5
    cl = np.zeros(meq + P); cu = np.concatenate((np.zeros(meq), np.full(P, 2.0e19)))

    nlp = cp.Problem(n=5*J+5, m=meq+P, problem_obj=_NestedNLP(WS, θ, args), lb=lb, ub=ub, cl=cl, cu=cu)
    opts = {'mu_strategy': 'adaptive', 'linear_solver': 'mumps', 'print_level': 0, 'sb': 'yes'}
    if options: opts.update(options)
    for k_, v_ in opts.items():
        nlp.add_option(k_, v_)
    if exact is True:
        nlp.add_option('hessian_approximation', 'exact')
    else:
        nlp.add_option('hessian_approximation', 'limited-memory')
        nlp.add_option('limited_memory_max_history', 10)

    z_opt, info = nlp.solve(z_0)
    return z_opt, info['status']



# -------------- #
# Working Set    #
# -------------- #

def build_working_set(z, args, IC_k, IC_slack):
    "Initial IC working set: neighbors in the MRS ordering and near-binding pairs (Processing_Functions.build_working_set)"
    n, Y_bar, δ, g, A_j, A_k, ζ, ν, σ, ρ, β, var_θ, φ, ε, x_bar, J = args
    c0, l, x, m, a, b, y, p, q, k = _unpack(z, J)
    c1 = c0 * np.exp(m); w = np.exp(a)
    β_t = β / (1 - β * (1+g)**(1-var_θ)); β_l = β / (1-β)
    MRS_order = (β_l / β_t) * (c1**var_θ) * φ * (l**(1/ε)) / w
    order = np.argsort(MRS_order)

    keep = np.zeros((J, J), dtype=bool)
    for pos in range(J):
        i = order[pos]
        for q_ in range(max(0, pos - IC_k), min(J, pos + IC_k + 1)):
            if q_ != pos:
                keep[i, order[q_]] = True

    dev = ic_matrix(z, args) / (c0**(-var_θ) * c0)[:, None]
    keep |= (dev < IC_slack)
    np.fill_diagonal(keep, False)
    return np.argwhere(keep).astype(np.int64)


def verify_working_set(z, WS, args, tol, max_add=None):
    "Add violated ICs outside the working set, worst first"
    J = args[-1]; var_θ = args[-5]
    c0 = z[:J]
    dev = -np.minimum(ic_matrix(z, args), 0.0) / (c0**(-var_θ) * c0)[:, None]
    in_set = np.zeros((J, J), dtype=bool); in_set[WS[:, 0], WS[:, 1]] = True
    cand = np.argwhere((dev > tol) & (~in_set))
    if cand.shape[0] == 0:
        return WS, 0
    order = np.argsort(-dev[cand[:, 0], cand[:, 1]])
    if max_add is not None:
        order = order[:max_add]
    add = cand[order].astype(np.int64)
    return np.unique(np.vstack((WS, add)), axis=0), add.shape[0]



# ------ #
# Driver #
# ------ #

def Mirrlees_Lagr(Econ, E, θ_on, tol=1e-6, max_iter=10_000, exact=True, θ_init=0.15, verbose=False):
    "Solve the non-linear tax problem in the nested economy (Economy.Mirrlees_Lagr with the full-space planner); E = (c_0, c_1, l, x) start"

    Y_0 = Econ.Y_sq / (1+Econ.g); K_0 = Econ.K_sq / (1+Econ.g)
    Y_bar = Y_0 + (1-Econ.δ) * K_0
    args = (Econ.n, Y_bar, Econ.δ, Econ.g, Econ.A_j, Econ.A_k, Econ.ζ, Econ.ν, Econ.σ, Econ.ρ,
            Econ.β, Econ.var_θ, Econ.φ, Econ.ε, Econ.x_bar, Econ.J)
    IC_k = 2
    J = Econ.J

    def solve_at_θ(z_init, θ, WS=None, use_exact=False):
        z = z_init.copy()
        exactness_set = [False, True] if use_exact else [False]
        WS_set = 0
        for exact_ in exactness_set:
            IC_slack = 0.01 if exact_ else 0.05
            WS = build_working_set(z, args, IC_k, IC_slack) if (WS is None or WS_set > 0) else WS
            WS_set += 1
            converged = False
            for _ in range(max_iter):
                z, status = solve_planner(z, θ, args, WS, exact_)
                if verbose: print(f'  [θ={θ:.4f} exact={exact_}] status={status}, -obj={-obj(z, args):.5f}, |WS|={WS.shape[0]}', flush=True)
                if status in (0, 1, -1):
                    WS, added = verify_working_set(z, WS, args, tol, max_add=300)
                    if verbose: print(f'     added {added} ICs', flush=True)
                    if added == 0:
                        converged = True
                        break
                else:
                    break
        c0, l, x, m, a, b, y, p, q, k = _unpack(z, J)
        c1 = c0 * np.exp(m)
        MRS_c = fn.cap_MRS(c0, c1, Econ.β, Econ.var_θ, Econ.g)
        τ_k = np.median(1 - (MRS_c + Econ.g) / (np.exp(b) - Econ.δ))
        return z, τ_k, converged, WS

    def out(z):
        c0, l, x, m, a, b, y, p, q, k = _unpack(z, J)
        return (c0, c0*np.exp(m), l, np.exp(k), x, None)

    z = Start_Vector(E, args)
    if θ_on == 0:
        z, τ_k, converged, _ = solve_at_θ(z, 0.0, use_exact=exact)
        c0, c1, l, K, x, _ = out(z)
        return (c0, c1, l, K, x, τ_k), converged, z

    θ_0 = θ_init
    z, τ_0, _, WS = solve_at_θ(z, θ_0)
    θ_1 = θ_0 + 0.01
    z, τ_1, converged, WS = solve_at_θ(z, θ_1, WS)
    for _ in range(max_iter):
        if abs(τ_1) < tol: break
        dτ = τ_1 - τ_0
        if abs(dτ) < 1e-12: break
        θ_2 = min(max(θ_1 - τ_1 * (θ_1 - θ_0) / dτ, 0), 5.0)
        θ_0, τ_0 = θ_1, τ_1
        θ_1 = θ_2
        z_new, τ_1, converged, WS_new = solve_at_θ(z, θ_1, WS)
        if converged:
            z = z_new; WS = WS_new
        if verbose: print(f'θ={θ_1:.5f} τ_k={τ_1:.3g}', flush=True)
    converged = converged and (abs(τ_1) < tol)
    c0, c1, l, K, x, _ = out(z)
    return (c0, c1, l, K, x, τ_1), θ_1, converged, z



# ------------ #
# Coarse Start #
# ------------ #

def Mirrlees_Start(Econ, θ_on, J_c=100, n_starts=12, x_noise=0.3):
    "Starting allocations for the non-linear tax problem (Economy.Mirrlees_Start for the nested economy): converged random-automation starts on a coarse-grained economy, best first, mapped to occupations"

    ACS16_df = pd.read_pickle(f'{Econ.Directory}/Clean Data/ACS16.pkl')
    Webb_df = pd.read_pickle(f'{Econ.Directory}/Clean Data/Webb.pkl')
    groups = gpf.cluster_occupations(ACS16_df, Webb_df, J_c)

    E_c = ne.Nested_Economy(J_c, Econ.ρ, groups); E_c.σ, E_c.Σ_k = Econ.σ, Econ.Σ_k; E_c.Calibrate()
    x_sq = E_c.var_κ * E_c.x_bar
    E_init = np.concatenate((E_c.c_0_sq, E_c.c_1_sq, E_c.l_j_sq, x_sq))
    alloc_args = (E_c.A_j, E_c.A_k, E_c.x_bar, E_c.ζ, E_c.ν, E_c.σ, E_c.ρ, J_c, E_c.n, E_c.β, E_c.var_θ, E_c.ε, E_c.δ, E_c.g, E_c.φ)
    E_0 = np.concatenate(ne.solve_eqbm(0, 0.0, E_c.Ψ, E_c.ψ, np.sum(E_c.n * E_c.y_0), E_init, *alloc_args))

    starts = []
    for s in range(n_starts):
        E_s = E_0.copy()
        x_s = E_s[3*J_c:] * np.exp(x_noise * np.random.default_rng(s).standard_normal(J_c))
        E_s[3*J_c:] = np.minimum(x_s, 0.95 * E_c.x_bar)
        out = Mirrlees_Lagr(E_c, E_s, θ_on, exact=False, max_iter=30)
        (c_0, c_1, l, _, x, τ_k), converged = out[0], out[2 if θ_on == 1 else 1]
        θ = out[1] if θ_on == 1 else 0.0
        W = np.sum(E_c.n * fn.V(c_0, c_1, l, E_c.β, E_c.var_θ, E_c.φ, E_c.ε, E_c.g))
        print(f'Coarse start {s}: converged={converged}, θ={θ:.4g}, τ_k={τ_k:.4g}, welfare={W:.5f}', flush=True)
        if converged and (0 < τ_k < 1 if θ_on == 0 else 0 < θ < 5):
            E_start = np.concatenate((c_0[groups] * Econ.c_0_sq / E_c.c_0_sq[groups],
                                      c_1[groups] * Econ.c_1_sq / E_c.c_1_sq[groups],
                                      l[groups] * Econ.l_j_sq / E_c.l_j_sq[groups],
                                      np.minimum(x[groups], 0.95 * Econ.x_bar)))
            starts.append((W, θ, E_start))
    if not starts:
        raise RuntimeError('No coarse start converged')
    starts.sort(key=lambda start: -start[0])
    return [(θ, E_start) for _, θ, E_start in starts]
