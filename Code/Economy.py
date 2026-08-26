"""""""""""
Economy Module

Notes: This file defines a class for the economy of "Optimal Taxation with Automation".
    
"""""""""""

import numpy as np
import pandas as pd
import scipy as sp
import quantecon as qe
from pathlib import Path
import Roots as rt
import Production_Functions as fn
import Processing_Functions as gpf
import Perturbations as pr



class Economy:
    
    def __init__(self, J):
        "Initialize Economy Object"
        
        self.Directory = Path(__file__).resolve().parent.parent
        
        
        # --------------------------------------- #
        # Define Externally Calibrated Parameters #
        # --------------------------------------- #
        self.J = J #Number of Occupations
        self.var_θ = 1 #Marginal Utility Curvature
        self.ε = 0.75 #Frisch Elasticity
        self.σ = 0.5 #Task-Level Elasticity of Substitution
        self.δ = 0.07 #Depreciation Rate
        

        # -------------------------- #
        # Define Calibration Targets #
        # -------------------------- #
        self.COR = 3.5 #Capital-Output Ratio
        self.S_k = 0.4 #Capital Income Share
        self.Y_sq = 100 #Status Quo Output
        self.G = 0.3 #Government Revenue Share
        self.x_bar = np.ones(J)*100 #Task Interval Upper Bounds
        self.var_κ = 0.15 #Normalized proportion of initial task
        self.g = 0.02 #TFP Growth
        self.τ_k = 0.1 #Status Quo Capital Income Tax
        self.ψ = 0.181 #Status Quo Labor Tax Shape
        self.Σ_k = 1.25 #Aggregate Elasticity of Substitution
        
        
        # --------------------------------------- #
        # Define Internally Calibrated Parameters #
        # --------------------------------------- #
        self.Θ = 0 #Wealth Share Convexity
        self.K_sq = self.COR * self.Y_sq #Status Quo Capital
        self.r_sq = self.S_k / self.COR #Status Quo Rent
        self.n = np.zeros(J) #Population
        self.l_j_sq = np.zeros(J) #Status Quo Labor Supply per Person
        self.L_j_sq = np.zeros(J) #Status Quo Labor Supply
        self.w_j_sq = np.zeros(J) #Status Quo Wage
        self.S_j_sq = np.zeros(J) #Status Quo Labor Share
        self.S_jk_sq = np.zeros(J) #Status Quo Wealth Share
        self.c_0_sq = np.zeros(J) #Initial Status Quo Consumption
        self.c_1_sq = np.zeros(J) #Final Status Quo Consumption
        self.y_0 = np.zeros(J) #Initial Post-Tax Income
        self.Ψ = 0 #Status Quo Labor Tax Scale
        self.φ = np.zeros(J) #Labor Disutility
        self.β = 0 #Discount Factor
        self.χ = np.zeros(J) #Webb Exposure Measures
        self.Γ = 0 #Automation Exposure Effect
        self.ν = np.zeros(J) #Absolute Advantage
        self.ζ = np.zeros(J) #Comparative Advantage
        self.A_j = np.zeros(J) #Labor-Augmenting Technology
        self.A_k = 0 #Capital-Augmenting Technology
        


    def Calibrate(self):
        "Calibrate Parameters and Status Quo Allocation"
        
        # --------- #
        # Load Data #
        # --------- #
        SCF_df = pd.read_pickle(f'{self.Directory}/Clean Data/SCF_2016.pkl')
        ACS16_df = pd.read_pickle(f'{self.Directory}/Clean Data/ACS16.pkl')
        Webb_df = pd.read_pickle(f'{self.Directory}/Clean Data/Webb.pkl')


        # ----------------------------- #
        # Wealth Distribution Convexity #
        # ----------------------------- #
        YS = gpf.compute_decile_shares(SCF_df, 'Labor Income')
        WS = gpf.compute_decile_shares(SCF_df, 'Wealth')
        
        Θ_cal = sp.optimize.root(rt.WealthShapeRoot, 1.88,
                                  args=(WS, YS),
                                  method='lm')
        
        self.Θ = Θ_cal.x[0]
        
        
        # ------------ #
        # Labor Market #
        # ------------ #
        ACS16_df['wL'] = ACS16_df['w'] * ACS16_df['L']
        ACS16_df['S'] = (1-self.S_k) * ACS16_df['wL'] / ACS16_df['wL'].sum()
        
        n_16 = ACS16_df['n'].to_numpy()
        l_16 = ACS16_df['l'].to_numpy()
        L_16 = ACS16_df['L'].to_numpy()
        S_16 = ACS16_df['S'].to_numpy()
        
        self.n = n_16
        self.l_j_sq = l_16
        self.L_j_sq = L_16
        self.S_j_sq = S_16
        
        self.S_jk_sq = self.S_j_sq**self.Θ / np.sum(self.S_j_sq**self.Θ)
        
        self.χ = (Webb_df['pct_software'].to_numpy() + Webb_df['pct_robot'].to_numpy()) / 2
        
        
        # ------------------------------ #
        # Consumption & Labor Disutility #
        # ------------------------------ #
        self.w_j_sq = self.S_j_sq * self.Y_sq / self.L_j_sq
        
        Y_0 = self.Y_sq / (1+self.g)
        K_0 = self.K_sq / (1+self.g)
        
        κ_0 = self.S_jk_sq * K_0 / self.n
        wl_1 = self.w_j_sq * self.l_j_sq
        wl_0 = wl_1 / (1+self.g)
        
        self.Ψ = self.Y_sq * ((1-self.S_k) + self.τ_k * (self.r_sq - self.δ) * self.COR - self.G) / (np.sum(self.n * (wl_1)**(1-self.ψ)))
        
        ωY_bar = (self.S_j_sq + self.S_jk_sq * (self.S_k + (1-self.δ) * self.COR)) * Y_0 / self.n
        
        Ψ_0 = self.Ψ * (1+self.g)**(-self.ψ)
        D_0 = self.G * Y_0
        Tax_0 = wl_0 - Ψ_0  * (wl_0)**(1-self.ψ) + self.τ_k * (self.r_sq - self.δ) * κ_0 - D_0
        
        self.y_0 = ωY_bar - Tax_0
        
        D_1 = self.G * self.Y_sq
        
        R_tilde = (1-self.τ_k) * (self.r_sq - self.δ) - self.g
        β_g = 1 - R_tilde
        c_0_g = (R_tilde*self.y_0 + self.Ψ * (wl_1)**(1-self.ψ) + D_1) / (R_tilde + (R_tilde * β_g / (1-β_g))**(1/self.var_θ))
        
        CSQ_g = np.concatenate((c_0_g, np.array([β_g])))
        
        CSQ = sp.optimize.root(rt.ConCalRoot, CSQ_g,
                      args=(self.J, self.Y_sq, self.K_sq, self.G, self.n, self.w_j_sq, self.l_j_sq, self.y_0, self.r_sq, self.δ, self.g, self.τ_k, self.Ψ, self.ψ, self.var_θ),
                      method='lm')
                
        self.c_0_sq = CSQ.x[:self.J]
        self.β = CSQ.x[-1]
        
        beta_tilde = self.β / (1 - self.β * (1+self.g)**(1-self.var_θ))
        self.c_1_sq = self.c_0_sq * (R_tilde * beta_tilde)**(1/self.var_θ)
        
        keep_l = fn.Heath_keep(self.w_j_sq, self.l_j_sq, self.Ψ, self.ψ)
        self.φ = (keep_l * self.w_j_sq) / (self.l_j_sq**(1/self.ε) * self.c_1_sq**(self.var_θ))
        

        # ---------------------------------- #
        # Task-Level Productivity Parameters #
        # ---------------------------------- #
        E_sq = np.concatenate((self.c_0_sq, self.c_1_sq, self.l_j_sq, self.var_κ * self.x_bar))
        args = (E_sq, self.w_j_sq, self.r_sq, self.COR, self.var_κ, self.Σ_k, self.χ, self.x_bar, self.σ, self.S_k, self.S_j_sq, self.J, self.n, self.y_0, κ_0, self.Ψ, self.ψ, self.β, self.var_θ, self.ε, self.τ_k, self.δ, self.g, self.φ)
        
        a = 0
        b = -np.log(1000) / 100 #Lower bound for ζ of 1/1000
        
        Gamma_Σ = lambda x: rt.GammaRoot(x, *args)
        Γ = gpf.secant_scalar(Gamma_Σ, a, b)
 
        self.Γ = Γ 
        self.ζ = np.exp(self.Γ * self.χ)
        
        nu_g = np.ones(self.J) * self.var_κ
        
        nu = sp.optimize.root(rt.νRoot, nu_g,
                      args=(self.Γ, self.χ, self.var_κ, self.x_bar, self.σ, self.S_j_sq, self.S_k),
                      method='lm')
        
        self.ν = nu.x
        
        x_j = self.var_κ * self.x_bar
        Λ_k = fn.Lamba_k(x_j, self.x_bar, self.ζ, self.ν, self.σ)
        self.A_k = self.r_sq**(self.σ / (self.σ-1)) * (self.COR / Λ_k)**(1 / (self.σ-1))
        
        self.A_j = (self.w_j_sq / x_j**(self.ζ)) / (self.r_sq / self.A_k)
        
        
        
    def Para_Solver(self, θ_on, τ_on, HC_on=0, y_0=None, E=None, θ_init=0.0, τ_k_init=None, damp=2/3, tol=1e-4, max_iter=10_000):
        "Solve Parametric Policy Problem"
        
        if y_0 is None:
            y_0 = self.y_0
        args = (self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ, self.J, self.n, y_0, self.β, self.var_θ, self.ε, self.δ, self.g, self.φ)

        if E is None:
            E = np.concatenate((self.c_0_sq, self.c_1_sq, self.l_j_sq, self.var_κ * self.x_bar))
        θ = θ_init
        τ_k = self.τ_k if τ_k_init is None else τ_k_init
        Ψ = self.Ψ
        ψ = self.ψ
        
        
        # ---------- #
        # Outer Loop #
        # ---------- #
        qe.tic()
        for _ in range(max_iter):
            
            cache_θ = [E.copy()]
            cache_τ = [E.copy()]
            cache_ψ = [E.copy()]
            
            
            # ---------------- #
            # Update Threshold #
            # ---------------- #
            if θ_on == 1:
                Optimal_θ_Root = lambda x: rt.Optimal_Para_Root(x, τ_k, Ψ, ψ, 'theta', E, *args, _cache=cache_θ)
                θ_lower = θ/2
                θ_upper = np.maximum(θ * 1.5, 1/3)
                θ_new = gpf.secant_scalar(Optimal_θ_Root, θ_lower, θ_upper, lb=-0.999, ub=2, expansion='additive')
                #print(θ_new)
            else:
                θ_new = 0
            
            
            # ------------------ #
            # Update Capital Tax #
            # ------------------ #
            if τ_on == 1:
                Optimal_τ_Root = lambda x: rt.Optimal_Para_Root(θ, x, Ψ, ψ, 'tau', E, *args, _cache=cache_τ)
                τ_new = gpf.secant_scalar(Optimal_τ_Root, τ_k/2, τ_k * 1.5, ub=1, expansion='additive')
                #print(τ_new)
            else:
                τ_new = self.τ_k
            
            
            # ---------------- #
            # Update Labor Tax #
            # ---------------- #
            if HC_on == 1:
                Optimal_Ψ_Root = lambda x: rt.Optimal_Para_Root(θ, τ_k, x, ψ, 'Psi', E, *args)
                Ψ_new = gpf.secant_scalar(Optimal_Ψ_Root, Ψ/2, Ψ * 1.5, lb=0)
                #print(Ψ_new)
                
                Optimal_ψ_Root = lambda x: rt.Optimal_Para_Root(θ, τ_k, Ψ, x, 'psi', E, *args, _cache=cache_ψ)
                ψ_new = gpf.secant_scalar(Optimal_ψ_Root, ψ/2, ψ * 1.5, ub=1, expansion='additive')
                #print(ψ_new)
            else:
                Ψ_new = self.Ψ
                ψ_new = self.ψ
                
            # ---------------------------- #
            # Check Convergence and Update #
            # ---------------------------- #
            single = θ_on + τ_on + HC_on
            if single == 1:
                θ = θ_new
                τ_k = τ_new
                Ψ = Ψ_new
                ψ = ψ_new
                break
            
            error_θ = np.abs(θ - θ_new)
            #print(f'Threshold Rule Error: {error_θ}')
            
            error_τ = np.abs(τ_k - τ_new)
            #print(f'Capital Tax Error: {error_τ}')
            
            error_Ψ = np.abs(Ψ - Ψ_new)
            #print(f'Labor Tax Scale Error: {error_Ψ}')
            
            error_ψ = np.abs(ψ - ψ_new)
            #print(f'Labor Tax Curvature Error: {error_ψ}')
                
            if error_θ < tol and error_τ < tol and error_Ψ < tol and error_ψ < tol:
                break
            
            θ = θ * (1-damp) + θ_new * damp
            τ_k = τ_k * (1-damp) + τ_new * damp
            Ψ = Ψ * (1-damp) + Ψ_new * damp
            ψ = ψ * (1-damp) + ψ_new * damp
            
            Eqbm = sp.optimize.root(rt.Eqbm_Root, E,
                          args=(θ, τ_k, Ψ, ψ, *args),
                          jac=pr.δH_δclx)
            
            E = Eqbm.x
            
        qe.toc()
        print("Parametric Policy Found")
        
        Out = ()
        
        if θ_on == 1:
            Out += (θ,)
        if τ_on == 1:
            Out += (τ_k,)
        if HC_on == 1:
            Out += (Ψ,)
            Out += (ψ,)
        
        return Out
        
        
        
    def Mirrlees_Lagr(self, E, θ_on, warm=True, max_iter=10_000):
        "Solve Non-Linear Tax Problem"
        
        Y_0 = self.Y_sq / (1+self.g); K_0 = self.K_sq / (1+self.g)
        Y_bar = Y_0 + (1-self.δ) * K_0
        args = (self.n, Y_bar, self.δ, self.g, self.A_j, self.A_k, self.ζ, self.ν, self.σ,
                self.β, self.var_θ, self.φ, self.ε, self.x_bar, self.J)
        IC_k = 2
        J = self.J
        
        if warm == True:
            IC_slack = 0.01
            tol = 1e-4
        else:
            IC_slack = 0.05
            tol = 1e-6
                    
        
        # ----------- #
        # Solve Inner #
        # ----------- #
        def solve_at_θ(E_init, θ):
            E = E_init.copy()
            c_0 = E[:J]; l = E[2*J:3*J]; x = E[3*J:4*J]
            K = Y_bar - np.sum(self.n * c_0); L = self.n * l
            w = fn.Wages(x, L, K, self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ)
            WS = gpf.build_working_set(E, w, args, IC_k, IC_slack)

            converged = False
            minus_one_count = 0
            for _ in range(max_iter):
                print(f'Active ICs: {WS.shape[0]}')
                E, status = gpf.solve_planner(E, θ, args, WS, warm)
                c_0 = E[:J]; l = E[2*J:3*J]; x = E[3*J:4*J]
                K = Y_bar - np.sum(self.n * c_0); L = self.n * l
                w = fn.Wages(x, L, K, self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ)
                
                if status in (0, 1):
                    WS, added = gpf.verify_working_set(E, w, WS, args, tol)
                    if added == 0:
                        converged = True
                        break
                elif status == -1:
                    minus_one_count += 1
                    if minus_one_count >= 2:
                        break
                    WS = gpf.build_working_set(E, w, args, IC_k, IC_slack)
                else:
                    break
                    

            c_0 = E[:J]; c_1 = E[J:2*J]; x = E[3*J:4*J]
            K = Y_bar - np.sum(self.n * c_0); L = self.n * l
            MRS_c = fn.cap_MRS(c_0, c_1, self.β, self.var_θ, self.ε, self.g)
            r = fn.Rents(x, L, K, self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ)
            τ_k = np.median(1 - (MRS_c + self.g) / (r - self.δ))
            return E, τ_k, converged
        
        
        # ----------- #
        # Solve Outer #
        # ----------- #
        qe.tic()
        if θ_on == 0:
            E, τ_k, converged = solve_at_θ(E, 0.0)
            c_0 = E[:J]; c_1 = E[J:2*J]; l = E[2*J:3*J]; x = E[3*J:4*J]
            K = Y_bar - np.sum(self.n * c_0)
            qe.toc(); print("Mirrlees Solution Found")
            return (c_0, c_1, l, K, x, τ_k), converged
        
        
        # Secant θ
        θ_0 = 0.10
        E, τ_0, _ = solve_at_θ(E, θ_0)
        θ_1 = θ_0 + 0.01                                   
        E, τ_1, converged = solve_at_θ(E, θ_1)

        for _ in range(max_iter):
            if abs(τ_1) < tol:
                break
            dτ = τ_1 - τ_0
            if abs(dτ) < 1e-12:
                break
            θ_2 = θ_1 - τ_1 * (θ_1 - θ_0) / dτ
            θ_2 = min(max(θ_2, 0), 5.0)           
            θ_0, τ_0 = θ_1, τ_1
            θ_1 = θ_2
            E_new, τ_1, converged = solve_at_θ(E, θ_1)
            if converged:
                E = E_new
            print(f'θ={θ_1:.5f}')
            print(f'τ_k={τ_1:.5f}')

        τ_k = τ_1; θ_star = θ_1
        converged = converged and (abs(τ_k) < tol)
        c_0 = E[:J]; c_1 = E[J:2*J]; l = E[2*J:3*J]; x = E[3*J:4*J]
        K = Y_bar - np.sum(self.n * c_0)
        
        δW = rt.Optimalθ_NL_Root(E, θ_star, *args)
        denom = np.sum(self.n * (c_0**(-self.var_θ) * c_0))
        δW_error = np.abs(δW) / denom
        if δW_error > 0.1:
            print(f"δW above tolerance: {δW_error:.5f}")
        
        qe.toc(); print(f"Mirrlees Solution Found  (θ={θ_star:.5f}, τ_k={τ_k:.3g})")
        return (c_0, c_1, l, K, x, τ_k), θ_star, converged
        
        
        
    def AI_economy(self, θ_on, τ_on, ζ_AI, ν_AI, A_k_AI_base, A_j_AI_base, LAT_frac, y_0, G_k, N_g, damp=2/3, tol=1e-4, max_iter=10_000):
        "Solve for Post-AI Economy"

        # -------------------------- #
        # Solve for Growth Scenarios #
        # -------------------------- #
        
        AI_growth = np.linspace(0, G_k, N_g)
        E = np.concatenate((self.c_0_sq, self.c_1_sq, self.l_j_sq, self.var_κ * self.x_bar))
        θ_AI = np.zeros(N_g)
        τ_k_AI = self.τ_k * np.ones(N_g)
        
        
        # ---------- #
        # Outer Loop #
        # ---------- #
        qe.tic()
        for n in range(N_g):
            
            A_k_AI = A_k_AI_base * (1 + AI_growth[n])
            A_j_AI = A_j_AI_base * (1 + AI_growth[n] * LAT_frac)
            
            g_y = self.S_k * AI_growth[n] + (1-self.S_k) * AI_growth[n] * LAT_frac
            Ψ_AI = self.Ψ * (1+g_y)**(self.ψ)
            
            args = (A_j_AI, A_k_AI, self.x_bar, ζ_AI, ν_AI, self.σ, self.J, self.n, y_0, self.β, self.var_θ, self.ε, self.δ, self.g, self.φ)
            
            θ = θ_AI[n-1]
            τ_k = τ_k_AI[n-1]
            
            for _ in range(max_iter):
                
                cache_θ = [E.copy()]
                cache_τ = [E.copy()]

                
                # ---------------- #
                # Update Threshold #
                # ---------------- #
                if θ_on == 1:
                    Optimal_θ_Root = lambda x: rt.Optimal_Para_Root(x, τ_k, Ψ_AI, self.ψ, 'theta', E, *args, _cache=cache_θ)
                    θ_lower = θ/2
                    θ_upper = np.maximum(θ * 1.5, 1/3)
                    θ_new = gpf.secant_scalar(Optimal_θ_Root, θ_lower, θ_upper, lb=-0.999, ub=2,  expansion='additive')
                    #print(θ_new)
                else:
                    θ_new = 0
                
                
                # ------------------ #
                # Update Capital Tax #
                # ------------------ #
                if τ_on == 1:
                    Optimal_τ_Root = lambda x: rt.Optimal_Para_Root(θ, x, Ψ_AI, self.ψ, 'tau', E, *args, _cache=cache_τ)
                    τ_new = gpf.secant_scalar(Optimal_τ_Root, τ_k/2, τ_k * 1.5, ub=1, expansion='additive')
                    #print(τ_new)
                else:
                    τ_new = self.τ_k
                    
                # ---------------------------- #
                # Check Convergence and Update #
                # ---------------------------- #
                single = θ_on + τ_on
                if single == 1:
                    θ_AI[n] = θ_new
                    τ_k_AI[n] = τ_new
                    break
                
                error_θ = np.abs(θ - θ_new)
                #print(f'Threshold Rule Error: {error_θ}')
                
                error_τ = np.abs(τ_k - τ_new)
                #print(f'Capital Tax Error: {error_τ}')
                    
                if error_θ < tol and error_τ < tol:
                    break
                
                θ = θ * (1-damp) + θ_new * damp
                τ_k = τ_k * (1-damp) + τ_new * damp
                
                Eqbm = sp.optimize.root(rt.Eqbm_Root, E,
                              args=(θ, τ_k, Ψ_AI, self.ψ, *args),
                              jac=pr.δH_δclx)
                
                E = Eqbm.x
                
            θ_AI[n] = θ_new
            #print(f'AI Experiment Threshold Rule: {θ_AI[n]}')
            τ_k_AI[n] = τ_new
            #print(f'AI Experiment Capital Tax: {τ_k_AI[n]}')
            
            
        qe.toc()
        print("AI Experiment Policy Path Computed")
        
        Out = ()
        
        if θ_on == 1:
            Out += (θ_AI,)
        if τ_on == 1:
            Out += (τ_k_AI,)
        
        return Out
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        