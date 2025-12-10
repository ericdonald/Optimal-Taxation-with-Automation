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
        self.var_κ = 0.2 #Normalized proportion of initial task
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
        
        R = (1-self.τ_k) * (self.r_sq - self.δ) - self.g
        β_g = 1 - R
        c_0_g = (R*self.y_0 + self.Ψ * (wl_1)**(1-self.ψ) + D_1) / (R + (R * β_g / (1-β_g))**(1/self.var_θ))
        
        CSQ_g = np.concatenate((c_0_g, np.array([β_g])))
        
        CSQ = sp.optimize.root(rt.ConCalRoot, CSQ_g,
                      args=(self.J, self.Y_sq, self.K_sq, self.G, self.n, self.w_j_sq, self.l_j_sq, self.y_0, self.r_sq, self.δ, self.g, self.τ_k, self.Ψ, self.ψ, self.var_θ),
                      method='lm')
                
        self.c_0_sq = CSQ.x[:self.J]
        self.β = CSQ.x[-1]
        
        self.c_1_sq = self.c_0_sq * (R * self.β / (1-self.β))**(1/self.var_θ)
        
        self.φ = (self.Ψ * (1-self.ψ) * (self.w_j_sq)**(1-self.ψ)) / (self.l_j_sq**(self.ψ + 1/self.ε) * self.c_1_sq**(self.var_θ))
        

        # ---------------------------------- #
        # Task-Level Productivity Parameters #
        # ---------------------------------- #
        args = (self.var_κ, self.Σ_k, self.χ, self.x_bar, self.σ, self.S_k, self.S_j_sq, self.J)
        
        a = 0
        b = -np.log(1000) / 100 #Lower bound for ζ of 1/1000
        
        Γ = gpf.bisect_scalar(rt.GammaRoot, a, b, args)
 
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
        
        
        
    def StatusQuo_θ(self, θ_lower, θ_upper):
        "Optimal Threshold Rule Finder with Status Quo Taxes"
        
        E_sq = np.concatenate((self.c_0_sq, self.c_1_sq, self.l_j_sq, self.var_κ * self.x_bar))
        args = (E_sq, self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ, self.J, self.n, self.y_0, self.Ψ, self.ψ, self.β, self.var_θ, self.ε, self.τ_k, self.δ, self.g, self.φ)
        
        qe.tic()
        θ = gpf.bisect_scalar(rt.Optimalθ_SQ_Root, θ_lower, θ_upper, args)
        qe.toc()
        print("Status Quo θ Found")
        
        return θ
    
    
    
    def Mirrlees_Lagr_NT(self, damp=1/5, tol=1e-5, max_iter=10000):
        "Solve Non-Linear Tax Problem without Threshold Rule"
            
        E = np.concatenate((self.c_0_sq, self.c_1_sq, self.l_j_sq))
        x = self.var_κ * self.x_bar
        
        Y_0 = self.Y_sq / (1+self.g)
        K_0 = self.K_sq / (1+self.g)
        Y_bar = Y_0 + (1-self.δ) * K_0
        
        args = (self.n, Y_bar, self.δ, self.g, self.A_j, self.A_k, self.β, self.var_θ, self.φ, self.ε, self.J)
        
        w = self.w_j_sq
        r = self.r_sq
        
        
        qe.tic()
        # ---------- #
        # Outer Loop #
        # ---------- #
        for _ in range(max_iter):
        
            
            # ---------------- #
            # Solve Inner Loop #
            # ---------------- #
            E = gpf.inner_solve(w, r, E, args)
            c_0 = E[:self.J]
            c_1 = E[self.J:2*self.J]
            l = E[2*self.J:3*self.J]
            K = Y_bar - np.sum(self.n * c_0)
            
            
            # -------------------- #
            # Update Factor Prices #
            # -------------------- #
            L = self.n * l
            w = fn.Wages(x, L, K, self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ)
            r = fn.Rents(x, L, K, self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ)
            
            
            # -------------------- #
            # Solve for Thresholds #
            # -------------------- #
            x_new = ((w / self.A_j) / (r / self.A_k))**(1/self.ζ)
            
            
            # ---------------------------- #
            # Check Convergence and Update #
            # ---------------------------- #
            error_x = np.max(np.abs(x - x_new))
            #print(error_x)
            
            if error_x < tol:
                break
            
            x = x * (1-damp) + x_new * damp
            
            w = fn.Wages(x, L, K, self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ)
            r = fn.Rents(x, L, K, self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ)
            
            
        qe.toc()
        print("Mirrlees Capital Tax Solution Found")
        
        return (c_0, c_1, l, K, x)
        
        
        
    def Mirrlees_Lagr_θ(self, E, x, θ_lower, θ_upper, θ=0.25, damp=1/10, tol=1e-8, max_iter=1000):
        "Solve Non-Linear Tax Problem with Threshold Rule"
                    
        Y_0 = self.Y_sq / (1+self.g)
        K_0 = self.K_sq / (1+self.g)
        Y_bar = Y_0 + (1-self.δ) * K_0
        
        args = (self.n, Y_bar, self.δ, self.g, self.A_j, self.A_k, self.β, self.var_θ, self.φ, self.ε, self.J)
        
        c_0 = E[:self.J]
        l = E[2*self.J:3*self.J]
        K = Y_bar - np.sum(self.n * c_0)
        L = self.n * l
        
        w = fn.Wages(x, L, K, self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ)
        r = fn.Rents(x, L, K, self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ)
        
        
        qe.tic()
        # ---------- #
        # Outer Loop #
        # ---------- #
        for _ in range(max_iter):
        
            
            # ---------------- #
            # Solve Inner Loop #
            # ---------------- #
            E = gpf.inner_solve(w, r, E, args)
            c_0 = E[:self.J]
            c_1 = E[self.J:2*self.J]
            l = E[2*self.J:3*self.J]
            K = Y_bar - np.sum(self.n * c_0)
            
            
            # -------------------- #
            # Update Factor Prices #
            # -------------------- #
            L = self.n * l
            w = fn.Wages(x, L, K, self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ)
            r = fn.Rents(x, L, K, self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ)
            
            
            # -------------------- #
            # Solve for Thresholds #
            # -------------------- #
            θ_args = (E, x, w, r, self.n, Y_bar, self.δ, self.g, self.A_j, self.A_k, self.β, self.var_θ, self.φ, self.ε, self.J, self.x_bar, self.ζ, self.ν, self.σ)
            θ_new = gpf.bisect_scalar(rt.Optimalθ_NL_Root, θ_lower, θ_upper, θ_args)
            x_new = ((w / self.A_j) / ((1+θ_new) * r / self.A_k))**(1/self.ζ)
            
            
            # ---------------------------- #
            # Check Convergence and Update #
            # ---------------------------- #
            error_x = np.max(np.abs(x - x_new))
            print(error_x)
            error_θ = np.abs(θ - θ_new)
            print(error_θ)
            
            if error_θ < tol and error_x < tol:
                break
            
            θ = θ * (1-damp) + θ_new * damp
            x = x * (1-damp) + x_new * damp
            
            w = fn.Wages(x, L, K, self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ)
            r = fn.Rents(x, L, K, self.A_j, self.A_k, self.x_bar, self.ζ, self.ν, self.σ)
            
        
        qe.toc()
        print("Mirrlees Threshold Rule Solution Found")
        
        return (c_0, c_1, l, K, x, θ)    
        
        
