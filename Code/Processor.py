"""""""""""
Processor Module

Notes: This file defines a class for processing the economy of "Optimal Taxation with Automation".
        
"""""""""""

import pandas as pd
import requests as api
import numpy as np
from ipumspy import IpumsApiClient, MicrodataExtract
import scipy as sp
from pathlib import Path
import sys
import pickle

import importlib.metadata as md
import Roots as rt
import Production_Functions as fn
import Perturbations as pr
import Processing_Functions as gpf



class Processor:
    
    def __init__(self, E):
        "Initialize Processor Object"
        
        self.E = E
        self.Directory = Path(__file__).resolve().parent.parent
        
        keys_path = self.Directory / ".keys"
        keys = {}
        with open(keys_path) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    keys[k.strip()] = v.strip()
        
        self.FRED_API = keys.get("FRED_API")
        self.IPUMS_API = keys.get("IPUMS_API")
                
        
        
    def Cleaner(self, ipums_extract=0):
        """""
        Clean Data
        
        Output: Clean Data/FRED_CPI.pkl
                Clean Data/SCF_2016.pkl
                Clean Data/OCC_Crosswalk.pkl
                Clean Data/Census80.pkl
                Clean Data/ACS16.pkl
                Clean Data/Webb.pkl
                Clean Data/CapbyOcc_ES_2d.pkl
        """""
   
        
        # -------- #
        # FRED CPI #
        # -------- #
        FRED = api.get(f'https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&frequency=a&api_key={self.FRED_API}&file_type=json')
        data = FRED.json()['observations']
        filtered_data = [{'date': entry['date'], 'value': entry['value']} for entry in data]
        CPI_df = pd.DataFrame(filtered_data)
        
        CPI_df['date'] = pd.to_datetime(CPI_df['date']).dt.year
        CPI_df.rename(columns={'date': 'year', 'value': 'CPI'}, inplace=True)
        CPI_df['CPI'] = pd.to_numeric(CPI_df['CPI'], errors='coerce')
        CPI_df['CPI'] = CPI_df['CPI'] / CPI_df.loc[CPI_df['year'] == 2016, 'CPI'].values[0] #CPI indexed to 1 in 2016
        
        CPI_df.to_pickle(f'{self.Directory}/Clean Data/FRED_CPI.pkl')
        
        
        # --- #
        # SCF #
        # --- #
        SCF_df = pd.read_csv(f'{self.Directory}/Raw Data/SCF_2016.csv', usecols=['WGT', 'NETWORTH', 'WAGEINC'])
        SCF_df.rename(columns={'WGT': 'Weight', 'NETWORTH': 'Wealth', 'WAGEINC': 'Labor Income'}, inplace=True)
        
        SCF_df.to_pickle(f'{self.Directory}/Clean Data/SCF_2016.pkl')
        
        
        # ----- #
        # IPUMS #
        # ----- #
        if ipums_extract==1:
            ipums = IpumsApiClient(self.IPUMS_API)
            extract = MicrodataExtract(
                collection="usa",
                description="1980 Census & 2016 ACS Extract",
                samples=["us1980a", "us2016a"],
                variables=["YEAR", "PERWT", "AGE", "OCC2010", "WKSWORK2", "UHRSWORK", "WORKEDYR", "INCWAGE"],
                data_format='stata',
                )
            ipums.submit_extract(extract)
        
        IPUMS_df = pd.read_stata(f'{self.Directory}/Raw Data/IPUMS.dta', convert_categoricals=False)
        IPUMS_df = IPUMS_df[['year', 'perwt', 'age', 'occ2010', 'wkswork2', 'uhrswork', 'workedyr', 'incwage']]
        IPUMS_df['occ'] =  IPUMS_df['occ2010'] #Webb maps to occ in the 2010 ACS

        IPUMS_df[['age', 'workedyr', 'wkswork2', 'uhrswork', 'incwage']] = IPUMS_df[['age', 'workedyr', 'wkswork2', 'uhrswork', 'incwage']].apply(pd.to_numeric, errors='coerce')
        IPUMS_df = IPUMS_df[(IPUMS_df['age'] >= 18) & (IPUMS_df['age'] <= 65)]
        IPUMS_df = IPUMS_df[IPUMS_df['workedyr'] == 3]
        IPUMS_df = IPUMS_df[IPUMS_df['wkswork2'] != 0]
        IPUMS_df = IPUMS_df[IPUMS_df['uhrswork'] != 0]
        IPUMS_df = IPUMS_df[IPUMS_df['incwage'] != 0]
        
        conditions = [
            IPUMS_df['wkswork2'] == 1,
            IPUMS_df['wkswork2'] == 2,
            IPUMS_df['wkswork2'] == 3,
            IPUMS_df['wkswork2'] == 4,
            IPUMS_df['wkswork2'] == 5,
            IPUMS_df['wkswork2'] == 6
        ]
        
        choices = [
            (13 + 1) / 2,
            (26 + 14) / 2,
            (39 + 27) / 2,
            (47 + 40) / 2,
            (49 + 48) / 2,
            (52 + 50) / 2
        ]
        
        IPUMS_df['wkswork'] = np.select(conditions, choices, default=0)
        IPUMS_df = IPUMS_df[['year', 'perwt', 'occ', 'wkswork', 'uhrswork', 'incwage']]
        IPUMS_df = IPUMS_df.apply(pd.to_numeric, errors='coerce')

        
        # ------------ #
        # Census & ACS #
        # ------------ #
        Crosswalk_df = pd.read_stata(f'{self.Directory}/Raw Data/onet_to_occ1990dd.dta')
        Crosswalk_df = Crosswalk_df[['occ1990dd', 'occ']]
        Crosswalk_df = Crosswalk_df.drop_duplicates()
        
        Census80_df = IPUMS_df[IPUMS_df['year'] == 1980]
        ACS16_df = IPUMS_df[IPUMS_df['year'] == 2016]
        
        Crosswalk_df = pd.merge(
            Crosswalk_df,
            Census80_df['occ'].drop_duplicates(),
            on='occ',
            how='inner'
        )
        Crosswalk_df = pd.merge(
            Crosswalk_df,
            ACS16_df['occ'].drop_duplicates(),
            on='occ',
            how='inner'
        ) #Only take intersection of available occupations 
        
        Crosswalk_df.to_pickle(f'{self.Directory}/Clean Data/OCC_Crosswalk.pkl')
        
        Census80_df = pd.merge(
            Census80_df,
            Crosswalk_df,
            on='occ',
            how='inner'
        )
        Census80_df = pd.merge(
            Census80_df,
            CPI_df,
            on='year',
            how='inner'
        )
        Census80_df['incwage'] = Census80_df['incwage'] / Census80_df['CPI'] #Make wages real
        
        Census80_df['workforce'] = Census80_df['perwt'].sum()
        Census80_df['n'] = Census80_df.groupby('occ1990dd')['perwt'].transform('sum') / Census80_df['workforce']
        
        Census80_df['person hours'] = Census80_df['perwt'] * Census80_df['wkswork'] * Census80_df['uhrswork']
        Census80_df['l'] = Census80_df.groupby('occ1990dd')['person hours'].transform('sum') / Census80_df.groupby('occ1990dd')['perwt'].transform('sum')
        
        Census80_df['person wages'] = Census80_df['perwt'] * Census80_df['incwage'] / Census80_df['wkswork'] / Census80_df['uhrswork']
        Census80_df['w'] = Census80_df.groupby('occ1990dd')['person wages'].transform('sum') / Census80_df.groupby('occ1990dd')['perwt'].transform('sum')
        
        Census80_df['L'] = Census80_df['n'] * Census80_df['l']
        
        Census80_df = Census80_df[['occ1990dd', 'workforce', 'w', 'L']]
        Census80_df = Census80_df.drop_duplicates()
        Census80_df.sort_values(by='occ1990dd', inplace=True)
        
        Census80_df.to_pickle(f'{self.Directory}/Clean Data/Census80.pkl')
        
        ACS16_df = pd.merge(
            ACS16_df,
            Crosswalk_df,
            on='occ',
            how='inner'
        )
        ACS16_df['workforce'] = ACS16_df['perwt'].sum()
        ACS16_df['n'] = ACS16_df.groupby('occ1990dd')['perwt'].transform('sum') / ACS16_df['workforce']
        
        ACS16_df['person hours'] = ACS16_df['perwt'] * ACS16_df['wkswork'] * ACS16_df['uhrswork']
        ACS16_df['l'] = ACS16_df.groupby('occ1990dd')['person hours'].transform('sum') / ACS16_df.groupby('occ1990dd')['perwt'].transform('sum')
        
        ACS16_df['person wages'] = ACS16_df['perwt'] * ACS16_df['incwage'] / ACS16_df['wkswork'] / ACS16_df['uhrswork']
        ACS16_df['w'] = ACS16_df.groupby('occ1990dd')['person wages'].transform('sum') / ACS16_df.groupby('occ1990dd')['perwt'].transform('sum')
        
        ACS16_df['L'] = ACS16_df['n'] * ACS16_df['l']
        
        ACS16_df = ACS16_df[['occ1990dd', 'workforce', 'n', 'l', 'w', 'L']]
        ACS16_df = ACS16_df.drop_duplicates()
        ACS16_df.sort_values(by='occ1990dd', inplace=True)
        
        ACS16_df.to_pickle(f'{self.Directory}/Clean Data/ACS16.pkl')
        
        # ---- #
        # Webb #
        # ---- #
        Webb_df = pd.read_csv(f'{self.Directory}/Raw Data/Webb.csv')
        Webb_df = Webb_df.drop('lswt2010', axis=1)
                
        Webb_df = pd.merge(
            Webb_df,
            Crosswalk_df['occ1990dd'].drop_duplicates(),
            on='occ1990dd',
            how='inner'
        )
        
        Webb_df.to_pickle(f'{self.Directory}/Clean Data/Webb.pkl')
        
        # --------------------- #
        # Capital by Occupation #
        # --------------------- #
        CapbyOcc_ES_2d_df = pd.read_csv(f'{self.Directory}/Raw Data/CapbyOcc_ES_2d.csv')
        
        CapbyOcc_ES_2d_df.rename(columns={'occp': 'occ1990dd_2d_title',
                                          'rhobas_IVall': 'ES'}, inplace=True)
        CapbyOcc_ES_2d_df = CapbyOcc_ES_2d_df[['occ1990dd_2d_title', 'ES']]
        
        CapbyOcc_ES_2d_df.to_pickle(f'{self.Directory}/Clean Data/CapbyOcc_ES_2d.pkl')

        

    def Calibrate(self):
        """""
        Calibrate Parameters and Status Quo Allocation
        
        Output: Results/Figures/Wealth_Convexity.csv
                Results/Tables/Calibrate_Results.csv
        """""
        
        self.E.Calibrate()
        Calibrate_Results = gpf.ResultsTable()
        
        Calibrate_Results.add('Status Quo Labor Tax Scale', gpf.clean_round(self.E.Ψ, 2))
        Calibrate_Results.add('Automation Exposure Effect', gpf.clean_round(self.E.Γ, 3))
        Calibrate_Results.add('Household Discount Factor', gpf.clean_round(self.E.β, 2))
        
        # ---------------------- #
        # Wealth Convexity Graph #
        # ---------------------- #
        Calibrate_Results.add('Wealth Convexity', gpf.clean_round(self.E.Θ, 2))
        
        SCF_df = pd.read_pickle(f'{self.Directory}/Clean Data/SCF_2016.pkl')
        
        YS = gpf.compute_decile_shares(SCF_df, 'Labor Income').reshape((-1,1))
        WS = gpf.compute_decile_shares(SCF_df, 'Wealth').reshape((-1,1))
        WS_hat = ( YS**self.E.Θ ) / ( sum(YS**self.E.Θ) ).reshape((-1,1))
        Deciles = np.arange(1,11).reshape((-1,1))
        
        DF_WC = pd.DataFrame(np.hstack((Deciles, YS, WS, WS_hat)), 
                             columns=['Deciles', 'Income Shares', 'Wealth Shares', 'Predicted Wealth Shares'])
        DF_WC.to_csv(f'{self.Directory}/Results/Figures/Wealth_Convexity.csv', index=False)
        
        Calibrate_Results.add('Top Wealth Share', gpf.clean_round(WS[-1,0]*100, 1))
        
        # ------------------------------------ #
        # Marginal Effect of Task Displacement #
        # ------------------------------------ #
        x_j = self.E.var_κ * self.E.x_bar
        rel_l = fn.relα(x_j, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 0)
        
        β_j = - (1 / self.E.σ) * rel_l * (self.E.x_bar - x_j)
        N = np.sum(self.E.L_j_sq)
        
        β_auto = np.sum(self.E.L_j_sq * β_j) / N
        
        Calibrate_Results.add('Task Displacement Effect', gpf.clean_round(β_auto, 2))
        
        # ---------------------- #
        # 50th Percentile Effect #
        # ---------------------- #
        ζ_50 = fn.zeta(self.E.Γ, 50)
        Fif = 1 / ζ_50
        
        Calibrate_Results.add('50th Percentile Effect', gpf.clean_round(Fif, 2))
        
        Calibrate_Results.to_csv(f'{self.Directory}/Results/Tables/Calibrate_Results.csv')
        
        
        
    def Validation(self):
        """""
        Validation Exercises
        
        Output: Results/Figures/CapbyOcc_ES_2d.csv
                Results/Tables/Validation_Results.csv
        """""
        
        Validation_Results = gpf.ResultsTable()
        
        # --------- #
        # Load Data #
        # --------- #
        Census80_df = pd.read_pickle(f'{self.Directory}/Clean Data/Census80.pkl')
        ACS16_df = pd.read_pickle(f'{self.Directory}/Clean Data/ACS16.pkl')
        CapbyOcc_ES_2d_df = pd.read_pickle(f'{self.Directory}/Clean Data/CapbyOcc_ES_2d.pkl')

        
        # --------------------- #
        # 1980-2016 Log Changes #
        # --------------------- #
        w_80 = Census80_df['w'].to_numpy()
        w_16 = ACS16_df['w'].to_numpy()
        L_16 = ACS16_df['L'].to_numpy()
        
        dln_w_j = np.log(w_16) - np.log(w_80)

                
        # ------------------------- #
        # Automation Regression Fit #
        # ------------------------- #
        N = np.sum(L_16)
        weight = L_16 / N
        
        E_dlnw = np.sum(weight * dln_w_j)
        Var_dlnw = np.sum(weight * (dln_w_j - E_dlnw)**2)
        
        x_j = self.E.var_κ * self.E.x_bar
        z_j = fn.relα(x_j, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 0) * x_j
        
        reg = z_j / self.E.ζ
        
        X = np.hstack((np.ones((self.E.J,1)), reg.reshape((-1,1))))
        W = np.diag(weight)
        
        dln_w_hat = (X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W @ dln_w_j.reshape((-1,1))).reshape(-1)
        error = dln_w_j - dln_w_hat
        
        SSR = np.sum(weight * error**2)
        
        R_squared = 1 - SSR / Var_dlnw
        
        Validation_Results.add('Regression Fit', gpf.clean_round(R_squared*100, 1))
        
        
        # -------------------------------------------- #
        # Occupation-Level Elasticities of Subsitution #
        # -------------------------------------------- #
        CapbyOcc_df = ACS16_df[['occ1990dd']]
        
        conditions = [
                    (CapbyOcc_df['occ1990dd'] >= 405) & (CapbyOcc_df['occ1990dd'] <= 408),
                    (CapbyOcc_df['occ1990dd'] == 415) | ((CapbyOcc_df['occ1990dd'] >= 417) & (CapbyOcc_df['occ1990dd'] <= 423)) | ((CapbyOcc_df['occ1990dd'] >= 425) & (CapbyOcc_df['occ1990dd'] <= 427)),
                    (CapbyOcc_df['occ1990dd'] >= 433) & (CapbyOcc_df['occ1990dd'] <= 444),
                    (CapbyOcc_df['occ1990dd'] >= 445) & (CapbyOcc_df['occ1990dd'] <= 447),
                    (CapbyOcc_df['occ1990dd'] >= 448) & (CapbyOcc_df['occ1990dd'] <= 455),
                    ((CapbyOcc_df['occ1990dd'] >= 457) & (CapbyOcc_df['occ1990dd'] <= 458)) | ((CapbyOcc_df['occ1990dd'] >= 459) & (CapbyOcc_df['occ1990dd'] <= 467)) | ((CapbyOcc_df['occ1990dd'] >= 469) & (CapbyOcc_df['occ1990dd'] <= 472)),
                    (CapbyOcc_df['occ1990dd'] == 468),
                    (CapbyOcc_df['occ1990dd'] >= 3) & (CapbyOcc_df['occ1990dd'] <= 22),
                    ((CapbyOcc_df['occ1990dd'] >= 23) & (CapbyOcc_df['occ1990dd'] <= 37)) | (CapbyOcc_df['occ1990dd'] == 489),
                    (CapbyOcc_df['occ1990dd'] >= 43) & (CapbyOcc_df['occ1990dd'] <= 200),
                    (CapbyOcc_df['occ1990dd'] >= 203) & (CapbyOcc_df['occ1990dd'] <= 235),
                    (CapbyOcc_df['occ1990dd'] >= 243) & (CapbyOcc_df['occ1990dd'] <= 258),
                    (CapbyOcc_df['occ1990dd'] >= 274) & (CapbyOcc_df['occ1990dd'] <= 283),
                    (CapbyOcc_df['occ1990dd'] >= 303) & (CapbyOcc_df['occ1990dd'] <= 389),
                    ((CapbyOcc_df['occ1990dd'] >= 503) & (CapbyOcc_df['occ1990dd'] <= 549)) | ((CapbyOcc_df['occ1990dd'] >= 803) & (CapbyOcc_df['occ1990dd'] <= 889)),
                    (CapbyOcc_df['occ1990dd'] >= 558) & (CapbyOcc_df['occ1990dd'] <= 599),
                    (CapbyOcc_df['occ1990dd'] >= 628) & (CapbyOcc_df['occ1990dd'] <= 699),
                    (CapbyOcc_df['occ1990dd'] >= 703) & (CapbyOcc_df['occ1990dd'] <= 799)
                ]

        values_2d = list(range(1, 19))
        
        CapbyOcc_df['occ1990dd_2d'] = np.select(conditions, values_2d, default=np.nan)

        
        z_jk = fn.relα(x_j, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 1) * x_j
        
        Σ_j = self.E.σ + (z_j + z_jk) / self.E.ζ
        
        ES_weight = self.E.S_j_sq * (self.E.Σ_k - self.E.σ) * self.E.ζ / z_jk
        
        CapbyOcc_df['Σ_j'] = Σ_j
        CapbyOcc_df['ES_weight'] = ES_weight
        CapbyOcc_df['norm'] = CapbyOcc_df.groupby('occ1990dd_2d')['ES_weight'].transform('sum')
        
        CapbyOcc_df['ES_summand'] = CapbyOcc_df['ES_weight'] * CapbyOcc_df['Σ_j'] / CapbyOcc_df['norm']
        CapbyOcc_df['Sigma_2d'] = CapbyOcc_df.groupby('occ1990dd_2d')['ES_summand'].transform('sum')

        CapbyOcc_df = CapbyOcc_df.dropna(subset=['occ1990dd_2d'])
        CapbyOcc_df = CapbyOcc_df.drop_duplicates('occ1990dd_2d')
        CapbyOcc_df = CapbyOcc_df.sort_values(by='occ1990dd_2d')
        
        norm = CapbyOcc_df['norm'].to_numpy()
        Sigma_model = CapbyOcc_df['Sigma_2d'].to_numpy()
        Sigma = CapbyOcc_ES_2d_df['ES'].to_numpy()
        
        X_0 = np.hstack((np.ones((18,1)), Sigma_model.reshape((-1,1))))
        X_1 = Sigma_model.reshape((-1,1))
        W = np.diag(norm)
        
        β_0 = np.linalg.inv(X_0.T @ W @ X_0) @ X_0.T @ W @ Sigma.reshape((-1,1))
        β_1 = np.linalg.inv(X_1.T @ W @ X_1) @ X_1.T @ W @ Sigma.reshape((-1,1))
        
        Sigma_hat = (X_1 @ β_1).reshape(-1)
        
        Validation_Results.add('Occ Regression Intercept', gpf.clean_round(β_0[0,0], 2))
        Validation_Results.add('Occ Regression Slope', gpf.clean_round(β_0[1,0], 2))
        Validation_Results.add('Occ Regression Coef', gpf.clean_round(β_1[0,0], 2))
        
        CapbyOcc_ES_2d_df['Sigma_2d'] = Sigma_model
        CapbyOcc_ES_2d_df['Sigma_2d_hat'] = Sigma_hat
        CapbyOcc_ES_2d_df['norm'] = norm
        CapbyOcc_ES_2d_df.loc[CapbyOcc_ES_2d_df['norm'] < CapbyOcc_ES_2d_df['norm'].quantile(5/9), 'occ1990dd_2d_title'] = ''
        
        CapbyOcc_ES_2d_df.to_csv(f'{self.Directory}/Results/Figures/CapbyOcc_ES_2d.csv', index=False)
        
        Validation_Results.to_csv(f'{self.Directory}/Results/Tables/Validation_Results.csv')
        
        
        
    def StatusQuo_Optimum(self):
        """""
        Optimal Threshold Rule for Status Quo Taxes
        
        Output: Results/Figures/StatusQuo_Covariance.csv
                Results/Tables/StatusQuo_Results.csv
        """""
        
        StatusQuo_Results = gpf.ResultsTable()
        
        θ = self.E.StatusQuo_θ(0,1/3)
        
        StatusQuo_Results.add('Optimal Status Quo Threshold Rule', gpf.clean_round(θ*100, 1))
        
        
        # ----------------- #
        # Derive Equilibria #
        # ----------------- #
        E_sq = np.concatenate((self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, self.E.var_κ * self.E.x_bar))
        
        Eqbm = sp.optimize.root(rt.Eqbm_Root, E_sq,
                      args=(θ, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, self.E.J, self.E.n, self.E.y_0, self.E.Ψ, self.E.ψ, self.E.β, self.E.var_θ, self.E.ε, self.E.τ_k, self.E.δ, self.E.g, self.E.φ),
                      jac=pr.δH_δclx)
        
        E = Eqbm.x
        
        c_0 = E[:self.E.J]
        c_1 = E[self.E.J:2*self.E.J]
        l = E[2*self.E.J:3*self.E.J]
        x = E[3*self.E.J:]
        
        
        # ------------------ #
        # Optimal Allocation #
        # ------------------ #
        L = self.E.n * l
        κ = self.E.y_0 - c_0
        K = np.sum(self.E.n * κ)
        Y = fn.Output(x, L, K, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ)
        
        λ = c_1**(-self.E.var_θ) / np.sum(self.E.n * c_1**(-self.E.var_θ))
        A_expos = x / self.E.ζ
        cov = np.sum(self.E.n * (λ-1) * A_expos)
        
        
        # --------------------- #
        # Status Quo Allocation #
        # --------------------- #
        λ_sq = self.E.c_1_sq**(-self.E.var_θ) / np.sum(self.E.n * self.E.c_1_sq**(-self.E.var_θ))
        A_expos_sq = self.E.var_κ * self.E.x_bar / self.E.ζ
        cov_sq = np.sum(self.E.n * (λ_sq-1) * A_expos_sq)
        
        
        # ---------------- #
        # Comparison Table #
        # ---------------- #
        ΔoptK = (K - self.E.K_sq) * 100 / self.E.K_sq
        ΔoptY = (Y - self.E.Y_sq) * 100 / self.E.Y_sq
        ΔoptCOV = (cov - cov_sq) * 100 / cov_sq
        
        CE = sp.optimize.root(rt.CERoot, 1,
                      args=(c_0, c_1, l, self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, self.E.n, self.E.β, self.E.var_θ, self.E.φ, self.E.ε, self.E.g),
                      method='lm')
        
        ConEquiv = (CE.x[0] - 1) * 100
        
        StatusQuo_Results.add('Optimal Status Quo DCapital', gpf.clean_round(ΔoptK, 1))
        StatusQuo_Results.add('Optimal Status Quo DOutput', gpf.clean_round(ΔoptY, 1))
        StatusQuo_Results.add('Optimal Status Quo DCOV', gpf.clean_round(ΔoptCOV, 1))
        StatusQuo_Results.add('Optimal Status Quo Consumption Equivalence', gpf.clean_round(ConEquiv, 1))

        
        # ----------------- #
        # Covariance Figure #
        # ----------------- #
        DF_Cov_sq = pd.DataFrame(np.hstack((self.E.n.reshape((-1,1)), A_expos.reshape((-1,1)), λ.reshape((-1,1)), A_expos_sq.reshape((-1,1)), λ_sq.reshape((-1,1)))), 
                             columns=['Weight', 'Automation Exposure Optimal', 'lambda Optimal',  'Automation Exposure Status Quo', 'lambda Status Quo'])
        DF_Cov_sq = DF_Cov_sq.sort_values("Weight", ascending=False)
        DF_Cov_sq.to_csv(f'{self.Directory}/Results/Figures/StatusQuo_Covariance.csv', index=False)
        
        
        StatusQuo_Results.to_csv(f'{self.Directory}/Results/Tables/StatusQuo_Results.csv')
        
        
        
    def Mirrlees_Optimum(self, first1=0, first2=0):
        """""
        Optimal Threshold Rule for Non-Linear Taxes
        
        Output: Results/Figures/Mirrlees_Covariance.csv
                Results/Tables/Mirrlees_Results.csv
        """""
        
        Mirrlees_Results = gpf.ResultsTable()
        
        Y_0 = self.E.Y_sq / (1+self.E.g)
        K_0 = self.E.K_sq / (1+self.E.g)
        Y_bar = Y_0 + (1-self.E.δ) * K_0
        
        
        # --------------------------------- #
        # Solve for Two Planner Allocations #
        # --------------------------------- #
        if first1 == 1:
            tup_NT = self.E.Mirrlees_Lagr_NT()
            
            with open(f'{self.Directory}/Results/E_NT.pkl', 'wb') as file:
                pickle.dump(tup_NT, file)
                (c_0_NT, c_1_NT, l_NT, K_NT, x_NT) = tup_NT
    
        else:
            with open(f'{self.Directory}/Results/E_NT.pkl', 'rb') as file:
                tup_NT = pickle.load(file)
                (c_0_NT, c_1_NT, l_NT, K_NT, x_NT) = tup_NT
        E_NT = np.concatenate((c_0_NT, c_1_NT, l_NT))
        
        if first2 == 1:
            tup = self.E.Mirrlees_Lagr_θ(E_NT, x_NT, -0.25, 0.5)
            
            with open(f'{self.Directory}/Results/E_theta.pkl', 'wb') as file:
                pickle.dump(tup, file)
                (c_0, c_1, l, K, x, θ) = tup
    
        else:
            with open(f'{self.Directory}/Results/E_theta.pkl', 'rb') as file:
                tup = pickle.load(file)
                (c_0, c_1, l, K, x, θ) = tup
        
        
        # -------------------------- #
        # Threshold Mirrlees Optimum #
        # -------------------------- #        
        L = self.E.n * l
        Y = fn.Output(x, L, K, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ)
        
        λ = c_1**(-self.E.var_θ) / np.sum(self.E.n * c_1**(-self.E.var_θ))
        A_expos = x / self.E.ζ
        cov = np.sum(self.E.n * (λ-1) * A_expos)
        
        Mirrlees_Results.add('Optimal Mirrlees Threshold Rule', gpf.clean_round(θ*100, 1))
        
        
        # ---------------------------- #
        # Capital Tax Mirrlees Optimum #
        # ---------------------------- #
        L_NT = self.E.n * l_NT
        Y_NT = fn.Output(x_NT, L_NT, K_NT, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ)
        
        λ_NT = c_1_NT**(-self.E.var_θ) / np.sum(self.E.n * c_1_NT**(-self.E.var_θ))
        A_expos_NT = x_NT / self.E.ζ
        cov_NT = np.sum(self.E.n * (λ_NT-1) * A_expos_NT)
        
        MRS_c_NT = fn.cap_MRS(c_0_NT, c_1_NT, self.E.β, self.E.var_θ, self.E.ε, self.E.g)
        r_NT = fn.Rents(x_NT, L_NT, K_NT, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ)
        τ_K_NT = np.sum(self.E.n * (1 - (MRS_c_NT + self.E.g) / (r_NT - self.E.δ)))
        Mirrlees_Results.add('Optimal Mirrlees Capital Tax NT', gpf.clean_round(τ_K_NT*100, 1))
        
        
        # ---------------- #
        # Comparison Table #
        # ---------------- #
        ΔoptK = (K - K_NT) * 100 / K_NT
        ΔoptY = (Y - Y_NT) * 100 / Y_NT
        ΔoptCOV = (cov - cov_NT) * 100 / cov_NT
        
        CE = sp.optimize.root(rt.CERoot, 1,
                      args=(c_0, c_1, l, c_0_NT, c_1_NT, l_NT, self.E.n, self.E.β, self.E.var_θ, self.E.φ, self.E.ε, self.E.g),
                      method='lm')
        
        ConEquiv = (CE.x[0] - 1) * 100
        
        Mirrlees_Results.add('Optimal Mirrlees DCapital', gpf.clean_round(ΔoptK, 2))
        Mirrlees_Results.add('Optimal Mirrlees DOutput', gpf.clean_round(ΔoptY, 2))
        Mirrlees_Results.add('Optimal Mirrlees DCOV', gpf.clean_round(ΔoptCOV, 2))
        Mirrlees_Results.add('Optimal Mirrlees Consumption Equivalence', gpf.clean_round(ConEquiv, 2))
        
        
        # ----------------- #
        # Covariance Figure #
        # ----------------- #
        DF_Cov_NT = pd.DataFrame(np.hstack((self.E.n.reshape((-1,1)), A_expos.reshape((-1,1)), λ.reshape((-1,1)), A_expos_NT.reshape((-1,1)), λ_NT.reshape((-1,1)))), 
                             columns=['Weight', 'Automation Exposure Optimal', 'lambda Optimal', 'Automation Exposure Capital Tax', 'lambda Capital Tax'])
        DF_Cov_NT = DF_Cov_NT.sort_values("Weight", ascending=False)
        DF_Cov_NT.to_csv(f'{self.Directory}/Results/Figures/Mirrlees_Covariance.csv', index=False)
        

        Mirrlees_Results.to_csv(f'{self.Directory}/Results/Tables/Mirrlees_Results.csv')
        
    
    def AI_experiment(self):
        """""
        Optimal Threshold Rule for AI Scenarios
        
        Output: 
        """""
        
        θ_AI = self.E.AI_economy(5, 0.25)
        
        
    def write_package_versions(self, packages):
        """""
        Table of Package Versions
    
        Output: Results/core_versions.txt
        """""
        
        filename=f'{self.Directory}/Results/core_versions.txt'
        
        # ---------------- #
        # Collect Packages #
        # ---------------- #
        rows = []
        for pkg in packages:
            ver = md.version(pkg)
            rows.append((pkg, ver))
    
    
        # ----------- #
        # Write Table #
        # ----------- #
        print(sys.version)
        
        with open(filename, "w") as f:
            f.write("| Package | Version |\n")
            f.write("|---------|---------|\n")
            for pkg, ver in rows:
                f.write(f"| {pkg} | {ver} |\n")
            