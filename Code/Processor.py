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
                
        
        
    def Cleaner(self, ipums_extract=0, CPI_year=2016):
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
        filtered_data = [{'year': entry['date'], 'CPI': entry['value']} for entry in data]
        CPI_df = pd.DataFrame(filtered_data)
        
        CPI_df['year'] = pd.to_datetime(CPI_df['year']).dt.year
        CPI_df['CPI'] = pd.to_numeric(CPI_df['CPI'], errors='coerce')
        CPI_df['CPI'] = CPI_df['CPI'] / CPI_df.loc[CPI_df['year'] == CPI_year, 'CPI'].values[0] #CPI indexed to 1 in CPI_year
        
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
        
        
        # ----------------------- #
        # Webb Correlation Matrix #
        # ----------------------- #
        Webb_df = pd.read_pickle(f'{self.Directory}/Clean Data/Webb.pkl')
        X_corr = np.vstack((self.E.w_j_sq, Webb_df['pct_software'].to_numpy(), Webb_df['pct_robot'].to_numpy(), Webb_df['pct_ai'].to_numpy()))
        corr_mat = np.corrcoef(X_corr)
        
        Calibrate_Results.add('Webb Wage Soft Corr', gpf.clean_round(corr_mat[0,1], 2))
        Calibrate_Results.add('Webb Wage Rob Corr', gpf.clean_round(corr_mat[0,2], 2))
        Calibrate_Results.add('Webb Wage AI Corr', gpf.clean_round(corr_mat[0,3], 2))
        
        Calibrate_Results.add('Webb Soft Rob Corr', gpf.clean_round(corr_mat[1,2], 2))
        Calibrate_Results.add('Webb Soft AI Corr', gpf.clean_round(corr_mat[1,3], 2))
        
        Calibrate_Results.add('Webb Rob AI Corr', gpf.clean_round(corr_mat[2,3], 2))
        
        
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
        
        θ = self.E.StatusQuo_θ(0, 1/3)
        
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
        
        ln_w = np.log(fn.Wages(x, L, K, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ))
        z_j = fn.relα(x, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 0) * x
        z_jk = fn.relα(x, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 1) * x
        Σ_j = (self.E.σ + (z_j + z_jk) / self.E.ζ).reshape((-1,1))
        cov = np.sum(self.E.n * Σ_j * ln_w) - np.sum(self.E.n * Σ_j) * np.sum(self.E.n * ln_w)
        
        
        # --------------------- #
        # Status Quo Allocation #
        # --------------------- #
        ln_w_sq = np.log(self.E.w_j_sq)
        x_sq = self.E.var_κ * self.E.x_bar
        z_j_sq = fn.relα(x_sq, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 0) * x_sq
        z_jk_sq = fn.relα(x_sq, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 1) * x_sq
        Σ_j_sq = (self.E.σ + (z_j_sq + z_jk_sq) / self.E.ζ).reshape((-1,1))
        cov_sq = np.sum(self.E.n * Σ_j_sq * ln_w_sq) - np.sum(self.E.n * Σ_j_sq) * np.sum(self.E.n * ln_w_sq)
        
        
        # ---------------- #
        # Comparison Table #
        # ---------------- #
        ΔoptY = (Y - self.E.Y_sq) * 100 / self.E.Y_sq
        ΔoptCOV = (cov - cov_sq) * 100 / cov_sq
        
        CE = sp.optimize.root(rt.CERoot, 1,
                      args=(c_0, c_1, l, self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, self.E.n, self.E.β, self.E.var_θ, self.E.φ, self.E.ε, self.E.g),
                      method='lm')
        
        ConEquiv = (CE.x[0] - 1) * 100
        
        StatusQuo_Results.add('Optimal Status Quo DOutput', gpf.clean_round(ΔoptY, 1))
        StatusQuo_Results.add('Optimal Status Quo DCOV', gpf.clean_round(ΔoptCOV, 1))
        StatusQuo_Results.add('Optimal Status Quo Consumption Equivalence', gpf.clean_round(ConEquiv, 1))

        
        # ----------------- #
        # Covariance Figure #
        # ----------------- #
        X = np.hstack((np.ones((self.E.J,1)), Σ_j))
        X_sq = np.hstack((np.ones((self.E.J,1)), Σ_j_sq))
        y = ln_w.reshape((-1,1))
        y_sq = ln_w_sq.reshape((-1,1))
        W = np.diag(self.E.n)
        
        β = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        β_sq = np.linalg.inv(X_sq.T @ W @ X_sq) @ X_sq.T @ W @ y_sq
        
        ln_w_hat = X @ β
        ln_w_hat_sq = X_sq @ β_sq
        
        DF_Cov_sq = pd.DataFrame(np.hstack((self.E.n.reshape((-1,1)), Σ_j, y, ln_w_hat, Σ_j_sq, y_sq, ln_w_hat_sq)), 
                             columns=['Weight', 'ES Optimal', 'Log Wages Optimal', 'Log Wages_hat Optimal', 'ES Status Quo', 'Log Wages Status Quo', 'Log Wages_hat Status Quo'])
        DF_Cov_sq = DF_Cov_sq.sort_values("Weight", ascending=False)
        DF_Cov_sq.to_csv(f'{self.Directory}/Results/Figures/StatusQuo_Covariance.csv', index=False)
        
        
        StatusQuo_Results.to_csv(f'{self.Directory}/Results/Tables/StatusQuo_Results.csv')
        
        
        
    def Mirrlees_Optimum(self):
        """""
        Optimal Threshold Rule for Non-Linear Taxes
        
        Output: Results/Figures/Mirrlees_Covariance.csv
                Results/Tables/Mirrlees_Results.csv
        """""
        
        Mirrlees_Results = gpf.ResultsTable()
        
        
        # --------------------------------- #
        # Solve for Two Planner Allocations #
        # --------------------------------- #
        (c_0_NT, c_1_NT, l_NT, K_NT, x_NT) = self.E.Mirrlees_Lagr_NT()
        E_NT = np.concatenate((c_0_NT, c_1_NT, l_NT))
        
        (c_0, c_1, l, K, x, θ) = self.E.Mirrlees_Lagr_θ(E_NT, x_NT, -0.25, 0.5)
        
        
        # -------------------------- #
        # Threshold Mirrlees Optimum #
        # -------------------------- #        
        L = self.E.n * l
        Y = fn.Output(x, L, K, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ)
        
        ln_w = np.log(fn.Wages(x, L, K, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ))
        z_j = fn.relα(x, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 0) * x
        z_jk = fn.relα(x, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 1) * x
        Σ_j = (self.E.σ + (z_j + z_jk) / self.E.ζ).reshape((-1,1))
        cov = np.sum(self.E.n * Σ_j * ln_w) - np.sum(self.E.n * Σ_j) * np.sum(self.E.n * ln_w)
        
        Mirrlees_Results.add('Optimal Mirrlees Threshold Rule', gpf.clean_round(θ*100, 1))
        
        
        # ---------------------------- #
        # Capital Tax Mirrlees Optimum #
        # ---------------------------- #
        L_NT = self.E.n * l_NT
        Y_NT = fn.Output(x_NT, L_NT, K_NT, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ)
        
        ln_w_NT = np.log(fn.Wages(x_NT, L_NT, K_NT, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ))
        z_j_NT = fn.relα(x_NT, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 0) * x_NT
        z_jk_NT = fn.relα(x_NT, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 1) * x_NT
        Σ_j_NT = (self.E.σ + (z_j_NT + z_jk_NT) / self.E.ζ).reshape((-1,1))
        cov_NT = np.sum(self.E.n * Σ_j_NT * ln_w_NT) - np.sum(self.E.n * Σ_j_NT) * np.sum(self.E.n * ln_w_NT)
        
        MRS_c_NT = fn.cap_MRS(c_0_NT, c_1_NT, self.E.β, self.E.var_θ, self.E.ε, self.E.g)
        r_NT = fn.Rents(x_NT, L_NT, K_NT, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ)
        τ_K_NT = np.sum(self.E.n * (1 - (MRS_c_NT + self.E.g) / (r_NT - self.E.δ)))
        Mirrlees_Results.add('Optimal Mirrlees Capital Tax', gpf.clean_round(τ_K_NT*100, 1))
        
        Return_NT = 1 + r_NT - self.E.δ
        Return_tilde_NT = 1 + (1-τ_K_NT)*(r_NT-self.E.δ)
        τ_wealth_NT = 1 - Return_tilde_NT / Return_NT
        Mirrlees_Results.add('Optimal Mirrlees Wealth Tax', gpf.clean_round(τ_wealth_NT*100, 1))
        
        
        # ---------------- #
        # Comparison Table #
        # ---------------- #
        ΔoptY = (Y - Y_NT) * 100 / Y_NT
        ΔoptCOV = (cov - cov_NT) * 100 / cov_NT
        
        CE = sp.optimize.root(rt.CERoot, 1,
                      args=(c_0, c_1, l, c_0_NT, c_1_NT, l_NT, self.E.n, self.E.β, self.E.var_θ, self.E.φ, self.E.ε, self.E.g),
                      method='lm')
        
        ConEquiv = (CE.x[0] - 1) * 100
        
        Mirrlees_Results.add('Optimal Mirrlees DOutput', gpf.clean_round(ΔoptY, 2))
        Mirrlees_Results.add('Optimal Mirrlees DCOV', gpf.clean_round(ΔoptCOV, 2))
        Mirrlees_Results.add('Optimal Mirrlees Consumption Equivalence', gpf.clean_round(ConEquiv, 2))
        
        
        # ----------------- #
        # Covariance Figure #
        # ----------------- #
        X = np.hstack((np.ones((self.E.J,1)), Σ_j))
        X_NT = np.hstack((np.ones((self.E.J,1)), Σ_j_NT))
        y = ln_w.reshape((-1,1))
        y_NT = ln_w_NT.reshape((-1,1))
        W = np.diag(self.E.n)
        
        β = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        β_NT = np.linalg.inv(X_NT.T @ W @ X_NT) @ X_NT.T @ W @ y_NT
        
        ln_w_hat = X @ β
        ln_w_hat_NT = X_NT @ β_NT
        
        DF_Cov_NT = pd.DataFrame(np.hstack((self.E.n.reshape((-1,1)), Σ_j, y, ln_w_hat, Σ_j_NT, y_NT, ln_w_hat_NT)), 
                             columns=['Weight', 'ES Optimal', 'Log Wages Optimal', 'Log Wages_hat Optimal', 'ES Capital Tax', 'Log Wages Capital Tax', 'Log Wages_hat Capital Tax'])
        DF_Cov_NT = DF_Cov_NT.sort_values("Weight", ascending=False)
        DF_Cov_NT.to_csv(f'{self.Directory}/Results/Figures/Mirrlees_Covariance.csv', index=False)
        

        Mirrlees_Results.to_csv(f'{self.Directory}/Results/Tables/Mirrlees_Results.csv')
        
    
    def AI_Experiment(self, G_k, N_g):
        """""
        Optimal Threshold Rule for AI Scenarios
        
        Output: Results/Figures/AI_Experiment.csv
                Results/Tables/AI_Experiment_Results.csv
        """""
        
        AI_Experiment_Results = gpf.ResultsTable()
        
        # ------------------- #
        # Post-AI Calibration #
        # ------------------- #
        Webb_df = pd.read_pickle(f'{self.Directory}/Clean Data/Webb.pkl')
        χ_AI = (Webb_df['pct_software'].to_numpy() + Webb_df['pct_robot'].to_numpy() + Webb_df['pct_ai'].to_numpy()) / 3
        
        ζ_AI = np.exp(self.E.Γ * χ_AI)
       
        nu_g = np.ones(self.E.J) * self.E.var_κ
        
        nu = sp.optimize.root(rt.νRoot, nu_g,
                      args=(self.E.Γ, χ_AI, self.E.var_κ, self.E.x_bar, self.E.σ, self.E.S_j_sq, self.E.S_k),
                      method='lm')
        
        ν_AI = nu.x
        
        x_AI = self.E.var_κ * self.E.x_bar
        Λ_k_AI = fn.Lamba_k(x_AI, self.E.x_bar, ζ_AI, ν_AI, self.E.σ)
        A_k_AI_base = self.E.r_sq**(self.E.σ / (self.E.σ-1)) * (self.E.COR / Λ_k_AI)**(1 / (self.E.σ-1))
        
        A_j_AI_base = (self.E.w_j_sq / x_AI**(ζ_AI)) / (self.E.r_sq / A_k_AI_base)
        
        
        # ----------------- #
        # Solve θ Sequences #
        # ----------------- #
        θ_AI_10 = self.E.AI_economy(ζ_AI, ν_AI, A_k_AI_base, A_j_AI_base, G_k, 0.1, N_g)
        θ_AI_25 = self.E.AI_economy(ζ_AI, ν_AI, A_k_AI_base, A_j_AI_base, G_k, 0.25, N_g)
        θ_AI_50 = self.E.AI_economy(ζ_AI, ν_AI, A_k_AI_base, A_j_AI_base, G_k, 0.5, N_g)
        
        # ----------------------- #
        # Consumption Equivalence #
        # ----------------------- #
        ConEquiv_10 = np.empty(N_g)
        ConEquiv_25 = np.empty(N_g)
        ConEquiv_50 = np.empty(N_g)
        AI_growth = np.linspace(0, G_k, N_g)
        
        E_θ_10 = np.concatenate((self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, self.E.var_κ * self.E.x_bar))
        E_θ_25 = np.concatenate((self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, self.E.var_κ * self.E.x_bar))
        E_θ_50 = np.concatenate((self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, self.E.var_κ * self.E.x_bar))
        
        for g in range(N_g):
            A_k_AI = A_k_AI_base * (1 + AI_growth[g])
            A_j_AI_10 = A_j_AI_base * (1 + AI_growth[g] * 0.1)
            A_j_AI_25 = A_j_AI_base * (1 + AI_growth[g] * 0.25)
            A_j_AI_50 = A_j_AI_base * (1 + AI_growth[g] * 0.5)
            
            g_y_10 = self.E.S_k * AI_growth[g] + (1-self.E.S_k) * AI_growth[g] * 0.25
            Ψ_AI_10 = self.E.Ψ * (1+g_y_10)**(self.E.ψ)
            g_y_25 = self.E.S_k * AI_growth[g] + (1-self.E.S_k) * AI_growth[g] * 0.25
            Ψ_AI_25 = self.E.Ψ * (1+g_y_25)**(self.E.ψ)
            g_y_50 = self.E.S_k * AI_growth[g] + (1-self.E.S_k) * AI_growth[g] * 0.5
            Ψ_AI_50 = self.E.Ψ * (1+g_y_50)**(self.E.ψ)
            
            Eqbm_θ_10 = sp.optimize.root(rt.Eqbm_Root, E_θ_10,
                          args=(θ_AI_10[g], A_j_AI_10, A_k_AI, self.E.x_bar, ζ_AI, ν_AI, self.E.σ, self.E.J, self.E.n, self.E.y_0, Ψ_AI_10, self.E.ψ, self.E.β, self.E.var_θ, self.E.ε, self.E.τ_k, self.E.δ, self.E.g, self.E.φ),
                          jac=pr.δH_δclx)
            E_θ_10 = Eqbm_θ_10.x
            c_0_θ_10 = E_θ_10[:self.E.J]
            c_1_θ_10 = E_θ_10[self.E.J:2*self.E.J]
            l_θ_10 = E_θ_10[2*self.E.J:3*self.E.J]
            Eqbm_θ_25 = sp.optimize.root(rt.Eqbm_Root, E_θ_25,
                          args=(θ_AI_25[g], A_j_AI_25, A_k_AI, self.E.x_bar, ζ_AI, ν_AI, self.E.σ, self.E.J, self.E.n, self.E.y_0, Ψ_AI_25, self.E.ψ, self.E.β, self.E.var_θ, self.E.ε, self.E.τ_k, self.E.δ, self.E.g, self.E.φ),
                          jac=pr.δH_δclx)
            E_θ_25 = Eqbm_θ_25.x
            c_0_θ_25 = E_θ_25[:self.E.J]
            c_1_θ_25 = E_θ_25[self.E.J:2*self.E.J]
            l_θ_25 = E_θ_25[2*self.E.J:3*self.E.J]
            Eqbm_θ_50 = sp.optimize.root(rt.Eqbm_Root, E_θ_50,
                          args=(θ_AI_50[g], A_j_AI_50, A_k_AI, self.E.x_bar, ζ_AI, ν_AI, self.E.σ, self.E.J, self.E.n, self.E.y_0, Ψ_AI_50, self.E.ψ, self.E.β, self.E.var_θ, self.E.ε, self.E.τ_k, self.E.δ, self.E.g, self.E.φ),
                          jac=pr.δH_δclx)
            E_θ_50 = Eqbm_θ_50.x
            c_0_θ_50 = E_θ_50[:self.E.J]
            c_1_θ_50 = E_θ_50[self.E.J:2*self.E.J]
            l_θ_50 = E_θ_50[2*self.E.J:3*self.E.J]
            
            Eqbm_LF_10 = sp.optimize.root(rt.Eqbm_Root, E_θ_10,
                          args=(0, A_j_AI_10, A_k_AI, self.E.x_bar, ζ_AI, ν_AI, self.E.σ, self.E.J, self.E.n, self.E.y_0, Ψ_AI_10, self.E.ψ, self.E.β, self.E.var_θ, self.E.ε, self.E.τ_k, self.E.δ, self.E.g, self.E.φ),
                          jac=pr.δH_δclx)
            E_LF_10 = Eqbm_LF_10.x
            c_0_LF_10 = E_LF_10[:self.E.J]
            c_1_LF_10 = E_LF_10[self.E.J:2*self.E.J]
            l_LF_10 = E_LF_10[2*self.E.J:3*self.E.J]
            Eqbm_LF_25 = sp.optimize.root(rt.Eqbm_Root, E_θ_25,
                          args=(0, A_j_AI_25, A_k_AI, self.E.x_bar, ζ_AI, ν_AI, self.E.σ, self.E.J, self.E.n, self.E.y_0, Ψ_AI_25, self.E.ψ, self.E.β, self.E.var_θ, self.E.ε, self.E.τ_k, self.E.δ, self.E.g, self.E.φ),
                          jac=pr.δH_δclx)
            E_LF_25 = Eqbm_LF_25.x
            c_0_LF_25 = E_LF_25[:self.E.J]
            c_1_LF_25 = E_LF_25[self.E.J:2*self.E.J]
            l_LF_25 = E_LF_25[2*self.E.J:3*self.E.J]
            Eqbm_LF_50 = sp.optimize.root(rt.Eqbm_Root, E_θ_50,
                          args=(0, A_j_AI_50, A_k_AI, self.E.x_bar, ζ_AI, ν_AI, self.E.σ, self.E.J, self.E.n, self.E.y_0, Ψ_AI_50, self.E.ψ, self.E.β, self.E.var_θ, self.E.ε, self.E.τ_k, self.E.δ, self.E.g, self.E.φ),
                          jac=pr.δH_δclx)
            E_LF_50 = Eqbm_LF_50.x
            c_0_LF_50 = E_LF_50[:self.E.J]
            c_1_LF_50 = E_LF_50[self.E.J:2*self.E.J]
            l_LF_50 = E_LF_50[2*self.E.J:3*self.E.J]
          
            CE_10 = sp.optimize.root(rt.CERoot, 1,
                          args=(c_0_θ_10, c_1_θ_10, l_θ_10, c_0_LF_10, c_1_LF_10, l_LF_10, self.E.n, self.E.β, self.E.var_θ, self.E.φ, self.E.ε, self.E.g),
                          method='lm')
            ConEquiv_10[g] = (CE_10.x[0] - 1) * 100
            CE_25 = sp.optimize.root(rt.CERoot, 1,
                          args=(c_0_θ_25, c_1_θ_25, l_θ_25, c_0_LF_25, c_1_LF_25, l_LF_25, self.E.n, self.E.β, self.E.var_θ, self.E.φ, self.E.ε, self.E.g),
                          method='lm')
            ConEquiv_25[g] = (CE_25.x[0] - 1) * 100
            CE_50 = sp.optimize.root(rt.CERoot, 1,
                          args=(c_0_θ_50, c_1_θ_50, l_θ_50, c_0_LF_50, c_1_LF_50, l_LF_50, self.E.n, self.E.β, self.E.var_θ, self.E.φ, self.E.ε, self.E.g),
                          method='lm')
            ConEquiv_50[g] = (CE_50.x[0] - 1) * 100
        
        # ---- #
        # Plot #
        # ---- #
        AI_growth *= 100
        θ_AI_10 *= 100
        θ_AI_25 *= 100
        θ_AI_50 *= 100
        
        DF_AI = pd.DataFrame(np.hstack((AI_growth.reshape((-1,1)), θ_AI_10.reshape((-1,1)), ConEquiv_10.reshape((-1,1)), θ_AI_25.reshape((-1,1)), ConEquiv_25.reshape((-1,1)), θ_AI_50.reshape((-1,1)), ConEquiv_50.reshape((-1,1)))), 
                             columns=['Growth', 'Threshold Rule 10', 'Consumption Equivalence 10', 'Threshold Rule 25', 'Consumption Equivalence 25', 'Threshold Rule 50', 'Consumption Equivalence 50'])
        DF_AI.to_csv(f'{self.Directory}/Results/Figures/AI_Experiment.csv', index=False)
        
        AI_Experiment_Results.add('Doubling Threshold Rule', gpf.clean_round(θ_AI_25[-1], 1))
        AI_Experiment_Results.add('Doubling Consumption Equivalence', gpf.clean_round(ConEquiv_25[-1], 1))
        
        
        AI_Experiment_Results.to_csv(f'{self.Directory}/Results/Tables/AI_Experiment_Results.csv')
        
        
        
    def Σ_robust(self, Σ_low, σ_low):
        """""
        Robustness with Elasticities of Substitution
    
        Output: Results/Figures/Σ_robust_StatusQuo_Covariance.csv
                Results/Figures/Σ_robust_Mirrlees_Covariance.csv
                Results/Tables/Σ_robust_Results.csv
        """""
        
        Σ_robust_Results = gpf.ResultsTable()
        self.E.Σ_k = Σ_low
        self.E.σ = σ_low
        
        
        # ----------- #
        # Recalibrate #
        # ----------- #
        self.E.Calibrate()
        
        
        # ------------------------- #
        # Status Quo Threshold Rule #
        # ------------------------- #
        θ = self.E.StatusQuo_θ(0, 1/3)
        
        Σ_robust_Results.add('Low ES Status Quo Threshold Rule', gpf.clean_round(θ*100, 1))
        
        # Derive Equilibria
        E_sq = np.concatenate((self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, self.E.var_κ * self.E.x_bar))
        
        Eqbm = sp.optimize.root(rt.Eqbm_Root, E_sq,
                      args=(θ, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, self.E.J, self.E.n, self.E.y_0, self.E.Ψ, self.E.ψ, self.E.β, self.E.var_θ, self.E.ε, self.E.τ_k, self.E.δ, self.E.g, self.E.φ),
                      jac=pr.δH_δclx)
        
        E = Eqbm.x
        
        c_0 = E[:self.E.J]
        c_1 = E[self.E.J:2*self.E.J]
        l = E[2*self.E.J:3*self.E.J]
        x = E[3*self.E.J:]
        
        # Optimal Allocation 
        L = self.E.n * l
        κ = self.E.y_0 - c_0
        K = np.sum(self.E.n * κ)
        Y = fn.Output(x, L, K, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ)
        
        ln_w = np.log(fn.Wages(x, L, K, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ))
        z_j = fn.relα(x, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 0) * x
        z_jk = fn.relα(x, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 1) * x
        Σ_j = (self.E.σ + (z_j + z_jk) / self.E.ζ).reshape((-1,1))
        cov = np.sum(self.E.n * Σ_j * ln_w) - np.sum(self.E.n * Σ_j) * np.sum(self.E.n * ln_w)
        
        # Status Quo Allocation
        ln_w_sq = np.log(self.E.w_j_sq)
        x_sq = self.E.var_κ * self.E.x_bar
        z_j_sq = fn.relα(x_sq, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 0) * x_sq
        z_jk_sq = fn.relα(x_sq, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 1) * x_sq
        Σ_j_sq = (self.E.σ + (z_j_sq + z_jk_sq) / self.E.ζ).reshape((-1,1))
        cov_sq = np.sum(self.E.n * Σ_j_sq * ln_w_sq) - np.sum(self.E.n * Σ_j_sq) * np.sum(self.E.n * ln_w_sq)
        
        # Comparison Table 
        ΔoptY = (Y - self.E.Y_sq) * 100 / self.E.Y_sq
        ΔoptCOV = (cov - cov_sq) * 100 / cov_sq
        
        CE = sp.optimize.root(rt.CERoot, 1,
                      args=(c_0, c_1, l, self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, self.E.n, self.E.β, self.E.var_θ, self.E.φ, self.E.ε, self.E.g),
                      method='lm')
        
        ConEquiv = (CE.x[0] - 1) * 100
        
        Σ_robust_Results.add('Low ES Status Quo DOutput', gpf.clean_round(ΔoptY, 1))
        Σ_robust_Results.add('Low ES Status Quo DCOV', gpf.clean_round(ΔoptCOV, 1))
        Σ_robust_Results.add('Low ES Status Quo Consumption Equivalence', gpf.clean_round(ConEquiv, 1))

        # Covariance Figure
        X = np.hstack((np.ones((self.E.J,1)), Σ_j))
        X_sq = np.hstack((np.ones((self.E.J,1)), Σ_j_sq))
        y = ln_w.reshape((-1,1))
        y_sq = ln_w_sq.reshape((-1,1))
        W = np.diag(self.E.n)
        
        β = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        β_sq = np.linalg.inv(X_sq.T @ W @ X_sq) @ X_sq.T @ W @ y_sq
        
        ln_w_hat = X @ β
        ln_w_hat_sq = X_sq @ β_sq
        
        DF_Cov_sq = pd.DataFrame(np.hstack((self.E.n.reshape((-1,1)), Σ_j, y, ln_w_hat, Σ_j_sq, y_sq, ln_w_hat_sq)), 
                             columns=['Weight', 'ES Optimal', 'Log Wages Optimal', 'Log Wages_hat Optimal', 'ES Status Quo', 'Log Wages Status Quo', 'Log Wages_hat Status Quo'])
        DF_Cov_sq = DF_Cov_sq.sort_values("Weight", ascending=False)
        DF_Cov_sq.to_csv(f'{self.Directory}/Results/Figures/Σ_robust_StatusQuo_Covariance.csv', index=False)
        
        
        # ---------------- #
        # Mirrlees Problem #
        # ---------------- #
        (c_0_NT, c_1_NT, l_NT, K_NT, x_NT) = self.E.Mirrlees_Lagr_NT()
        E_NT = np.concatenate((c_0_NT, c_1_NT, l_NT))
        
        (c_0, c_1, l, K, x, θ) = self.E.Mirrlees_Lagr_θ(E_NT, x_NT, -0.25, 0.5)
        
        # Threshold Mirrlees Optimum
        L = self.E.n * l
        Y = fn.Output(x, L, K, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ)
        
        ln_w = np.log(fn.Wages(x, L, K, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ))
        z_j = fn.relα(x, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 0) * x
        z_jk = fn.relα(x, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 1) * x
        Σ_j = (self.E.σ + (z_j + z_jk) / self.E.ζ).reshape((-1,1))
        cov = np.sum(self.E.n * Σ_j * ln_w) - np.sum(self.E.n * Σ_j) * np.sum(self.E.n * ln_w)
        
        Σ_robust_Results.add('Low ES Mirrlees Threshold Rule', gpf.clean_round(θ*100, 1))
        
        # Capital Tax Mirrlees Optimum
        L_NT = self.E.n * l_NT
        Y_NT = fn.Output(x_NT, L_NT, K_NT, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ)
        
        ln_w_NT = np.log(fn.Wages(x_NT, L_NT, K_NT, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ))
        z_j_NT = fn.relα(x_NT, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 0) * x_NT
        z_jk_NT = fn.relα(x_NT, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, 1) * x_NT
        Σ_j_NT = (self.E.σ + (z_j_NT + z_jk_NT) / self.E.ζ).reshape((-1,1))
        cov_NT = np.sum(self.E.n * Σ_j_NT * ln_w_NT) - np.sum(self.E.n * Σ_j_NT) * np.sum(self.E.n * ln_w_NT)
        
        MRS_c_NT = fn.cap_MRS(c_0_NT, c_1_NT, self.E.β, self.E.var_θ, self.E.ε, self.E.g)
        r_NT = fn.Rents(x_NT, L_NT, K_NT, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ)
        τ_K_NT = np.sum(self.E.n * (1 - (MRS_c_NT + self.E.g) / (r_NT - self.E.δ)))
        Σ_robust_Results.add('Low ES Mirrlees Capital Tax', gpf.clean_round(τ_K_NT*100, 1))
    
        # Comparison Table 
        ΔoptY = (Y - Y_NT) * 100 / Y_NT
        ΔoptCOV = (cov - cov_NT) * 100 / cov_NT
        
        CE = sp.optimize.root(rt.CERoot, 1,
                      args=(c_0, c_1, l, c_0_NT, c_1_NT, l_NT, self.E.n, self.E.β, self.E.var_θ, self.E.φ, self.E.ε, self.E.g),
                      method='lm')
        
        ConEquiv = (CE.x[0] - 1) * 100
        
        Σ_robust_Results.add('Low ES Mirrlees DOutput', gpf.clean_round(ΔoptY, 2))
        Σ_robust_Results.add('Low ES Mirrlees DCOV', gpf.clean_round(ΔoptCOV, 2))
        Σ_robust_Results.add('Low ES Mirrlees Consumption Equivalence', gpf.clean_round(ConEquiv, 2))
        
        # Covariance Figure
        X = np.hstack((np.ones((self.E.J,1)), Σ_j))
        X_NT = np.hstack((np.ones((self.E.J,1)), Σ_j_NT))
        y = ln_w.reshape((-1,1))
        y_NT = ln_w_NT.reshape((-1,1))
        W = np.diag(self.E.n)
        
        β = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        β_NT = np.linalg.inv(X_NT.T @ W @ X_NT) @ X_NT.T @ W @ y_NT
        
        ln_w_hat = X @ β
        ln_w_hat_NT = X_NT @ β_NT
        
        DF_Cov_NT = pd.DataFrame(np.hstack((self.E.n.reshape((-1,1)), Σ_j, y, ln_w_hat, Σ_j_NT, y_NT, ln_w_hat_NT)), 
                             columns=['Weight', 'ES Optimal', 'Log Wages Optimal', 'Log Wages_hat Optimal', 'ES Capital Tax', 'Log Wages Capital Tax', 'Log Wages_hat Capital Tax'])
        DF_Cov_NT = DF_Cov_NT.sort_values("Weight", ascending=False)
        DF_Cov_NT.to_csv(f'{self.Directory}/Results/Figures/Mirrlees_Covariance.csv', index=False)
        
        
        Σ_robust_Results.to_csv(f'{self.Directory}/Results/Tables/Σ_robust_Results.csv')
        
        
        
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
            