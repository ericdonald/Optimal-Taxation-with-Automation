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
import sys, pickle
import importlib.metadata as md
import Roots as rt
import Production_Functions as fn
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
                
        
        
    def Cleaner(self, API, CPI_year=2016):
        """""
        Clean Data
        
        Output: Raw Data/FRED_CPI.pkl 
                Clean Data/FRED_CPI.pkl
                Clean Data/SCF_2016.pkl
                Clean Data/Census80.pkl
                Clean Data/ACS16.pkl
                Clean Data/Webb.pkl
                Clean Data/CapbyOcc_ES_2d.pkl
                Raw Data/Felten.pkl
                Clean Data/Felten.pkl
                Raw Data/Elondou.pkl
                Clean Data/Elondou.pkl
        """""
   
        
        # -------- #
        # FRED CPI #
        # -------- #
        if API==1:
            FRED = api.get(f'https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&frequency=a&api_key={self.FRED_API}&file_type=json')
            data = FRED.json()['observations']
            filtered_data = [{'year': entry['date'], 'CPI': entry['value']} for entry in data]
            CPI_df = pd.DataFrame(filtered_data)
            CPI_df.to_pickle(f'{self.Directory}/Raw Data/FRED_CPI.pkl')
        else:
            CPI_df = pd.read_pickle(f'{self.Directory}/Raw Data/FRED_CPI.pkl')
        
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
        if API==1:
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
        Crosswalk_soc_df= Crosswalk_df[['occ1990dd', 'onetsoccode']].drop_duplicates()
        Crosswalk_occ_df = Crosswalk_df[['occ1990dd', 'occ']].drop_duplicates()        
        
        Census80_df = IPUMS_df[IPUMS_df['year'] == 1980]
        ACS16_df = IPUMS_df[IPUMS_df['year'] == 2016]
        
        Crosswalk_occ_df = pd.merge(
            Crosswalk_occ_df,
            Census80_df['occ'].drop_duplicates(),
            on='occ',
            how='inner'
        )
        Crosswalk_occ_df = pd.merge(
            Crosswalk_occ_df,
            ACS16_df['occ'].drop_duplicates(),
            on='occ',
            how='inner'
        ) #Only take intersection of available occupations 
                
        Census80_df = pd.merge(
            Census80_df,
            Crosswalk_occ_df,
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
            Crosswalk_occ_df,
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
                
        Webb_df = pd.merge(
            Webb_df,
            Crosswalk_occ_df['occ1990dd'].drop_duplicates(),
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
        
        
        # --------------------- #
        # Felten et al Exposure #
        # --------------------- #
        if API==1:
            Felten_df = pd.read_excel("https://raw.githubusercontent.com/AIOE-Data/AIOE/main/AIOE_DataAppendix.xlsx", sheet_name="Appendix A")
            Felten_df.to_pickle(f'{self.Directory}/Raw Data/Felten.pkl')
        else:
            Felten_df = pd.read_pickle(f'{self.Directory}/Raw Data/Felten.pkl')
        
        Crosswalk_soc_df['SOC Code'] = Crosswalk_soc_df["onetsoccode"].str[:7]
        Felten_df = Felten_df.merge(Crosswalk_soc_df[['SOC Code', 'occ1990dd']].drop_duplicates()
                                    , on='SOC Code', how='inner')
        Felten_df['AIOE_expos'] = Felten_df.groupby('occ1990dd')['AIOE'].transform('mean')
        Felten_df = Felten_df[['occ1990dd', 'AIOE_expos']].drop_duplicates()
        
        Felten_df = Felten_df.merge(Webb_df[['occ1990dd', 'lswt2010']],
                                    on='occ1990dd',how='inner')
        
        Felten_df = Felten_df.sort_values('AIOE_expos')

        Felten_df['cum_weight'] = Felten_df['lswt2010'].cumsum()
        Felten_df['percentile'] = 100 * Felten_df['cum_weight'] / Felten_df['lswt2010'].sum()
        
        Felten_df = Felten_df[['occ1990dd', 'percentile']].sort_values('occ1990dd')
        Felten_df.to_pickle(f'{self.Directory}/Clean Data/Felten.pkl')

        
        # ---------------------- #
        # Elondou et al Exposure #
        # ---------------------- #
        if API==1:
            Elondou_df = pd.read_csv("https://github.com/openai/GPTs-are-GPTs/raw/refs/heads/main/data/occ_level.csv")
            Elondou_df.to_pickle(f'{self.Directory}/Raw Data/Elondou.pkl')
        else:
            Elondou_df = pd.read_pickle(f'{self.Directory}/Raw Data/Elondou.pkl')
            
        Elondou_df.rename(columns={'O*NET-SOC Code': 'onetsoccode'}, inplace=True)
        Elondou_df = pd.merge(
            Crosswalk_soc_df,
            Elondou_df[['onetsoccode', 'dv_rating_beta', 'human_rating_beta']],
            on='onetsoccode',
            how='inner'
        )
        
        Elondou_df['β_expos'] = (Elondou_df.groupby('occ1990dd')['dv_rating_beta'].transform('mean') + Elondou_df.groupby('occ1990dd')['human_rating_beta'].transform('mean')) / 2
        Elondou_df = Elondou_df[['occ1990dd', 'β_expos']].drop_duplicates()
        
        Elondou_df = pd.merge(
            Elondou_df,
            Webb_df[['occ1990dd', 'lswt2010']],
            on='occ1990dd',
            how='inner'
        )
        
        Elondou_df = Elondou_df.sort_values('β_expos')

        Elondou_df['cum_weight'] = Elondou_df['lswt2010'].cumsum()
        Elondou_df['percentile'] = 100 * Elondou_df['cum_weight'] / Elondou_df['lswt2010'].sum()
        
        Elondou_df = pd.merge(
            Elondou_df,
            Webb_df[['occ1990dd', 'pct_software', 'pct_robot', 'pct_ai']],
            on='occ1990dd',
            how='outer'
        )
        
        W_sft_exp = Elondou_df.dropna()['pct_software'].to_numpy()
        W_rbt_exp = Elondou_df.dropna()['pct_robot'].to_numpy()
        W_ai_exp = Elondou_df.dropna()['pct_ai'].to_numpy()
        E_exp = Elondou_df.dropna()['percentile'].to_numpy()
        
        X = np.hstack((np.ones((W_ai_exp.size,1)), W_sft_exp.reshape((-1,1)), W_rbt_exp.reshape((-1,1)), W_ai_exp.reshape((-1,1))))
        β = np.linalg.inv(X.T @ X) @ X.T @ E_exp.reshape((-1,1))
        
        Elondou_df['percentile'] = Elondou_df['percentile'].fillna(
            β[0,0] + β[1,0] * Elondou_df['pct_software'] + β[2,0] * Elondou_df['pct_robot'] + β[3,0] * Elondou_df['pct_ai']
            )
        
        Elondou_df = Elondou_df[['occ1990dd', 'percentile']].sort_values('occ1990dd')
        Elondou_df.to_pickle(f'{self.Directory}/Clean Data/Elondou.pkl')



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
        cov = np.cov(X_corr, aweights=self.E.n)

        std = np.sqrt(np.diag(cov))
        corr_mat = cov / np.outer(std, std)
                
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
        
        X = np.hstack((np.ones((18,1)), Sigma_model.reshape((-1,1))))
        
        β = np.linalg.inv(X.T @ X) @ X.T @ Sigma.reshape((-1,1))
        
        Sigma_hat = (X @ β).reshape(-1)
        
        Validation_Results.add('Occ Regression Slope', gpf.clean_round(β[1,0], 2))
        
        CapbyOcc_ES_2d_df['Sigma_2d'] = Sigma_model
        CapbyOcc_ES_2d_df['Sigma_2d_hat'] = Sigma_hat
        CapbyOcc_ES_2d_df['norm'] = norm
        CapbyOcc_ES_2d_df.loc[CapbyOcc_ES_2d_df['norm'] < CapbyOcc_ES_2d_df['norm'].quantile(5/9), 'occ1990dd_2d_title'] = ''
        
        CapbyOcc_ES_2d_df.to_csv(f'{self.Directory}/Results/Figures/CapbyOcc_ES_2d.csv', index=False)
        
        Validation_Results.to_csv(f'{self.Directory}/Results/Tables/Validation_Results.csv')
        
        
        
    def Parametric_Optimum(self):
        """""
        Optimal Parametric Policy Tools
        
        Output: Results/Figures/StatusQuo_Covariance.csv
                Results/Figures/StatusQuo_VAT_Graph.csv
                Results/Tables/Parametric_Results.csv
        """""
        
        Parametric_Results = gpf.ResultsTable()
        
        
        # ----------------------------------------------------------------

        # Status quo tax function optimum.

        # ----------------------------------------------------------------
        (θ_sq,) = self.E.Para_Solver(1, 0)
        (τ_k_sq,) = self.E.Para_Solver(0, 1)
        (θ_sq_both, τ_k_sq_both) = self.E.Para_Solver(1, 1)
        
        Parametric_Results.add('Optimal Status Quo Threshold Rule', gpf.clean_round(θ_sq*100, 1))
        Parametric_Results.add('Optimal Status Quo Capital Tax', gpf.clean_round(τ_k_sq*100, 1))
        Parametric_Results.add('Optimal Status Quo Threshold Rule, Both', gpf.clean_round(θ_sq_both*100, 1))
        Parametric_Results.add('Optimal Status Quo Capital Tax, Both', gpf.clean_round(τ_k_sq_both*100, 1))
        
        
        # ----------------- #
        # Derive Equilibria #
        # ----------------- #
        x_sq = self.E.var_κ * self.E.x_bar
        E_init = np.concatenate((self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, x_sq))
        alloc_args = (self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, self.E.J, self.E.n, self.E.β, self.E.var_θ, self.E.ε, self.E.δ, self.E.g, self.E.φ)
        
        c_0_θ,    c_1_θ,    l_θ,    x_θ    = gpf.solve_eqbm(θ_sq, self.E.τ_k, self.E.Ψ, self.E.ψ, self.E.y_0, E_init, *alloc_args)
        c_0_τ,    c_1_τ,    l_τ,    x_τ    = gpf.solve_eqbm(0, τ_k_sq, self.E.Ψ, self.E.ψ, self.E.y_0, E_init, *alloc_args)
        c_0_both, c_1_both, l_both, x_both = gpf.solve_eqbm(θ_sq_both, τ_k_sq_both, self.E.Ψ, self.E.ψ, self.E.y_0, E_init, *alloc_args)

        stats_τ    = gpf.alloc_stats(c_0_τ,    c_1_τ,    l_τ,    x_τ, self.E.y_0, *alloc_args)
        stats_both = gpf.alloc_stats(c_0_both, c_1_both, l_both, x_both, self.E.y_0, *alloc_args)
        
       
        # -------------------- #
        # Status Quo to Policy #
        # -------------------- #
        ConEquiv_θ = gpf.consumption_equiv(c_0_θ, c_1_θ, l_θ,
                                self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, *alloc_args)
        Parametric_Results.add('Optimal Status Quo Consumption Equivalence, Theta', gpf.clean_round(ConEquiv_θ, 2))
        
        ConEquiv_τ = gpf.consumption_equiv(c_0_τ, c_1_τ, l_τ,
                                self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, *alloc_args)
        Parametric_Results.add('Optimal Status Quo Consumption Equivalence, Capital Tax', gpf.clean_round(ConEquiv_τ, 2))
        
        
        # ------------------- #
        # Capital Tax to Both #
        # ------------------- #
        Δ_ln_Λ_both, Δ_var_λ_both, Δ_cov_both = gpf.deltas(stats_both, stats_τ)
        ConEquiv_both = gpf.consumption_equiv(c_0_both, c_1_both, l_both,
                                           c_0_τ, c_1_τ, l_τ, *alloc_args)
        
        Parametric_Results.add('Optimal Status Quo DLambda, Both', gpf.clean_round(Δ_ln_Λ_both, 1))
        Parametric_Results.add('Optimal Status Quo DvarWW, Both', gpf.clean_round(Δ_var_λ_both, 1))
        Parametric_Results.add('Optimal Status Quo DCOV, Both', gpf.clean_round(Δ_cov_both, 1))
        Parametric_Results.add('Optimal Status Quo Consumption Equivalence, Both', gpf.clean_round(ConEquiv_both, 2))

        gpf.cov_dataframe(stats_both, 'Both', stats_τ, 'tau_k', self.E.n, self.E.J).to_csv(
                        f'{self.Directory}/Results/Figures/StatusQuo_Covariance.csv', index=False)
        
        
        # ----------------------------------------------------------------

        # Status quo tax function optimum with VAT.

        # ----------------------------------------------------------------
        avg_y_0 = np.sum(self.E.n * self.E.y_0)
        
        VAT_cases = [5, 25, 45, 65]
        vat_rows  = []
        
        for τ_vat in VAT_cases:
            y_0_vat = self.E.y_0 * (100 - τ_vat)/100 + avg_y_0 * τ_vat/100
            
            (τ_k_sq_vat,) = self.E.Para_Solver(0, 1, y_0=y_0_vat)
            (θ_sq_vat_both, τ_k_sq_vat_both) = self.E.Para_Solver(1, 1, y_0=y_0_vat)
                                    
            
            # ----------------- #
            # Derive Equilibria #
            # ----------------- #
            c_0_sq_vat,    c_1_sq_vat,    l_sq_vat,    x_sq_vat    = gpf.solve_eqbm(0, τ_k_sq_vat, self.E.Ψ, self.E.ψ, y_0_vat, E_init, *alloc_args)
            c_0_sq_vat_both, c_1_sq_vat_both, l_sq_vat_both, x_sq_vat_both = gpf.solve_eqbm(θ_sq_vat_both, τ_k_sq_vat_both, self.E.Ψ, self.E.ψ, y_0_vat, E_init, *alloc_args)

            stats_sq_vat    = gpf.alloc_stats(c_0_sq_vat,    c_1_sq_vat,    l_sq_vat,    x_sq_vat, y_0_vat, *alloc_args)
            stats_sq_vat_both = gpf.alloc_stats(c_0_sq_vat_both, c_1_sq_vat_both, l_sq_vat_both, x_sq_vat_both, y_0_vat, *alloc_args)
            
            
            # ---------- #
            # Comparison #
            # ---------- #
            ConEquiv_sq_vat = gpf.consumption_equiv(c_0_sq_vat_both, c_1_sq_vat_both, l_sq_vat_both,
                                               c_0_sq_vat, c_1_sq_vat, l_sq_vat, *alloc_args)
            
            vat_rows.append({'VAT': τ_vat,
                            'Capital Tax': gpf.clean_round(τ_k_sq_vat*100, 1),
                            'Capital Tax Both': gpf.clean_round(τ_k_sq_vat_both*100, 1),
                            'dτ_k':             gpf.clean_round((τ_k_sq_vat - τ_k_sq_vat_both)*100, 1),
                            'Threshold Rule':   gpf.clean_round(θ_sq_vat_both*100, 1),
                            'CE Welfare Gain':  gpf.clean_round(ConEquiv_sq_vat, 2),})
            
            if τ_vat == 25:
                Parametric_Results.add('Optimal Status Quo VAT Capital Tax', gpf.clean_round(τ_k_sq_vat*100, 1))
                Parametric_Results.add('Optimal Status Quo VAT Threshold Rule, Both', gpf.clean_round(θ_sq_vat_both*100, 1))
                Parametric_Results.add('Optimal Status Quo VAT Capital Tax, Both', gpf.clean_round(τ_k_sq_vat_both*100, 1))
                
                Δ_ln_Λ_vat_both, Δ_var_λ_vat_both, Δ_cov_vat_both = gpf.deltas(stats_sq_vat_both, stats_sq_vat)
                
                Parametric_Results.add('Optimal Status Quo VAT DLambda, Both', gpf.clean_round(Δ_ln_Λ_vat_both, 1))
                Parametric_Results.add('Optimal Status Quo VAT DvarWW, Both', gpf.clean_round(Δ_var_λ_vat_both, 1))
                Parametric_Results.add('Optimal Status Quo VAT DCOV, Both', gpf.clean_round(Δ_cov_vat_both, 1))
                Parametric_Results.add('Optimal Status Quo VAT Consumption Equivalence, Both', gpf.clean_round(ConEquiv_sq_vat, 2))

        
        VAT_df = pd.DataFrame(vat_rows)
        VAT_df.to_csv(f'{self.Directory}/Results/Figures/StatusQuo_VAT_Graph.csv', index=False)
        
        Parametric_Results.add('Optimal Status Quo VAT Threshold Rule, Start', gpf.clean_round(VAT_df['Threshold Rule'].iloc[0], 1))
        Parametric_Results.add('Optimal Status Quo VAT Threshold Rule, End', gpf.clean_round(VAT_df['Threshold Rule'].iloc[-1], 1))
        
        Parametric_Results.add('Optimal Status Quo VAT Capital Tax Difference, Start', gpf.clean_round(VAT_df['dτ_k'].iloc[0], 1))
        Parametric_Results.add('Optimal Status Quo VAT Capital Tax Difference, End', gpf.clean_round(VAT_df['dτ_k'].iloc[-1], 1))
        
        Parametric_Results.add('Optimal Status Quo VAT Consumption Equivalence, Start', gpf.clean_round(VAT_df['CE Welfare Gain'].iloc[0], 2))
        Parametric_Results.add('Optimal Status Quo VAT Consumption Equivalence, End', gpf.clean_round(VAT_df['CE Welfare Gain'].iloc[-1], 2))
        
        
        Parametric_Results.to_csv(f'{self.Directory}/Results/Tables/Parametric_Results.csv')
        
        
        
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
        x_sq = self.E.var_κ * self.E.x_bar
        E_init = np.concatenate((self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, x_sq))
        avg_y_0 = np.sum(self.E.n * self.E.y_0)
        alloc_args = (self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, self.E.J, self.E.n, self.E.β, self.E.var_θ, self.E.ε, self.E.δ, self.E.g, self.E.φ)
        
        c_0_0,    c_1_0,    l_0,    x_0    = gpf.solve_eqbm(0, self.E.τ_k, self.E.Ψ, self.E.ψ, avg_y_0, E_init, *alloc_args)
        E_0 = np.concatenate((c_0_0, c_1_0, l_0, x_0))
        (c_0_NT, c_1_NT, l_NT, K_NT, x_NT, τ_k_NT), _ = self.E.Mirrlees_Lagr(E_0, 0)
        E_mirr = np.concatenate((c_0_NT, c_1_NT, l_NT, x_NT))
        with open(f'{self.Directory}/Clean Data/E_mirr.pkl', 'wb') as file:
            pickle.dump(E_mirr, file)
        
        E_NT = np.concatenate((c_0_NT, c_1_NT, l_NT, x_NT, np.zeros(1)))
        (c_0, c_1, l, K, x, τ_k), θ, _ = self.E.Mirrlees_Lagr(E_NT, 1)
        
        Mirrlees_Results.add('Optimal Mirrlees Capital Tax', gpf.clean_round(τ_k_NT*100, 1))
        
        L_NT = self.E.n * l_NT
        r_NT = fn.Rents(x_NT, L_NT, K_NT, self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ)
        Return_NT = 1 + r_NT - self.E.δ
        Return_tilde_NT = 1 + (1-τ_k_NT)*(r_NT-self.E.δ)
        τ_wealth_NT = 1 - Return_tilde_NT / Return_NT
        Mirrlees_Results.add('Optimal Mirrlees Wealth Tax', gpf.clean_round(τ_wealth_NT*100, 2))
        
        Mirrlees_Results.add('Optimal Mirrlees Threshold Rule', gpf.clean_round(θ*100, 1))
        
        
        # ---------------- #
        # Comparison Table #
        # ---------------- #
        stats_NT    = gpf.alloc_stats(c_0_NT, c_1_NT, l_NT, x_NT, avg_y_0, *alloc_args)
        stats = gpf.alloc_stats(c_0, c_1, l, x, avg_y_0, *alloc_args)
        
        Δ_ln_Λ, Δ_var_λ, Δ_cov = gpf.deltas(stats, stats_NT)
        ConEquiv = gpf.consumption_equiv(c_0, c_1, l,
                                           c_0_NT, c_1_NT, l_NT, *alloc_args)
        
        Mirrlees_Results.add('Optimal Mirrlees DLambda', gpf.clean_round(Δ_ln_Λ, 1))
        Mirrlees_Results.add('Optimal Mirrlees DvarWW', gpf.clean_round(Δ_var_λ, 1))
        Mirrlees_Results.add('Optimal Mirrlees DCOV', gpf.clean_round(Δ_cov, 1))
        Mirrlees_Results.add('Optimal Mirrlees Consumption Equivalence', gpf.clean_round(ConEquiv, 2))
        
        gpf.cov_dataframe(stats, 'Both', stats_NT, 'tau_k', self.E.n, self.E.J).to_csv(
                        f'{self.Directory}/Results/Figures/Mirrlees_Covariance.csv', index=False)
        

        Mirrlees_Results.to_csv(f'{self.Directory}/Results/Tables/Mirrlees_Results.csv')
        
        
    
    def AI_Experiment(self, G_k, N_g):
        """""
        Optimal Threshold Rule for AI Scenarios
        
        Output: Results/Figures/AI_Experiment.csv
                Results/Figures/AI_Experiment_LAT.csv
                Results/Figures/AI_Experiment_VAT.csv
                Results/Figures/AI_Experiment_Felten.csv
                Results/Figures/AI_Experiment_Elondou.csv
                Results/Tables/AI_Experiment_Results.csv
        """""
        
        AI_Experiment_Results = gpf.ResultsTable()
        
        def _ConEquiv_seq(τ_k_AI, θ_AI_both, τ_k_AI_both, ζ_AI, ν_AI, A_k_AI_base, A_j_AI_base, LAT_frac, y_0):
            
            ConEquiv_AI = np.empty(N_g)
            x_sq = self.E.var_κ * self.E.x_bar
            E_init_AI = np.concatenate((self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, x_sq))
            E_init_AI_both = np.concatenate((self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, x_sq))
            
            for n in range(N_g):
                A_k_AI = A_k_AI_base * (1 + AI_growth[n])
                A_j_AI = A_j_AI_base * (1 + AI_growth[n] * LAT_frac)
                
                g_y = self.E.S_k * AI_growth[n] + (1-self.E.S_k) * AI_growth[n] * LAT_frac
                Ψ_AI = self.E.Ψ * (1+g_y)**(self.E.ψ)
                
                alloc_args = (A_j_AI, A_k_AI, self.E.x_bar, ζ_AI, ν_AI, self.E.σ, self.E.J, self.E.n, self.E.β, self.E.var_θ, self.E.ε, self.E.δ, self.E.g, self.E.φ)
                
                c_0_AI, c_1_AI, l_AI, x_AI = gpf.solve_eqbm(0, τ_k_AI[n], Ψ_AI, self.E.ψ, y_0, E_init_AI, *alloc_args)
                c_0_AI_both, c_1_AI_both, l_AI_both, x_AI_both = gpf.solve_eqbm(θ_AI_both[n], τ_k_AI_both[n], Ψ_AI, self.E.ψ, y_0, E_init_AI_both, *alloc_args)
                
                ConEquiv_AI[n] = gpf.consumption_equiv(c_0_AI_both, c_1_AI_both, l_AI_both,
                                                       c_0_AI, c_1_AI, l_AI, *alloc_args)
                
                E_init_AI = np.concatenate((c_0_AI, c_1_AI, l_AI, x_AI))
                E_init_AI_both = np.concatenate((c_0_AI_both, c_1_AI_both, l_AI_both, x_AI_both))
                
            return ConEquiv_AI
        
        
        def _stat_helper(τ_k_AI, τ_k_AI_both, θ_AI_both, ConEquiv_AI, case=''):
            
            dτ_k_AI = (τ_k_AI - τ_k_AI_both) * 100 
            θ_AI_both *= 100
            
            AI_Experiment_Results.add(f'Optimal AI {case}Threshold Rule, Start', gpf.clean_round(θ_AI_both[0], 1))
            AI_Experiment_Results.add(f'Optimal AI {case}Threshold Rule, End', gpf.clean_round(θ_AI_both[1], 1))
            
            AI_Experiment_Results.add(f'Optimal AI {case}Capital Tax Difference, Start', gpf.clean_round(dτ_k_AI[0], 1))
            AI_Experiment_Results.add(f'Optimal AI {case}Capital Tax Difference, End', gpf.clean_round(dτ_k_AI[-1], 1))
            
            AI_Experiment_Results.add(f'Optimal AI {case}Consumption Equivalence, Start', gpf.clean_round(ConEquiv_AI[0], 2))
            AI_Experiment_Results.add(f'Optimal AI {case}Consumption Equivalence, End', gpf.clean_round(ConEquiv_AI[-1], 2))

        
        AI_growth = np.linspace(0, G_k, N_g)
        
        
        # ----------------------------------------------------------------

        # Webb (2020) AI exposure.

        # ----------------------------------------------------------------
        
        # ------------------- #
        # Post-AI Calibration #
        # ------------------- #
        Webb_df = pd.read_pickle(f'{self.Directory}/Clean Data/Webb.pkl')
        χ_AI_webb = (Webb_df['pct_software'].to_numpy() + Webb_df['pct_robot'].to_numpy() + Webb_df['pct_ai'].to_numpy()) / 3
        
        ζ_AI_webb = np.exp(self.E.Γ * χ_AI_webb)
       
        nu_g = np.ones(self.E.J) * self.E.var_κ
        
        nu = sp.optimize.root(rt.νRoot, nu_g,
                      args=(self.E.Γ, χ_AI_webb, self.E.var_κ, self.E.x_bar, self.E.σ, self.E.S_j_sq, self.E.S_k),
                      method='lm')
        
        ν_AI_webb = nu.x
        
        x_AI = self.E.var_κ * self.E.x_bar
        Λ_k_AI_webb = fn.Lamba_k(x_AI, self.E.x_bar, ζ_AI_webb, ν_AI_webb, self.E.σ)
        A_k_AI_webb_base = self.E.r_sq**(self.E.σ / (self.E.σ-1)) * (self.E.COR / Λ_k_AI_webb)**(1 / (self.E.σ-1))
        
        A_j_AI_webb_base = (self.E.w_j_sq / x_AI**(ζ_AI_webb)) / (self.E.r_sq / A_k_AI_webb_base)
        
        
        # --------- #
        # Base Case #
        # --------- #
        (τ_k_AI,) = self.E.AI_economy(0, 1, ζ_AI_webb, ν_AI_webb, A_k_AI_webb_base, A_j_AI_webb_base, 0.25, self.E.y_0, G_k, N_g)
        (θ_AI_both, τ_k_AI_both) = self.E.AI_economy(1, 1, ζ_AI_webb, ν_AI_webb, A_k_AI_webb_base, A_j_AI_webb_base, 0.25, self.E.y_0, G_k, N_g)
        
        ConEquiv_AI = _ConEquiv_seq(τ_k_AI, θ_AI_both, τ_k_AI_both, ζ_AI_webb, ν_AI_webb, A_k_AI_webb_base, A_j_AI_webb_base, 0.25, self.E.y_0)
        
        DF_AI = pd.DataFrame(100*np.hstack((AI_growth.reshape((-1,1)), θ_AI_both.reshape((-1,1)), τ_k_AI.reshape((-1,1)), τ_k_AI_both.reshape((-1,1)), ConEquiv_AI.reshape((-1,1)))), 
                             columns=['Growth', 'Threshold Rule', 'Capital Tax', 'Capital Tax Both', 'Consumption Equivalence'])
        DF_AI.to_csv(f'{self.Directory}/Results/Figures/AI_Experiment.csv', index=False)
        
        _stat_helper(τ_k_AI, τ_k_AI_both, θ_AI_both, ConEquiv_AI)
        
        
        # -------------- #
        # LAT Robustness #
        # -------------- #
        (τ_k_AI_10,) = self.E.AI_economy(0, 1, ζ_AI_webb, ν_AI_webb, A_k_AI_webb_base, A_j_AI_webb_base, 0.10, self.E.y_0, G_k, N_g)
        (θ_AI_both_10, τ_k_AI_both_10) = self.E.AI_economy(1, 1, ζ_AI_webb, ν_AI_webb, A_k_AI_webb_base, A_j_AI_webb_base, 0.10, self.E.y_0, G_k, N_g)
        
        ConEquiv_AI_10 = _ConEquiv_seq(τ_k_AI_10, θ_AI_both_10, τ_k_AI_both_10, ζ_AI_webb, ν_AI_webb, A_k_AI_webb_base, A_j_AI_webb_base, 0.10, self.E.y_0)
        
        (τ_k_AI_50,) = self.E.AI_economy(0, 1, ζ_AI_webb, ν_AI_webb, A_k_AI_webb_base, A_j_AI_webb_base, 0.50, self.E.y_0, G_k-0.02, N_g)
        (θ_AI_both_50, τ_k_AI_both_50) = self.E.AI_economy(1, 1, ζ_AI_webb, ν_AI_webb, A_k_AI_webb_base, A_j_AI_webb_base, 0.50, self.E.y_0, G_k, N_g)
        
        ConEquiv_AI_50 = _ConEquiv_seq(τ_k_AI_50, θ_AI_both_50, τ_k_AI_both_50, ζ_AI_webb, ν_AI_webb, A_k_AI_webb_base, A_j_AI_webb_base, 0.50, self.E.y_0)
        
        DF_AI_LAT = pd.DataFrame(100*np.hstack((AI_growth.reshape((-1,1)), θ_AI_both_10.reshape((-1,1)), τ_k_AI_10.reshape((-1,1)), τ_k_AI_both_10.reshape((-1,1)), ConEquiv_AI_10.reshape((-1,1)),
                                                                        θ_AI_both_50.reshape((-1,1)), τ_k_AI_50.reshape((-1,1)), τ_k_AI_both_50.reshape((-1,1)), ConEquiv_AI_50.reshape((-1,1)))), 
                             columns=['Growth', 'Threshold Rule 10', 'Capital Tax 10', 'Capital Tax Both 10', 'Consumption Equivalence 10', 
                                                'Threshold Rule 50', 'Capital Tax 50', 'Capital Tax Both 50', 'Consumption Equivalence 50'])
        DF_AI_LAT.to_csv(f'{self.Directory}/Results/Figures/AI_Experiment_LAT.csv', index=False)
        
        _stat_helper(τ_k_AI_10, τ_k_AI_both_10, θ_AI_both_10, ConEquiv_AI_10, case='LAT 10 ')
        _stat_helper(τ_k_AI_50, τ_k_AI_both_50, θ_AI_both_50, ConEquiv_AI_50, case='LAT 50 ')
        
        
        # -------------- #
        # VAT Robustness #
        # -------------- #
        avg_y_0 = np.sum(self.E.n * self.E.y_0)
        τ_vat = 25
        y_0_vat = self.E.y_0 * (100 - τ_vat)/100 + avg_y_0 * τ_vat/100
            
        (τ_k_AI_vat,) = self.E.AI_economy(0, 1, ζ_AI_webb, ν_AI_webb, A_k_AI_webb_base, A_j_AI_webb_base, 0.25, y_0_vat, G_k, N_g)
        (θ_AI_both_vat, τ_k_AI_both_vat) = self.E.AI_economy(1, 1, ζ_AI_webb, ν_AI_webb, A_k_AI_webb_base, A_j_AI_webb_base, 0.25, y_0_vat, G_k, N_g)
        
        ConEquiv_AI_vat = _ConEquiv_seq(τ_k_AI_vat, θ_AI_both_vat, τ_k_AI_both_vat, ζ_AI_webb, ν_AI_webb, A_k_AI_webb_base, A_j_AI_webb_base, 0.25, y_0_vat)
        
        DF_AI_vat = pd.DataFrame(100*np.hstack((AI_growth.reshape((-1,1)), θ_AI_both_vat.reshape((-1,1)), τ_k_AI_vat.reshape((-1,1)), τ_k_AI_both_vat.reshape((-1,1)), ConEquiv_AI_vat.reshape((-1,1)))), 
                             columns=['Growth', 'Threshold Rule VAT', 'Capital Tax VAT', 'Capital Tax Both VAT', 'Consumption Equivalence VAT'])
        DF_AI_vat.to_csv(f'{self.Directory}/Results/Figures/AI_Experiment_VAT.csv', index=False)
        
        _stat_helper(τ_k_AI_vat, τ_k_AI_both_vat, θ_AI_both_vat, ConEquiv_AI_vat, 'VAT ')
        
    
        # ----------------------------------------------------------------

        # Robustness with Felten et al. (2021) exposure.

        # ----------------------------------------------------------------
        
        # ------------------- #
        # Post-AI Calibration #
        # ------------------- #
        Felten_df = pd.read_pickle(f'{self.Directory}/Clean Data/Felten.pkl')
        χ_AI_Felt = Felten_df['percentile'].to_numpy()
        
        ζ_AI_Felt = np.exp(self.E.Γ * χ_AI_Felt)
               
        nu = sp.optimize.root(rt.νRoot, nu_g,
                      args=(self.E.Γ, χ_AI_Felt, self.E.var_κ, self.E.x_bar, self.E.σ, self.E.S_j_sq, self.E.S_k),
                      method='lm')
        
        ν_AI_Felt = nu.x
        
        Λ_k_AI_Felt = fn.Lamba_k(x_AI, self.E.x_bar, ζ_AI_Felt, ν_AI_Felt, self.E.σ)
        A_k_AI_Felt_base = self.E.r_sq**(self.E.σ / (self.E.σ-1)) * (self.E.COR / Λ_k_AI_Felt)**(1 / (self.E.σ-1))
        
        A_j_AI_Felt_base = (self.E.w_j_sq / x_AI**(ζ_AI_Felt)) / (self.E.r_sq / A_k_AI_Felt_base)
        
        
        # ----- #
        # Solve #
        # ----- #
        (τ_k_AI_Felt,) = self.E.AI_economy(0, 1, ζ_AI_Felt, ν_AI_Felt, A_k_AI_Felt_base, A_j_AI_Felt_base, 0.25, self.E.y_0, G_k, N_g)
        (θ_AI_Felt_both, τ_k_AI_Felt_both) = self.E.AI_economy(1, 1, ζ_AI_Felt, ν_AI_Felt, A_k_AI_Felt_base, A_j_AI_Felt_base, 0.25, self.E.y_0, G_k, N_g)
        
        ConEquiv_AI_Felt = _ConEquiv_seq(τ_k_AI_Felt, θ_AI_Felt_both, τ_k_AI_Felt_both, ζ_AI_Felt, ν_AI_Felt, A_k_AI_Felt_base, A_j_AI_Felt_base, 0.25, self.E.y_0)
        
        DF_AI_Felt = pd.DataFrame(100*np.hstack((AI_growth.reshape((-1,1)), θ_AI_Felt_both.reshape((-1,1)), τ_k_AI_Felt.reshape((-1,1)), τ_k_AI_Felt_both.reshape((-1,1)), ConEquiv_AI_Felt.reshape((-1,1)))), 
                             columns=['Growth', 'Threshold Rule Felten', 'Capital Tax Felten', 'Capital Tax Both Felten', 'Consumption Equivalence Felten'])
        DF_AI_Felt.to_csv(f'{self.Directory}/Results/Figures/AI_Experiment_Felten.csv', index=False)
        
        _stat_helper(τ_k_AI_Felt, τ_k_AI_Felt_both, θ_AI_Felt_both, ConEquiv_AI_Felt, 'Felten ')
        
        
        # ----------------------------------------------------------------

        # Robustness with Elondou et al. (2024) exposure.

        # ----------------------------------------------------------------
        
        # ------------------- #
        # Post-AI Calibration #
        # ------------------- #
        Elondou_df = pd.read_pickle(f'{self.Directory}/Clean Data/Elondou.pkl')
        χ_AI_Elond = Elondou_df['percentile'].to_numpy()
        
        ζ_AI_Elond = np.exp(self.E.Γ * χ_AI_Elond)
               
        nu = sp.optimize.root(rt.νRoot, nu_g,
                      args=(self.E.Γ, χ_AI_Elond, self.E.var_κ, self.E.x_bar, self.E.σ, self.E.S_j_sq, self.E.S_k),
                      method='lm')
        
        ν_AI_Elond = nu.x
        
        Λ_k_AI_Elond = fn.Lamba_k(x_AI, self.E.x_bar, ζ_AI_Elond, ν_AI_Elond, self.E.σ)
        A_k_AI_Elond_base = self.E.r_sq**(self.E.σ / (self.E.σ-1)) * (self.E.COR / Λ_k_AI_Elond)**(1 / (self.E.σ-1))
        
        A_j_AI_Elond_base = (self.E.w_j_sq / x_AI**(ζ_AI_Elond)) / (self.E.r_sq / A_k_AI_Elond_base)
        
        
        # ----- #
        # Solve #
        # ----- #
        (τ_k_AI_Elond,) = self.E.AI_economy(0, 1, ζ_AI_Elond, ν_AI_Elond, A_k_AI_Elond_base, A_j_AI_Elond_base, 0.25, self.E.y_0, G_k, N_g)
        (θ_AI_Elond_both, τ_k_AI_Elond_both) = self.E.AI_economy(1, 1, ζ_AI_Elond, ν_AI_Elond, A_k_AI_Elond_base, A_j_AI_Elond_base, 0.25, self.E.y_0, G_k, N_g)
        
        ConEquiv_AI_Elond = _ConEquiv_seq(τ_k_AI_Elond, θ_AI_Elond_both, τ_k_AI_Elond_both, ζ_AI_Elond, ν_AI_Elond, A_k_AI_Elond_base, A_j_AI_Elond_base, 0.25, self.E.y_0)
        
        DF_AI = pd.DataFrame(100*np.hstack((AI_growth.reshape((-1,1)), θ_AI_Elond_both.reshape((-1,1)), τ_k_AI_Elond.reshape((-1,1)), τ_k_AI_Elond_both.reshape((-1,1)), ConEquiv_AI_Elond.reshape((-1,1)))), 
                             columns=['Growth', 'Threshold Rule Elondou', 'Capital Tax Elondou', 'Capital Tax Both Elondou', 'Consumption Equivalence Elondou'])
        DF_AI.to_csv(f'{self.Directory}/Results/Figures/AI_Experiment_Elondou.csv', index=False)
        
        _stat_helper(τ_k_AI_Elond, τ_k_AI_Elond_both, θ_AI_Elond_both, ConEquiv_AI_Elond, 'Elondou ')
        
        
        AI_Experiment_Results.to_csv(f'{self.Directory}/Results/Tables/AI_Experiment_Results.csv')

                
        
    def Σ_robust(self, Σ_low, σ_low):
        """""
        Robustness with Elasticities of Substitution
    
        Output: Results/Tables/ES_robust_Results.csv
        """""
        
        ES_robust_Results = gpf.ResultsTable()
        σ_base = self.E.σ
        self.E.Σ_k = Σ_low
        self.E.σ = σ_low
        
        
        # ----------- #
        # Recalibrate #
        # ----------- #
        self.E.Calibrate()
        
        x_sq = self.E.var_κ * self.E.x_bar
        E_init = np.concatenate((self.E.c_0_sq, self.E.c_1_sq, self.E.l_j_sq, x_sq))
        alloc_args = (self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, self.E.J, self.E.n, self.E.β, self.E.var_θ, self.E.ε, self.E.δ, self.E.g, self.E.φ)

        
        # ------------------------ #
        # Status Quo Optimum + VAT #
        # ------------------------ #
        avg_y_0 = np.sum(self.E.n * self.E.y_0)
        τ_vat = 25
        y_0_vat = self.E.y_0 * (100 - τ_vat)/100 + avg_y_0 * τ_vat/100
            
        (τ_k_sq_vat,) = self.E.Para_Solver(0, 1, y_0=y_0_vat)
        (θ_sq_vat_both, τ_k_sq_vat_both) = self.E.Para_Solver(1, 1, y_0=y_0_vat, damp=1/3)
        
        ES_robust_Results.add('Low ES Status Quo VAT Capital Tax', gpf.clean_round(τ_k_sq_vat*100, 1))
        ES_robust_Results.add('Low ES Status Quo VAT Threshold Rule, Both', gpf.clean_round(θ_sq_vat_both*100, 1))
        ES_robust_Results.add('Low ES Status Quo VAT Capital Tax, Both', gpf.clean_round(τ_k_sq_vat_both*100, 1))
        
        # Derive Equilibria
        c_0_sq_vat,    c_1_sq_vat,    l_sq_vat,    x_sq_vat    = gpf.solve_eqbm(0, τ_k_sq_vat, self.E.Ψ, self.E.ψ, y_0_vat, E_init, *alloc_args)
        c_0_sq_vat_both, c_1_sq_vat_both, l_sq_vat_both, x_sq_vat_both = gpf.solve_eqbm(θ_sq_vat_both, τ_k_sq_vat_both, self.E.Ψ, self.E.ψ, y_0_vat, E_init, *alloc_args)

        stats_sq_vat    = gpf.alloc_stats(c_0_sq_vat,    c_1_sq_vat,    l_sq_vat,    x_sq_vat, y_0_vat, *alloc_args)
        stats_sq_vat_both = gpf.alloc_stats(c_0_sq_vat_both, c_1_sq_vat_both, l_sq_vat_both, x_sq_vat_both, y_0_vat, *alloc_args)
        
        # Comparison Table
        Δ_ln_Λ_vat_both, Δ_var_λ_vat_both, Δ_cov_vat_both = gpf.deltas(stats_sq_vat_both, stats_sq_vat)
        ConEquiv_sq_vat = gpf.consumption_equiv(c_0_sq_vat_both, c_1_sq_vat_both, l_sq_vat_both,
                                           c_0_sq_vat, c_1_sq_vat, l_sq_vat, *alloc_args)
    
        ES_robust_Results.add('Low ES Status Quo VAT DLambda', gpf.clean_round(Δ_ln_Λ_vat_both, 1))
        ES_robust_Results.add('Low ES Status Quo VAT DvarWW', gpf.clean_round(Δ_var_λ_vat_both, 1))
        ES_robust_Results.add('Low ES Status Quo VAT DCOV', gpf.clean_round(Δ_cov_vat_both, 1))
        ES_robust_Results.add('Low ES Status Quo VAT Consumption Equivalence', gpf.clean_round(ConEquiv_sq_vat, 2))
        
        
        # ------------------ #
        # Status Quo Optimum #
        # ------------------ #
        E_vat = np.concatenate((c_0_sq_vat_both, c_1_sq_vat_both, l_sq_vat_both, x_sq_vat_both))
        (τ_k_sq,) = self.E.Para_Solver(0, 1)
        (θ_sq_both, τ_k_sq_both) = self.E.Para_Solver(1, 1, E=E_vat, θ_init=θ_sq_vat_both, τ_k_init=τ_k_sq_vat_both, damp=1/3)
        
        ES_robust_Results.add('Low ES Status Quo Capital Tax', gpf.clean_round(τ_k_sq*100, 1))
        ES_robust_Results.add('Low ES Status Quo Threshold Rule, Both', gpf.clean_round(θ_sq_both*100, 1))
        ES_robust_Results.add('Low ES Status Quo Capital Tax, Both', gpf.clean_round(τ_k_sq_both*100, 1))
        
        # Derive Equilibria        
        c_0_τ,    c_1_τ,    l_τ,    x_τ    = gpf.solve_eqbm(0, τ_k_sq, self.E.Ψ, self.E.ψ, self.E.y_0, E_init, *alloc_args)
        c_0_both, c_1_both, l_both, x_both = gpf.solve_eqbm(θ_sq_both, τ_k_sq_both, self.E.Ψ, self.E.ψ, self.E.y_0, E_vat, *alloc_args)

        stats_τ    = gpf.alloc_stats(c_0_τ,    c_1_τ,    l_τ,    x_τ, self.E.y_0, *alloc_args)
        stats_both = gpf.alloc_stats(c_0_both, c_1_both, l_both, x_both, self.E.y_0, *alloc_args)
        
        # Comparison Table 
        Δ_ln_Λ_both, Δ_var_λ_both, Δ_cov_both = gpf.deltas(stats_both, stats_τ)
        ConEquiv_both = gpf.consumption_equiv(c_0_both, c_1_both, l_both,
                                           c_0_τ, c_1_τ, l_τ, *alloc_args)
        
        ES_robust_Results.add('Low ES Status Quo DLambda', gpf.clean_round(Δ_ln_Λ_both, 1))
        ES_robust_Results.add('Low ES Status Quo DvarWW', gpf.clean_round(Δ_var_λ_both, 1))
        ES_robust_Results.add('Low ES Status Quo DCOV', gpf.clean_round(Δ_cov_both, 1))
        ES_robust_Results.add('Low ES Status Quo Consumption Equivalence', gpf.clean_round(ConEquiv_both, 2))
       
        
        # ---------------- #
        # Mirrlees Problem #
        # ---------------- #
        
        # Warm Start Ramp
        dΣ = Σ_low - σ_low
        σ_prev = σ_base
        with open(f'{self.Directory}/Clean Data/E_mirr.pkl', 'rb') as file:
            E_0 = pickle.load(file)
        
        inter_n = 10
        targets = list(np.linspace(σ_base, σ_low, inter_n + 1))[1:]
        while targets:
            σ_try = targets[0]
            print(σ_try)
            self.E.σ = σ_try; self.E.Σ_k = σ_try + dΣ; self.E.Calibrate()
            out, converged = self.E.Mirrlees_Lagr(E_0, 0)
            if converged:
                c_0, c_1, l, K, x, τ_k = out
                E_0 = np.concatenate((c_0, c_1, l, x))
                σ_prev = σ_try
                targets.pop(0)
            else:
                σ_mid = 0.5 * (σ_prev + σ_try)
                if abs(σ_try - σ_mid) < 1e-4:
                    raise RuntimeError(f'σ-ramp stalled between {σ_prev:.4f} and {σ_try:.4f}')
                targets.insert(0, σ_mid) 
        
        alloc_args = (self.E.A_j, self.E.A_k, self.E.x_bar, self.E.ζ, self.E.ν, self.E.σ, self.E.J, self.E.n, self.E.β, self.E.var_θ, self.E.ε, self.E.δ, self.E.g, self.E.φ)
        (c_0_NT, c_1_NT, l_NT, K_NT, x_NT, τ_k_NT) = out
        
        E_NT = np.concatenate((c_0_NT, c_1_NT, l_NT, x_NT, np.zeros(1)))
        (c_0, c_1, l, K, x, θ, τ_k), _ = self.E.Mirrlees_Lagr(E_NT, 1)
        
        ES_robust_Results.add('Low ES Mirrlees Capital Tax', gpf.clean_round(τ_k_NT*100, 1))
        
        ES_robust_Results.add('Low ES Mirrlees Threshold Rule', gpf.clean_round(θ*100, 1))
    
        # Comparison Table 
        stats_NT    = gpf.alloc_stats(c_0_NT, c_1_NT, l_NT, x_NT, avg_y_0, *alloc_args)
        stats = gpf.alloc_stats(c_0, c_1, l, x, avg_y_0, *alloc_args)
        
        Δ_ln_Λ, Δ_var_λ, Δ_cov = gpf.deltas(stats, stats_NT)
        ConEquiv = gpf.consumption_equiv(c_0, c_1, l,
                                           c_0_NT, c_1_NT, l_NT, *alloc_args)
        
        ES_robust_Results.add('Low ES Mirrlees DLambda', gpf.clean_round(Δ_ln_Λ, 1))
        ES_robust_Results.add('Low ES Mirrlees DvarWW', gpf.clean_round(Δ_var_λ, 1))
        ES_robust_Results.add('Low ES Mirrlees DCOV', gpf.clean_round(Δ_cov, 1))
        ES_robust_Results.add('Low ES Mirrlees Consumption Equivalence', gpf.clean_round(ConEquiv, 2))
        
        
        ES_robust_Results.to_csv(f'{self.Directory}/Results/Tables/ES_robust_Results.csv')
        
        
        
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
            