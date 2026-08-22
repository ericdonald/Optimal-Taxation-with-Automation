"""""""""""
Executor

Notes: This file executes the code for "Optimal Taxation with Automation".
    
"""""""""""

import Economy as e
import Processor as p


# ----------------------------------------------------------------

# Define project objects.

# ----------------------------------------------------------------

E = e.Economy(300)
P = p.Processor(E)


# ----------------------------------------------------------------

# Run project methods.

# ----------------------------------------------------------------

# ---------- #
# Clean Data #
# ---------- #
#API = 0
#Set to 1 for new API download

#P.Cleaner(API)


# --------- #
# Calibrate #
# --------- #
P.Calibrate()


# ---------- #
# Validation #
# ---------- #
#P.Validation()


# --------------------------- #
# Optimal Parametric Policies #
# --------------------------- #
#P.Parametric_Optimum()


# ---------------------- #
# Non-Linear Tax Problem #
# ---------------------- #
P.Mirrlees_Optimum()


# ------------------ #
# Introduction of AI #
# ------------------ #
#P.AI_Experiment(0.5, 5)


# ---------------------------------- #
# Lower Elasticities of Substitution #
# ---------------------------------- #
P.Σ_robust(0.99, 0.3)


# ----------------------- #
# Record Package Versions #
# ----------------------- #
packages = ["cyipopt", "ipumspy", "numba", "numpy", "openpyxl", "pandas", "quantecon", "scipy"]
P.write_package_versions(packages)











