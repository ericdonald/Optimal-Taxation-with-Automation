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
P.Cleaner()


# --------- #
# Calibrate #
# --------- #
P.Calibrate()


# ---------- #
# Validation #
# ---------- #
P.Validation()


# ------------------------------------ #
# Optimal Threshold Rule in Status Quo #
# ------------------------------------ #
P.StatusQuo_Optimum()


# ---------------------- #
# Non-Linear Tax Problem #
# ---------------------- #
P.Mirrlees_Optimum()


# ------------------ #
# Introduction of AI #
# ------------------ #
P.AI_Experiment()


# ----------------------- #
# Record Package Versions #
# ----------------------- #
packages = ["cyipopt", "ipumspy", "numba", "numpy", "pandas", "quantecon", "scipy"]
P.write_package_versions(packages)











