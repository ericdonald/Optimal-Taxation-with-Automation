"""""""""""
Executor Module

Last Modified: Eric Donald 5/25

Notes:
    
Output:
"""""""""""

import Economy as e
import Processor as p



"Define Objects"
E = e.Economy(300)

P = p.Processor(E)


"Clean Data"
P.Cleaner()


"Calibrate"
P.Calibrate()


"Validation"
P.Validation()


"Optimal Threshold Rule in Status Quo"
P.StatusQuo_Optimum()


"Non-Linear Tax Problem"
P.Mirrlees_Optimum()