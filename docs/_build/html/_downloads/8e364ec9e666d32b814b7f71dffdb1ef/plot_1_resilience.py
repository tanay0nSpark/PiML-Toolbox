# -*- coding: utf-8 -*-
"""
Resilience - Performance: XGB
=====================================

PiML model diagnostics
"""

# %%
# Train model and Run 
from piml import Experiment
from xgboost import XGBRegressor

exp = Experiment()
exp.data_loader(data="BikeSharing")
exp.data_prepare()


# %%
# Train model and Run 
exp.model_train(model=XGBRegressor(max_depth=2, n_estimators=100), name="XGB-2")
exp.model_diagnose("XGB-2", show='resilience_perf', immu_feature=None)

