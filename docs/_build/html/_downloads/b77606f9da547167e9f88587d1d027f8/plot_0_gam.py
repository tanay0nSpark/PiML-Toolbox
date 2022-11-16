# -*- coding: utf-8 -*-
"""
Generalized additive model (GAM)
=====================================

"""

from piml import Experiment
from piml.models import GAMRegressor

exp = Experiment()
exp.data_loader(data="BikeSharing")
exp.data_prepare()

exp.model_train(model=GAMRegressor(), name="GAM")
exp.model_interpret("GAM", show="global_effect_plot", uni_feature="hr", figsize=(6, 4))