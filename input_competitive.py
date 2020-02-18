# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:49:53 2020

@author: Luke
"""

from __future__ import division
import os
import glob
from os.path import join
import pandas as pd
import numpy as np
import math
import time
from pyomo.environ import *
from pyomo.bilevel import *
from pyomo.opt import SolverFactory

import model_competitive

start_time = time.time()
cwd = os.getcwd()

def scenario_inputs(inputs_directory):
    data = DataPortal()
    
    data.load(filename=os.path.join(inputs_directory, "PJM_generators.csv"),
              index=model_competitive.dispatch_model.GENERATORS,
              param=(model_competitive.dispatch_model.fuelcost,
                     model_competitive.dispatch_model.pmin,
                     model_competitive.dispatch_model.startcost,
                     model_competitive.dispatch_model.canspin,
                     model_competitive.dispatch_model.cannonspin,
                     model_competitive.dispatch_model.minup,
                     model_competitive.dispatch_model.mindown,
                     model_competitive.dispatch_model.noloadcost)
              )
              
    data.load(filename=os.path.join(inputs_directory, "initialize_generators.csv"),
              param=(model_competitive.dispatch_model.commitinit,
                     model_competitive.dispatch_model.upinit,
                     model_competitive.dispatch_model.downinit)
              )

    data.load(filename=os.path.join(inputs_directory, "PJM_generators_scheduled_outage.csv"),
              param=(model_competitive.dispatch_model.scheduledavailable)
              )
    
    data.load(filename=os.path.join(inputs_directory, "PJM_generators_zone.csv"),
              param=(model_competitive.dispatch_model.capacity,
                     model_competitive.dispatch_model.ramp,
                     model_competitive.dispatch_model.rampstartuplimit,
                     model_competitive.dispatch_model.rampshutdownlimit)
              )
        
              
    data.load(filename=os.path.join(inputs_directory, "timepoints_index.csv"),
              index=model_competitive.dispatch_model.TIMEPOINTS,
              param=(model_competitive.dispatch_model.temperature)
              )

    data.load(filename=os.path.join(inputs_directory, "zones.csv"),
              index=model_competitive.dispatch_model.ZONES,
              param=(model_competitive.dispatch_model.windcap,
                     model_competitive.dispatch_model.solarcap)
              )

    data.load(filename=os.path.join(inputs_directory, "timepoints_zonal.csv"),
              param=(model_competitive.dispatch_model.GrossLoad,
                     model_competitive.dispatch_model.windcf,
                     model_competitive.dispatch_model.solarcf)
              ) 

    data.load(filename=os.path.join(inputs_directory,"generator_segments.csv"),
              index=model_competitive.dispatch_model.GENERATORSEGMENTS,
              param=(model_competitive.dispatch_model.generatorsegmentlength)
              )
    
    data.load(filename=os.path.join(inputs_directory,"generator_segment_marginalcost.csv"),
              param=(model_competitive.dispatch_model.generatormarginalcost)
              )
              
    return data
