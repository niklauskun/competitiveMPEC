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
              param=(model_competitive.dispatch_model.capacity,
                     model_competitive.dispatch_model.fuelcost,
                     model_competitive.dispatch_model.pmin,
                     model_competitive.dispatch_model.startcost,
                     model_competitive.dispatch_model.canspin,
                     model_competitive.dispatch_model.cannonspin,
                     model_competitive.dispatch_model.minup,
                     model_competitive.dispatch_model.mindown,
                     model_competitive.dispatch_model.noloadcost,
                     model_competitive.dispatch_model.ramp,
                     model_competitive.dispatch_model.zonelabel,
                     model_competitive.dispatch_model.genco_index)
              )
    
    data.load(filename=os.path.join(inputs_directory, "storage_resources.csv"),
              index=model_competitive.dispatch_model.STORAGE,
              param=(model_competitive.dispatch_model.discharge_max,
                     model_competitive.dispatch_model.charge_max,
                     model_competitive.dispatch_model.soc_max,
                     model_competitive.dispatch_model.storage_zone_label)
              )
              
    data.load(filename=os.path.join(inputs_directory, "initialize_generators.csv"),
              param=(model_competitive.dispatch_model.commitinit,
                     model_competitive.dispatch_model.upinit,
                     model_competitive.dispatch_model.downinit)
              )

    data.load(filename=os.path.join(inputs_directory, "PJM_generators_scheduled_outage.csv"),
              param=(model_competitive.dispatch_model.scheduled_available,
                     model_competitive.dispatch_model.capacity_time,
                     model_competitive.dispatch_model.fuel_cost_time)
              )
    
   # data.load(filename=os.path.join(inputs_directory, "PJM_generators_zone.csv"),
   #           param=(model_competitive.dispatch_model.ramp,
   #                  model_competitive.dispatch_model.rampstartuplimit,
   #                  model_competitive.dispatch_model.rampshutdownlimit,
   #                  model_competitive.dispatch_model.zonebool)
   #           )
        
              
    data.load(filename=os.path.join(inputs_directory, "timepoints_index.csv"),
              index=model_competitive.dispatch_model.TIMEPOINTS,
              param=(model_competitive.dispatch_model.reference_bus)
              )

    data.load(filename=os.path.join(inputs_directory, "zones.csv"),
              index=model_competitive.dispatch_model.ZONES,
              param=(model_competitive.dispatch_model.wind_cap,
                     model_competitive.dispatch_model.solar_cap,
                     model_competitive.dispatch_model.voltage_angle_max,
                     model_competitive.dispatch_model.voltage_angle_min)
              )

    data.load(filename=os.path.join(inputs_directory, "timepoints_zonal.csv"),
              param=(model_competitive.dispatch_model.gross_load,
                     model_competitive.dispatch_model.wind_cf,
                     model_competitive.dispatch_model.solar_cf)
              ) 
              
    data.load(filename=os.path.join(inputs_directory, "transmission_lines.csv"),
              index=model_competitive.dispatch_model.TRANSMISSION_LINE,
              param=(model_competitive.dispatch_model.susceptance)
              )
    
    data.load(filename=os.path.join(inputs_directory, "transmission_lines_hourly.csv"),
              param=(model_competitive.dispatch_model.transmission_from,
                     model_competitive.dispatch_model.transmission_to,
                     model_competitive.dispatch_model.transmission_from_capacity,
                     model_competitive.dispatch_model.transmission_to_capacity,
                     model_competitive.dispatch_model.hurdle_rate)
              )

    data.load(filename=os.path.join(inputs_directory,"generator_segments.csv"),
              index=model_competitive.dispatch_model.GENERATORSEGMENTS,
              param=(model_competitive.dispatch_model.generator_segment_length)
              )
    
    data.load(filename=os.path.join(inputs_directory,"generator_segment_marginalcost.csv"),
              param=(model_competitive.dispatch_model.generator_marginal_cost)
              )
              
    return data
