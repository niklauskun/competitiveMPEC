# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:55:18 2020

@author: Luke
"""


#general imports
from __future__ import division
import os
import glob
from os.path import join
import pandas as pd
import numpy as np
import math
import time
import sys
import datetime
from pyutilib.services import TempfileManager
from pyomo.environ import *
from pyomo.bilevel import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

#other scripts
#from main_competitive import case_folder
case_folder = "Oct_19_25_2017_TEST"

def diag_plots(scenario_results, dir_str):
    #PLOTS ONLY
    #plot some basic results with matplotlib
    scenario_results_np = np.reshape(scenario_results[0], (int(scenario_results[1]), int(len(scenario_results[0])/scenario_results[1])))
    start_results_np = np.reshape(scenario_results[5], (int(scenario_results[1]), int(len(scenario_results[5])/scenario_results[1])))
    shut_results_np = np.reshape(scenario_results[6], (int(scenario_results[1]), int(len(scenario_results[6])/scenario_results[1])))
  
    wind_results_np = np.reshape(scenario_results[2], (int(scenario_results[1]), int(len(scenario_results[2])/scenario_results[1])))
    solar_results_np = np.reshape(scenario_results[3], (int(scenario_results[1]), int(len(scenario_results[3])/scenario_results[1])))
    curtailment_results_np = np.reshape(scenario_results[4], (int(scenario_results[1]), int(len(scenario_results[4])/scenario_results[1])))
    lmp_duals_np = np.reshape(scenario_results[7], (int(scenario_results[1]), int(len(scenario_results[7])/scenario_results[1])))
    
    #read in the gen and zone types so aggregation can be done for plots
    gens = pd.read_csv(join(dir_str.INPUTS_DIRECTORY, 'PJM_generators_full.csv'))
    zones = pd.read_csv(join(dir_str.INPUTS_DIRECTORY, 'zones.csv'))
    #full_tx_lines = pd.read_csv(join(dir_str.INPUTS_DIRECTORY, 'transmission_lines_hourly.csv'))
    
    gens_list = []
    zones_list = []
    y = []
    start = []
    shut = []
    wind_power = []
    solar_power = []
    curtail_power = []
            
    
    
    for g in gens['Category'].unique():
        #print(g) #placeholder for now
        gen_type = (gens['Category']==g)
        
        start.append(np.dot(start_results_np,np.array(gen_type)))
        shut.append(np.dot(shut_results_np,np.array(gen_type)))
    
    for z in range(len(zones['zone'])):
        wind_power.append(wind_results_np[:,z])
        solar_power.append(solar_results_np[:,z])
        curtail_power.append(curtailment_results_np[:,z])
        for g in gens['Category'].unique():
            gen_type = (gens['Category']==g)
            y.append(np.dot(scenario_results_np[:,z*len(gen_type):(z+1)*len(gen_type)],np.array(gen_type)))

    # Your x and y axis
    x=range(1,int(scenario_results[1])+1)
    #y is made above
    
    # Basic stacked area chart by zone
    for z in range(len(zones['zone'])):
        
        adder = len(gens['Category'].unique())*z
        
        plt.plot([],[],color='b', label='Hydro', linewidth=5)
        plt.plot([],[],color='m', label='Nuclear', linewidth=5)
        plt.plot([],[],color='k', label='Large Coal', linewidth=5)
        plt.plot([],[],color='slategray', label='Small Coal', linewidth=5)
        plt.plot([],[],color='orange', label='Gas CC', linewidth=5)
        plt.plot([],[],color='sienna', label='Gas CT', linewidth=5)
        plt.plot([],[],color='g', label='Oil', linewidth=5)
        plt.plot([],[],color='silver', label='Demand Response', linewidth=5)
        plt.plot([],[],color='cyan', label='Wind', linewidth=5)
        plt.plot([],[],color='yellow', label='Solar', linewidth=5)
        plt.plot([],[],color='red', label='Curtailment', linewidth=5)
        
        if case_folder == "TOYCASE":
            plt.stackplot(x,y[adder+5],y[adder+6],y[adder+2],y[adder+4],y[adder+0],y[adder+1],y[adder+3],y[adder+7],
                          wind_power[z],solar_power[z],curtail_power[z],
                          colors=['b','m','k','slategray','orange','sienna','g','silver','cyan','yellow','red'])
        else:
            plt.stackplot(x,y[adder+4],y[adder+6],y[adder+2],y[adder+5],y[adder+0],y[adder+1],y[adder+3],y[adder+7],
                          wind_power[z],solar_power[z],curtail_power[z],
                          colors=['b','m','k','slategray','orange','sienna','g','silver','cyan','yellow','red'])
        
        plt.title('Zone ' + zones['zone'][z] + ' Generator Dispatch')
        plt.ylabel('Load (MW)')
        plt.xlabel('Hour')
        plt.legend(loc=4)
        plt.show()
    
    #do also for starts
    plt.plot([],[],color='b', label='Hydro', linewidth=5)
    plt.plot([],[],color='m', label='Nuclear', linewidth=5)
    plt.plot([],[],color='k', label='Large Coal', linewidth=5)
    plt.plot([],[],color='slategray', label='Small Coal', linewidth=5)
    plt.plot([],[],color='orange', label='Gas CC', linewidth=5)
    plt.plot([],[],color='sienna', label='Gas CT', linewidth=5)
    plt.plot([],[],color='g', label='Oil', linewidth=5)
    
    if case_folder == "TOYCASE":
        plt.stackplot(x,start[5],start[6],start[2],start[4],start[0],start[1],start[3],
                  colors=['b','m','k','slategray','orange','sienna','g'])
    else:
        plt.stackplot(x,start[4],start[6],start[2],start[5],start[0],start[1],start[3],
                  colors=['b','m','k','slategray','orange','sienna','g'])
    plt.ylabel('StartUps (# Plants)')
    plt.xlabel('Hour')
    plt.legend()
    plt.show()
    
    #and shuts
    plt.plot([],[],color='b', label='Hydro', linewidth=5)
    plt.plot([],[],color='m', label='Nuclear', linewidth=5)
    plt.plot([],[],color='k', label='Large Coal', linewidth=5)
    plt.plot([],[],color='slategray', label='Small Coal', linewidth=5)
    plt.plot([],[],color='orange', label='Gas CC', linewidth=5)
    plt.plot([],[],color='sienna', label='Gas CT', linewidth=5)
    plt.plot([],[],color='g', label='Oil', linewidth=5)
    
    if case_folder == "TOYCASE":
        plt.stackplot(x,shut[5],shut[6],shut[2],shut[4],shut[0],shut[1],shut[3],
                  colors=['b','m','k','slategray','orange','sienna','g'])
    else:
        plt.stackplot(x,shut[4],shut[6],shut[2],shut[5],shut[0],shut[1],shut[3],
                  colors=['b','m','k','slategray','orange','sienna','g'])
    plt.ylabel('Shutdowns (# Plants)')
    plt.xlabel('Hour')
    plt.legend()
    plt.show()


    #and finally, plot the energy LMP dual
    lmp_palette = ['r']
    legend_label = []
    for z in range(len(zones['zone'])):
        plt.plot(x, lmp_duals_np[:,z], color=lmp_palette[z])
        legend_label.append('Zone ' + zones['zone'][z])
    plt.ylabel('Energy Price ($/MWh)')
    plt.xlabel('Hour')
    plt.legend(legend_label, loc='upper left')
    plt.show()
    
