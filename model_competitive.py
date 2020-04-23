# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:49:32 2020

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
#from pyomo.bilevel import *
from pyomo.mpec import *
from pyomo.gdp import *

'''
this is the formulation of the Pyomo optimization model
'''

start_time = time.time()
cwd = os.getcwd()

dispatch_model = AbstractModel()


###########################
# ######## SETS ######### #
###########################

#time
dispatch_model.TIMEPOINTS = Set(domain=PositiveIntegers, ordered=True)

#generators
dispatch_model.GENERATORS = Set(ordered=True)

#zones
dispatch_model.ZONES = Set(doc="study zones", ordered=True)

#lines
dispatch_model.TRANSMISSION_LINE = Set(doc="tx lines", ordered=True)

#generator bid segments (creates piecewise heat rate curve)
dispatch_model.GENERATORSEGMENTS = Set(ordered=True)

#storage resources
dispatch_model.STORAGE = Set(ordered=True)

#case indexing
dispatch_model.CASE = Set(ordered=True) #this will only be needed for EPEC

###########################
# ####### PARAMS ######## #
###########################

#Params will be in lower case with underscores between words

#time and zone-dependent params
dispatch_model.gross_load = Param(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, within=NonNegativeReals)
dispatch_model.wind_cf = Param(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, within=NonNegativeReals)
dispatch_model.solar_cf = Param(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, within=NonNegativeReals)

#timepoint-dependent params
dispatch_model.reference_bus = Param(dispatch_model.TIMEPOINTS, within=dispatch_model.ZONES)
dispatch_model.reg_up_mw = Param(dispatch_model.TIMEPOINTS, within=NonNegativeReals)
dispatch_model.reg_down_mw = Param(dispatch_model.TIMEPOINTS, within=NonNegativeReals)
dispatch_model.flex_up_mw = Param(dispatch_model.TIMEPOINTS, within=NonNegativeReals)
dispatch_model.flex_down_mw = Param(dispatch_model.TIMEPOINTS, within=NonNegativeReals)


#zone-dependent params
dispatch_model.wind_cap = Param(dispatch_model.ZONES, within=NonNegativeReals)
dispatch_model.solar_cap = Param(dispatch_model.ZONES, within=NonNegativeReals)
dispatch_model.voltage_angle_max = Param(dispatch_model.ZONES, within=NonNegativeReals)
dispatch_model.voltage_angle_min = Param(dispatch_model.ZONES, within=Reals)

#dispatch_model.sub.wind_cap = Param(dispatch_model.sub.ZONES, within=NonNegativeReals)
#dispatch_model.sub.solar_cap = Param(dispatch_model.sub.ZONES, within=NonNegativeReals)

#generator-dependent params
dispatch_model.capacity = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.fuelcost = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.pmin = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.startcost = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.canspin = Param(dispatch_model.GENERATORS, within=Binary)
dispatch_model.cannonspin = Param(dispatch_model.GENERATORS, within=Binary)
dispatch_model.minup = Param(dispatch_model.GENERATORS, within=NonNegativeIntegers)
dispatch_model.mindown = Param(dispatch_model.GENERATORS, within=NonNegativeIntegers)
dispatch_model.noloadcost = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.ramp = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.tonneCO2perMWh = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.CO2price = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.CO2dollarsperMWh = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.zonelabel = Param(dispatch_model.GENERATORS, within=dispatch_model.ZONES)
dispatch_model.genco_index = Param(dispatch_model.GENERATORS, within=NonNegativeIntegers)

#storage params
dispatch_model.discharge_max = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.charge_max = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.soc_max = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.discharge_eff = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.charge_eff = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.storage_zone_label = Param(dispatch_model.STORAGE, within=dispatch_model.ZONES)
dispatch_model.storage_index = Param(dispatch_model.STORAGE, within=NonNegativeIntegers)

#generator-dependent initialization parameters
dispatch_model.commitinit = Param(dispatch_model.GENERATORS, within=Binary)
dispatch_model.upinit = Param(dispatch_model.GENERATORS, within=NonNegativeIntegers)
dispatch_model.downinit = Param(dispatch_model.GENERATORS, within=NonNegativeIntegers)

#time and zone-dependent params
dispatch_model.scheduled_available = Param(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, within=PercentFraction)
dispatch_model.capacity_time = Param(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.fuel_cost_time= Param(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, within=NonNegativeReals)


#generator and zone-dependent params ###DEFUNCT

#dispatch_model.rampstartuplimit = Param(dispatch_model.GENERATORS, dispatch_model.ZONES, within=NonNegativeReals) #special component of the ramping constraint on the startup hour
#dispatch_model.rampshutdownlimit = Param(dispatch_model.GENERATORS, dispatch_model.ZONES, within=NonNegativeReals) #special component of the ramping constraint on the shutdown hour ---- NEW

#transmission line only depedent params
dispatch_model.susceptance = Param(dispatch_model.TRANSMISSION_LINE, within=NonNegativeReals)

#time and transmission line-dependent params
dispatch_model.transmission_from = Param(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=dispatch_model.ZONES)
dispatch_model.transmission_to = Param(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=dispatch_model.ZONES)
dispatch_model.transmission_from_capacity = Param(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=Reals)
dispatch_model.transmission_to_capacity = Param(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=Reals)
dispatch_model.hurdle_rate = Param(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=NonNegativeReals)

#generator segment params
dispatch_model.base_generator_segment_length = Param(dispatch_model.GENERATORSEGMENTS, within=PercentFraction)

#generator and generator segment-dependent params
dispatch_model.generator_segment_length = Param(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.GENERATORSEGMENTS,within=PercentFraction)
dispatch_model.generator_marginal_cost = Param(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.GENERATORSEGMENTS, within=NonNegativeReals)
dispatch_model.previous_offer = Param(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.GENERATORSEGMENTS, within=NonNegativeReals)
dispatch_model.marginal_CO2 = Param(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.GENERATORSEGMENTS, within=NonNegativeReals)
dispatch_model.CO2_damage = Param(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.GENERATORSEGMENTS, within=NonNegativeReals)

#genco
dispatch_model.genco = Param(dispatch_model.CASE, within=NonNegativeIntegers)

###########################
# ###### SUBSETS ####### #
###########################

#probably want to subset generators owned by competitive genco eventually

#
#def storage_init(model):
#    storage_resources = list()
#    for g in model.GENERATORS:
#        if model.fuelcost[g]==0: #this should have different index but OK for now
#            storage_resources.append(g)
#    return storage_resources

#dispatch_model.STORAGE = Set(within=dispatch_model.GENERATORS, initialize=storage_init)

def strategic_gens_init(model):
    strategic_gens = list()
    for c in model.CASE:
        for g in model.GENERATORS:
            if model.genco_index[g] == model.genco[c]:
                strategic_gens.append(g)
    return strategic_gens
dispatch_model.STRATEGIC_GENERATORS = Set(within=dispatch_model.GENERATORS, initialize=strategic_gens_init)

def non_strategic_gens_init(model):
    non_strategic_gens = list()
    for c in model.CASE:
        for g in model.GENERATORS:
            if model.genco_index[g] != model.genco[c]:
                non_strategic_gens.append(g)
    return non_strategic_gens
dispatch_model.NON_STRATEGIC_GENERATORS = Set(within=dispatch_model.GENERATORS, initialize=non_strategic_gens_init)


def strategic_storage_init(model):
    strategic_storage = list()
    for c in model.CASE:
        for s in model.STORAGE:
            if model.storage_index[s] == model.genco[c]:
                strategic_storage.append(s)
    return strategic_storage
dispatch_model.STRATEGIC_STORAGE = Set(within=dispatch_model.STORAGE, initialize=strategic_storage_init)

def non_strategic_storage_init(model):
    non_strategic_storage = list()
    for c in model.CASE:
        for s in model.STORAGE:
            if model.storage_index[s] != model.genco[c]:
                non_strategic_storage.append(s)
    return non_strategic_storage
dispatch_model.NON_STRATEGIC_STORAGE = Set(within=dispatch_model.STORAGE, initialize=non_strategic_storage_init)


###########################
# ######## VARS ######### #
###########################

#Vars will be camel case

dispatch_model.segmentdispatch = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                     dispatch_model.GENERATORSEGMENTS,
                                     within=NonNegativeReals, initialize=0, bounds=(0,5000))

#dispatch_model.CO2_emissions = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
#                                   within=NonNegativeReals, initialize=0)

dispatch_model.windgen = Var(dispatch_model.TIMEPOINTS, dispatch_model.ZONES,
                              within = NonNegativeReals, initialize=0)

dispatch_model.solargen = Var(dispatch_model.TIMEPOINTS, dispatch_model.ZONES,
                              within = NonNegativeReals, initialize=0)

dispatch_model.curtailment = Var(dispatch_model.TIMEPOINTS,  dispatch_model.ZONES,
                                 within = NonNegativeReals, initialize=0)

dispatch_model.transmit_power_MW = Var(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE,
                                       within = Reals, initialize=0, bounds=(-5000,5000))

dispatch_model.voltage_angle = Var(dispatch_model.TIMEPOINTS, dispatch_model.ZONES,
                                       within = Reals, initialize=0)

#resource specific vars

dispatch_model.discharge = Var(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE,
                              within = NonNegativeReals, bounds=(0,5000), initialize=0)

dispatch_model.charge = Var(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE,
                              within = NonNegativeReals, bounds=(0,5000), initialize=0)

dispatch_model.soc = Var(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE,
                              within = NonNegativeReals, initialize=0)

dispatch_model.storagebool = Var(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE,
                                 within=Boolean, initialize=0)

dispatch_model.storagedispatch = Var(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE,
                                     within=Reals, bounds=(-5000,5000), initialize=0)
                            
                            
#the following vars will make problem integer when implemented
#for now they are linearized
dispatch_model.commitment = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                within=NonNegativeReals,bounds=(0,1), initialize=0)

dispatch_model.startup = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                               within=NonNegativeReals,bounds=(0,1), initialize=0)

dispatch_model.shutdown = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                               within=NonNegativeReals,bounds=(0,1), initialize=0)

#new vars for competitive version of model#
#duals of MO problem
dispatch_model.zonalprice = Var(dispatch_model.TIMEPOINTS, dispatch_model.ZONES,
                                within=Reals, initialize=0) #this is zonal load balance dual

dispatch_model.gensegmentmaxdual = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,dispatch_model.GENERATORSEGMENTS,
                                     within=NonNegativeReals, initialize=0, bounds=(0,5000))

dispatch_model.gensegmentmindual = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,dispatch_model.GENERATORSEGMENTS,
                                     within=NonNegativeReals, initialize=0, bounds=(0,5000))

dispatch_model.transmissionmaxdual = Var(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE,
                                     within=NonNegativeReals, initialize=0, bounds=(0,1000))

dispatch_model.transmissionmindual = Var(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE,
                                     within=NonNegativeReals, initialize=0, bounds=(0,1000))

dispatch_model.storagemaxdual = Var(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE,
                                     within=NonNegativeReals, initialize=0, bounds=(0,5000))

dispatch_model.storagemindual = Var(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE,
                                     within=NonNegativeReals, initialize=0, bounds=(0,5000))

dispatch_model.rampmaxdual = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                     within=NonNegativeReals, initialize=0)

dispatch_model.rampmindual = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                     within=NonNegativeReals, initialize=0)

dispatch_model.curtailmentdual = Var(dispatch_model.TIMEPOINTS, dispatch_model.ZONES,
                                     within=NonNegativeReals,initialize=0)

dispatch_model.winddual = Var(dispatch_model.TIMEPOINTS, dispatch_model.ZONES,
                              within=NonNegativeReals, initialize=0)

#offer-related variables (since generators no longer just offer at marginal cost)
dispatch_model.gensegmentoffer = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                     dispatch_model.GENERATORSEGMENTS,
                                     within=Reals)

dispatch_model.storageoffer = Var(dispatch_model.TIMEPOINTS,dispatch_model.STORAGE,
                                  within=Reals)


###########################
# ##### EXPRESSIONS ##### #
###########################
#build additional params or variables we'd like to record based on other param or variable values

def GeneratorDispatchRule(model,t,g):
    return sum(model.segmentdispatch[t,g,gs] for gs in model.GENERATORSEGMENTS)

dispatch_model.dispatch = Expression(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                                        rule=GeneratorDispatchRule)

def AvailableSegmentCapacityExpr(model,t,g,gs):
    return model.generator_segment_length[t,g,gs]*model.capacity_time[t,g]*model.scheduled_available[t,g]
dispatch_model.availablesegmentcapacity = Expression(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, 
                                                     dispatch_model.GENERATORSEGMENTS,rule=AvailableSegmentCapacityExpr)

def CO2EmittedExpr(model,t,g,gs):
    return model.segmentdispatch[t,g,gs]*model.marginal_CO2[t,g,gs]
dispatch_model.CO2_emissions = Expression(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                          dispatch_model.GENERATORSEGMENTS,rule=CO2EmittedExpr)

###########################
# ##### CONSTRAINTS ##### #
###########################

## RENEWABLES CONSTRAINTS ##

#wind output, should allow for curtailment but has $0 cost for now
def WindRule(model, t, z):
    return (model.wind_cap[z]*model.wind_cf[t,z] == model.windgen[t,z])
dispatch_model.WindMaxConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=WindRule)

#solar output, should allow for curtailment but has $0 cost for now
def SolarRule(model, t, z):
    return (model.solar_cap[z]*model.solar_cf[t,z] == model.solargen[t,z])
dispatch_model.SolarMaxConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=SolarRule)

#curtailment probably won't get used, but let's put it in for now
def CurtailmentRule(model, t, z):
    return (model.curtailment[t,z] == (model.wind_cap[z]*model.wind_cf[t,z]-model.windgen[t,z]) + (model.solar_cap[z]*model.solar_cf[t,z]-model.solargen[t,z]))
#dispatch_model.CurtailmentConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=CurtailmentRule)

## STORAGE CONSTRAINTS ##

def StorageDischargeRule(model,t,s):
    return  model.discharge_max[s]*model.storagebool[t,s] >= model.discharge[t,s]
    #return model.discharge_max[s]*model.storagebool[t,s] >= model.storagedispatch[t,s]
dispatch_model.StorageDischargeConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE,rule=StorageDischargeRule)

def StorageChargeRule(model,t,s):
    return model.charge_max[s]*(1-model.storagebool[t,s]) >= model.charge[t,s]
    #return model.storagedispatch[t,s] >= -model.charge_max[s]*(1-model.storagebool[t,s])
dispatch_model.StorageChargeConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, rule=StorageChargeRule)

def StorageDispatchRule(model,t,s):
    return model.storagedispatch[t,s] == model.discharge[t,s]-model.charge[t,s]
dispatch_model.StorageDispatchConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, rule=StorageDispatchRule)

def SOCChangeRule(model,t,s):
    if t==1:
        return model.soc[t,s] == model.charge[t,s]*model.charge_eff[s] - model.discharge[t,s]*model.discharge_eff[s] #start half charged?
        #return model.soc[t,s] == -model.storagedispatch[t,s]
    else:
        return model.soc[t,s] == model.soc[t-1,s] + model.charge[t,s]*model.charge_eff[s] - model.discharge[t,s]*model.discharge_eff[s]
        #return model.soc[t,s] == model.soc[t-1,s] - model.storagedispatch[t,s]
dispatch_model.SOCChangeConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, rule=SOCChangeRule)

def SOCMaxRule(model,t,s):
    return model.soc_max[s] >= model.soc[t,s]
dispatch_model.SOCMaxConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, rule=SOCMaxRule)
    
def BindFinalSOCRule(model,s):
    return model.soc_max[s]*0 == model.soc[model.TIMEPOINTS[-1],s]
dispatch_model.BindFinalSOCConstraint = Constraint(dispatch_model.STORAGE, rule=BindFinalSOCRule)

## TRANSMISSION LINES ##

#flow rules, implemented as DCOPF but could add additional rules about hourly ramp

def TxFromRule(model, t, line):
    return (model.transmit_power_MW[t,line] >= model.transmission_from_capacity[t, line])
dispatch_model.TxFromConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, 
                                             rule=TxFromRule, name='TxFrom')

def TxToRule(model, t, line):
    return (model.transmission_to_capacity[t, line] >= model.transmit_power_MW[t,line])
dispatch_model.TxToConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, 
                                           rule=TxToRule, name='TxTo')

#dcopf rules, first, bound voltage angle above and below

def VoltageAngleMaxRule(model,t,z):
    return model.voltage_angle_max[z] >= model.voltage_angle[t,z]
dispatch_model.VoltageAngleMaxConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES,
                                                      rule=VoltageAngleMaxRule, name='VoltageAngleMax')

def VoltageAngleMinRule(model,t,z):
    return model.voltage_angle[t,z] >= model.voltage_angle_min[z]
dispatch_model.VoltageAngleMinConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, 
                                                      rule=VoltageAngleMinRule, name='VoltageAngleMin')

#then set the reference bus
def SetReferenceBusRule(model,t,z):
    if z==model.reference_bus[t]:
        return model.voltage_angle[t,z]==0
    else:
        return Constraint.Skip
dispatch_model.SetReferenceBusConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, 
                                                      rule=SetReferenceBusRule, name='RefBus')

#then, bind transmission flows between lines based on voltage angle

def DCOPFRule(model,t,line):
    zone_to = model.transmission_to[t,line]
    zone_from = model.transmission_from[t,line]
    return model.transmit_power_MW[t,line] == model.susceptance[line]*(model.voltage_angle[t,zone_to]-model.voltage_angle[t,zone_from])

dispatch_model.DCOPFConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, 
                                            rule=DCOPFRule, name='DCOPF')

## LOAD BALANCE ##

#load/gen balance
def LoadRule(model, t, z):
    #implement total tx flow
    imports_exports = 0
    zonal_generation = 0
    zonal_storage = 0
    for line in model.TRANSMISSION_LINE:
        if model.transmission_to[t, line] == z or model.transmission_from[t, line] == z:
            if model.transmission_to[t, line] == z:
                imports_exports += model.transmit_power_MW[t, line]
            elif model.transmission_from[t, line] == z:
                imports_exports -= model.transmit_power_MW[t, line]
            #add additional note to dec import/exports by line losses
            #no, this will just be done as a hurdle rate 
    for g in model.GENERATORS:
        if model.zonelabel[g] == z:
            zonal_generation += sum(model.segmentdispatch[t,g,gs] for gs in model.GENERATORSEGMENTS)
    for s in model.STORAGE:
        if model.storage_zone_label[s] == z:
            #zonal_storage += model.discharge[t,s]
            #zonal_storage -= model.charge[t,s]
            zonal_storage += model.storagedispatch[t,s]
    #full constraint, with tx flow now
    #(sum(sum(model.segmentdispatch[t,g,z,gs] for gs in model.GENERATORSEGMENTS) for g in model.GENERATORS)+\
    return zonal_generation + imports_exports + zonal_storage == model.gross_load[t,z]
dispatch_model.LoadConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=LoadRule)

## CONVENTIONAL GENERATORS CONSTRAINTS ##

#gen capacity with scheduled outage factored in
def CapacityMaxRule(model, t, g, z):
    return (model.capacity[g]*model.commitment[t,g]*model.scheduled_available[t,g] >= model.dispatch[t,g,z])
#dispatch_model.CapacityMaxConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.ZONES, rule=CapacityMaxRule)

#pmin: INACTIVE
def PminRule(model,t,g,z):
    return (model.dispatch[t,g,z] >= model.capacity[g]*model.commitment[t,g]*model.scheduled_available[t,g]*model.pmin[g])
#dispatch_model.PminConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.ZONES, rule=PminRule)

### GENERATOR BID SEGMENT DISPATCH ###

#max on segment
def GeneratorSegmentDispatchMax(model, t, g, gs):
    #if model.availablesegmentcapacity[t,g,gs].expr!=0:
    #    print(t,g,gs)
    #    print(model.availablesegmentcapacity[t,g,gs].expr)
    return model.availablesegmentcapacity[t,g,gs] >= model.segmentdispatch[t,g,gs]
    #if z==
    #try:
    #    return ((model.zonebool[g,z]*model.generator_segment_length[gs]*model.capacity[g]*model.scheduled_available[t,g]) >= model.segmentdispatch[t,g,z,gs])
    #except ValueError:
    #    return 0==model.segmentdispatch[t,g,z,gs] #non-existent must be zero
dispatch_model.GeneratorSegmentMaxConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.GENERATORSEGMENTS,
                                                          rule=GeneratorSegmentDispatchMax, name='SegMax')

### GENERATOR RAMP ###

def GeneratorRampUpRule(model,t,g):
    if t==1:
        return Constraint.Skip
    else:
        return (model.dispatch[t-1,g] + model.ramp[g]*model.commitment[t-1,g] >= model.dispatch[t,g])
#dispatch_model.GeneratorRampUpConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, rule=GeneratorRampUpRule)

def GeneratorRampDownRule(model,t,g):
    if t==1:
        return Constraint.Skip 
    else:
        return (model.dispatch[t,g] >= model.dispatch[t-1,g] - model.ramp[g])
#dispatch_model.GeneratorRampDownConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, rule=GeneratorRampDownRule)



### DUAL CONSTRAINTS ###
def BindGeneratorOfferDual(model,t,g,gs):
    return model.gensegmentoffer[t,g,gs]+model.gensegmentmaxdual[t,g,gs]-model.gensegmentmindual[t,g,gs]-model.zonalprice[t,model.zonelabel[g]]==0
dispatch_model.GeneratorOfferDualConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                                dispatch_model.GENERATORSEGMENTS,rule=BindGeneratorOfferDual,name='OfferDual')

def BindStorageOfferDual(model,t,s):
    return model.storageoffer[t,s]+model.storagemaxdual[t,s]-model.storagemindual[t,s]-model.zonalprice[t,model.storage_zone_label[s]]==0
dispatch_model.StorageOfferDualConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE,
                                                       rule=BindStorageOfferDual, name='StorageOfferDual')

def BindRampOfferDual(model,t,g,gs):
    return model.gensegmentoffer[t,g,gs]+model.rampmaxdual[t,g]-model.rampmindual[t,g]-model.zonalprice[t,model.zonelabel[g]]==0
#dispatch_model.RampOfferDualConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
#                                                    rule=BindRampOfferDual)

def BindFlowDual(model,t,z):
    maxdual = 0
    mindual = 0
    lmp_delta = 0
    for line in model.TRANSMISSION_LINE:
        
        if model.transmission_to[t, line] == z:
            sink_zone = model.transmission_from[t,line]
            #if t==1:
            #    print(sink_zone)
            maxdual += model.susceptance[line]*model.transmissionmaxdual[t,line]
            mindual += model.susceptance[line]*model.transmissionmindual[t,line]
            lmp_delta += model.susceptance[line]*model.zonalprice[t,z]
            lmp_delta -= model.susceptance[line]*model.zonalprice[t,sink_zone]
            
        elif model.transmission_from[t, line] == z:
            sink_zone = model.transmission_to[t,line]
            #if t==1:
            #    print(sink_zone)
            maxdual -= model.susceptance[line]*model.transmissionmaxdual[t,line]
            mindual -= model.susceptance[line]*model.transmissionmindual[t,line]
            lmp_delta += model.susceptance[line]*model.zonalprice[t,z]
            lmp_delta -= model.susceptance[line]*model.zonalprice[t,sink_zone]
    #if t==8 and z==1:
    #    print(maxdual,mindual,lmp_delta)
    return maxdual-mindual==lmp_delta
dispatch_model.FlowDualConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES,
                                               rule=BindFlowDual,name='FlowDual')

### COMPLEMENTARITY CONSTRAINTS ###

def BindMaxDispatchComplementarity(model,t,g,gs):
    return complements(model.availablesegmentcapacity[t,g,gs]-model.segmentdispatch[t,g,gs]>=0, model.gensegmentmaxdual[t,g,gs]>=0)

dispatch_model.MaxDispatchComplementarity = Complementarity(dispatch_model.TIMEPOINTS,dispatch_model.GENERATORS,dispatch_model.GENERATORSEGMENTS,
                                                            rule=BindMaxDispatchComplementarity, name='MaxDispatchComp')

def BindMinDispatchComplementarity(model,t,g,gs):
    return complements(model.segmentdispatch[t,g,gs]>=0, model.gensegmentmindual[t,g,gs]>=0)
dispatch_model.MinDispatchComplementarity = Complementarity(dispatch_model.TIMEPOINTS,dispatch_model.GENERATORS, dispatch_model.GENERATORSEGMENTS,
                                                            rule=BindMinDispatchComplementarity, name='MinDispatchComp')

def BindMaxTransmissionComplementarity(model,t,line):
    return complements(model.transmission_to_capacity[t,line]-model.transmit_power_MW[t,line]>=0,model.transmissionmaxdual[t,line]>=0)
dispatch_model.MaxTransmissionComplementarity = Complementarity(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE,
                                                                rule=BindMaxTransmissionComplementarity, name='MaxTxComp')

def BindMinTransmissionComplementarity(model,t,line):
    return complements(-model.transmission_from_capacity[t,line]+model.transmit_power_MW[t,line]>=0,model.transmissionmindual[t,line]>=0)
dispatch_model.MinTransmissionComplementarity = Complementarity(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE,
                                                                rule=BindMinTransmissionComplementarity, name='MinTxComp')

def BindMaxStorageComplementarity(model,t,s):
    return complements(model.discharge_max[s]*model.storagebool[t,s]-model.storagedispatch[t,s]>=0, model.storagemaxdual[t,s]>=0)
dispatch_model.MaxStorageComplementarity = Complementarity(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE,
                                                           rule=BindMaxStorageComplementarity, name='MaxStorageComp')

def BindMinStorageComplementarity(model,t,s):
    return complements(model.charge_max[s]*(1-model.storagebool[t,s])+model.storagedispatch[t,s]>=0, model.storagemindual[t,s]>=0)
dispatch_model.MinStorageComplementarity = Complementarity(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE,
                                                           rule=BindMinStorageComplementarity, name='MinStorageComp')

def BindMaxRampComplementarity(model,t,g):
    if t==1:
        return Complementarity.Skip
    else:
        return complements(model.ramp[t,g]-(model.dispatch[t,g]-model.dispatch[t-1,g])>=0, model.rampmaxdual[t,g]>=0)
#dispatch_model.MaxRampComplementarity = Complementarity(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
#                                                        rule=BindMaxRampComplementarity)

def BindMinRampComplementarity(model,t,g):
    if t==1:
        return Complementarity.Skip
    else:
        return complements()
#dispatch_model.MinRampComplementarity = Complementarity(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
#                                                        rule=BindMinRampComplementarity)

## SEGMENT DISPATCH INCREASING ##
#forces dispatch on first segment first, etc.
'''
def IncreasingDispatch(model,t,g,gs):
    if gs==0:
        return Constraint.Skip
    elif 0.001>=model.availablesegmentcapacity[t,g,0]-model.segmentdispatch[t,g,0]:
        return Constraint.Skip #generator committed
    else:
        print(t,g,gs)
        return 0.001>=model.segmentdispatch[t,g,gs]  
    #else:
    #    return model.segmentdispatch[t,g,gs-1]>=model.segmentdispatch[t,g,gs]
dispatch_model.IncreasingDispatchConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                                         dispatch_model.GENERATORSEGMENTS, rule=IncreasingDispatch)
'''
## OFFER CURVE INCREASING ## 
#it's just generally a market rule that offers of a single generator be increasing, due to heat rate curves and necessity of convexity

def IncreasingOfferCurve(model,t,g,gs):
    if gs==0:
        return Constraint.Skip #offer whatever you want on the first segment of your offer curve
    else:
        return model.gensegmentoffer[t,g,gs]>=model.gensegmentoffer[t,g,gs-1]
dispatch_model.IncreasingOfferCurveConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                                           dispatch_model.GENERATORSEGMENTS, rule=IncreasingOfferCurve,name='IncOfferCurve')

### MARKET-BASED CONSTRAINTS ###
#only needed for now because genco owns all generators and could easily increase price to cap
def OfferCap(model,t,g,gs):
    return model.previous_offer[t,g,gs]*2>=model.gensegmentoffer[t,g,gs] #caps offer at 2x cost for now
dispatch_model.OfferCapConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.STRATEGIC_GENERATORS,dispatch_model.GENERATORSEGMENTS,
                                                       rule=OfferCap, name='OfferCap')

def OfferCap2(model,t,g,gs):
    return model.previous_offer[t,g,gs]>=model.gensegmentoffer[t,g,gs] #caps offer at cost to make it param
dispatch_model.OfferCap2Constraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.NON_STRATEGIC_GENERATORS,dispatch_model.GENERATORSEGMENTS,
                                                       rule=OfferCap2, name='OfferCap2')

def OfferMin(model,t,g,gs):
    return model.gensegmentoffer[t,g,gs]>=model.generator_marginal_cost[t,g,gs] #must offer at least marginal cost
dispatch_model.OfferMinConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.GENERATORSEGMENTS,
                                                       rule=OfferMin,name='OfferMin')

def MarketPriceCap(model,t,z):
    return 2000>=model.zonalprice[t,z]
dispatch_model.ZonalPriceConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, 
                                                 rule=MarketPriceCap, name='MktPriceCap')


## GENERATOR STARTUP/SHUTDOWN ##
'''
#startups
def StartUpRule(model,t,g):
    if t==1:
        return 1-model.commitinit[g] >= model.startup[t,g]
    else:
        return (1-model.commitment[t-1,g] >= model.startup[t,g])
dispatch_model.StartUpConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, rule=StartUpRule)

#shutdowns
def ShutDownRule(model,t,g):
    if t==1:
        return model.commitinit[g] >= model.shutdown[t,g]
    else:
        return (model.commitment[t-1,g] >= model.shutdown[t,g])
dispatch_model.ShutDownConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, rule=ShutDownRule)

#assign shuts and starts
def AssignStartShutRule(model,t,g):
    if t==1: #binds commitment in first hour based on initialization commitment from input (could be last timeperiod of previous run)
        return model.commitment[t,g] - model.commitinit[g] == model.startup[t,g] - model.shutdown[t,g]
    else: #general rule
        return (model.commitment[t,g] - model.commitment[t-1,g] == model.startup[t,g] - model.shutdown[t,g])
dispatch_model.AssignStartShutConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, rule=AssignStartShutRule)

#force de-comitted generator if unit unavailable due to scheduled outage
def scheduled_availableRule(model,t,g):
    if model.scheduled_available[t,g]==0:
        return (model.scheduled_available[t,g] == model.commitment[t,g])
    else:
        return Constraint.Skip
dispatch_model.scheduled_availableConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, rule=scheduled_availableRule)

## GENERATOR MIN UP AND DOWN ##

#min uptime constraint
def MinUpRule(model,t,g):
    recent_start_bool = float() #initialize our tracker; boolean because you'll just never see multiple starts
    
    #allow minup to be overruled if generator is scheduled not available
    if model.scheduled_available[t,g] == 0:
        return Constraint.Skip

    if t - model.minup[g] <1: #i.e. in the lookback period to the initialization condition
        for tp in range(1,t+1):
            if model.scheduled_available[tp,g]==0: #if the generator previously could have scheduled shutdown, nullify this constraint
                return Constraint.Skip
        if model.minup[g] >= t+model.upinit[g] and model.commitinit[g]==1: #if generator started online, and hasn't been up long enough
            return model.commitment[t,g] >= model.commitinit[g]
        else:
            return Constraint.Skip

    else: #define subperiod
        for tp in range(1, model.minup[g]+1): #b/c exclusive upper bound!
            recent_start_bool += model.startup[t-tp,g]
        return model.commitment[t,g] >= recent_start_bool
dispatch_model.MinUpConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, rule=MinUpRule)

#min downtime constraint
def MinDownRule(model,t,g):
    recent_shut_bool = float()

    if t - model.mindown[g] <1: 
        if model.mindown[g] >= t+model.downinit[g] and model.commitinit[g]==0:
            return model.commitinit[g] >= model.commitment[t,g]
        else:
            return Constraint.Skip
    else:
        for tp in range(1, model.mindown[g]+1):
            recent_shut_bool += model.shutdown[t-tp,g]
        return (1-recent_shut_bool) >= model.commitment[t,g]
dispatch_model.MinDownConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, rule=MinDownRule)
'''


###########################
# ###### OBJECTIVE ###### #
###########################

def objective_rule(model): 
    #min dispatch cost for objective
    return sum(sum(sum(model.segmentdispatch[t,g,gs]*model.generator_marginal_cost[t,g,gs] for t in model.TIMEPOINTS) for g in model.GENERATORS) for gs in model.GENERATORSEGMENTS)+\
           sum(sum(model.commitment[t,g] for t in model.TIMEPOINTS)*model.noloadcost[g] for g in model.GENERATORS)+\
           sum(sum(model.startup[t,g] for t in model.TIMEPOINTS)*model.startcost[g] for g in model.GENERATORS)

    #DESCRIPTION OF OBJECTIVE
    #(1) dispatch cost
    #(2) no load cost of committed gen
    #(3) start up costs when generators brought online
dispatch_model.TotalCost = Objective(rule=objective_rule, sense=minimize)

def objective_rule2(model): 
    #min dispatch cost for objective
    return sum(sum(sum(model.segmentdispatch[t,g,gs]*model.generator_marginal_cost[t,g,gs] for t in model.TIMEPOINTS) for g in model.GENERATORS) for gs in model.GENERATORSEGMENTS)+\
           sum(sum(sum(model.segmentdispatch[t,g,gs]*model.marginal_CO2[t,g,gs]*model.CO2_damage[t,g,gs] for t in model.TIMEPOINTS) for g in model.GENERATORS) for gs in model.GENERATORSEGMENTS)
    #DESCRIPTION OF OBJECTIVE
    #(1) dispatch cost
dispatch_model.TotalCost2 = Objective(rule=objective_rule2, sense=minimize)

def objective_profit(model):
    return sum(sum(model.gross_load[t,z]*model.zonalprice[t,z] for z in model.ZONES) for t in model.TIMEPOINTS)-\
           sum(sum(sum(model.segmentdispatch[t,g,gs]*model.generator_marginal_cost[t,g,gs] for t in model.TIMEPOINTS) for g in model.GENERATORS) for gs in model.GENERATORSEGMENTS)
dispatch_model.GeneratorProfit = Objective(rule=objective_profit,sense=maximize)

def objective_profit_dual(model):
    #return sum(sum(sum(model.availablesegmentcapacity[t,g,gs]*model.gensegmentmaxdual[t,g,gs] for t in model.TIMEPOINTS) for g in model.STRATEGIC_GENERATORS) for gs in model.GENERATORSEGMENTS)-\
    return sum(sum(sum(-model.availablesegmentcapacity[t,g,gs]*model.gensegmentmaxdual[t,g,gs] for t in model.TIMEPOINTS) for g in model.NON_STRATEGIC_GENERATORS) for gs in model.GENERATORSEGMENTS)-\
           sum(sum(model.discharge_max[s]*model.storagemaxdual[t,s] for t in model.TIMEPOINTS) for s in model.NON_STRATEGIC_STORAGE)-\
           sum(sum(model.charge_max[s]*model.storagemindual[t,s] for t in model.TIMEPOINTS) for s in model.NON_STRATEGIC_STORAGE)-\
           sum(sum(sum(model.segmentdispatch[t,g,gs] * model.generator_marginal_cost[t,g,gs] for t in model.TIMEPOINTS) for g in model.GENERATORS) for gs in model.GENERATORSEGMENTS)-\
           sum(sum(model.transmission_to_capacity[t,line]*model.transmissionmaxdual[t,line] for t in model.TIMEPOINTS) for line in model.TRANSMISSION_LINE)+\
           sum(sum(model.transmission_from_capacity[t,line]*model.transmissionmindual[t,line] for t in model.TIMEPOINTS) for line in model.TRANSMISSION_LINE)+\
           sum(sum(model.gross_load[t,z]*model.zonalprice[t,z] for t in model.TIMEPOINTS) for z in model.ZONES)

#sum(sum(model.discharge_max[s]*model.storagebool[t,s]*model.storagemaxdual[t,s] for t in model.TIMEPOINTS) for s in model.NON_STRATEGIC_STORAGE)-\
#sum(sum(model.charge_max[s]*(1-model.storagebool[t,s])*model.storagemindual[t,s] for t in model.TIMEPOINTS) for s in model.NON_STRATEGIC_STORAGE)-\

#sum(sum(sum((model.generator_segment_length[gs]*model.capacity[g]*model.scheduled_available[t,g])*model.gensegmentmindual[t,g,gs] for t in model.TIMEPOINTS) for g in model.GENERATORS) for gs in model.GENERATORSEGMENTS)-\
#sum(sum(sum(model.totalsegmentdispatch[t,g,gs]*model.gensegmentmindual[t,g,gs] for t in model.TIMEPOINTS) for g in model.GENERATORS) for gs in model.GENERATORSEGMENTS)
#sum(sum(sum(sum(model.gensegmentoffer[t,g,gs]*model.segmentdispatch[t,g,z,gs] for z in model.ZONES) for t in model.TIMEPOINTS)*model.generator_marginal_cost[g,gs] for g in model.GENERATORS) for gs in model.GENERATORSEGMENTS)
dispatch_model.GeneratorProfitDual = Objective(rule=objective_profit_dual,sense=maximize)