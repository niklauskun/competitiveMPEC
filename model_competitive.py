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

###########################
# ####### PARAMS ######## #
###########################

#Params will be in CamelCase

#time and zone-dependent params
dispatch_model.GrossLoad = Param(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, within=NonNegativeReals)
dispatch_model.windcf = Param(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, within=NonNegativeReals)
dispatch_model.solarcf = Param(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, within=NonNegativeReals)

#timepoint-dependent params
dispatch_model.reference_bus = Param(dispatch_model.TIMEPOINTS, within=dispatch_model.ZONES)

#zone-dependent params
dispatch_model.windcap = Param(dispatch_model.ZONES, within=NonNegativeReals)
dispatch_model.solarcap = Param(dispatch_model.ZONES, within=NonNegativeReals)
dispatch_model.voltage_angle_max = Param(dispatch_model.ZONES, within=NonNegativeReals)
dispatch_model.voltage_angle_min = Param(dispatch_model.ZONES, within=Reals)

#dispatch_model.sub.windcap = Param(dispatch_model.sub.ZONES, within=NonNegativeReals)
#dispatch_model.sub.solarcap = Param(dispatch_model.sub.ZONES, within=NonNegativeReals)

#generator-dependent params
dispatch_model.fuelcost = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.pmin = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.startcost = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.canspin = Param(dispatch_model.GENERATORS, within=Binary)
dispatch_model.cannonspin = Param(dispatch_model.GENERATORS, within=Binary)
dispatch_model.minup = Param(dispatch_model.GENERATORS, within=NonNegativeIntegers)
dispatch_model.mindown = Param(dispatch_model.GENERATORS, within=NonNegativeIntegers)
dispatch_model.noloadcost = Param(dispatch_model.GENERATORS, within=NonNegativeReals)

#generator-dependent initialization parameters
dispatch_model.commitinit = Param(dispatch_model.GENERATORS, within=Binary)
dispatch_model.upinit = Param(dispatch_model.GENERATORS, within=NonNegativeIntegers)
dispatch_model.downinit = Param(dispatch_model.GENERATORS, within=NonNegativeIntegers)

#time and zone-dependent params
dispatch_model.scheduledavailable = Param(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, within=PercentFraction)

#generator and zone-dependent params
dispatch_model.capacity = Param(dispatch_model.GENERATORS, dispatch_model.ZONES, within=NonNegativeReals)
dispatch_model.ramp = Param(dispatch_model.GENERATORS, dispatch_model.ZONES, within=NonNegativeReals) #rate is assumed to be equal up and down
dispatch_model.rampstartuplimit = Param(dispatch_model.GENERATORS, dispatch_model.ZONES, within=NonNegativeReals) #special component of the ramping constraint on the startup hour
dispatch_model.rampshutdownlimit = Param(dispatch_model.GENERATORS, dispatch_model.ZONES, within=NonNegativeReals) #special component of the ramping constraint on the shutdown hour ---- NEW

#transmission line only depedent params
dispatch_model.susceptance = Param(dispatch_model.TRANSMISSION_LINE, within=NonNegativeReals)

#time and transmission line-dependent params
dispatch_model.transmission_from = Param(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=dispatch_model.ZONES)
dispatch_model.transmission_to = Param(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=dispatch_model.ZONES)
dispatch_model.transmission_from_capacity = Param(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=Reals)
dispatch_model.transmission_to_capacity = Param(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=Reals)
dispatch_model.hurdle_rate = Param(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=NonNegativeReals)

#generator segment params
dispatch_model.generatorsegmentlength = Param(dispatch_model.GENERATORSEGMENTS, within=PercentFraction)

#generator and generator segment-dependent params
dispatch_model.generatormarginalcost = Param(dispatch_model.GENERATORS, dispatch_model.GENERATORSEGMENTS, within=NonNegativeReals)

###########################
# ###### SUBSETS ####### #
###########################

#probably want to subset generators owned by competitive genco eventually

#subsets hydro resources so can determine different operational characteristics for them
def storage_init(model):
    storage_resources = list()
    for g in model.GENERATORS:
        if model.fuelcost[g]==0: #this should have different index but OK for now
            storage_resources.append(g)
    return storage_resources

dispatch_model.STORAGE = Set(within=dispatch_model.GENERATORS, initialize=storage_init)


###########################
# ######## VARS ######### #
###########################

#Vars will be lower case

dispatch_model.segmentdispatch = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                     dispatch_model.ZONES, dispatch_model.GENERATORSEGMENTS,
                                     within=NonNegativeReals, initialize=0, bounds=(0,5000))

dispatch_model.windgen = Var(dispatch_model.TIMEPOINTS, dispatch_model.ZONES,
                              within = NonNegativeReals, initialize=0)

dispatch_model.solargen = Var(dispatch_model.TIMEPOINTS, dispatch_model.ZONES,
                              within = NonNegativeReals, initialize=0)

dispatch_model.curtailment = Var(dispatch_model.TIMEPOINTS,  dispatch_model.ZONES,
                                 within = NonNegativeReals, initialize=0)

dispatch_model.transmit_power_MW = Var(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE,
                                       within = Reals, initialize=0)

dispatch_model.voltage_angle = Var(dispatch_model.TIMEPOINTS, dispatch_model.ZONES,
                                       within = Reals, initialize=0)

#resource specific vars

dispatch_model.discharge = Var(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, dispatch_model.ZONES,
                              within = NonNegativeReals, initialize=0)

dispatch_model.charge = Var(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, dispatch_model.ZONES,
                              within = NonNegativeReals, initialize=0)

dispatch_model.soc = Var(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, dispatch_model.ZONES,
                              within = NonNegativeReals, initialize=0)
                            
                            

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
                                within=NonNegativeReals, initialize=0) #this is zonal load balance dual

dispatch_model.gensegmentmaxdual = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                     dispatch_model.ZONES, dispatch_model.GENERATORSEGMENTS,
                                     within=NonNegativeReals, initialize=0, bounds=(0,5000))

dispatch_model.gensegmentmindual = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                     dispatch_model.ZONES, dispatch_model.GENERATORSEGMENTS,
                                     within=NonNegativeReals, initialize=0, bounds=(0,5000))

dispatch_model.rampmaxdual = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                     dispatch_model.ZONES,
                                     within=NonNegativeReals, initialize=0)

dispatch_model.rampmindual = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                     dispatch_model.ZONES,
                                     within=NonNegativeReals, initialize=0)

dispatch_model.curtailmentdual = Var(dispatch_model.TIMEPOINTS, dispatch_model.ZONES,
                                     within=NonNegativeReals,initialize=0)

dispatch_model.winddual = Var(dispatch_model.TIMEPOINTS, dispatch_model.ZONES,
                              within=NonNegativeReals, initialize=0)

#offer-related variables (since generators no longer just offer at marginal cost)
dispatch_model.gensegmentoffer = Var(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                     dispatch_model.ZONES, dispatch_model.GENERATORSEGMENTS,
                                     within=NonNegativeReals)


###########################
# ##### EXPRESSIONS ##### #
###########################
#build additional variables we'd like to record based on other variable values

def GeneratorDispatchRule(model,t,g):
    return sum(sum(model.segmentdispatch[t,g,z,gs] for gs in model.GENERATORSEGMENTS) for z in model.ZONES)

dispatch_model.dispatch = Expression(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                                        rule=GeneratorDispatchRule)

###########################
# ##### CONSTRAINTS ##### #
###########################

## RENEWABLES CONSTRAINTS ##

#wind output, should allow for curtailment but has $0 cost for now
def WindRule(model, t, z):
    return (model.windcap[z]*model.windcf[t,z] == model.windgen[t,z])
dispatch_model.WindMaxConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=WindRule)

#solar output, should allow for curtailment but has $0 cost for now
def SolarRule(model, t, z):
    return (model.solarcap[z]*model.solarcf[t,z] == model.solargen[t,z])
dispatch_model.SolarMaxConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=SolarRule)

#curtailment probably won't get used, but let's put it in for now
def CurtailmentRule(model, t, z):
    return (model.curtailment[t,z] == (model.windcap[z]*model.windcf[t,z]-model.windgen[t,z]) + (model.solarcap[z]*model.solarcf[t,z]-model.solargen[t,z]))
#dispatch_model.CurtailmentConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=CurtailmentRule)

## STORAGE CONSTRAINTS ##

def StorageDischargeRule(model, t, s, z):
    return  model.DischargeMax[t,s,z] >= model.discharge[t,s,z]
#dispatch_model.StorageDischargeConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, dispatch_model.ZONES, rule=StorageDischargeRule)

def StorageChargeRule(model, t, s, z):
    return model.ChargeMax[t,s,z] >= model.charge[t,s,z]
#dispatch_model.StorageChargeConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, dispatch_model.ZONES, rule=StorageChargeRule)

def SOCRule(model,t,s,z):
    if t==1:
        return Constraint.Skip
    else:
        return model.soc[t] == model.soc[t-1]
    
def SOCMaxRule(model,t,s,z):
    return model.SOCMax[t,s,z] >= model.soc[t,s,z]
#dispatch_model.SOCMaxConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, dispatch_model.ZONES, rule=SOCMaxRule)
    
# def BindInitSOCRule(model,s,z)


## TRANSMISSION LINES ##

#flow rules, simple for now but could eventually include line ramp limits or etc.
#do want to try to implement dc opf

def TxFromRule(model, t, line):
    return (model.transmit_power_MW[t,line] >= model.transmission_from_capacity[t, line])
dispatch_model.TxFromConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, rule=TxFromRule)

def TxToRule(model, t, line):
    return (model.transmission_to_capacity[t, line] >= model.transmit_power_MW[t,line])
dispatch_model.TxToConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, rule=TxToRule)

#dcopf rules, first, bound voltage angle above and below

def VoltageAngleMaxRule(model,t,z):
    return model.voltage_angle_max[z] >= model.voltage_angle[t,z]
dispatch_model.VoltageAngleMaxConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=VoltageAngleMaxRule)

def VoltageAngleMinRule(model,t,z):
    return model.voltage_angle[t,z] >= model.voltage_angle_min[z]
dispatch_model.VoltageAngleMinConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=VoltageAngleMinRule)

#then set the reference bus
def SetReferenceBusRule(model,t,z):
    if z==model.reference_bus[t]:
        return model.voltage_angle[t,z]==0
    else:
        return Constraint.Skip
dispatch_model.SetReferenceBusConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=SetReferenceBusRule)

#then, bind transmission flows between lines based on voltage angle

def DCOPFRule(model,t,line):
    zone_to = model.transmission_to[t,line]
    zone_from = model.transmission_from[t,line]
    return model.transmit_power_MW[t,line] == model.susceptance[line]*(model.voltage_angle[t,zone_to]-model.voltage_angle[t,zone_from])

dispatch_model.DCOPFConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, rule=DCOPFRule)

## LOAD BALANCE ##

#load/gen balance
def LoadRule(model, t, z):
    #implement total tx flow
    imports_exports = 0
    for line in model.TRANSMISSION_LINE:
        if model.transmission_to[t, line] == z or model.transmission_from[t, line] == z:
            if model.transmission_to[t, line] == z:
                imports_exports += model.transmit_power_MW[t, line]
            elif model.transmission_from[t, line] == z:
                imports_exports -= model.transmit_power_MW[t, line]
            #add additional note to dec import/exports by line losses
            #no, this will just be done as a hurdle rate 
    #full constraint, with tx flow now
    return (sum(sum(model.segmentdispatch[t,g,z,gs] for gs in model.GENERATORSEGMENTS) for g in model.GENERATORS)+\
            model.windgen[t,z] + model.solargen[t,z] + imports_exports == model.GrossLoad[t,z])
dispatch_model.LoadConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=LoadRule)


## CONVENTIONAL GENERATORS CONSTRAINTS ##

#gen capacity with scheduled outage factored in
def CapacityMaxRule(model, t, g, z):
    return (model.capacity[g,z]*model.commitment[t,g]*model.scheduledavailable[t,g] >= model.dispatch[t,g,z])
#dispatch_model.CapacityMaxConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.ZONES, rule=CapacityMaxRule)

#pmin: INACTIVE
def PminRule(model,t,g,z):
    return (model.dispatch[t,g,z] >= model.capacity[g,z]*model.commitment[t,g]*model.scheduledavailable[t,g]*model.pmin[g])
#dispatch_model.PminConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.ZONES, rule=PminRule)

### GENERATOR SEGMENT DISPATCH ###

#max on segment
def GeneratorSegmentDispatchMax(model, t, g, z, gs):
    try:
        return ((model.generatorsegmentlength[gs]*model.capacity[g,z]*model.scheduledavailable[t,g]) >= model.segmentdispatch[t,g,z,gs])
    except ValueError:
        return 0==model.segmentdispatch[t,g,z,gs] #non-existent must be zero
dispatch_model.GeneratorSegmentMaxConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                                       dispatch_model.ZONES, dispatch_model.GENERATORSEGMENTS
                                                       ,rule=GeneratorSegmentDispatchMax)

'''
### DUAL CONSTRAINTS ###
def BindOfferDual(model,t,g,z,gs):
    return model.gensegmentoffer[t,g,z,gs]+model.gensegmentmaxdual[t,g,z,gs]-model.gensegmentmindual[t,g,z,gs]-model.zonalprice[t,z]==0
dispatch_model.OfferDualConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                                       dispatch_model.ZONES, dispatch_model.GENERATORSEGMENTS
                                                       ,rule=BindOfferDual)

### COMPLEMENTARITY CONSTRAINTS ###

def BindMaxDispatchComplementarity(model,t,g,z,gs):
    return complements(model.generatorsegmentlength[gs]*model.capacity[g,z]*model.scheduledavailable[t,g]-model.segmentdispatch[t,g,z,gs]>=0, model.gensegmentmaxdual[t,g,z,gs]>=0)
dispatch_model.MaxDispatchComplementarity = Complementarity(dispatch_model.TIMEPOINTS,dispatch_model.GENERATORS,
                                                            dispatch_model.ZONES, dispatch_model.GENERATORSEGMENTS,
                                                            rule=BindMaxDispatchComplementarity)

def BindMinDispatchComplementarity(model,t,g,z,gs):
    return complements(model.segmentdispatch[t,g,z,gs]>=0, model.gensegmentmindual[t,g,z,gs]>=0)
dispatch_model.MinDispatchComplementarity = Complementarity(dispatch_model.TIMEPOINTS,dispatch_model.GENERATORS,
                                                            dispatch_model.ZONES, dispatch_model.GENERATORSEGMENTS,
                                                            rule=BindMinDispatchComplementarity)

### MARKET-BASED CONSTRAINTS ###
def OfferCap(model,t,g,z,gs):
    return model.generatormarginalcost[g,gs]*2>=model.gensegmentoffer[t,g,z,gs]
dispatch_model.OfferCapConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                                       dispatch_model.ZONES, dispatch_model.GENERATORSEGMENTS,
                                                       rule=OfferCap)

def OfferMin(model,t,g,z,gs):
    return model.gensegmentoffer[t,g,z,gs]>=model.generatormarginalcost[g,gs]
dispatch_model.OfferMinConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                                       dispatch_model.ZONES, dispatch_model.GENERATORSEGMENTS,
                                                       rule=OfferMin)

def MarketPriceCap(model,t,z):
    return 2000>=model.zonalprice[t,z]
dispatch_model.ZonalPriceConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=MarketPriceCap)

#sum of generator segment dispatch equivalent to total generator dispatch
#we are implicitly assuming the first segment will be the most efficient
#which is generally true when bids are constrained to be monotonic
def GeneratorSegmentDispatchSegmentSummation(model,t,g,z):
    return model.dispatch[t,g,z] == sum(model.segmentdispatch[t,g,z,gs] for gs in model.GENERATORSEGMENTS)
dispatch_model.GeneratorSegmentSumConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
                                                       dispatch_model.ZONES,rule=GeneratorSegmentDispatchSegmentSummation)

## GENERATOR RAMP ##

def GeneratorRampUpRule(model,t,g,z):
    if t==1:
        return Constraint.Skip
    else:
        return (model.dispatch[t-1,g,z] + model.ramp[g,z]*model.commitment[t-1,g] + model.startup[t,g]*model.rampstartuplimit[g,z]  >= model.dispatch[t,g,z])
#dispatch_model.GeneratorRampUpConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.ZONES, rule=GeneratorRampUpRule)

def GeneratorRampDownRule(model,t,g,z): ### NEW
    if t==1: ##guessing it's worthwhile to have this to guard against weirdness, even though a generator will never "get shutdown" in hour 1... 
        return Constraint.Skip 
    else:
        return (model.dispatch[t,g,z] >= model.dispatch[t-1,g,z] - model.ramp[g,z]*model.commitment[t-1,g] + model.shutdown[t,g]*model.rampshutdownlimit[g,z])
#dispatch_model.GeneratorRampDownConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.ZONES, rule=GeneratorRampDownRule)


#CUT

## GENERATOR STARTUP/SHUTDOWN ##

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
def ScheduledAvailableRule(model,t,g):
    if model.scheduledavailable[t,g]==0:
        return (model.scheduledavailable[t,g] == model.commitment[t,g])
    else:
        return Constraint.Skip
dispatch_model.ScheduledAvailableConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, rule=ScheduledAvailableRule)

## GENERATOR MIN UP AND DOWN ##

#min uptime constraint
def MinUpRule(model,t,g):
    recent_start_bool = float() #initialize our tracker; boolean because you'll just never see multiple starts
    
    #allow minup to be overruled if generator is scheduled not available
    if model.scheduledavailable[t,g] == 0:
        return Constraint.Skip

    if t - model.minup[g] <1: #i.e. in the lookback period to the initialization condition
        for tp in range(1,t+1):
            if model.scheduledavailable[tp,g]==0: #if the generator previously could have scheduled shutdown, nullify this constraint
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
    return sum(sum(sum(sum(model.segmentdispatch[t,g,z,gs] for z in model.ZONES) for t in model.TIMEPOINTS)*model.generatormarginalcost[g,gs] for g in model.GENERATORS) for gs in model.GENERATORSEGMENTS)+\
           sum(sum(model.commitment[t,g] for t in model.TIMEPOINTS)*model.noloadcost[g] for g in model.GENERATORS)+\
           sum(sum(model.startup[t,g] for t in model.TIMEPOINTS)*model.startcost[g] for g in model.GENERATORS)

    #DESCRIPTION OF OBJECTIVE
    #(1) dispatch cost
    #(2) no load cost of committed gen
    #(3) start up costs when generators brought online
dispatch_model.TotalCost = Objective(rule=objective_rule, sense=minimize)

def objective_rule2(model): 
    #min dispatch cost for objective
    return sum(sum(sum(sum(model.segmentdispatch[t,g,z,gs] for z in model.ZONES) for t in model.TIMEPOINTS)*model.generatormarginalcost[g,gs] for g in model.GENERATORS) for gs in model.GENERATORSEGMENTS)
    #DESCRIPTION OF OBJECTIVE
    #(1) dispatch cost
dispatch_model.TotalCost2 = Objective(rule=objective_rule2, sense=minimize)

def objective_profit(model):
    return sum(sum(model.GrossLoad[t,z]*model.zonalprice[t,z] for z in model.ZONES) for t in model.TIMEPOINTS)-\
           sum(sum(sum(sum(model.segmentdispatch[t,g,z,gs] for z in model.ZONES) for t in model.TIMEPOINTS)*model.generatormarginalcost[g,gs] for g in model.GENERATORS) for gs in model.GENERATORSEGMENTS)
dispatch_model.GeneratorProfit = Objective(rule=objective_profit,sense=maximize)

