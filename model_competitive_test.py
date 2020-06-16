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
from pyomo.environ import (
    AbstractModel,
    Set,
    Param,
    Var,
    Expression,
    Constraint,
    Objective,
    minimize,
    maximize,
    Boolean,
    Reals,
    Binary,
    NonNegativeIntegers,
    PositiveIntegers,
    NonNegativeReals,
    PercentFraction,
)

from pyomo.mpec import complements, Complementarity

"""
This is the formulation of the Pyomo optimization model.
It's in this script that we'll add new constraints and functionality to the model itself
"""

start_time = time.time()
cwd = os.getcwd()

dispatch_model = AbstractModel()


###########################
# ######## SETS ######### #
###########################

# sets are in ALLCAPS without spaces

# time
dispatch_model.TIMEPOINTS = Set(domain=PositiveIntegers, ordered=True)

# generators
dispatch_model.GENERATORS = Set(ordered=True)

# zones
dispatch_model.ZONES = Set(doc="study zones", ordered=True)

# lines
dispatch_model.TRANSMISSION_LINE = Set(doc="tx lines", ordered=True)

# generator bid segments (creates piecewise heat rate curve)
dispatch_model.GENERATORSEGMENTS = Set(ordered=True)

# storage resources
dispatch_model.STORAGE = Set(ordered=True)

# case indexing, needed for changing ownership index in EPEC
dispatch_model.CASE = Set(ordered=True)

###########################
# ###### SUBSETS ####### #
###########################


def strategic_gens_init(model):
    """Subsets generators owned by strategic agent

    Arguments:
        model {Pyomo model} -- the Pyomo model instance

    Returns:
        the subset of generators owned by the strategic agent in the case
    """
    strategic_gens = list()
    for c in model.CASE:
        for g in model.GENERATORS:
            if model.genco_index[g] == model.genco[c]:
                strategic_gens.append(g)
    return strategic_gens


dispatch_model.STRATEGIC_GENERATORS = Set(
    within=dispatch_model.GENERATORS, initialize=strategic_gens_init
)  # implements strategic_gens_init()


def non_strategic_gens_init(model):
    """Subsets generators NOT owned by strategic agent

    Arguments:
        model {Pyomo model} -- the Pyomo model instance

    Returns:
        the subset of generators NOT owned by the strategic agent in the case
    """
    non_strategic_gens = list()
    for c in model.CASE:
        for g in model.GENERATORS:
            if model.genco_index[g] != model.genco[c]:
                non_strategic_gens.append(g)
    return non_strategic_gens


dispatch_model.NON_STRATEGIC_GENERATORS = Set(
    within=dispatch_model.GENERATORS, initialize=non_strategic_gens_init
)  # implements non_strategic_gens_init()


def strategic_storage_init(model):
    strategic_storage = list()
    for c in model.CASE:
        for s in model.STORAGE:
            if model.storage_index[s] == model.genco[c]:
                strategic_storage.append(s)
    return strategic_storage


dispatch_model.STRATEGIC_STORAGE = Set(
    within=dispatch_model.STORAGE, initialize=strategic_storage_init
)  # implements strategic_storage_init()


def non_strategic_storage_init(model):
    non_strategic_storage = list()
    for c in model.CASE:
        for s in model.STORAGE:
            if model.storage_index[s] != model.genco[c]:
                non_strategic_storage.append(s)
    return non_strategic_storage


dispatch_model.NON_STRATEGIC_STORAGE = Set(
    within=dispatch_model.STORAGE, initialize=non_strategic_storage_init
)  # implements non_strategic_storage_init()

###########################
# ####### PARAMS ######## #
###########################

# Params will be in lower case with underscores between words
# Params are separated below by their indexing

# time and zone-indexed params
dispatch_model.gross_load = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.ZONES, within=NonNegativeReals
)

# timepoint-indexed params
dispatch_model.reference_bus = Param(
    dispatch_model.TIMEPOINTS, within=dispatch_model.ZONES
)

# zone-indexed params
dispatch_model.voltage_angle_max = Param(dispatch_model.ZONES, within=NonNegativeReals)
dispatch_model.voltage_angle_min = Param(dispatch_model.ZONES, within=Reals)

# generator-indexed params
dispatch_model.capacity = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.pmin = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.startcost = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
#dispatch_model.minup = Param(dispatch_model.GENERATORS, within=NonNegativeIntegers)
#dispatch_model.mindown = Param(dispatch_model.GENERATORS, within=NonNegativeIntegers)
dispatch_model.noloadcost = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.ramp = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.zonelabel = Param(dispatch_model.GENERATORS, within=dispatch_model.ZONES)
dispatch_model.genco_index = Param(
    dispatch_model.GENERATORS, within=NonNegativeIntegers, mutable=True
)

# storage-indexed params (will be subset from other generators)
dispatch_model.discharge_max = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.charge_max = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.soc_max = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.discharge_eff = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.charge_eff = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.storage_zone_label = Param(
    dispatch_model.STORAGE, within=dispatch_model.ZONES
)
dispatch_model.storage_index = Param(dispatch_model.STORAGE, within=NonNegativeIntegers)

# time and zone-indexed params
dispatch_model.scheduled_available = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, within=PercentFraction
)
dispatch_model.capacity_time = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, within=NonNegativeReals
)
dispatch_model.fuel_cost_time = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, within=NonNegativeReals
)

# transmission line indexed params
dispatch_model.susceptance = Param(
    dispatch_model.TRANSMISSION_LINE, within=NonNegativeReals
)

# time and transmission line-indexed params
dispatch_model.transmission_from = Param(
    dispatch_model.TIMEPOINTS,
    dispatch_model.TRANSMISSION_LINE,
    within=dispatch_model.ZONES,
)
dispatch_model.transmission_to = Param(
    dispatch_model.TIMEPOINTS,
    dispatch_model.TRANSMISSION_LINE,
    within=dispatch_model.ZONES,
)
dispatch_model.transmission_from_capacity = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=Reals
)
dispatch_model.transmission_to_capacity = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=Reals
)
#dispatch_model.hurdle_rate = Param(
#    dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=NonNegativeReals
#)

# generator segment indexed params
dispatch_model.base_generator_segment_length = Param(
    dispatch_model.GENERATORSEGMENTS, within=PercentFraction
)

# generator and generator segment-indexed params
dispatch_model.generator_segment_length = Param(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    within=PercentFraction,
)
dispatch_model.generator_marginal_cost = Param(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    within=NonNegativeReals,
)
dispatch_model.previous_offer = Param(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    within=NonNegativeReals,
)

# genco integer for the case. Defines which generators are owned by agent and bid competitively.
dispatch_model.genco = Param(dispatch_model.CASE, within=NonNegativeIntegers)

###########################
# ######## VARS ######### #
###########################

# Vars will be CamelCase without underscore

dispatch_model.segmentdispatch = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000),
)

dispatch_model.transmit_power_MW = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.TRANSMISSION_LINE,
    within=Reals,
    initialize=0,
    bounds=(-1000, 1000),
)

dispatch_model.voltage_angle = Var(
    dispatch_model.TIMEPOINTS, dispatch_model.ZONES, within=Reals, initialize=0
)

# resource specific vars

dispatch_model.discharge = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    within=NonNegativeReals,
    bounds=(0, 1000),
    initialize=0,
)

dispatch_model.charge = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    within=NonNegativeReals,
    bounds=(0, 1000),
    initialize=0,
)

dispatch_model.soc = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    within=NonNegativeReals,
    initialize=0,
)

# should now be inactive because storage is linearized
# dispatch_model.storagebool = Var(
#    dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, within=Boolean, initialize=0
# )

dispatch_model.storagedispatch = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    within=Reals,
    bounds=(-1000, 1000),
    initialize=0,
)


# the following vars can make problem integer when implemented
# for now relevant constraints are unimplemented, so there is no commitment
dispatch_model.commitment = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    within=NonNegativeReals,
    bounds=(0, 1),
    initialize=0,
)

dispatch_model.startup = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    within=NonNegativeReals,
    bounds=(0, 1),
    initialize=0,
)

dispatch_model.shutdown = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    within=NonNegativeReals,
    bounds=(0, 1),
    initialize=0,
)

# new vars for competitive version of model#
# duals of MO problem
# bounds help reduce feasible space for BigM method, but they should be high enough to not bind
dispatch_model.zonalprice = Var(
    dispatch_model.TIMEPOINTS, dispatch_model.ZONES, within=Reals, initialize=0
)  # this is zonal load balance dual

dispatch_model.gensegmentmaxdual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000),
)

dispatch_model.gensegmentmindual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000),
)

dispatch_model.transmissionmaxdual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.TRANSMISSION_LINE,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000),
)

dispatch_model.transmissionmindual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.TRANSMISSION_LINE,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000),
)

dispatch_model.storagemaxdual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000),
)

dispatch_model.storagemindual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000),
)

dispatch_model.rampmaxdual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    within=(0, 1000),
    initialize=0,
)

dispatch_model.rampmindual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    within=(0, 1000),
    initialize=0,
)

dispatch_model.voltageanglemaxdual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.ZONES,
    within=(0, 1000),
    initialize=0,
)

dispatch_model.voltageanglemindual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.ZONES,
    within=(0, 1000),
    initialize=0,
)

dispatch_model.storagetightdual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    within=(0, 1000),
    initialize=0,
)

dispatch_model.chargedual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    within=(0, 1000),
    initialize=0,
)

dispatch_model.dischargedual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    within=(0, 1000),
    initialize=0,
)

dispatch_model.onecycledual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    within=(0, 1000),
    initialize=0,
)

dispatch_model.gendispatchmaxdual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    within=(0, 1000),
    initialize=0,
)

dispatch_model.gendispatchmindual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    within=(0, 1000),
    initialize=0,
)

dispatch_model.startupshutdowndual = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    within=(0, 1000),
    initialize=0,
)

# offer-related variables (since generators no longer just offer at marginal cost)
dispatch_model.gensegmentoffer = Var(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    within=Reals,
)

dispatch_model.storageoffer = Var(
    dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, within=Reals
)


###########################
# ##### EXPRESSIONS ##### #
###########################
# build additional params or variables we'd like to record based on other param or variable values

def GeneratorDispatchRule(model, t, g):
    return sum(model.segmentdispatch[t, g, gs] for gs in model.GENERATORSEGMENTS)


dispatch_model.dispatch = Expression(
    dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, rule=GeneratorDispatchRule
)  # implement GeneratorDispatchRule


def GeneratorPminRule(model, t, g):
    return model.capacity_time[t, g] * model.commitment[t, g] * model.pmin[g] * model.scheduled_available[t, g]


# dispatch_model.gpmin = Expression(
#    dispatch.TIMEPOINTS, dispatch_model.GENERATORS, rule=GeneratorPminRule
# )


def AvailableSegmentCapacityExpr(model, t, g, gs):
    return (
        model.generator_segment_length[t, g, gs]
        * model.capacity_time[t, g]
        * model.scheduled_available[t, g]
    )


dispatch_model.availablesegmentcapacity = Expression(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    rule=AvailableSegmentCapacityExpr,
)  # implement AvailableSegmentCapacityExpr()

###########################
# ##### CONSTRAINTS ##### #
###########################

## RENEWABLES CONSTRAINTS ##
# No special renewables constraints. They're implemented as generators.

## STORAGE CONSTRAINTS ##
# additional constraints applied only to storage resources


def StorageTightRule(model, t, s):
    """Combine storage charge and discharge capacity limits into one, limit the possibility of charge and discharge happens simutaneously.

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        s {str} -- storage resource index
    """
    return (
        model.charge_max[s] * model.discharge_max[s]
        >= model.charge_max[s] * model.discharge[t, s]
        + model.discharge_max[s] * model.charge[t, s]
    )


dispatch_model.StorageTightConstraint = Constraint(
    dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, rule=StorageTightRule
)


def SOCChangeRule(model, t, s):
    """State of charge of storage changes based on dispatch
    this is where we should add roundtrip efficiency param when implemented

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        s {str} -- storage resource index
    """
    if t == 1:
        return (
            model.soc[t, s]
            == model.charge[t, s] * model.charge_eff[s]
            - model.discharge[t, s] * model.discharge_eff[s]
        )  # start half charged?
        # return model.soc[t,s] == -model.storagedispatch[t,s]
    else:
        return (
            model.soc[t, s]
            == model.soc[t - 1, s]
            + model.charge[t, s] * model.charge_eff[s]
            - model.discharge[t, s] * model.discharge_eff[s]
        )
        # return model.soc[t,s] == model.soc[t-1,s] - model.storagedispatch[t,s]


dispatch_model.SOCChangeConstraint = Constraint(
    dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, rule=SOCChangeRule
)  # implements SOCChangeConstraint


def SOCMaxRule(model, t, s):
    """Storage state of charge cannot exceed its max state of charge

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        s {str} -- storage resource index
    """
    return model.soc_max[s] >= model.soc[t, s]


dispatch_model.SOCMaxConstraint = Constraint(
    dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, rule=SOCMaxRule
)  # implements SOCMaxConstraint


def BindFinalSOCRule(model, s):
    """Storage state of charge in final timestep must be equal to user-defined final SOC value
    I've input this to be 0 for now. This will at least make each day symmetric (0 initial and final SOC)

    Arguments:
        model -- Pyomo model
        s {str} -- storage resource index
    """
    return model.soc_max[s] * 0 == model.soc[model.TIMEPOINTS[-1], s]


dispatch_model.BindFinalSOCConstraint = Constraint(
    dispatch_model.STORAGE, rule=BindFinalSOCRule
)  # implements BindFinalSOCConstraint


def OneCycleRule(model, s):
    return model.soc_max[s] >= sum(model.discharge[t, s] for t in model.TIMEPOINTS)


dispatch_model.OneCycleConstraint = Constraint(
    dispatch_model.STORAGE, rule=OneCycleRule
)  # implements BindFinalSOCConstraint

## TRANSMISSION LINES ##

# flow rules, implemented as DCOPF

# first are the from/to capacity rules from the old hub/spoke version of the model
def TxFromRule(model, t, line):
    """Real power flow on line must be greater than from capacity
    (note from capacity is negative by convention in the model

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        line {str} -- transmission line index
    """
    return model.transmit_power_MW[t, line] >= model.transmission_from_capacity[t, line]


dispatch_model.TxFromConstraint = Constraint(
    dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, rule=TxFromRule
)  # implements TxFromConstraint


def TxToRule(model, t, line):
    """Real power flow on line must be less than to capacity
    (note to capacity is positive by convention in the model

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        line {str} -- transmission line index
    """
    return model.transmission_to_capacity[t, line] >= model.transmit_power_MW[t, line]


dispatch_model.TxToConstraint = Constraint(
    dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, rule=TxToRule
)  # implements TxToConstraint

# then the dcopf rules
# first, bound voltage angle above and below


def VoltageAngleMaxRule(model, t, z):
    """Bus voltage angle must be less than max bus voltage angle

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        z {int} -- bus index (z by convention holdover from old zonal model)
    """
    return model.voltage_angle_max[z] >= model.voltage_angle[t, z]


dispatch_model.VoltageAngleMaxConstraint = Constraint(
    dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=VoltageAngleMaxRule
)  # implements VoltageAngleMaxConstraint


def VoltageAngleMinRule(model, t, z):
    """Bus voltage angle must be greater than min bus voltage angle

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        z {int} -- bus index (z by convention holdover from old zonal model)
    """
    return model.voltage_angle[t, z] >= model.voltage_angle_min[z]


dispatch_model.VoltageAngleMinConstraint = Constraint(
    dispatch_model.TIMEPOINTS,
    dispatch_model.ZONES,
    rule=VoltageAngleMinRule,
    name="VoltageAngleMin",
)  # implements VoltageAngleMaxConstraint

# then set the reference bus
def SetReferenceBusRule(model, t, z):
    """Binds voltage angle of the system reference bus to be 0

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        z {int} -- bus index (z by convention holdover from old zonal model)
    """
    if z == model.reference_bus[t]:
        return model.voltage_angle[t, z] == 0
    else:
        return Constraint.Skip


dispatch_model.SetReferenceBusConstraint = Constraint(
    dispatch_model.TIMEPOINTS,
    dispatch_model.ZONES,
    rule=SetReferenceBusRule,
    name="RefBus",
)  # implements SetReferenceBusConstraint

# then, bind transmission flows between lines based on voltage angle


def DCOPFRule(model, t, line):
    """Power flow defined by angle between buses and line susceptance

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        z {int} -- bus index (z by convention holdover from old zonal model)
    """
    zone_to = model.transmission_to[t, line]
    zone_from = model.transmission_from[t, line]
    return model.transmit_power_MW[t, line] == model.susceptance[line] * (
        model.voltage_angle[t, zone_to] - model.voltage_angle[t, zone_from]
    )


dispatch_model.DCOPFConstraint = Constraint(
    dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, rule=DCOPFRule
)  # implements DCOPFConstraint

## LOAD BALANCE ##

# load/gen balance at all buses
def LoadRule(model, t, z):
    """It's long but does what it sounds like: load must be served at all buses in the system
    so the sum of generation at, storage at, and transmission flows into the bus
    again, some things get "zonal" or "z" labels/indices based on old zonal model
    
    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        z {int} -- bus index (z by convention holdover from old zonal model)
    """
    # implement total tx flow
    imports_exports = 0
    zonal_generation = 0
    zonal_storage = 0
    for line in model.TRANSMISSION_LINE:
        if model.transmission_to[t, line] == z or model.transmission_from[t, line] == z:
            if model.transmission_to[t, line] == z:
                imports_exports += model.transmit_power_MW[t, line]
            elif model.transmission_from[t, line] == z:
                imports_exports -= model.transmit_power_MW[t, line]
            # add additional note to dec import/exports by line losses
            # no, this will just be done as a hurdle rate
    for g in model.GENERATORS:
        if model.zonelabel[g] == z:
            zonal_generation += sum(
                model.segmentdispatch[t, g, gs] for gs in model.GENERATORSEGMENTS
            )
    for s in model.STORAGE:
        if model.storage_zone_label[s] == z:
            # zonal_storage += model.discharge[t,s]
            # zonal_storage -= model.charge[t,s]
            zonal_storage += model.storagedispatch[t, s]
    # full constraint, with tx flow now
    # (sum(sum(model.segmentdispatch[t,g,z,gs] for gs in model.GENERATORSEGMENTS) for g in model.GENERATORS)+\
    return zonal_generation + imports_exports + zonal_storage == model.gross_load[t, z]


dispatch_model.LoadConstraint = Constraint(
    dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=LoadRule
)  # implements load constraint

## CONVENTIONAL GENERATORS CONSTRAINTS ##

# gen capacity with scheduled outage factored in: INACTIVE
def CapacityMaxRule(model, t, g):
    """ Generator dispatch cannot exceed (available) capacity
    this is actually disabled by defaul right now since it's implement by bid segment
    update docstring if enabled

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
    """
    return (
        model.capacity_time[t,g] * model.commitment[t, g] * model.scheduled_available[t, g]
        >= model.dispatch[t, g]
    )


# dispatch_model.CapacityMaxConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.ZONES, rule=CapacityMaxRule)

# pmin: INACTIVE
def CapacityMinRule(model, t, g):
    """ Generator dispatch must be above minimum stable level if generator is dispatched
    this is actually disabled by default right now since I linearized this version of the model
    update docstring if enabled

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
    """
    return (
        model.dispatch[t, g]
        >= model.gpmin[t,g]
    )


# dispatch_model.PminConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, dispatch_model.ZONES, rule=CapacityMinRule)

### GENERATOR SEGMENT DISPATCH ###
# basically I reimplemented most generator constraints segment-wise, below
# this allows generators to bid multiple segments and have a heat rate curve

# max on segment
def GeneratorSegmentDispatchMax(model, t, g, gs):
    """ Generator segment dispatch cannot exceed (available) capacity on segment

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
        gs {int} -- generator segment index
    """
    return model.availablesegmentcapacity[t, g, gs] >= model.segmentdispatch[t, g, gs]


dispatch_model.GeneratorSegmentMaxConstraint = Constraint(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    rule=GeneratorSegmentDispatchMax,
)  # implements GeneratorSegmentMaxConstraint

### GENERATOR RAMP ###
# these are currently inactive but may be a helfpul template for implementing TRUC


def GeneratorRampUpRule(model, t, g):
    """ Increase in generator dispatch between timepoints cannot exceed upward ramp rate
    Note this isn't implemented in the first timepoint, so any initialization is allowed
    There are a couple ways around this, one of the more common/simple is looping the day back on itself

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
    """
    if t == 1:
        return Constraint.Skip
    else:
        return (
            model.dispatch[t - 1, g] - model.gpmin[t - 1, g] + model.ramp[g] * model.commitment[t - 1, g]
            >= model.dispatch[t, g] - model.gpmin[t, g]
        )


# dispatch_model.GeneratorRampUpConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, rule=GeneratorRampUpRule)


def GeneratorRampDownRule(model, t, g):
    """ Decrease in generator dispatch between timepoints cannot exceed downward ramp rate
    Note this isn't implemented in the first timepoint, so any initialization is allowed
    There are a couple ways around this, one of the more common/simple is looping the day back on itself

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
    """
    if t == 1:
        return Constraint.Skip
    else:
        return model.dispatch[t, g] - model.gpmin[t, g] >= model.dispatch[t - 1, g] - model.gpmin[t - 1, g] - model.ramp[g] * model.commitment[t, g]


# dispatch_model.GeneratorRampDownConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, rule=GeneratorRampDownRule)


## GENERATOR STARTUP/SHUTDOWN ##
def GeneratorStartupShutdownRule(model, t, g):
    """ Generator segment dispatch cannot exceed (available) capacity on segment

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
        gs {int} -- generator segment index
    """
    return model.commitment[t, g] - model.commitment[t - 1, g] == model.startup[t, g] - model.shutdown[t, g]


# dispatch_model.GeneratorSegmentMaxConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, rule=GeneratorStartupShutdownRule)



### DUAL CONSTRAINTS ###
# all of this applies only in the competitive version of the model


def BindGeneratorOfferDual(model, t, g, gs):
    """ Duals associated with generator max, min, and load balance constraints equal offer
    If you want to read a paper on reformulating using strong duality, let me know

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
        gs {int} -- generator segment index
    """
    return (
        model.gensegmentoffer[t, g, gs]
        + model.gensegmentmaxdual[t, g, gs]
        - model.gensegmentmindual[t, g, gs]
        + model.gendispatchmaxdual[t, g, gs]
        - model.gendispatchmindual[t, g, gs]
        + model.rampmaxdual[t, g]
        - model.rampmindual[t, g]
        - model.zonalprice[t, model.zonelabel[g]]
        == 0
    )


dispatch_model.GeneratorOfferDualConstraint = Constraint(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    rule=BindGeneratorOfferDual,
    name="OfferDual",
)  # implements GeneratorOfferDualConstraint


def BindStorageDischargeDual(model, t, s):
    """ Duals associated with storage dispatch max, min, and load balance constraints equal offer

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        s {str} -- storage index
    """
    return (
        model.storageoffer[t, s]
        + model.charge_max[s] * model.storagetightdual[t, s]
        - model.dischargedual[t, s]
        - model.discharge_eff[s] * model.storagemaxdual[t, s]
        + model.discharge_eff[s] * model.storagemindual[t, s]
        + model.onecycledual[s]
        - model.zonalprice[t, model.storage_zone_label[s]]
        == 0
    )


dispatch_model.StorageDischargeDualConstraint = Constraint(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    rule=BindStorageDischargeDual,
    name="StorageDischargeDual",
)  

def BindStorageChargeDual(model, t, s):
    """ Duals associated with storage dispatch max, min, and load balance constraints equal offer

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        s {str} -- storage index
    """
    return (
        model.discharge_max[s] * model.storagetightdual[t, s]
        - model.chargedual[t, s]
        + model.charge_eff[s] * model.storagemaxdual[t, s]
        - model.charge_eff[s] * model.storagemindual[t, s]
        + model.zonalprice[t, model.storage_zone_label[s]]
        == 0
    )


dispatch_model.StorageChargeDualConstraint = Constraint(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    rule=BindStorageChargeDual,
    name="StorageChargeDual",
)  


def BindStartupDual(model, t, g):
    return (
        model.startcost[g] - model.startupshutdowndual[t, g]
    )


# dispatch_model.StartupDualConstraint = Constraint(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
#                                                    rule=StartupDual)


def BindComittmentDual(model,t,g):
    return(
        model.noloadcost[g] 
        - model.scheduled_available[t, g] * model.capacity_time[t, g] * model. gendispatchmaxdual[t, g]
        - model.scheduled_available[t, g] * model.capacity_time[t, g] * model. gendispatchmindual[t, g] * model.pmin[g]
        - model.startupshutdowndual[t, g]
        - (model.ramp[g] + model.pmin[g] * model.scheduled_available[t, g] * model.capacity_time[t, g]) * model.rampmaxdual[t, g]
        - (model.ramp[g] - model.pmin[g] * model.scheduled_available[t, g] * model.capacity_time[t, g]) * model.rampmindual[t, g]
    )

def BindFlowDual(model, t, z):
    """ Duals associated with transmission min and max flows equals different between 
    prices at the buses connected by the transmission line.

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        z {str} -- bus index
    """
    maxdual = 0
    mindual = 0
    lmp_delta = 0
    for line in model.TRANSMISSION_LINE:
    
        if model.transmission_to[t, line] == z:
            sink_zone = model.transmission_from[t, line]
            # if t==1:
            #    print(sink_zone)
            maxdual += model.susceptance[line] * model.transmissionmaxdual[t, line] + model.voltageanglemaxdual[t,z]
            mindual += model.susceptance[line] * model.transmissionmindual[t, line] + model.voltageanglemindual[t,z]
            lmp_delta += model.susceptance[line] * model.zonalprice[t, z]
            lmp_delta -= model.susceptance[line] * model.zonalprice[t, sink_zone]

        elif model.transmission_from[t, line] == z:
            sink_zone = model.transmission_to[t, line]
            # if t==1:
            #    print(sink_zone)
            maxdual -= model.susceptance[line] * model.transmissionmaxdual[t, line] + model.voltageanglemaxdual[t,z]
            mindual -= model.susceptance[line] * model.transmissionmindual[t, line] + model.voltageanglemindual[t,z]
            lmp_delta += model.susceptance[line] * model.zonalprice[t, z]
            lmp_delta -= model.susceptance[line] * model.zonalprice[t, sink_zone]
    # if t==8 and z==1:
    #    print(maxdual,mindual,lmp_delta)
    return maxdual - mindual == lmp_delta


dispatch_model.FlowDualConstraint = Constraint(
    dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=BindFlowDual, name="FlowDual"
)  # implements FlowDualConstraint

### COMPLEMENTARITY CONSTRAINTS ###
# these are needed for MPEC reformulation
# the upshot is at least one of the two complements must bind (i.e., be an equality)
# this ends up properly constraining the dual variables, though you may prefer to understand it in math


def BindStorageTightComplementarity(model,t,s):
    return complements(
        model.discharge_max[s] * model.charge_max[s] - model.discharge_max[s] * model.charge[t, s] - model.charge_max[s] * model.discharge[t,s] >= 0,
        model.storagetightdual[t, s] >= 0,
    )


dispatch_model.StorageTightComplementarity = Complementarity(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    rule=BindStorageTightComplementarity,
)


def BindStorageChargeComplementarity(model, t, s):
    return complements(
        model.charge[t, s] >= 0,
        model.chargedual[t, s] >= 0,
    )


dispatch_model.StorageChargeComplementarity = Complementarity(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    rule=BindStorageChargeComplementarity,
)


def BindStorageDischargeComplementarity(model, t, s):
    return complements(
        model.discharge[t, s] >= 0,
        model.dischargedual[t, s] >= 0,
    )


dispatch_model.StorageDischargeComplementarity = Complementarity(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    rule=BindStorageDischargeComplementarity,
)


def BindMaxStorageComplementarity(model, t, s):
    return complements(
        model.soc_max[s] - model.soc[t, s] >= 0,
        model.storagemaxdual[t, s] >= 0,
    )


dispatch_model.MaxStorageComplementarity = Complementarity(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    rule=BindMaxStorageComplementarity,
) 


def BindMinStorageComplementarity(model, t, s):
    return complements(
        model.soc[t, s] >= 0,
        model.storagemindual[t, s] >= 0,
    )


dispatch_model.MinStorageComplementarity = Complementarity(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STORAGE,
    rule=BindMinStorageComplementarity,
) 


def BindOneCycleComplementarity(model, s):
    return complements(
        model.soc_max[s] - sum(model.discharge[t,s] for t in model.TIMEPOINTS) >= 0,
        model.onecycledual[s] >= 0,
    )


dispatch_model.OneCycleComplementarity = Complementarity(
    dispatch_model.STORAGE,
    rule=BindOneCycleComplementarity,
) 


def BindMaxTransmissionComplementarity(model, t, line):
    """ Transmission line power flow is either (1) at its max, or (2) dual is zero (or both)
    Upshot: transmissionmaxdual can only be nonzero when line flow is at its max

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        line {str} -- transmission line index
    """
    return complements(
        model.transmission_to_capacity[t, line] - model.transmit_power_MW[t, line] >= 0,
        model.transmissionmaxdual[t, line] >= 0,
    )


dispatch_model.MaxTransmissionComplementarity = Complementarity(
    dispatch_model.TIMEPOINTS,
    dispatch_model.TRANSMISSION_LINE,
    rule=BindMaxTransmissionComplementarity,
)  # implements MaxTransmissionComplementarity


def BindMinTransmissionComplementarity(model, t, line):
    """ Transmission line power flow is either (1) at its min, or (2) min dual is zero (or both)
    Upshot: transmissionmindual can only be nonzero when line flow is at its min (max in from/negative direction)

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        line {str} -- transmission line index
    """
    return complements(
        -model.transmission_from_capacity[t, line] + model.transmit_power_MW[t, line]
        >= 0,
        model.transmissionmindual[t, line] >= 0,
    )


dispatch_model.MinTransmissionComplementarity = Complementarity(
    dispatch_model.TIMEPOINTS,
    dispatch_model.TRANSMISSION_LINE,
    rule=BindMinTransmissionComplementarity,
)  # implements MinTransmissionComplementarity


def BindMaxVoltageAngleComplementarity(model, t, z):
    return complements(
        model.voltage_angle_max[z] - model.voltage_angle[t, z] >= 0,
        model.voltageanglemaxdual[t, z] >= 0,
    )


dispatch_model.MaxVoltageAngleComplementarity = Complementarity(
    dispatch_model.TIMEPOINTS,
    dispatch_model.ZONES,
    rule=BindMaxVoltageAngleComplementarity,
) 


def BindMinVoltageAngleComplementarity(model, t, z):
    return complements(
        model.voltage_angle[t, z] - model.voltage_angle_min[z]>= 0,
        model.voltageanglemindual[t, z] >= 0,
    )


dispatch_model.MinVoltageAngleComplementarity = Complementarity(
    dispatch_model.TIMEPOINTS,
    dispatch_model.ZONES,
    rule=BindMinVoltageAngleComplementarity,
) 


def BindMaxDispatchComplementarity(model, t, g, gs):
    """ Generator segment dispatch is either (1) at its max,
    or (2) the dual variable associated with max segment dispatch is zero.
    Or both can be equalities
    Upshot: gensegmentmaxdual can only be nonzero when generator segment is dispatch to its max

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
        gs {int} -- generator segment index
    """
    return complements(
        model.availablesegmentcapacity[t, g, gs] - model.segmentdispatch[t, g, gs] >= 0,
        model.gensegmentmaxdual[t, g, gs] >= 0,
    )


dispatch_model.MaxDispatchComplementarity = Complementarity(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    rule=BindMaxDispatchComplementarity,
)  # implements MaxDispatchComplementarity


def BindMinDispatchComplementarity(model, t, g, gs):
    """ Generator segment dispatch is either (1) at its min,
    or (2) the dual variable associated with min segment dispatch is zero.
    Or both can be equalities
    Upshot: gensegmentmindual can only be nonzero when generator segment is dispatched to its min (i.e., 0)

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
        gs {int} -- generator segment index
    """
    return complements(
        model.segmentdispatch[t, g, gs] >= 0, model.gensegmentmindual[t, g, gs] >= 0
    )


dispatch_model.MinDispatchComplementarity = Complementarity(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    rule=BindMinDispatchComplementarity,
)  # implements MinDispatchComplementarity


def BindMaxRampComplementarity(model, t, g):
    """ Generator ramp is either (1) at its max, or (2) dual is zero (or both)
    Upshot: rampmaxdual can only be nonzero when generator ramp is at its max
    INACTIVE: update docstring if activated

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
    """
    if t == 1:
        return Complementarity.Skip
    else:
        return complements(
            model.ramp[t, g] - (model.dispatch[t, g] - model.dispatch[t - 1, g]) >= 0,
            model.rampmaxdual[t, g] >= 0,
        )


# dispatch_model.MaxRampComplementarity = Complementarity(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
#                                                        rule=BindMaxRampComplementarity)


def BindMinRampComplementarity(model, t, g):
    # INACTIVE: I never finished writing this and integrating it in the objective(s)
    if t == 1:
        return Complementarity.Skip
    else:
        return complements()


# dispatch_model.MinRampComplementarity = Complementarity(dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS,
#                                                        rule=BindMinRampComplementarity)


## OFFER CURVE INCREASING ##
# it's just generally a market rule that offers of a single generator be increasing, due to heat rate curves and necessity of convexity


def IncreasingOfferCurve(model, t, g, gs):
    """ Enforces that each segment is offered at greater or equal to the cost of the previous segments
    For marginal costs convexity should enforce this inherently, but it's not guaranteed when you 
    allow generators to submit arbitrary bids. It is a market rule, though, so may as well put it in

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
        gs {int} -- generator segment index
    """
    if gs == 0:
        return (
            Constraint.Skip
        )  # offer whatever you want on the first segment of your offer curve
    else:
        return model.gensegmentoffer[t, g, gs] >= model.gensegmentoffer[t, g, gs - 1]


dispatch_model.IncreasingOfferCurveConstraint = Constraint(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    rule=IncreasingOfferCurve,
)  # implements IncreasingOfferCurveConstraint

### MARKET-BASED CONSTRAINTS ###
# these generally just help the model solve with reasonable values in a reasonable timeframe
# though of course in practice markets do have price caps, and extremely uncompetitive offers would be hard to hide


def OfferCap(model, t, g, gs):
    """ To keep problem from running away, I cap strategic generators to offer at 2x their cost for now
    This is arbitrary and could be dropped, particularly because I also implement a market price cap

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
        gs {int} -- generator segment index
    """
    return model.previous_offer[t, g, gs] * 2 >= model.gensegmentoffer[t, g, gs]


dispatch_model.OfferCapConstraint = Constraint(
    dispatch_model.TIMEPOINTS,
    dispatch_model.STRATEGIC_GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    rule=OfferCap,
)  # implements OfferCapConstraint


def OfferCap2(model, t, g, gs):
    """ This only ends up really mattering for iterated cases (EPEC), but it caps offer at cost
    for non-competitive generators. In iterated cases it's important the "previous offer"
    be updated to reflect each genco's behavior, making this a bit more complicated.

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
        gs {int} -- generator segment index
    """
    return (
        model.previous_offer[t, g, gs] >= model.gensegmentoffer[t, g, gs]
    )  # caps offer at cost to make it param


dispatch_model.OfferCap2Constraint = Constraint(
    dispatch_model.TIMEPOINTS,
    dispatch_model.NON_STRATEGIC_GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    rule=OfferCap2,
)  # implements OfferCap2Constraint


def OfferMin(model, t, g, gs):
    """ Generators must offer at least at cost. This probably doesn't matter much but 
    I didn't want any kind of weird outcomes where generators offer under cost to get a 
    "deal" for their storage assets, as this would be difficult to pull off. 

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
        gs {int} -- generator segment index
    """
    return (
        model.gensegmentoffer[t, g, gs] >= model.generator_marginal_cost[t, g, gs]
    )  # must offer at least marginal cost


dispatch_model.OfferMinConstraint = Constraint(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    rule=OfferMin,
)


def MarketPriceCap(model, t, z):
    """ Caps bus price at $2000/MWh. This is about where the energy price cap is in most RTOs
    other than ERCOT under normal operations without scarcity adders. The purpose here is just
    to avoid cases that are unbounded above, even if constraints on generator bids are removed.

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        z {str} -- bus index (z labeling is from old zonal model)
    """
    return 2000 >= model.zonalprice[t, z]


dispatch_model.ZonalPriceConstraint = Constraint(
    dispatch_model.TIMEPOINTS, dispatch_model.ZONES, rule=MarketPriceCap
)

###########################
# ###### OBJECTIVES ##### #
###########################

# There are a few objectives here depending on how the user implements the model
# in general it'll be important to only have one active objective, so if you add more objectives
# keep that in mind (objective activation passes happen elsewhere)


def objective_rule(model):
    """Old unit commitment system operator objective (includes no-load and start-up cost)

    Arguments:
        model  -- Pyomo model
    """
    return (
        sum(
            sum(
                sum(
                    model.segmentdispatch[t, g, gs]
                    * model.generator_marginal_cost[t, g, gs]
                    for t in model.TIMEPOINTS
                )
                for g in model.GENERATORS
            )
            for gs in model.GENERATORSEGMENTS
        )
        + sum(
            sum(model.commitment[t, g] for t in model.TIMEPOINTS) * model.noloadcost[g]
            for g in model.GENERATORS
        )
        + sum(
            sum(model.startup[t, g] for t in model.TIMEPOINTS) * model.startcost[g]
            for g in model.GENERATORS
        )
    )

    # DESCRIPTION OF OBJECTIVE
    # (1) dispatch cost
    # (2) no load cost of committed gen
    # (3) start up costs when generators brought online


dispatch_model.TotalCost = Objective(rule=objective_rule, sense=minimize)


def objective_rule2(model):
    """system operator dispatch-only objective (considers only marginal cost of generators)

    Arguments:
        model  -- Pyomo model
    """
    return sum(
        sum(
            sum(
                model.segmentdispatch[t, g, gs]
                * model.generator_marginal_cost[t, g, gs]
                for t in model.TIMEPOINTS
            )
            for g in model.GENERATORS
        )
        for gs in model.GENERATORSEGMENTS
    ) + sum(
        sum(
            sum(
                model.segmentdispatch[t, g, gs]
                * model.marginal_CO2[t, g, gs]
                * model.CO2_damage[t, g, gs]
                for t in model.TIMEPOINTS
            )
            for g in model.GENERATORS
        )
        for gs in model.GENERATORSEGMENTS
    )
    # DESCRIPTION OF OBJECTIVE
    # (1) dispatch cost


dispatch_model.TotalCost2 = Objective(rule=objective_rule2, sense=minimize)


def objective_profit(model):
    """Simple objective I wrote to look at generator profits as only objective for model vetting
    Not currently used in any cases

    Arguments:
        model  -- Pyomo model
    """
    return sum(
        sum(model.gross_load[t, z] * model.zonalprice[t, z] for z in model.ZONES)
        for t in model.TIMEPOINTS
    ) - sum(
        sum(
            sum(
                model.segmentdispatch[t, g, gs]
                * model.generator_marginal_cost[t, g, gs]
                for t in model.TIMEPOINTS
            )
            for g in model.GENERATORS
        )
        for gs in model.GENERATORSEGMENTS
    )


dispatch_model.GeneratorProfit = Objective(rule=objective_profit, sense=maximize)


def objective_profit_dual(model):
    """Full objective for MPEC reformulated as MIP using BigM

    Arguments:
        model  -- Pyomo model
    """

    return (
        sum(
            sum(
                sum(
                    -model.availablesegmentcapacity[t, g, gs]
                    * model.gensegmentmaxdual[t, g, gs]
                    for t in model.TIMEPOINTS
                )
                for g in model.NON_STRATEGIC_GENERATORS
            )
            for gs in model.GENERATORSEGMENTS
        )
        - sum(
            sum(
                model.discharge_max[s] * model.storagemaxdual[t, s]
                for t in model.TIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                model.charge_max[s] * model.storagemindual[t, s]
                for t in model.TIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                sum(
                    model.segmentdispatch[t, g, gs]
                    * model.generator_marginal_cost[t, g, gs]
                    for t in model.TIMEPOINTS
                )
                for g in model.GENERATORS
            )
            for gs in model.GENERATORSEGMENTS
        )
        - sum(
            sum(
                model.transmission_to_capacity[t, line]
                * model.transmissionmaxdual[t, line]
                for t in model.TIMEPOINTS
            )
            for line in model.TRANSMISSION_LINE
        )
        + sum(
            sum(
                model.transmission_from_capacity[t, line]
                * model.transmissionmindual[t, line]
                for t in model.TIMEPOINTS
            )
            for line in model.TRANSMISSION_LINE
        )
        + sum(
            sum(
                model.gross_load[t, z] * model.zonalprice[t, z]
                for t in model.TIMEPOINTS
            )
            for z in model.ZONES
        )
    )


dispatch_model.GeneratorProfitDual = Objective(
    rule=objective_profit_dual, sense=maximize
)


def objective_profit_dual_pre(model):
    """Pre-processing version of full objective for MPEC reformulated as MIP using BigM
    Only difference is iteration over *ALL* generators rather than just non-strategic generators
    for the first term in the objective (involving gensegmentmaxdual). This just forces
    all generators to be considered non-competitive for purposes of the objective
    which makes an easier problem to solve as only storage resources can bid competitively

    Arguments:
        model  -- Pyomo model
    """
    return (
        sum(
            sum(
                sum(
                    -model.availablesegmentcapacity[t, g, gs]
                    * model.gensegmentmaxdual[t, g, gs]
                    for t in model.TIMEPOINTS
                )
                for g in model.GENERATORS
            )
            for gs in model.GENERATORSEGMENTS
        )
        - sum(
            sum(
                model.discharge_max[s] * model.storagemaxdual[t, s]
                for t in model.TIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                model.charge_max[s] * model.storagemindual[t, s]
                for t in model.TIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                sum(
                    model.segmentdispatch[t, g, gs]
                    * model.generator_marginal_cost[t, g, gs]
                    for t in model.TIMEPOINTS
                )
                for g in model.GENERATORS
            )
            for gs in model.GENERATORSEGMENTS
        )
        - sum(
            sum(
                model.transmission_to_capacity[t, line]
                * model.transmissionmaxdual[t, line]
                for t in model.TIMEPOINTS
            )
            for line in model.TRANSMISSION_LINE
        )
        + sum(
            sum(
                model.transmission_from_capacity[t, line]
                * model.transmissionmindual[t, line]
                for t in model.TIMEPOINTS
            )
            for line in model.TRANSMISSION_LINE
        )
        + sum(
            sum(
                model.gross_load[t, z] * model.zonalprice[t, z]
                for t in model.TIMEPOINTS
            )
            for z in model.ZONES
        )
    )


dispatch_model.GeneratorProfitDualPre = Objective(
    rule=objective_profit_dual_pre, sense=maximize
)
