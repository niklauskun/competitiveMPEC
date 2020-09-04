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

# active time
dispatch_model.ACTIVETIMEPOINTS = Set(
    domain=PositiveIntegers, ordered=True, within=dispatch_model.TIMEPOINTS
)

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
# ####### PARAMS ######## #
###########################

# Params will be in lower case with underscores between words
# Params are separated below by their indexing

# time and zone-indexed params
dispatch_model.GrossLoad = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.ZONES, within=NonNegativeReals
)

# timepoint-indexed params
dispatch_model.ReferenceBus = Param(
    dispatch_model.TIMEPOINTS, within=dispatch_model.ZONES
)
dispatch_model.Hours = Param(dispatch_model.TIMEPOINTS, within=NonNegativeReals)

# active timepoint-indexed params
dispatch_model.FirstTimepoint = Param(
    dispatch_model.ACTIVETIMEPOINTS, within=dispatch_model.TIMEPOINTS
)

# zone-indexed params
dispatch_model.VoltageAngleMax = Param(dispatch_model.ZONES, within=NonNegativeReals)
dispatch_model.VoltageAngleMin = Param(dispatch_model.ZONES, within=Reals)

# generator-indexed params
dispatch_model.capacity = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.fuelcost = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.Pmin = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.StartCost = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.NoLoadCost = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.RampRate = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.TonneCO2PerMWh = Param(
    dispatch_model.GENERATORS, within=NonNegativeReals
)
dispatch_model.CO2Price = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.CO2DollarsPerMWh = Param(
    dispatch_model.GENERATORS, within=NonNegativeReals
)
dispatch_model.ZoneLabel = Param(dispatch_model.GENERATORS, within=dispatch_model.ZONES)
dispatch_model.GencoIndex = Param(
    dispatch_model.GENERATORS, within=NonNegativeIntegers, mutable=True
)
dispatch_model.UCIndex = Param(dispatch_model.GENERATORS, within=NonNegativeReals)
dispatch_model.HybridIndex = Param(dispatch_model.GENERATORS, within=NonNegativeReals)

# storage-indexed params (will be subset from other generators)
dispatch_model.DischargeMax = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.ChargeMax = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.SocMax = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.DischargeEff = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.ChargeEff = Param(dispatch_model.STORAGE, within=NonNegativeReals)
dispatch_model.StorageZoneLabel = Param(
    dispatch_model.STORAGE, within=dispatch_model.ZONES
)
dispatch_model.StorageIndex = Param(dispatch_model.STORAGE, within=NonNegativeIntegers)
dispatch_model.HybridStorageIndex = Param(
    dispatch_model.STORAGE, within=NonNegativeIntegers
)

# storage and time-indexed params
dispatch_model.ChargeMaxOffer = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, within=Reals
)
dispatch_model.DischargeMaxOffer = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, within=Reals
)

dispatch_model.DischargeOffer = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, within=Reals
)

dispatch_model.ChargeOffer = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, within=Reals
)

dispatch_model.SOCInitDA = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, within=Reals
)

dispatch_model.ChargeInitDA = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, within=Reals
)

dispatch_model.DischargeInitDA = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.STORAGE, within=Reals
)

# time and zone-indexed params
dispatch_model.ScheduledAvailable = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, within=PercentFraction
)
dispatch_model.CapacityTime = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, within=NonNegativeReals
)
dispatch_model.fuel_cost_time = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.GENERATORS, within=NonNegativeReals
)

# transmission line indexed params
dispatch_model.Susceptance = Param(
    dispatch_model.TRANSMISSION_LINE, within=NonNegativeReals
)

# time and transmission line-indexed params
dispatch_model.TransmissionFrom = Param(
    dispatch_model.TIMEPOINTS,
    dispatch_model.TRANSMISSION_LINE,
    within=dispatch_model.ZONES,
)
dispatch_model.TransmissionTo = Param(
    dispatch_model.TIMEPOINTS,
    dispatch_model.TRANSMISSION_LINE,
    within=dispatch_model.ZONES,
)
dispatch_model.TransmissionFromCapacity = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=Reals
)
dispatch_model.TransmissionToCapacity = Param(
    dispatch_model.TIMEPOINTS, dispatch_model.TRANSMISSION_LINE, within=Reals
)

# generator segment indexed params
dispatch_model.base_GeneratorSegmentLength = Param(
    dispatch_model.GENERATORSEGMENTS, within=PercentFraction
)

# generator and generator segment-indexed params
dispatch_model.GeneratorSegmentLength = Param(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    within=PercentFraction,
)
dispatch_model.GeneratorMarginalCost = Param(
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
dispatch_model.marginal_CO2 = Param(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    within=NonNegativeReals,
)
dispatch_model.CO2_damage = Param(
    dispatch_model.TIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    within=NonNegativeReals,
)


# genco integer for the case. Defines which generators are owned by agent and bid competitively.
dispatch_model.genco = Param(dispatch_model.CASE, within=NonNegativeIntegers)


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
            if model.GencoIndex[g] == model.genco[c]:
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
            if model.GencoIndex[g] != model.genco[c]:
                non_strategic_gens.append(g)
    return non_strategic_gens


dispatch_model.NON_STRATEGIC_GENERATORS = Set(
    within=dispatch_model.GENERATORS, initialize=non_strategic_gens_init
)  # implements non_strategic_gens_init()


def strategic_storage_init(model):
    strategic_storage = list()
    for c in model.CASE:
        for s in model.STORAGE:
            if model.StorageIndex[s] == model.genco[c]:
                strategic_storage.append(s)
    return strategic_storage


dispatch_model.STRATEGIC_STORAGE = Set(
    within=dispatch_model.STORAGE, initialize=strategic_storage_init
)  # implements strategic_storage_init()


def non_strategic_storage_init(model):
    non_strategic_storage = list()
    for c in model.CASE:
        for s in model.STORAGE:
            if model.StorageIndex[s] != model.genco[c]:
                non_strategic_storage.append(s)
    return non_strategic_storage


dispatch_model.NON_STRATEGIC_STORAGE = Set(
    within=dispatch_model.STORAGE, initialize=non_strategic_storage_init
)  # implements non_strategic_storage_init()


def uc_generators_init(model):
    uc_generators = list()
    for g in model.GENERATORS:
        if model.UCIndex[g] == 2:
            uc_generators.append(g)
    return uc_generators


dispatch_model.UC_GENS = Set(
    within=dispatch_model.GENERATORS, initialize=uc_generators_init
)


def nuc_generators_init(model):
    nuc_generators = list()
    for g in model.GENERATORS:
        if model.UCIndex[g] == 1:
            nuc_generators.append(g)
    return nuc_generators


dispatch_model.NUC_GENS = Set(
    within=dispatch_model.GENERATORS, initialize=nuc_generators_init
)


def hybrid_generators_init(model):
    hybrid_generators = list()
    for g in model.GENERATORS:
        if model.HybridIndex[g] == 1:
            hybrid_generators.append(g)
    return hybrid_generators


dispatch_model.HYBRID_GENS = Set(
    within=dispatch_model.GENERATORS, initialize=hybrid_generators_init
)


def hybrid_storage_init(model):
    hybrid_storage = list()
    for s in model.STORAGE:
        if model.HybridStorageIndex[s] == 1:
            hybrid_storage.append(s)
    return hybrid_storage


dispatch_model.HYBRID_STORAGE = Set(
    within=dispatch_model.STORAGE, initialize=hybrid_storage_init
)

###########################
# ######## VARS ######### #
###########################

# Vars will be CamelCase without underscore

dispatch_model.gsd = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 5000),
)

dispatch_model.nucgd = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 5000),
)


dispatch_model.txmwh = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.TRANSMISSION_LINE,
    within=Reals,
    initialize=0,
    bounds=(-5000, 5000),
)

dispatch_model.va = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.ZONES,
    within=Reals,
    initialize=0,
    bounds=(-5000, 5000),
)

# resource specific vars

dispatch_model.sd = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    within=NonNegativeReals,
    bounds=(0, 5000),
    initialize=0,
)

dispatch_model.sc = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    within=NonNegativeReals,
    bounds=(0, 5000),
    initialize=0,
)

dispatch_model.soc = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    within=NonNegativeReals,
    bounds=(0, 5000),
    initialize=0,
)


# the following vars can make problem integer when implemented
# for now relevant constraints are unimplemented, so there is no commitment
dispatch_model.gopstat = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    within=NonNegativeReals,
    bounds=(0, 1),
    initialize=0,
)

dispatch_model.gup = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    within=NonNegativeReals,
    bounds=(0, 1),
    initialize=0,
)

dispatch_model.gdn = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    within=NonNegativeReals,
    bounds=(0, 1),
    initialize=0,
)

# new vars for competitive version of model#
# duals of MO problem
# bounds help reduce feasible space for BigM method, but they should be high enough to not bind
dispatch_model.zonalprice = Var(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.ZONES, within=Reals, initialize=0
)  # this is zonal load balance dual

dispatch_model.gensegmentmax_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.gensegmentmin_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.transmissionmax_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.TRANSMISSION_LINE,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.transmissionmin_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.TRANSMISSION_LINE,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.socmax_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.socmin_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.rampmax_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.rampmin_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.voltageanglemax_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.ZONES,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.voltageanglemin_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.ZONES,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.storagetight_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.chargemin_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.dischargemin_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.finalsoc_dual = Var(
    dispatch_model.STORAGE, within=NonNegativeReals, initialize=0, bounds=(0, 1000000),
)

dispatch_model.onecycle_dual = Var(
    dispatch_model.STORAGE, within=NonNegativeReals, initialize=0, bounds=(0, 1000000),
)

dispatch_model.bindonecycle_dual = Var(
    dispatch_model.STORAGE, within=NonNegativeReals, initialize=0, bounds=(0, 1000000),
)

dispatch_model.gendispatchmax_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.gendispatchmin_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.startupshutdown_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.nucdispatchmax_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

dispatch_model.nucdispatchmin_dual = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    within=NonNegativeReals,
    initialize=0,
    bounds=(0, 1000000),
)

# offer-related variables (since generators no longer just offer at marginal cost)
dispatch_model.gso = Var(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    within=Reals,
)

dispatch_model.go = Var(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.GENERATORS, within=Reals,
)

dispatch_model.sodischarge = Var(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.STORAGE, within=Reals
)

dispatch_model.socharge = Var(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.STORAGE, within=Reals
)


###########################
# ##### EXPRESSIONS ##### #
###########################
# build additional params or variables we'd like to record based on other param or variable values


def GeneratorDispatchRule(model, t, g):
    if g in model.UC_GENS:
        return sum(model.gsd[t, g, gs] for gs in model.GENERATORSEGMENTS)
    elif g in model.NUC_GENS:
        return model.nucgd[t, g]


dispatch_model.gd = Expression(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    rule=GeneratorDispatchRule,
)  # implement GeneratorDispatchRule


def GeneratorPminRule(model, t, g):
    return (
        model.CapacityTime[t, g]
        * model.gopstat[t, g]
        * model.Pmin[g]
        * model.ScheduledAvailable[t, g]
    )


dispatch_model.gpmin = Expression(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.UC_GENS, rule=GeneratorPminRule
)


def AvailableSegmentCapacityExpr(model, t, g, gs):
    return (
        model.GeneratorSegmentLength[t, g, gs]
        * model.CapacityTime[t, g]
        * model.ScheduledAvailable[t, g]
    )


dispatch_model.availablesegmentcapacity = Expression(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.UC_GENS,
    dispatch_model.GENERATORSEGMENTS,
    rule=AvailableSegmentCapacityExpr,
)  # implement AvailableSegmentCapacityExpr()


def CO2EmittedExpr(model, t, g, gs):
    return model.gsd[t, g, gs] * model.marginal_CO2[t, g, gs]


dispatch_model.CO2_emissions = Expression(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    dispatch_model.GENERATORSEGMENTS,
    rule=CO2EmittedExpr,
)  # implement CO2EmittedExpr


def ZoneChargeExpr(model, t, z):
    zonal_charge = 0
    for s in model.STRATEGIC_STORAGE:
        if model.StorageZoneLabel[s] == z:
            zonal_charge += model.sc[t, s]
    return zonal_charge


dispatch_model.zonalcharge = Expression(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.ZONES, rule=ZoneChargeExpr
)


def TotalDischargeExpr(model, t, s):
    if s in model.HYBRID_STORAGE:
        hybrid_dispatch = 0
        for g in model.HYBRID_GENS:
            if model.ZoneLabel[g] == model.StorageZoneLabel[s]:
                hybrid_dispatch += model.gd[t, g]
            else:
                raise ValueError(
                    "Generator is trying to be hybridized with storage that is not in its zone."
                )
        return hybrid_dispatch + model.sd[t, s]
    else:
        return model.sd[t, s]


dispatch_model.totaldischarge = Expression(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.STORAGE, rule=TotalDischargeExpr
)


def GeneratorTotalDispatchRule(model, t, g):
    if g in model.HYBRID_GENS:
        return 0
    else:
        return model.gd[t, g]


dispatch_model.totaldispatch = Expression(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.GENERATORS,
    rule=GeneratorTotalDispatchRule,
)

###########################
# ##### CONSTRAINTS ##### #
###########################

## RENEWABLES CONSTRAINTS ##


def NonUCDispatchRule(model, t, g):
    return (
        model.CapacityTime[t, g] * model.ScheduledAvailable[t, g] * model.Hours[t]
        >= model.nucgd[t, g]
    )


dispatch_model.NonUCDispatchConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.NUC_GENS, rule=NonUCDispatchRule
)


def REOfferCap(model, t, g):
    if g in model.NUC_GENS:
        return model.go[t, g] == 0.0
    else:
        return Constraint.Skip


dispatch_model.REOfferCapConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.GENERATORS, rule=REOfferCap,
)
# dispatch_model.NON_STRATEGIC_GENERATORS

## STORAGE OFFER CONSTRAINTS ##
# added by Luke 6.30.20
# will eventually want to move lower down with other upper-level offer mitigation constraints
def MitigateChargeOffer(model, t, s):
    return model.ChargeMaxOffer[t, s] >= model.socharge[t, s]


dispatch_model.MitigateChargeOfferConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.STORAGE, rule=MitigateChargeOffer
)


def MitigateDischargeOffer(model, t, s):
    return model.DischargeMaxOffer[t, s] >= model.sodischarge[t, s]


dispatch_model.MitigateDischargeOfferConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.STORAGE, rule=MitigateDischargeOffer
)


def ForceBindDischargeOffer(model, t, s):
    """This constraint should only be active in RT cases, and only when user select it to be active
    It FORCES RT offers to equal DA offers from previously run case
    """
    return model.DischargeOffer[t, s] == model.sodischarge[t, s]


dispatch_model.ForceBindDischargeOfferConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    rule=ForceBindDischargeOffer,
)


def ForceBindChargeOffer(model, t, s):
    """This constraint should only be active in RT cases, and only when user select it to be active
    It FORCES RT offers to equal DA offers from previously run case
    """
    return model.ChargeOffer[t, s] == model.socharge[t, s]


dispatch_model.ForceBindChargeOfferConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.STORAGE, rule=ForceBindChargeOffer,
)


def DischargeOfferExceedsChargeOffer(model, t, s):
    return model.sodischarge[t, s] >= model.socharge[t, s]


dispatch_model.StorageOfferConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    rule=DischargeOfferExceedsChargeOffer,
)


## STORAGE CONSTRAINTS ##
# additional constraints applied only to storage resources


def HybirdCapacityRule(model, t, s):
    for g in model.HYBRID_GENS:
        hybrid_dispatch = 0
        hybrid_capacity = 0
        if model.ZoneLabel[g] == model.StorageZoneLabel[s]:
            hybrid_dispatch += model.nucgd[t, s]
            hybrid_capacity += model.CapacityTime[t, g] * model.ScheduledAvailable[t, g]
    return hybrid_capacity * model.Hours[t] >= hybrid_dispatch + model.sd[t, s]


dispatch_model.HybirdCapacityConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.HYBRID_STORAGE,
    rule=HybirdCapacityRule,
)


def StorageTightRule(model, t, s):
    """Combine storage charge and discharge capacity limits into one, limit the possibility of charge and discharge happens simutaneously.

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        s {str} -- storage resource index
    """
    return model.ChargeMax[s] * model.DischargeMax[s] * model.Hours[t] >= (
        model.ChargeMax[s] * model.sd[t, s] + model.DischargeMax[s] * model.sc[t, s]
    )


dispatch_model.StorageTightConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.STORAGE, rule=StorageTightRule
)


def SOCChangeRule(model, t, s):
    """State of charge of storage changes based on dispatch
    this is where we should add roundtrip efficiency param when implemented

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        s {str} -- storage resource index
    """

    if t == model.FirstTimepoint[t]:
        return (
            model.soc[t, s]
            == model.SocMax[s] * 0
            + model.sc[t, s] * model.ChargeEff[s]
            - model.sd[t, s] * model.DischargeEff[s]
        )  # start half charged?
        # return model.soc[t,s] == -model.sd[t,s]
    else:
        return (
            model.soc[t, s]
            == model.soc[t - 1, s]
            + model.sc[t, s] * model.ChargeEff[s]
            - model.sd[t, s] * model.DischargeEff[s]
        )
        # return model.soc[t,s] == model.soc[t-1,s] - model.sd[t,s]


dispatch_model.SOCChangeConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.STORAGE, rule=SOCChangeRule
)  # implements SOCChangeConstraint


def BindDASOCChangeRule(model, t, s):
    if t == model.FirstTimepoint[t]:
        return (
            model.soc[t, s]
            == model.SOCInitDA[model.ACTIVETIMEPOINTS[1], s]
            - model.ChargeInitDA[model.ACTIVETIMEPOINTS[1], s] * model.ChargeEff[s]
            + model.DischargeInitDA[model.ACTIVETIMEPOINTS[1], s]
            * model.DischargeEff[s]
            + model.sc[t, s] * model.ChargeEff[s]
            - model.sd[t, s] * model.DischargeEff[s]
        )  # start half charged?
        # return model.soc[t,s] == -model.sd[t,s]
    else:
        return (
            model.soc[t, s]
            == model.soc[t - 1, s]
            + model.sc[t, s] * model.ChargeEff[s]
            - model.sd[t, s] * model.DischargeEff[s]
        )
        # return model.soc[t,s] == model.soc[t-1,s] - model.sd[t,s]


dispatch_model.BindDASOCChangeConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.STORAGE, rule=BindDASOCChangeRule
)  # implements SOCChangeConstraint


def SOCMaxRule(model, t, s):
    """Storage state of charge cannot exceed its max state of charge

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        s {str} -- storage resource index
    """

    return model.SocMax[s] >= model.soc[t, s]
    # return model.SocMax[s] >= model.soc[t, s]


dispatch_model.SOCMaxConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.STORAGE, rule=SOCMaxRule
)  # implements SOCMaxConstraint


def BindFinalSOCRule(model, s):
    """Storage state of charge in final timestep must be equal to user-defined final SOC value
    I've input this to be 0 for now. This will at least make each day symmetric (0 initial and final SOC)

    Arguments:
        model -- Pyomo model
        s {str} -- storage resource index
    """

    tmp_soc = 0
    for tp in range(model.ACTIVETIMEPOINTS[1], model.ACTIVETIMEPOINTS[-1] + 1):
        tmp_soc += (
            model.sc[tp, s] * model.ChargeEff[s]
            - model.sd[tp, s] * model.DischargeEff[s]
        )
    return model.SocMax[s] * -0 == tmp_soc


dispatch_model.BindFinalSOCConstraint = Constraint(
    dispatch_model.STORAGE, rule=BindFinalSOCRule
)  # implements BindFinalSOCConstraint


def BindDAFinalSOCRule(model, s):
    tmp_soc = 0
    for tp in range(model.ACTIVETIMEPOINTS[1], model.ACTIVETIMEPOINTS[-1] + 1):
        tmp_soc += (
            model.sc[tp, s] * model.ChargeEff[s]
            - model.sd[tp, s] * model.DischargeEff[s]
        )
    return (
        model.SOCInitDA[model.ACTIVETIMEPOINTS[-1], s]
        - model.SOCInitDA[model.ACTIVETIMEPOINTS[1], s]
        + model.ChargeInitDA[model.ACTIVETIMEPOINTS[1], s] * model.ChargeEff[s]
        - model.DischargeInitDA[model.ACTIVETIMEPOINTS[1], s] * model.DischargeEff[s]
        == tmp_soc
    )


dispatch_model.BindDAFinalSOCConstraint = Constraint(
    dispatch_model.STORAGE, rule=BindDAFinalSOCRule
)


def OneCycleRule(model, s):
    return model.SocMax[s] >= sum(model.sd[t, s] for t in model.ACTIVETIMEPOINTS)


dispatch_model.OneCycleConstraint = Constraint(
    dispatch_model.STORAGE, rule=OneCycleRule
)


def BindDAOneCycleRule(model, s):
    print(sum(model.DischargeInitDA[t, s] for t in model.ACTIVETIMEPOINTS))
    print(sum(model.DischargeInitDA[t, s] for t in model.ACTIVETIMEPOINTS) / 12)

    return sum(model.DischargeInitDA[t, s] for t in model.ACTIVETIMEPOINTS) / 12 == sum(
        model.sd[t, s] for t in model.ACTIVETIMEPOINTS
    )


dispatch_model.BindDAOneCycleConstraint = Constraint(
    dispatch_model.STORAGE, rule=BindDAOneCycleRule
)


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
    return (
        model.txmwh[t, line] >= model.TransmissionFromCapacity[t, line] * model.Hours[t]
    )


dispatch_model.TxFromConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.TRANSMISSION_LINE, rule=TxFromRule
)  # implements TxFromConstraint


def TxToRule(model, t, line):
    """Real power flow on line must be less than to capacity
    (note to capacity is positive by convention in the model

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        line {str} -- transmission line index
    """
    return (
        model.TransmissionToCapacity[t, line] * model.Hours[t] >= model.txmwh[t, line]
    )


dispatch_model.TxToConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.TRANSMISSION_LINE, rule=TxToRule
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
    return model.VoltageAngleMax[z] >= model.va[t, z]


dispatch_model.VoltageAngleMaxConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.ZONES, rule=VoltageAngleMaxRule
)  # implements VoltageAngleMaxConstraint


def VoltageAngleMinRule(model, t, z):
    """Bus voltage angle must be greater than min bus voltage angle

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        z {int} -- bus index (z by convention holdover from old zonal model)
    """
    return model.va[t, z] >= model.VoltageAngleMin[z]


dispatch_model.VoltageAngleMinConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
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
    if z == model.ReferenceBus[t]:
        return model.va[t, z] == 0
    else:
        return Constraint.Skip


dispatch_model.SetReferenceBusConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
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
    zone_to = model.TransmissionTo[t, line]
    zone_from = model.TransmissionFrom[t, line]
    return model.txmwh[t, line] == model.Hours[t] * model.Susceptance[line] * (
        model.va[t, zone_to] - model.va[t, zone_from]
    )


dispatch_model.DCOPFConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.TRANSMISSION_LINE, rule=DCOPFRule
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
        if model.TransmissionTo[t, line] == z or model.TransmissionFrom[t, line] == z:
            if model.TransmissionTo[t, line] == z:
                imports_exports += model.txmwh[t, line]
            elif model.TransmissionFrom[t, line] == z:
                imports_exports -= model.txmwh[t, line]
            # add additional note to dec import/exports by line losses
            # no, this will just be done as a hurdle rate
    for g in model.GENERATORS:
        if model.ZoneLabel[g] == z:
            zonal_generation += model.totaldispatch[t, g]
    for s in model.STORAGE:
        if model.StorageZoneLabel[s] == z:
            zonal_storage += model.totaldischarge[t, s]
            zonal_storage -= model.sc[t, s]
            # zonal_storage += model.sd[t, s]
    # full constraint, with tx flow now
    # (sum(sum(model.gsd[t,g,z,gs] for gs in model.GENERATORSEGMENTS) for g in model.GENERATORS)+\
    return (
        zonal_generation + imports_exports + zonal_storage
        == model.GrossLoad[t, z] * model.Hours[t]
    )


dispatch_model.LoadConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.ZONES, rule=LoadRule
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
        model.CapacityTime[t, g]
        * model.gopstat[t, g]
        * model.ScheduledAvailable[t, g]
        * model.Hours[t]
        >= model.gd[t, g]
    )


dispatch_model.CapacityMaxConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.UC_GENS, rule=CapacityMaxRule
)

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
    return model.gd[t, g] >= model.gpmin[t, g] * model.Hours[t]


dispatch_model.PminConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.UC_GENS, rule=CapacityMinRule
)

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
    return (
        model.availablesegmentcapacity[t, g, gs] * model.Hours[t] >= model.gsd[t, g, gs]
    )


dispatch_model.GeneratorSegmentMaxConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.UC_GENS,
    dispatch_model.GENERATORSEGMENTS,
    rule=GeneratorSegmentDispatchMax,
)  # implements GeneratorSegmentMaxConstraint

### GENERATOR RAMP ###
# these are currently inactive but may be a helfpul template for implementing TRUC


def GeneratorRampUpRule(model, t, g):
    """ Increase in generator dispatch between ACTIVETIMEPOINTS cannot exceed upward ramp rate
    Note this isn't implemented in the first timepoint, so any initialization is allowed
    There are a couple ways around this, one of the more common/simple is looping the day back on itself

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
    """
    if t == model.FirstTimepoint[t]:
        return Constraint.Skip
    else:
        return (
            model.gd[t - 1, g]
            - model.gpmin[t - 1, g] * model.Hours[t]
            + model.RampRate[g] * model.gopstat[t - 1, g] * model.Hours[t]
            >= model.gd[t, g] - model.gpmin[t, g] * model.Hours[t]
        )


dispatch_model.GeneratorRampUpConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.UC_GENS, rule=GeneratorRampUpRule
)


def GeneratorRampDownRule(model, t, g):
    """ Decrease in generator dispatch between ACTIVETIMEPOINTS cannot exceed downward ramp rate
    Note this isn't implemented in the first timepoint, so any initialization is allowed
    There are a couple ways around this, one of the more common/simple is looping the day back on itself

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
    """
    if t == model.FirstTimepoint[t]:
        return Constraint.Skip
    else:
        return (
            model.gd[t, g] - model.gpmin[t, g] * model.Hours[t]
            >= model.gd[t - 1, g]
            - model.gpmin[t - 1, g] * model.Hours[t]
            - model.RampRate[g] * model.gopstat[t, g] * model.Hours[t]
        )


dispatch_model.GeneratorRampDownConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.UC_GENS, rule=GeneratorRampDownRule
)


## GENERATOR STARTUP/SHUTDOWN ##
def GeneratorStartupShutdownRule(model, t, g):
    """ Generator segment dispatch cannot exceed (available) capacity on segment

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        g {str} -- generator index
        gs {int} -- generator segment index
    """
    if t == model.FirstTimepoint[t]:
        return Constraint.Skip
    else:
        return (
            model.gopstat[t, g] - model.gopstat[t - 1, g]
            == model.gup[t, g] - model.gdn[t, g]
        )


dispatch_model.GeneratorStartupShutdownConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.UC_GENS,
    rule=GeneratorStartupShutdownRule,
)


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
        model.gso[t, g, gs]
        + model.gensegmentmax_dual[t, g, gs]
        - model.gensegmentmin_dual[t, g, gs]
        + model.gendispatchmax_dual[t, g]
        - model.gendispatchmin_dual[t, g]
        + model.rampmax_dual[t, g]
        - model.rampmin_dual[t, g]
        - model.zonalprice[t, model.ZoneLabel[g]]
        == 0
    )


dispatch_model.GeneratorOfferDualConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.UC_GENS,
    dispatch_model.GENERATORSEGMENTS,
    rule=BindGeneratorOfferDual,
    name="OfferDual",
)  # implements GeneratorOfferDualConstraint


def BindNonUCDual(model, t, g):
    return (
        model.go[t, g]
        + model.nucdispatchmax_dual[t, g]
        - model.nucdispatchmin_dual[t, g]
        - model.zonalprice[t, model.ZoneLabel[g]]
        == 0
    )


dispatch_model.NonUCDualConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.NUC_GENS, rule=BindNonUCDual,
)


def BindStorageDischargeDual(model, t, s):
    """ Duals associated with storage dispatch max, min, and load balance constraints equal offer

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        s {str} -- storage index
    """
    return (
        model.sodischarge[t, s]
        + model.ChargeMax[s] * model.storagetight_dual[t, s]
        - model.dischargemin_dual[t, s]
        - sum(
            model.DischargeEff[s]
            * (model.socmax_dual[tp, s] - model.socmin_dual[tp, s])
            for tp in range(t, model.ACTIVETIMEPOINTS[-1] + 1)
        )
        + model.onecycle_dual[s]
        + model.DischargeEff[s] * model.finalsoc_dual[s]
        - model.zonalprice[t, model.StorageZoneLabel[s]]
        == 0
    )


# + model.onecycle_dual[s]

dispatch_model.StorageDischargeDualConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STRATEGIC_STORAGE,
    rule=BindStorageDischargeDual,
    name="StorageDischargeDual",
)


def RTBindStorageDischargeDual(model, t, s):
    """ Duals associated with storage dispatch max, min, and load balance constraints equal offer

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        s {str} -- storage index
    """
    return (
        model.sodischarge[t, s]
        + model.ChargeMax[s] * model.storagetight_dual[t, s]
        - model.dischargemin_dual[t, s]
        - sum(
            model.DischargeEff[s]
            * (model.socmax_dual[tp, s] - model.socmin_dual[tp, s])
            for tp in range(t, model.ACTIVETIMEPOINTS[-1] + 1)
        )
        + model.onecycle_dual[s]
        + model.bindonecycle_dual[s]
        + model.DischargeEff[s] * model.finalsoc_dual[s]
        - model.zonalprice[t, model.StorageZoneLabel[s]]
        == 0
    )


# + model.onecycle_dual[s]

dispatch_model.RTStorageDischargeDualConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STRATEGIC_STORAGE,
    rule=RTBindStorageDischargeDual,
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
        -model.socharge[t, s]
        + model.DischargeMax[s] * model.storagetight_dual[t, s]
        - model.chargemin_dual[t, s]
        + sum(
            model.ChargeEff[s] * (model.socmax_dual[tp, s] - model.socmin_dual[tp, s])
            for tp in range(t, model.ACTIVETIMEPOINTS[-1] + 1)
        )
        - model.ChargeEff[s] * model.finalsoc_dual[s]
        + model.zonalprice[t, model.StorageZoneLabel[s]]
        == 0
    )


dispatch_model.StorageChargeDualConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STRATEGIC_STORAGE,
    rule=BindStorageChargeDual,
    name="StorageChargeDual",
)


def BindNSStorageDischargeDual(model, t, s):

    return (
        model.ChargeMax[s] * model.storagetight_dual[t, s]
        - model.dischargemin_dual[t, s]
        - sum(
            model.DischargeEff[s]
            * (model.socmax_dual[tp, s] - model.socmin_dual[tp, s])
            for tp in range(t, model.ACTIVETIMEPOINTS[-1] + 1)
        )
        + model.onecycle_dual[s]
        + model.DischargeEff[s] * model.finalsoc_dual[s]
        - model.zonalprice[t, model.StorageZoneLabel[s]]
        == 0
    )


# + model.onecycle_dual[s]

dispatch_model.StorageNSDischargeDualConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.NON_STRATEGIC_STORAGE,
    rule=BindNSStorageDischargeDual,
    name="StorageNSDischargeDual",
)


def RTBindNSStorageDischargeDual(model, t, s):
    return (
        model.ChargeMax[s] * model.storagetight_dual[t, s]
        - model.dischargemin_dual[t, s]
        - sum(
            model.DischargeEff[s]
            * (model.socmax_dual[tp, s] - model.socmin_dual[tp, s])
            for tp in range(t, model.ACTIVETIMEPOINTS[-1] + 1)
        )
        + model.onecycle_dual[s]
        + model.bindonecycle_dual[s]
        + model.DischargeEff[s] * model.finalsoc_dual[s]
        - model.zonalprice[t, model.StorageZoneLabel[s]]
        == 0
    )


# + model.onecycle_dual[s]

dispatch_model.RTStorageNSDischargeDualConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.NON_STRATEGIC_STORAGE,
    rule=RTBindNSStorageDischargeDual,
    name="StorageNSDischargeDual",
)


def BindNSStorageChargeDual(model, t, s):
    return (
        model.DischargeMax[s] * model.storagetight_dual[t, s]
        - model.chargemin_dual[t, s]
        + sum(
            model.ChargeEff[s] * (model.socmax_dual[tp, s] - model.socmin_dual[tp, s])
            for tp in range(t, model.ACTIVETIMEPOINTS[-1] + 1)
        )
        - model.ChargeEff[s] * model.finalsoc_dual[s]
        + model.zonalprice[t, model.StorageZoneLabel[s]]
        == 0
    )


dispatch_model.StorageNSChargeDualConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.NON_STRATEGIC_STORAGE,
    rule=BindNSStorageChargeDual,
    name="StorageNSChargeDual",
)


def BindStartupDual(model, t, g):
    return model.StartCost[g] - model.startupshutdown_dual[t, g] == 0


dispatch_model.StartupDualConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.UC_GENS, rule=BindStartupDual
)


def BindComittmentDual(model, t, g):
    return (
        model.NoLoadCost[g]
        - model.ScheduledAvailable[t, g]
        * model.CapacityTime[t, g]
        * model.gendispatchmax_dual[t, g]
        * model.Hours[t]
        + model.ScheduledAvailable[t, g]
        * model.CapacityTime[t, g]
        * model.gendispatchmin_dual[t, g]
        * model.Pmin[g]
        * model.Hours[t]
        + model.startupshutdown_dual[t, g]
        - (
            model.RampRate[g]
            + model.Pmin[g] * model.ScheduledAvailable[t, g] * model.CapacityTime[t, g]
        )
        * model.rampmax_dual[t, g]
        * model.Hours[t]
        - (
            model.RampRate[g]
            - model.Pmin[g] * model.ScheduledAvailable[t, g] * model.CapacityTime[t, g]
        )
        * model.rampmin_dual[t, g]
        * model.Hours[t]
        == 0
    )


dispatch_model.CommitmentConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.UC_GENS, rule=BindComittmentDual
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

        if model.TransmissionTo[t, line] == z:
            sink_zone = model.TransmissionFrom[t, line]
            # if t==1:
            #    print(sink_zone)
            maxdual += (
                model.Susceptance[line] * model.transmissionmax_dual[t, line]
                + model.voltageanglemax_dual[t, z]
            )
            mindual += (
                model.Susceptance[line] * model.transmissionmin_dual[t, line]
                + model.voltageanglemin_dual[t, z]
            )
            lmp_delta += model.Susceptance[line] * model.zonalprice[t, z]
            lmp_delta -= model.Susceptance[line] * model.zonalprice[t, sink_zone]

        elif model.TransmissionFrom[t, line] == z:
            sink_zone = model.TransmissionTo[t, line]
            # if t==1:
            #    print(sink_zone)
            maxdual -= (
                model.Susceptance[line] * model.transmissionmax_dual[t, line]
                + model.voltageanglemax_dual[t, z]
            )
            mindual -= (
                model.Susceptance[line] * model.transmissionmin_dual[t, line]
                + model.voltageanglemin_dual[t, z]
            )
            lmp_delta += model.Susceptance[line] * model.zonalprice[t, z]
            lmp_delta -= model.Susceptance[line] * model.zonalprice[t, sink_zone]
    # if t==8 and z==1:
    #    print(maxdual,mindual,lmp_delta)
    return maxdual - mindual == lmp_delta


dispatch_model.FlowDualConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.ZONES,
    rule=BindFlowDual,
    name="FlowDual",
)  # implements FlowDualConstraint

### COMPLEMENTARITY CONSTRAINTS ###
# these are needed for MPEC reformulation
# the upshot is at least one of the two complements must bind (i.e., be an equality)
# this ends up properly constraining the dual variables, though you may prefer to understand it in math


def BindStorageTightComplementarity(model, t, s):

    mysum = (
        model.DischargeMax[s] * model.ChargeMax[s] * model.Hours[t]
        - model.DischargeMax[s] * model.sc[t, s]
        - model.ChargeMax[s] * model.sd[t, s]
    )
    return complements(mysum >= 0, model.storagetight_dual[t, s] >= 0,)


dispatch_model.StorageTightComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    rule=BindStorageTightComplementarity,
)


def BindStorageChargeMinComplementarity(model, t, s):
    return complements(model.sc[t, s] >= 0, model.chargemin_dual[t, s] >= 0,)


dispatch_model.StorageChargeMinComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    rule=BindStorageChargeMinComplementarity,
)


def BindStorageDischargeMinComplementarity(model, t, s):
    return complements(model.sd[t, s] >= 0, model.dischargemin_dual[t, s] >= 0,)


dispatch_model.StorageDischargeMinComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    rule=BindStorageDischargeMinComplementarity,
)

"""
def DischargeMinDual(model,t,s):
    return 1000.*(1.-model.sdbool[t,s]) >= model.sd[t,s] 
dispatch_model.DischargeDualConstraint = Constraint(dispatch_model.ACTIVETIMEPOINTS, dispatch_model.STORAGE,rule=DischargeDual)


def DischargeDual(model,t,s):
    return 1000.*model.sdbool[t,s] >= model.sd[t,s] 
dispatch_model.DischargeDualConstraint = Constraint(dispatch_model.ACTIVETIMEPOINTS, dispatch_model.STORAGE,rule=DischargeDual)
"""


def BindMaxStorageComplementarity(model, t, s):
    return complements(
        model.SocMax[s] - model.soc[t, s] >= 0, model.socmax_dual[t, s] >= 0,
    )


dispatch_model.MaxStorageComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    rule=BindMaxStorageComplementarity,
)


def BindMinStorageComplementarity(model, t, s):
    return complements(model.soc[t, s] >= 0, model.socmin_dual[t, s] >= 0,)


dispatch_model.MinStorageComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    rule=BindMinStorageComplementarity,
)


def RTBindMaxStorageComplementarity(model, t, s):
    tmp_soc = 0
    for tp in range(model.ACTIVETIMEPOINTS[1], t + 1):
        tmp_soc += (
            model.sc[tp, s] * model.ChargeEff[s]
            - model.sd[tp, s] * model.DischargeEff[s]
        )
    return complements(
        model.SocMax[s]
        - model.SOCInitDA[model.ACTIVETIMEPOINTS[1], s]
        + model.ChargeInitDA[model.ACTIVETIMEPOINTS[1], s] * model.ChargeEff[s]
        - model.DischargeInitDA[model.ACTIVETIMEPOINTS[1], s]
        - tmp_soc
        >= 0,
        model.socmax_dual[t, s] >= 0,
    )


dispatch_model.RTMaxStorageComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    rule=RTBindMaxStorageComplementarity,
)


def RTBindMinStorageComplementarity(model, t, s):
    tmp_soc = 0
    for tp in range(model.ACTIVETIMEPOINTS[1], t + 1):
        tmp_soc += (
            model.sc[tp, s] * model.ChargeEff[s]
            - model.sd[tp, s] * model.DischargeEff[s]
        )
    return complements(
        model.SOCInitDA[model.ACTIVETIMEPOINTS[1], s]
        - model.ChargeInitDA[model.ACTIVETIMEPOINTS[1], s] * model.ChargeEff[s]
        + model.DischargeInitDA[model.ACTIVETIMEPOINTS[1], s]
        + tmp_soc
        >= 0,
        model.socmin_dual[t, s] >= 0,
    )


dispatch_model.RTMinStorageComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.STORAGE,
    rule=RTBindMinStorageComplementarity,
)


def BindOneCycleComplementarity(model, s):
    return complements(
        model.SocMax[s] - sum(model.sd[t, s] for t in model.ACTIVETIMEPOINTS) >= 0,
        model.onecycle_dual[s] >= 0,
    )


dispatch_model.OneCycleComplementarity = Complementarity(
    dispatch_model.STORAGE, rule=BindOneCycleComplementarity,
)


def BindMaxTransmissionComplementarity(model, t, line):
    """ Transmission line power flow is either (1) at its max, or (2) dual is zero (or both)
    Upshot: transmissionmax_dual can only be nonzero when line flow is at its max

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        line {str} -- transmission line index
    """
    return complements(
        model.TransmissionToCapacity[t, line] * model.Hours[t] - model.txmwh[t, line]
        >= 0,
        model.transmissionmax_dual[t, line] >= 0,
    )


dispatch_model.MaxTransmissionComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.TRANSMISSION_LINE,
    rule=BindMaxTransmissionComplementarity,
)  # implements MaxTransmissionComplementarity


def BindMinTransmissionComplementarity(model, t, line):
    """ Transmission line power flow is either (1) at its min, or (2) min dual is zero (or both)
    Upshot: transmissionmin_dual can only be nonzero when line flow is at its min (max in from/negative direction)

    Arguments:
        model -- Pyomo model
        t {int} -- timepoint index
        line {str} -- transmission line index
    """
    return complements(
        -model.TransmissionFromCapacity[t, line] * model.Hours[t] + model.txmwh[t, line]
        >= 0,
        model.transmissionmin_dual[t, line] >= 0,
    )


dispatch_model.MinTransmissionComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.TRANSMISSION_LINE,
    rule=BindMinTransmissionComplementarity,
)  # implements MinTransmissionComplementarity


def BindMaxVoltageAngleComplementarity(model, t, z):
    return complements(
        model.VoltageAngleMax[z] - model.va[t, z] >= 0,
        model.voltageanglemax_dual[t, z] >= 0,
    )


dispatch_model.MaxVoltageAngleComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.ZONES,
    rule=BindMaxVoltageAngleComplementarity,
)


def BindMinVoltageAngleComplementarity(model, t, z):
    return complements(
        model.va[t, z] - model.VoltageAngleMin[z] >= 0,
        model.voltageanglemin_dual[t, z] >= 0,
    )


dispatch_model.MinVoltageAngleComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.ZONES,
    rule=BindMinVoltageAngleComplementarity,
)


def BindMaxDispatchComplementarity(model, t, g):
    return complements(
        model.CapacityTime[t, g]
        * model.gopstat[t, g]
        * model.ScheduledAvailable[t, g]
        * model.Hours[t]
        - model.gd[t, g]
        >= 0,
        model.gendispatchmax_dual[t, g] >= 0,
    )


dispatch_model.MaxDispatchComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.UC_GENS,
    rule=BindMaxDispatchComplementarity,
)  # implements MaxDispatchComplementarity


def BindMinDispatchComplementarity(model, t, g):
    return complements(
        model.gd[t, g] - model.gpmin[t, g] * model.Hours[t] >= 0,
        model.gendispatchmin_dual[t, g] >= 0,
    )


dispatch_model.MinDispatchComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.UC_GENS,
    rule=BindMinDispatchComplementarity,
)  # implements MinDispatchComplementarity


def BindMaxSegmentComplementarity(model, t, g, gs):
    return complements(
        model.availablesegmentcapacity[t, g, gs] * model.Hours[t] - model.gsd[t, g, gs]
        >= 0,
        model.gensegmentmax_dual[t, g, gs] >= 0,
    )


dispatch_model.MaxSegmentComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.UC_GENS,
    dispatch_model.GENERATORSEGMENTS,
    rule=BindMaxSegmentComplementarity,
)


def BindMinSegmentComplementarity(model, t, g, gs):
    return complements(
        model.gsd[t, g, gs] >= 0, model.gensegmentmin_dual[t, g, gs] >= 0,
    )


dispatch_model.MinSegmentComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.UC_GENS,
    dispatch_model.GENERATORSEGMENTS,
    rule=BindMinSegmentComplementarity,
)


def BindMaxRampComplementarity(model, t, g):
    if t == model.FirstTimepoint[t]:
        return Complementarity.Skip
    else:
        return complements(
            model.gd[t - 1, g]
            - model.gpmin[t - 1, g] * model.Hours[t]
            + model.RampRate[g] * model.gopstat[t - 1, g] * model.Hours[t]
            - model.gd[t, g]
            + model.gpmin[t, g] * model.Hours[t]
            >= 0,
            model.rampmax_dual[t, g] >= 0,
        )


dispatch_model.MaxRampComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.UC_GENS,
    rule=BindMaxRampComplementarity,
)


def BindMinRampComplementarity(model, t, g):
    # INACTIVE: I never finished writing this and integrating it in the objective(s)
    if t == model.FirstTimepoint[t]:
        return Complementarity.Skip
    else:
        return complements(
            model.gd[t, g]
            - model.gpmin[t, g] * model.Hours[t]
            - model.gd[t - 1, g]
            + model.gpmin[t - 1, g] * model.Hours[t]
            + model.RampRate[g] * model.gopstat[t, g] * model.Hours[t]
            >= 0,
            model.rampmin_dual[t, g] >= 0,
        )


dispatch_model.MinRampComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.UC_GENS,
    rule=BindMinRampComplementarity,
)


def BindMaxNonUCComplementarity(model, t, g):
    return complements(
        model.CapacityTime[t, g] * model.ScheduledAvailable[t, g] * model.Hours[t]
        - model.nucgd[t, g]
        >= 0,
        model.nucdispatchmax_dual[t, g] >= 0,
    )


dispatch_model.MaxNonUCComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.NUC_GENS,
    rule=BindMaxNonUCComplementarity,
)


def BindMinNonUCComplementarity(model, t, g):
    return complements(model.nucgd[t, g] >= 0, model.nucdispatchmin_dual[t, g] >= 0,)


dispatch_model.MinNonUCComplementarity = Complementarity(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.NUC_GENS,
    rule=BindMinNonUCComplementarity,
)


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
        return model.gso[t, g, gs] >= model.gso[t, g, gs - 1]


dispatch_model.IncreasingOfferCurveConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.UC_GENS,
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
    if g in model.UC_GENS:
        return model.previous_offer[t, g, gs] * 2 >= model.gso[t, g, gs]
    else:
        return Constraint.Skip


dispatch_model.OfferCapConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
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
    if g in model.UC_GENS:
        return (
            model.previous_offer[t, g, gs] >= model.gso[t, g, gs]
        )  # caps offer at cost to make it param
    else:
        return Constraint.Skip


dispatch_model.OfferCap2Constraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
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
        model.gso[t, g, gs] >= model.GeneratorMarginalCost[t, g, gs]
    )  # must offer at least marginal cost


dispatch_model.OfferMinConstraint = Constraint(
    dispatch_model.ACTIVETIMEPOINTS,
    dispatch_model.UC_GENS,
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
    dispatch_model.ACTIVETIMEPOINTS, dispatch_model.ZONES, rule=MarketPriceCap
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
                    model.gsd[t, g, gs] * model.GeneratorMarginalCost[t, g, gs]
                    for t in model.ACTIVETIMEPOINTS
                )
                for g in model.GENERATORS
            )
            for gs in model.GENERATORSEGMENTS
        )
        + sum(
            sum(model.gopstat[t, g] for t in model.ACTIVETIMEPOINTS)
            * model.NoLoadCost[g]
            for g in model.GENERATORS
        )
        + sum(
            sum(model.gup[t, g] for t in model.ACTIVETIMEPOINTS) * model.StartCost[g]
            for g in model.GENERATORS
        )
        + sum(
            sum(model.sc[t, s] for t in model.ACTIVETIMEPOINTS) for s in model.STORAGE
        )  # charge penalty
    )

    # DESCRIPTION OF OBJECTIVE
    # (1) dispatch cost
    # (2) no load cost of committed gen
    # (3) start up costs when generators brought online
    # (4) penalty for storage charging


dispatch_model.TotalCost = Objective(rule=objective_rule, sense=minimize)


def objective_rule2(model):
    """system operator dispatch-only objective (considers only marginal cost of generators)
    Arguments:
        model  -- Pyomo model
    """
    return sum(
        sum(
            sum(
                model.gsd[t, g, gs] * model.GeneratorMarginalCost[t, g, gs]
                for t in model.ACTIVETIMEPOINTS
            )
            for g in model.GENERATORS
        )
        for gs in model.GENERATORSEGMENTS
    ) + sum(
        sum(
            sum(
                model.gsd[t, g, gs]
                * model.marginal_CO2[t, g, gs]
                * model.CO2_damage[t, g, gs]
                for t in model.ACTIVETIMEPOINTS
            )
            for g in model.GENERATORS
        )
        for gs in model.GENERATORSEGMENTS
    )
    -sum(
        sum(model.sc[t, s] for t in model.ACTIVETIMEPOINTS) for s in model.STORAGE
    )  # charge penalty
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
        sum(model.GrossLoad[t, z] * model.zonalprice[t, z] for z in model.ZONES)
        for t in model.ACTIVETIMEPOINTS
    ) - sum(
        sum(
            sum(
                model.gsd[t, g, gs] * model.GeneratorMarginalCost[t, g, gs]
                for t in model.ACTIVETIMEPOINTS
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
                    * model.Hours[t]
                    * model.gensegmentmax_dual[t, g, gs]
                    for t in model.ACTIVETIMEPOINTS
                )
                for g in model.NON_STRATEGIC_GENERATORS.intersection(model.UC_GENS)
            )
            for gs in model.GENERATORSEGMENTS
        )
        + sum(
            sum(
                -model.CapacityTime[t, g]
                * model.ScheduledAvailable[t, g]
                * model.nucdispatchmax_dual[t, g]
                * model.Hours[t]
                for t in model.ACTIVETIMEPOINTS
            )
            for g in model.NON_STRATEGIC_GENERATORS
        )
        - sum(
            sum(
                model.DischargeMax[s]
                * model.ChargeMax[s]
                * model.storagetight_dual[t, s]
                for t in model.ACTIVETIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                (model.SocMax[s] - model.SocMax[s] * 0) * model.socmax_dual[t, s]
                for t in model.ACTIVETIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                model.SocMax[s] * 0 * model.socmin_dual[t, s]
                for t in model.ACTIVETIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            model.SocMax[s] * -0 * model.finalsoc_dual[s]
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            model.SocMax[s] * model.onecycle_dual[s]
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                sum(
                    model.gsd[t, g, gs] * model.GeneratorMarginalCost[t, g, gs]
                    for t in model.ACTIVETIMEPOINTS
                )
                for g in model.UC_GENS
            )
            for gs in model.GENERATORSEGMENTS
        )
        - sum(
            sum(model.gopstat[t, g] for t in model.ACTIVETIMEPOINTS)
            * model.NoLoadCost[g]
            for g in model.UC_GENS
        )
        - sum(
            sum(model.gup[t, g] for t in model.ACTIVETIMEPOINTS) * model.StartCost[g]
            for g in model.UC_GENS
        )
        - sum(
            sum(
                model.TransmissionToCapacity[t, line]
                * model.Hours[t]
                * model.transmissionmax_dual[t, line]
                for t in model.ACTIVETIMEPOINTS
            )
            for line in model.TRANSMISSION_LINE
        )
        + sum(
            sum(
                model.TransmissionFromCapacity[t, line]
                * model.Hours[t]
                * model.transmissionmin_dual[t, line]
                for t in model.ACTIVETIMEPOINTS
            )
            for line in model.TRANSMISSION_LINE
        )
        - sum(
            sum(
                model.VoltageAngleMax[z] * model.voltageanglemax_dual[t, z]
                for t in model.ACTIVETIMEPOINTS
            )
            for z in model.ZONES
        )
        + sum(
            sum(
                model.VoltageAngleMin[z] * model.voltageanglemin_dual[t, z]
                for t in model.ACTIVETIMEPOINTS
            )
            for z in model.ZONES
        )
        + sum(
            sum(
                (model.GrossLoad[t, z] * model.Hours[t]) * model.zonalprice[t, z]
                for t in model.ACTIVETIMEPOINTS
            )
            for z in model.ZONES
        )
        #        - sum(
        #            sum(model.sc[t, s] for t in model.ACTIVETIMEPOINTS) for s in model.STORAGE
        #        )  # charge penalty
    )


dispatch_model.GeneratorProfitDual = Objective(
    rule=objective_profit_dual, sense=maximize
)
# - model.zonalcharge[t, z]


def objective_profit_dual_pre(model):
    """Pre-processing version of full objective for MPEC reformulated as MIP using BigM
    Only difference is iteration over *ALL* generators rather than just non-strategic generators
    for the first term in the objective (involving gensegmentmax_dual). This just forces
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
                    * model.Hours[t]
                    * model.gensegmentmax_dual[t, g, gs]
                    for t in model.ACTIVETIMEPOINTS
                )
                for g in model.UC_GENS
            )
            for gs in model.GENERATORSEGMENTS
        )
        + sum(
            sum(
                -model.CapacityTime[t, g]
                * model.ScheduledAvailable[t, g]
                * model.nucdispatchmax_dual[t, g]
                * model.Hours[t]
                for t in model.ACTIVETIMEPOINTS
            )
            for g in model.NUC_GENS
        )
        - sum(
            sum(
                model.DischargeMax[s]
                * model.ChargeMax[s]
                * model.storagetight_dual[t, s]
                for t in model.ACTIVETIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                (model.SocMax[s] - model.SocMax[s] * 0) * model.socmax_dual[t, s]
                for t in model.ACTIVETIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                model.SocMax[s] * 0 * model.socmin_dual[t, s]
                for t in model.ACTIVETIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            model.SocMax[s] * -0 * model.finalsoc_dual[s]
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            model.SocMax[s] * model.onecycle_dual[s]
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                sum(
                    model.gsd[t, g, gs] * model.GeneratorMarginalCost[t, g, gs]
                    for t in model.ACTIVETIMEPOINTS
                )
                for g in model.UC_GENS
            )
            for gs in model.GENERATORSEGMENTS
        )
        - sum(
            sum(model.gopstat[t, g] for t in model.ACTIVETIMEPOINTS)
            * model.NoLoadCost[g]
            for g in model.UC_GENS
        )
        - sum(
            sum(model.gup[t, g] for t in model.ACTIVETIMEPOINTS) * model.StartCost[g]
            for g in model.UC_GENS
        )
        - sum(
            sum(
                model.TransmissionToCapacity[t, line]
                * model.Hours[t]
                * model.transmissionmax_dual[t, line]
                for t in model.ACTIVETIMEPOINTS
            )
            for line in model.TRANSMISSION_LINE
        )
        + sum(
            sum(
                model.TransmissionFromCapacity[t, line]
                * model.Hours[t]
                * model.transmissionmin_dual[t, line]
                for t in model.ACTIVETIMEPOINTS
            )
            for line in model.TRANSMISSION_LINE
        )
        - sum(
            sum(
                model.VoltageAngleMax[z] * model.voltageanglemax_dual[t, z]
                for t in model.ACTIVETIMEPOINTS
            )
            for z in model.ZONES
        )
        + sum(
            sum(
                model.VoltageAngleMin[z] * model.voltageanglemin_dual[t, z]
                for t in model.ACTIVETIMEPOINTS
            )
            for z in model.ZONES
        )
        + sum(
            sum(
                (model.GrossLoad[t, z] * model.Hours[t]) * model.zonalprice[t, z]
                for t in model.ACTIVETIMEPOINTS
            )
            for z in model.ZONES
        )
        #        - sum(
        #            sum(model.sc[t, s] for t in model.ACTIVETIMEPOINTS) for s in model.STORAGE
        #        )  # charge penalty
    )


dispatch_model.GeneratorProfitDualPre = Objective(
    rule=objective_profit_dual_pre, sense=maximize
)
# - model.zonalcharge[t, z]


def rt_objective_profit_dual(model):
    """Full objective for MPEC reformulated as MIP using BigM

    Arguments:
        model  -- Pyomo model
    """

    return (
        sum(
            sum(
                sum(
                    -model.availablesegmentcapacity[t, g, gs]
                    * model.Hours[t]
                    * model.gensegmentmax_dual[t, g, gs]
                    for t in model.ACTIVETIMEPOINTS
                )
                for g in model.NON_STRATEGIC_GENERATORS.intersection(model.UC_GENS)
            )
            for gs in model.GENERATORSEGMENTS
        )
        + sum(
            sum(
                -model.CapacityTime[t, g]
                * model.ScheduledAvailable[t, g]
                * model.nucdispatchmax_dual[t, g]
                * model.Hours[t]
                for t in model.ACTIVETIMEPOINTS
            )
            for g in model.NON_STRATEGIC_GENERATORS
        )
        - sum(
            sum(
                model.DischargeMax[s]
                * model.ChargeMax[s]
                * model.Hours[t]
                * model.storagetight_dual[t, s]
                for t in model.ACTIVETIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                (
                    model.SocMax[s]
                    - (
                        model.SOCInitDA[model.ACTIVETIMEPOINTS[1], s]
                        + model.ChargeInitDA[model.ACTIVETIMEPOINTS[1], s]
                        * model.ChargeEff[s]
                        - model.DischargeInitDA[model.ACTIVETIMEPOINTS[1], s]
                        * model.DischargeEff[s]
                    )
                    * 0.0
                )
                * model.socmax_dual[t, s]
                for t in model.ACTIVETIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                abs(
                    (
                        model.SOCInitDA[model.ACTIVETIMEPOINTS[1], s]
                        - model.ChargeInitDA[model.ACTIVETIMEPOINTS[1], s]
                        * model.ChargeEff[s]
                        + model.DischargeInitDA[model.ACTIVETIMEPOINTS[1], s]
                        * model.DischargeEff[s]
                    )
                )
                * 1.0
                * model.socmin_dual[t, s]
                for t in model.ACTIVETIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            model.SocMax[s] * model.onecycle_dual[s]
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(model.DischargeInitDA[t, s] for t in model.ACTIVETIMEPOINTS)
            / 12.0
            * model.bindonecycle_dual[s]
            for s in model.NON_STRATEGIC_STORAGE
        )
        + sum(
            (
                (
                    model.SOCInitDA[model.ACTIVETIMEPOINTS[-1], s]
                    - model.SOCInitDA[model.ACTIVETIMEPOINTS[1], s]
                    + model.ChargeInitDA[model.ACTIVETIMEPOINTS[1], s]
                    * model.ChargeEff[s]
                    - model.DischargeInitDA[model.ACTIVETIMEPOINTS[1], s]
                    * model.DischargeEff[s]
                )
            )
            * 1.0
            * model.finalsoc_dual[s]
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                sum(
                    model.gsd[t, g, gs] * model.GeneratorMarginalCost[t, g, gs]
                    for t in model.ACTIVETIMEPOINTS
                )
                for g in model.UC_GENS
            )
            for gs in model.GENERATORSEGMENTS
        )
        - sum(
            sum(model.gopstat[t, g] for t in model.ACTIVETIMEPOINTS)
            * model.NoLoadCost[g]
            for g in model.UC_GENS
        )
        - sum(
            sum(model.gup[t, g] for t in model.ACTIVETIMEPOINTS) * model.StartCost[g]
            for g in model.UC_GENS
        )
        - sum(
            sum(
                model.TransmissionToCapacity[t, line]
                * model.Hours[t]
                * model.transmissionmax_dual[t, line]
                for t in model.ACTIVETIMEPOINTS
            )
            for line in model.TRANSMISSION_LINE
        )
        + sum(
            sum(
                model.TransmissionFromCapacity[t, line]
                * model.Hours[t]
                * model.transmissionmin_dual[t, line]
                for t in model.ACTIVETIMEPOINTS
            )
            for line in model.TRANSMISSION_LINE
        )
        - sum(
            sum(
                model.VoltageAngleMax[z] * model.voltageanglemax_dual[t, z]
                for t in model.ACTIVETIMEPOINTS
            )
            for z in model.ZONES
        )
        + sum(
            sum(
                model.VoltageAngleMin[z] * model.voltageanglemin_dual[t, z]
                for t in model.ACTIVETIMEPOINTS
            )
            for z in model.ZONES
        )
        + sum(
            sum(
                (model.GrossLoad[t, z] * model.Hours[t]) * model.zonalprice[t, z]
                for t in model.ACTIVETIMEPOINTS
            )
            for z in model.ZONES
        )
        #        - sum(
        #            sum(model.sc[t, s] for t in model.ACTIVETIMEPOINTS) for s in model.STORAGE
        #        )  # charge penalty
    )


dispatch_model.RTGeneratorProfitDual = Objective(
    rule=rt_objective_profit_dual, sense=maximize
)
# - model.zonalcharge[t, z]


def rt_objective_profit_dual_pre(model):
    """Pre-processing version of full objective for MPEC reformulated as MIP using BigM
    Only difference is iteration over *ALL* generators rather than just non-strategic generators
    for the first term in the objective (involving gensegmentmax_dual). This just forces
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
                    * model.Hours[t]
                    * model.gensegmentmax_dual[t, g, gs]
                    for t in model.ACTIVETIMEPOINTS
                )
                for g in model.UC_GENS
            )
            for gs in model.GENERATORSEGMENTS
        )
        + sum(
            sum(
                -model.CapacityTime[t, g]
                * model.ScheduledAvailable[t, g]
                * model.nucdispatchmax_dual[t, g]
                * model.Hours[t]
                for t in model.ACTIVETIMEPOINTS
            )
            for g in model.NUC_GENS
        )
        - sum(
            sum(
                model.DischargeMax[s]
                * model.ChargeMax[s]
                * model.Hours[t]
                * model.storagetight_dual[t, s]
                for t in model.ACTIVETIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                (
                    model.SocMax[s]
                    - 0.0
                    * (
                        model.SOCInitDA[model.ACTIVETIMEPOINTS[1], s]
                        + model.ChargeInitDA[model.ACTIVETIMEPOINTS[1], s]
                        * model.ChargeEff[s]
                        - model.DischargeInitDA[model.ACTIVETIMEPOINTS[1], s]
                        * model.DischargeEff[s]
                    )
                )
                * model.socmax_dual[t, s]
                for t in model.ACTIVETIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                abs(
                    (
                        model.SOCInitDA[model.ACTIVETIMEPOINTS[1], s]
                        - model.ChargeInitDA[model.ACTIVETIMEPOINTS[1], s]
                        * model.ChargeEff[s]
                        + model.DischargeInitDA[model.ACTIVETIMEPOINTS[1], s]
                        * model.DischargeEff[s]
                    )
                )
                * 1.0
                * model.socmin_dual[t, s]
                for t in model.ACTIVETIMEPOINTS
            )
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            model.SocMax[s] * model.onecycle_dual[s]
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(model.DischargeInitDA[t, s] for t in model.ACTIVETIMEPOINTS)
            / 12.0
            * model.bindonecycle_dual[s]
            for s in model.NON_STRATEGIC_STORAGE
        )
        + sum(
            (
                (
                    model.SOCInitDA[model.ACTIVETIMEPOINTS[-1], s]
                    - model.SOCInitDA[model.ACTIVETIMEPOINTS[1], s]
                    + model.ChargeInitDA[model.ACTIVETIMEPOINTS[1], s]
                    * model.ChargeEff[s]
                    - model.DischargeInitDA[model.ACTIVETIMEPOINTS[1], s]
                    * model.DischargeEff[s]
                )
            )
            * 1.0
            * model.finalsoc_dual[s]
            for s in model.NON_STRATEGIC_STORAGE
        )
        - sum(
            sum(
                sum(
                    model.gsd[t, g, gs] * model.GeneratorMarginalCost[t, g, gs]
                    for t in model.ACTIVETIMEPOINTS
                )
                for g in model.UC_GENS
            )
            for gs in model.GENERATORSEGMENTS
        )
        - sum(
            sum(model.gopstat[t, g] for t in model.ACTIVETIMEPOINTS)
            * model.NoLoadCost[g]
            for g in model.UC_GENS
        )
        - sum(
            sum(model.gup[t, g] for t in model.ACTIVETIMEPOINTS) * model.StartCost[g]
            for g in model.UC_GENS
        )
        - sum(
            sum(
                model.TransmissionToCapacity[t, line]
                * model.Hours[t]
                * model.transmissionmax_dual[t, line]
                for t in model.ACTIVETIMEPOINTS
            )
            for line in model.TRANSMISSION_LINE
        )
        + sum(
            sum(
                model.TransmissionFromCapacity[t, line]
                * model.Hours[t]
                * model.transmissionmin_dual[t, line]
                for t in model.ACTIVETIMEPOINTS
            )
            for line in model.TRANSMISSION_LINE
        )
        - sum(
            sum(
                model.VoltageAngleMax[z] * model.voltageanglemax_dual[t, z]
                for t in model.ACTIVETIMEPOINTS
            )
            for z in model.ZONES
        )
        + sum(
            sum(
                model.VoltageAngleMin[z] * model.voltageanglemin_dual[t, z]
                for t in model.ACTIVETIMEPOINTS
            )
            for z in model.ZONES
        )
        + sum(
            sum(
                (model.GrossLoad[t, z] * model.Hours[t]) * model.zonalprice[t, z]
                for t in model.ACTIVETIMEPOINTS
            )
            for z in model.ZONES
        )
        #        - sum(
        #            sum(model.sc[t, s] for t in model.ACTIVETIMEPOINTS) for s in model.STORAGE
        #        )  # charge penalty
    )


dispatch_model.RTGeneratorProfitDualPre = Objective(
    rule=rt_objective_profit_dual_pre, sense=maximize
)
