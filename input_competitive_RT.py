# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:49:53 2020

@author: Luke
"""
import os
from pyomo.environ import DataPortal

import model_competitive_test


def scenario_inputs(inputs_directory):
    """loads in scenario data from csvs and formats as Pyomo DataPortal for abstract model input
    NOTE: if you create more params in the model, you'll have to make sure they get properly loaded here
    
    Arguments:
        inputs_directory {filepath} -- filepath of case directory with input csvs

    Returns:
        <class 'pyomo.dataportal.DataPortal.DataPortal'> -- Pyomo DataPortal
    """
    data = DataPortal()

    data.load(
        filename=os.path.join(inputs_directory, "generators_rt.csv"),
        index=model_competitive_test.dispatch_model.GENERATORS,
        param=(
            model_competitive_test.dispatch_model.capacity,
            model_competitive_test.dispatch_model.fuelcost,
            model_competitive_test.dispatch_model.Pmin,
            model_competitive_test.dispatch_model.StartCost,
            model_competitive_test.dispatch_model.NoLoadCost,
            model_competitive_test.dispatch_model.RampRate,
            model_competitive_test.dispatch_model.TonneCO2PerMWh,
            model_competitive_test.dispatch_model.CO2Price,
            model_competitive_test.dispatch_model.CO2DollarsPerMWh,
            model_competitive_test.dispatch_model.ZoneLabel,
            model_competitive_test.dispatch_model.GencoIndex,
            model_competitive_test.dispatch_model.UCIndex,
            model_competitive_test.dispatch_model.HybridIndex,
        ),
    )

    data.load(
        filename=os.path.join(inputs_directory, "storage_resources_rt.csv"),
        index=model_competitive_test.dispatch_model.STORAGE,
        param=(
            model_competitive_test.dispatch_model.DischargeMax,
            model_competitive_test.dispatch_model.ChargeMax,
            model_competitive_test.dispatch_model.SocMax,
            model_competitive_test.dispatch_model.DischargeEff,
            model_competitive_test.dispatch_model.ChargeEff,
            model_competitive_test.dispatch_model.StorageZoneLabel,
            model_competitive_test.dispatch_model.StorageIndex,
            model_competitive_test.dispatch_model.HybridStorageIndex,
        ),
    )

    data.load(
        filename=os.path.join(
            inputs_directory, "generators_scheduled_availability_rt.csv"
        ),
        param=(
            model_competitive_test.dispatch_model.ScheduledAvailable,
            model_competitive_test.dispatch_model.CapacityTime,
            model_competitive_test.dispatch_model.fuel_cost_time,
        ),
    )

    data.load(
        filename=os.path.join(inputs_directory, "timepoints_index_rt.csv"),
        index=model_competitive_test.dispatch_model.TIMEPOINTS,
        param=(model_competitive_test.dispatch_model.ReferenceBus,),
    )

    data.load(
        filename=os.path.join(inputs_directory, "timepoints_index_subset_rt.csv"),
        index=model_competitive_test.dispatch_model.ACTIVETIMEPOINTS,
        param=(model_competitive_test.dispatch_model.FirstTimepoint,),
    )

    data.load(
        filename=os.path.join(inputs_directory, "zones.csv"),
        index=model_competitive_test.dispatch_model.ZONES,
        param=(
            model_competitive_test.dispatch_model.VoltageAngleMax,
            model_competitive_test.dispatch_model.VoltageAngleMin,
        ),
    )

    data.load(
        filename=os.path.join(inputs_directory, "timepoints_zonal_rt.csv"),
        param=(model_competitive_test.dispatch_model.GrossLoad,),
    )

    data.load(
        filename=os.path.join(inputs_directory, "transmission_lines.csv"),
        index=model_competitive_test.dispatch_model.TRANSMISSION_LINE,
        param=(model_competitive_test.dispatch_model.Susceptance),
    )

    data.load(
        filename=os.path.join(inputs_directory, "transmission_lines_hourly_rt.csv"),
        param=(
            model_competitive_test.dispatch_model.TransmissionFrom,
            model_competitive_test.dispatch_model.TransmissionTo,
            model_competitive_test.dispatch_model.TransmissionFromCapacity,
            model_competitive_test.dispatch_model.TransmissionToCapacity,
        ),
    )

    data.load(
        filename=os.path.join(inputs_directory, "generator_segments.csv"),
        index=model_competitive_test.dispatch_model.GENERATORSEGMENTS,
        param=(model_competitive_test.dispatch_model.base_GeneratorSegmentLength),
    )

    data.load(
        filename=os.path.join(
            inputs_directory, "generator_segment_marginalcost_rt.csv"
        ),
        param=(
            model_competitive_test.dispatch_model.GeneratorSegmentLength,
            model_competitive_test.dispatch_model.GeneratorMarginalCost,
            model_competitive_test.dispatch_model.previous_offer,
            model_competitive_test.dispatch_model.marginal_CO2,
            model_competitive_test.dispatch_model.CO2_damage,
        ),
    )

    data.load(
        filename=os.path.join(inputs_directory, "case_index.csv"),
        index=model_competitive_test.dispatch_model.CASE,
        param=(model_competitive_test.dispatch_model.genco),
    )

    data.load(
        filename=os.path.join(inputs_directory, "storage_offers.csv"),
        param=(
            model_competitive_test.dispatch_model.charge_max_offer,
            model_competitive_test.dispatch_model.discharge_max_offer,
        ),
    )

    return data
