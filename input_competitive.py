# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:49:53 2020

@author: Luke
"""
import os
from pyomo.environ import DataPortal

import model_competitive


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
        filename=os.path.join(inputs_directory, "generators.csv"),
        index=model_competitive.dispatch_model.GENERATORS,
        param=(
            model_competitive.dispatch_model.capacity,
            model_competitive.dispatch_model.fuelcost,
            model_competitive.dispatch_model.pmin,
            model_competitive.dispatch_model.startcost,
            model_competitive.dispatch_model.canspin,
            model_competitive.dispatch_model.cannonspin,
            model_competitive.dispatch_model.minup,
            model_competitive.dispatch_model.mindown,
            model_competitive.dispatch_model.noloadcost,
            model_competitive.dispatch_model.ramp,
            model_competitive.dispatch_model.tonneCO2perMWh,
            model_competitive.dispatch_model.CO2price,
            model_competitive.dispatch_model.CO2dollarsperMWh,
            model_competitive.dispatch_model.zonelabel,
            model_competitive.dispatch_model.genco_index,
            model_competitive.dispatch_model.highCO2price,
        ),
    )

    data.load(
        filename=os.path.join(inputs_directory, "storage_resources.csv"),
        index=model_competitive.dispatch_model.STORAGE,
        param=(
            model_competitive.dispatch_model.discharge_max,
            model_competitive.dispatch_model.charge_max,
            model_competitive.dispatch_model.soc_max,
            model_competitive.dispatch_model.discharge_eff,
            model_competitive.dispatch_model.charge_eff,
            model_competitive.dispatch_model.storage_zone_label,
            model_competitive.dispatch_model.storage_index,
        ),
    )

    data.load(
        filename=os.path.join(inputs_directory, "storage_two_resources.csv"),
        index=model_competitive.dispatch_model.STORAGETWO,
        param=(model_competitive.dispatch_model.discharge_two),
    )

    data.load(
        filename=os.path.join(inputs_directory, "initialize_generators.csv"),
        param=(
            model_competitive.dispatch_model.commitinit,
            model_competitive.dispatch_model.upinit,
            model_competitive.dispatch_model.downinit,
        ),
    )

    data.load(
        filename=os.path.join(
            inputs_directory, "generators_scheduled_availability.csv"
        ),
        param=(
            model_competitive.dispatch_model.scheduled_available,
            model_competitive.dispatch_model.capacity_time,
            model_competitive.dispatch_model.fuel_cost_time,
        ),
    )

    data.load(
        filename=os.path.join(inputs_directory, "timepoints_index.csv"),
        index=model_competitive.dispatch_model.TIMEPOINTS,
        param=(
            model_competitive.dispatch_model.reference_bus,
            model_competitive.dispatch_model.reg_up_mw,
            model_competitive.dispatch_model.reg_down_mw,
            model_competitive.dispatch_model.flex_up_mw,
            model_competitive.dispatch_model.flex_down_mw,
        ),
    )

    data.load(
        filename=os.path.join(inputs_directory, "zones.csv"),
        index=model_competitive.dispatch_model.ZONES,
        param=(
            model_competitive.dispatch_model.wind_cap,
            model_competitive.dispatch_model.solar_cap,
            model_competitive.dispatch_model.voltage_angle_max,
            model_competitive.dispatch_model.voltage_angle_min,
        ),
    )

    data.load(
        filename=os.path.join(inputs_directory, "timepoints_zonal.csv"),
        param=(
            model_competitive.dispatch_model.gross_load,
            model_competitive.dispatch_model.wind_cf,
            model_competitive.dispatch_model.solar_cf,
        ),
    )

    data.load(
        filename=os.path.join(inputs_directory, "transmission_lines.csv"),
        index=model_competitive.dispatch_model.TRANSMISSION_LINE,
        param=(model_competitive.dispatch_model.susceptance),
    )

    data.load(
        filename=os.path.join(inputs_directory, "transmission_lines_hourly.csv"),
        param=(
            model_competitive.dispatch_model.transmission_from,
            model_competitive.dispatch_model.transmission_to,
            model_competitive.dispatch_model.transmission_from_capacity,
            model_competitive.dispatch_model.transmission_to_capacity,
            model_competitive.dispatch_model.hurdle_rate,
        ),
    )

    data.load(
        filename=os.path.join(inputs_directory, "generator_segments.csv"),
        index=model_competitive.dispatch_model.GENERATORSEGMENTS,
        param=(model_competitive.dispatch_model.base_generator_segment_length),
    )

    data.load(
        filename=os.path.join(inputs_directory, "generator_segment_marginalcost.csv"),
        param=(
            model_competitive.dispatch_model.generator_segment_length,
            model_competitive.dispatch_model.generator_marginal_cost,
            model_competitive.dispatch_model.previous_offer,
            model_competitive.dispatch_model.marginal_CO2,
            model_competitive.dispatch_model.CO2_damage,
        ),
    )

    data.load(
        filename=os.path.join(inputs_directory, "case_index.csv"),
        index=model_competitive.dispatch_model.CASE,
        param=(model_competitive.dispatch_model.genco),
    )

    return data
