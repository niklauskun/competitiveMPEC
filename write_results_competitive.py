# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:11:07 2020

@author: Luke
"""

import os
from os.path import join
import traceback
import sys
import pdb
import pandas as pd
import numpy as np


def export_results(instance, results, results_directory, is_MPEC, debug_mode):
    """Retrieves the relevant sets over which it will loop, then call functions to export different result categories.
    If an exception is encountered, log the error traceback. If not in debug mode, exit. If in debug mode,
    open an interactive Python session that will make it possible to try to correct the error without having to re-run
    problem (to save time!); quitting the interactive session will resume running the next export function, not exit.

    Arguments:
        instance{pyomo.core.base.PyomoModel.ConcreteModel} -- a Pyomo model instance
        results {<class 'pyomo.opt.results.results_.SolverResults'>} -- a pyomo model solution
        results_directory {filepath} -- string of filepath where you stored scenario results
        is_MPEC {bool} -- whether results are competitive model format (affects how model records dual variables)
        debug_mode {int} -- defines functioning of debugger (default=1)

    Returns:
        None -- N/A
    """

    print("Exporting results... ")

    # First, load solution
    load_solution(instance, results)

    # Get model sets for indexing
    # Sort the sets to return a predictable format of results files
    timepoints_set = sorted(instance.TIMEPOINTS)
    generators_set = (
        instance.GENERATORS
    )  # don't sort these so order is preserved for future cases
    ucgenerators_set = instance.UC_GENS
    nuc_set = instance.NUC_GENS
    transmission_lines_set = instance.TRANSMISSION_LINE
    zones_set = sorted(instance.ZONES)
    generatorsegment_set = sorted(instance.GENERATORSEGMENTS)
    storage_set = sorted(instance.STORAGE)

    # Call various export functions, throw debug errors if there's an issue
    # Export generator dispatch
    try:
        export_generator_dispatch(
            instance, timepoints_set, generators_set, results_directory
        )
    except Exception as err:
        msg = "ERROR exporting generator dispatch! Check export_generator_dispatch()."
        handle_exception(msg, debug_mode)

    # Export segmented generator dispatch
    try:
        export_generator_segment_dispatch(
            instance,
            timepoints_set,
            ucgenerators_set,
            zones_set,
            generatorsegment_set,
            results_directory,
        )
    except Exception as err:
        msg = "ERROR exporting segmented generator dispatch! Check export_generator_segment_dispatch()."
        handle_exception(msg, debug_mode)

    try:
        export_generator_segment_offer(
            instance,
            timepoints_set,
            ucgenerators_set,
            zones_set,
            generatorsegment_set,
            results_directory,
            is_MPEC,
        )
    except Exception as err:
        msg = "ERROR exporting segmented generator offers! Check export_generator_segment_offer()."
        handle_exception(msg, debug_mode)

    try:
        export_nuc_generator_dispatch(
            instance, timepoints_set, nuc_set, is_MPEC, results_directory
        )
    except Exception as err:
        msg = "ERROR exporting NUC generator offers! Check export_nuc_generator_dispatch()."
        handle_exception(msg, debug_mode)

    # export tx lines
    try:
        export_lines(
            instance, timepoints_set, transmission_lines_set, results_directory, is_MPEC
        )
    except Exception as err:
        msg = "ERROR exporting transmission lines! Check export_lines()."
        handle_exception(msg, debug_mode)

    # export zonal prices
    try:
        export_zonal_price(
            instance, timepoints_set, zones_set, results_directory, is_MPEC
        )
    except Exception as err:
        msg = "ERROR exporting zonal prices! Check export_zonal_price()."
        handle_exception(msg, debug_mode)

    # export storage
    try:
        export_storage(
            instance, timepoints_set, storage_set, results_directory, is_MPEC
        )
    except Exception as err:
        msg = "ERROR exporting storage! Check export_storage()."
        handle_exception(msg, debug_mode)
    # export VRE
    try:
        export_VREs(instance, results_directory)
    except Exception as err:
        msg = "ERROR exporting VRE results! Check export_VREs()."
        handle_exception(msg, debug_mode)

    # return call
    return None


def format_6f(input_data):
    """returns an input value rounded to six decimal places

    Arguments:
        input_data {float} -- value to format

    Returns:
        [float] -- formatted value to six decimal places
    """
    if input_data is None:
        formatted_data = None
    else:
        formatted_data = "{:0.6f}".format(input_data)
        # if formatted_data is negative but rounds to zero, it will be printed as a negative zero
        # this gets rid of the negative zero
        if formatted_data == "0.00" or formatted_data == "-0.00":
            formatted_data = 0
    return formatted_data


def handle_exception(message, debug):
    """ How to handle exceptions. First print a custom message and the traceback.
    If in debug mode, open the Python debugger (PDB), else exit. To execute multi-line statements in PDB, type this
    to launch an interactive Python session with all the local variables available.
    This function is used by export_results() to handle exceptions.

    Arguments:
        message {str} -- error message to print when exception is handled
        debug {bool} -- debug param
    """
    print(message)
    print(traceback.format_exc())
    if debug:
        print(
            """
           Launching Python debugger (PDB)...
           To execute multi-line statements in PDB, type this:
                import code; code.interact(local=vars())
           (The command launches an interactive Python session with all the local variables available.)
           To exit the interactive session and continue script execution, press Ctrl+D.
           """
        )
        tp, value, tb = sys.exc_info()
        pdb.post_mortem(tb)
    else:
        print("Debug option not chosen, so exiting.")
        sys.exit()


def load_solution(instance, results):
    """Load pyomo model results. This function is called by export_results().

    Arguments:
        instance {[type]} -- [description]
        results {[type]} -- [description]
    """
    instance.solutions.load_from(results)


# #################### Data Exports #################### #
# these functions all work quite similarly, so I'm only going to document the first one.
# You will have to change them though if you want to format outputs differently, or add new ones


def export_generator_dispatch(
    instance, timepoints_set, generators_set, results_directory
):
    """formats and exports generator dispatch to csv, indexed by generator and timepoint

    Arguments:
        instance{pyomo.core.base.PyomoModel.ConcreteModel} -- a Pyomo model instance
        timepoints_set {list} -- sorted list of model timepoints
        generators_set {list} -- sorted list of model generators
        results_directory {filepath} -- string of directory in which to write csv
    """
    results_dispatch, index_name = [], []
    # for z in zones_set:
    for g in generators_set:
        index_name.append(str(g))
        for t in timepoints_set:
            results_dispatch.append(format_6f(instance.dispatch[t, g]()))
    results_dispatch_np = np.reshape(
        results_dispatch,
        (int(len(results_dispatch) / len(timepoints_set)), int(len(timepoints_set))),
    )
    df = pd.DataFrame(results_dispatch_np, index=pd.Index(index_name))
    df.to_csv(os.path.join(results_directory, "generator_dispatch.csv"))


def export_generator_segment_dispatch(
    instance,
    timepoints_set,
    ucgenerators_set,
    zones_set,
    generatorsegment_set,
    results_directory,
):

    results_dispatch, index_name = [], []
    for g in ucgenerators_set:
        for gs in generatorsegment_set:
            index_name.append(str(g) + "-" + str(gs))
            for t in timepoints_set:
                results_dispatch.append(
                    format_6f(instance.segmentdispatch[t, g, gs].value)
                )
    results_dispatch_np = np.reshape(
        results_dispatch,
        (int(len(results_dispatch) / len(timepoints_set)), int(len(timepoints_set))),
    )
    df = pd.DataFrame(results_dispatch_np, index=pd.Index(index_name))
    df.to_csv(os.path.join(results_directory, "generator_segment_dispatch.csv"))


def export_generator_segment_offer(
    instance,
    timepoints_set,
    ucgenerators_set,
    zones_set,
    generatorsegment_set,
    results_directory,
    is_MPEC,
):

    results_offer = []
    index_name = []
    timepoints_list = []
    total_dispatch = []
    total_emissions = []
    commitment = []
    start = []
    shut = []
    max_dual = []
    min_dual = []
    dmax_dual = []
    dmin_dual = []
    rmax_dual = []
    rmin_dual = []
    start_shut_dual = []
    zone_names = []
    lmp = []
    marginal_cost = []
    previous_offer = []

    for t in timepoints_set:
        for g in ucgenerators_set:
            for gs in generatorsegment_set:
                timepoints_list.append(t)
                index_name.append(str(g) + "-" + str(gs))
                results_offer.append(
                    format_6f(instance.gensegmentoffer[t, g, gs].value)
                )
                total_dispatch.append(format_6f(instance.segmentdispatch[t, g, gs]()))
                total_emissions.append(format_6f(instance.CO2_emissions[t, g, gs]()))
                commitment.append(format_6f(instance.commitment[t, g]()))
                start.append(format_6f(instance.startup[t, g]()))
                shut.append(format_6f(instance.startup[t, g]()))
                max_dual.append(format_6f(instance.gensegmentmaxdual[t, g, gs].value))
                min_dual.append(format_6f(instance.gensegmentmindual[t, g, gs].value))
                dmax_dual.append(format_6f(instance.gendispatchmaxdual[t, g].value))
                dmin_dual.append(format_6f(instance.gendispatchmindual[t, g].value))
                rmax_dual.append(format_6f(instance.rampmaxdual[t, g].value))
                rmin_dual.append(format_6f(instance.rampmindual[t, g].value))
                start_shut_dual.append(
                    format_6f(instance.startupshutdowndual[t, g].value)
                )
                marginal_cost.append(
                    format_6f(instance.generator_marginal_cost[t, g, gs])
                )
                previous_offer.append(format_6f(instance.previous_offer[t, g, gs]))
                zone_names.append(instance.zonelabel[g])
                if is_MPEC:
                    lmp.append(
                        format_6f(instance.zonalprice[t, instance.zonelabel[g]].value)
                    )
                else:
                    lmp.append(
                        format_6f(
                            instance.dual[
                                instance.LoadConstraint[t, instance.zonelabel[g]]
                            ]
                        )
                    )
    profit = [
        (float(a_i) - float(b_i)) * float(c_i)
        for a_i, b_i, c_i in zip(lmp, marginal_cost, total_dispatch)
    ]
    col_names = [
        "hour",
        "SegmentOffer",
        "SegmentDispatch",
        "SegmentEmissions",
        "Commitment",
        "Startup",
        "Shutdown",
        "MaxDual",
        "MinDual",
        "DMaxDual",
        "DMinDual",
        "RampMaxDual",
        "RampMinDual",
        "StartShutDual",
        "Zone",
        "LMP",
        "MarginalCost",
        "Profit",
        "PreviousOffer",
    ]
    df = pd.DataFrame(
        data=np.column_stack(
            (
                np.asarray(timepoints_list),
                np.asarray(results_offer),
                np.asarray(total_dispatch),
                np.asarray(total_emissions),
                np.asarray(commitment),
                np.asarray(start),
                np.asarray(shut),
                np.asarray(max_dual),
                np.asarray(min_dual),
                np.asarray(dmax_dual),
                np.asarray(dmin_dual),
                np.asarray(rmax_dual),
                np.asarray(rmin_dual),
                np.asarray(start_shut_dual),
                np.asarray(zone_names),
                np.asarray(lmp),
                np.asarray(marginal_cost),
                np.asarray(profit),
                np.asarray(previous_offer),
            )
        ),
        columns=col_names,
        index=pd.Index(index_name),
    )
    df.to_csv(os.path.join(results_directory, "generator_segment_offer.csv"))


def export_nuc_generator_dispatch(
    instance, timepoints_set, nuc_set, is_MPEC, results_directory
):
    index_name = []
    timepoints_list, dispatch, dual, offer, lmp = [], [], [], [], []
    for t in timepoints_set:
        for g in nuc_set:
            index_name.append(str(g) + "-" + str(t))
            timepoints_list.append(t)
            dispatch.append(format_6f(instance.nucdispatch[t, g]()))
            dual.append(format_6f(instance.nucdispatchmaxdual[t, g].value))
            offer.append(format_6f(instance.genoffer[t, g].value))
            if is_MPEC:
                lmp.append(
                    format_6f(instance.zonalprice[t, instance.zonelabel[g]].value)
                )
            else:
                lmp.append(
                    format_6f(
                        instance.dual[instance.LoadConstraint[t, instance.zonelabel[g]]]
                    )
                )
    col_names = ["hour", "dispatch", "dual", "offer", "lmp"]
    df = pd.DataFrame(
        data=np.column_stack(
            (
                np.asarray(timepoints_list),
                np.asarray(dispatch),
                np.asarray(dual),
                np.asarray(offer),
                np.asarray(lmp),
            )
        ),
        columns=col_names,
        index=pd.Index(index_name),
    )
    df.to_csv(os.path.join(results_directory, "nuc_offer.csv"))


def export_zonal_price(instance, timepoints_set, zones_set, results_directory, is_MPEC):

    results_prices = []
    index_name = []
    timepoints_list = []
    voltage_angle_list = []
    voltage_angle_dual_max = []
    voltage_angle_dual_min = []
    load = []

    for z in zones_set:
        for t in timepoints_set:
            index_name.append(z)
            if is_MPEC:
                results_prices.append(format_6f(instance.zonalprice[t, z].value))
            else:
                results_prices.append(
                    format_6f(instance.dual[instance.LoadConstraint[t, z]])
                )

            timepoints_list.append(t)
            voltage_angle_list.append(format_6f(instance.voltage_angle[t, z].value))
            voltage_angle_dual_max.append(
                format_6f(instance.voltageanglemaxdual[t, z].value)
            )
            voltage_angle_dual_min.append(
                format_6f(instance.voltageanglemindual[t, z].value)
            )
            load.append(format_6f(instance.gross_load[t, z]))

    load_payment = [float(a) * float(b) for a, b in zip(results_prices, load)]
    col_names = [
        "hour",
        "LMP",
        "VoltageAngle",
        " VoltageAngleDualMax",
        "VoltageAngleDualMin",
        "Load",
        "LoadPayment",
    ]
    df = pd.DataFrame(
        data=np.column_stack(
            (
                np.asarray(timepoints_list),
                np.asarray(results_prices),
                np.asarray(voltage_angle_list),
                np.asarray(voltage_angle_dual_max),
                np.asarray(voltage_angle_dual_min),
                np.asarray(load),
                np.asarray(load_payment),
            )
        ),
        columns=col_names,
        index=pd.Index(index_name),
    )

    df.to_csv(os.path.join(results_directory, "zonal_prices.csv"))


def export_lines(
    instance, timepoints_set, transmission_lines_set, results_directory, is_MPEC
):

    transmission_duals_from = []
    transmission_duals_to = []
    results_transmission_line_flow = []
    # dc_opf_dual = []
    index_name = []
    for line in transmission_lines_set:
        for t in timepoints_set:
            index_name.append(line + "-" + str(t))
            if is_MPEC:
                transmission_duals_from.append(
                    format_6f(instance.transmissionmindual[t, line].value)
                )
                transmission_duals_to.append(
                    format_6f(instance.transmissionmaxdual[t, line].value)
                )
            else:
                transmission_duals_from.append(
                    format_6f(instance.dual[instance.TxFromConstraint[t, line]])
                )
                transmission_duals_to.append(
                    format_6f(instance.dual[instance.TxToConstraint[t, line]])
                )
            results_transmission_line_flow.append(
                format_6f(instance.transmit_power_MW[t, line].value)
            )
            # dc_opf_dual.append(format_6f(instance.dual[instance.DCOPFConstraint[t,line]]))
    col_names = [
        "flow (MW)",
        "congestion price from ($/MW)",
        "congestion price to ($/MW)",
    ]
    df = pd.DataFrame(
        data=np.column_stack(
            (
                np.asarray(results_transmission_line_flow),
                np.asarray(transmission_duals_from),
                np.asarray(transmission_duals_to),
            )
        ),
        columns=col_names,
        index=pd.Index(index_name),
    )
    df.to_csv(os.path.join(results_directory, "tx_flows.csv"))


def export_generator_commits_reserves(
    instance, timepoints_set, ucgenerators_set, results_directory
):

    results_gens = []
    results_time = []
    results_commitment = []
    results_starts = []
    results_shuts = []
    results_hourson = []
    results_hoursoff = []
    results_primarysynchreserves = []
    results_primarynonsynchreserves = []
    results_allreserves = []
    results_secondaryreserves = []
    index_name = []
    for g in ucgenerators_set:
        for t in timepoints_set:
            index_name.append(str(g) + "," + str(t))
            results_gens.append(g)
            results_time.append(t)
            results_commitment.append(instance.commitment[t, g].value)
            results_starts.append(instance.startup[t, g].value)
            results_shuts.append(instance.shutdown[t, g].value)

            if t == 1 and instance.commitinit[g] == instance.commitment[t, g].value:
                results_hourson.append(
                    instance.commitinit[g] * instance.upinit[g]
                    + instance.commitment[t, g].value
                )
                results_hoursoff.append(
                    (1 - instance.commitinit[g]) * instance.downinit[g]
                    + (1 - instance.commitment[t, g].value)
                )
            elif (
                instance.startup[t, g].value == 1 or instance.shutdown[t, g].value == 1
            ):
                results_hourson.append(instance.commitment[t, g].value)
                results_hoursoff.append((1 - instance.commitment[t, g].value))
            else:
                results_hourson.append(
                    results_hourson[-1] + instance.commitment[t, g].value
                )
                results_hoursoff.append(
                    results_hoursoff[-1] + (1 - instance.commitment[t, g].value)
                )

            results_primarysynchreserves.append(
                format_6f(instance.synchreserves[t, g].value)
            )
            results_primarynonsynchreserves.append(
                format_6f(instance.nonsynchreserves[t, g].value)
            )
            results_allreserves.append(
                format_6f(
                    instance.synchreserves[t, g].value
                    + instance.nonsynchreserves[t, g].value
                )
            )
            results_secondaryreserves.append(
                format_6f(instance.secondaryreserves[t, g].value)
            )

    col_names = [
        "Gen_Index",
        "timepoint",
        "Committed",
        "Started",
        "Shut",
        "TimeOn",
        "TimeOff",
        "Total Held as Primary Synch Reserves (MW)",
        "Total Held as Primary NonSynch Reserves (MW)",
        "Total Held as Primary Reserves (MW)",
        "Total Held as Secondary Reserves (MW)",
    ]
    df = pd.DataFrame(
        data=np.column_stack(
            (
                np.asarray(results_gens),
                np.asarray(results_time),
                np.asarray(results_commitment),
                np.asarray(results_starts),
                np.asarray(results_shuts),
                np.asarray(results_hourson),
                np.asarray(results_hoursoff),
                np.asarray(results_primarysynchreserves),
                np.asarray(results_primarynonsynchreserves),
                np.asarray(results_allreserves),
                np.asarray(results_secondaryreserves),
            )
        ),
        columns=col_names,
        index=pd.Index(index_name),
    )
    df.to_csv(
        os.path.join(results_directory, "generator_commits_reserves.csv"), index=False
    )


def export_reserve_segment_commits(
    instance, timepoints_set, ordc_segments_set, results_directory
):

    results_synch_segments = []
    results_nonsynch_segments = []
    results_secondary_segments = []
    index_name = []
    for s in ordc_segments_set:
        for t in timepoints_set:
            index_name.append(str(s) + "," + str(t))
            results_synch_segments.append(
                format_6f(instance.segmentreserves[t, s].value)
            )
            results_nonsynch_segments.append(
                format_6f(instance.nonsynchsegmentreserves[t, s].value)
            )
            results_secondary_segments.append(
                format_6f(instance.secondarysegmentreserves[t, s].value)
            )
    col_names = [
        "MW on primary synch reserve segment",
        "MW on primary nonsynch reserve segment",
        "MW on secondary reserve segment",
    ]
    df = pd.DataFrame(
        data=np.column_stack(
            (
                np.asarray(results_synch_segments),
                np.asarray(results_nonsynch_segments),
                np.asarray(results_secondary_segments),
            )
        ),
        columns=col_names,
        index=pd.Index(index_name),
    )
    df.to_csv(os.path.join(results_directory, "reserve_segment_commit.csv"))


def export_storage(instance, timepoints_set, storage_set, results_directory, is_MPEC):

    index_name = []
    results_time = []
    storage_dispatch = []
    storage_charge = []
    storage_discharge = []
    soc = []
    storage_offer = []
    storage_tight_dual = []
    storage_max_dual = []
    storage_min_dual = []
    storage_charge_dual = []
    storage_discharge_dual = []
    cycle_dual = []
    node = []
    lmp = []

    for t in timepoints_set:
        for s in storage_set:
            index_name.append(s)
            results_time.append(t)
            storage_charge.append(format_6f(instance.charge[t, s].value))
            storage_discharge.append(format_6f(instance.discharge[t, s].value))
            storage_dispatch.append(format_6f(instance.storagedispatch[t, s].value))
            soc.append(format_6f(instance.soc[t, s].value))
            storage_offer.append(format_6f(instance.storageoffer[t, s].value))
            storage_tight_dual.append(format_6f(instance.storagetightdual[t, s].value))
            storage_max_dual.append(format_6f(instance.storagemaxdual[t, s].value))
            storage_min_dual.append(format_6f(instance.storagemindual[t, s].value))
            storage_charge_dual.append(format_6f(instance.chargedual[t, s].value))
            storage_discharge_dual.append(format_6f(instance.dischargedual[t, s].value))
            cycle_dual.append(format_6f(instance.onecycledual[s].value))
            node.append(instance.storage_zone_label[s])
            if is_MPEC:
                lmp.append(
                    format_6f(
                        instance.zonalprice[t, instance.storage_zone_label[s]].value
                    )
                )
            else:
                lmp.append(
                    format_6f(
                        instance.dual[
                            instance.LoadConstraint[t, instance.storage_zone_label[s]]
                        ]
                    )
                )

    profit = [float(c) * float(price) for c, price in zip(storage_dispatch, lmp)]
    col_names = [
        "time",
        "charge",
        "discharge",
        "dispatch",
        "soc",
        "offer",
        "tightdual",
        "maxdual",
        "mindual",
        "chargedual",
        "dischargedual",
        "cycledual",
        "node",
        "lmp",
        "profit",
    ]
    df = pd.DataFrame(
        data=np.column_stack(
            (
                np.asarray(results_time),
                np.asarray(storage_charge),
                np.asarray(storage_discharge),
                np.asarray(storage_dispatch),
                np.asarray(soc),
                np.asarray(storage_offer),
                np.asarray(storage_tight_dual),
                np.asarray(storage_max_dual),
                np.asarray(storage_min_dual),
                np.asarray(storage_charge_dual),
                np.asarray(storage_discharge_dual),
                np.asarray(cycle_dual),
                np.asarray(node),
                np.asarray(lmp),
                np.asarray(profit),
            )
        ),
        columns=col_names,
        index=pd.Index(index_name),
    )

    df.to_csv(os.path.join(results_directory, "storage_dispatch.csv"))


def export_VREs(instance, results_directory):

    results_wind = []
    results_solar = []
    results_curtailment = []
    tmps = []
    zones = []

    for t in instance.TIMEPOINTS:
        for z in instance.ZONES:
            tmps.append(t)
            zones.append(z)
            results_wind.append(instance.windgen[t, z].value)
            results_solar.append(instance.solargen[t, z].value)
            results_curtailment.append(instance.curtailment[t, z].value)

    VRE = pd.DataFrame(
        {
            "timepoint": tmps,
            "zone": zones,
            "wind": results_wind,
            "solar": results_solar,
            "curtailment": results_curtailment,
        }
    )
    VRE.to_csv(os.path.join(results_directory, "renewable_generation.csv"), index=False)
