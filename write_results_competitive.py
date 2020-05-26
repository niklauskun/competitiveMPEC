<<<<<<< HEAD
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
            generators_set,
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
            generators_set,
            zones_set,
            generatorsegment_set,
            results_directory,
            is_MPEC,
        )
    except Exception as err:
        msg = "ERROR exporting segmented generator offers! Check export_generator_segment_offer()."
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
    generators_set,
    zones_set,
    generatorsegment_set,
    results_directory,
):

    results_dispatch, index_name = [], []
    for g in generators_set:
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
    generators_set,
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
    max_dual = []
    min_dual = []
    zone_names = []
    lmp = []
    marginal_cost = []
    previous_offer = []

    for t in timepoints_set:
        for g in generators_set:
            for gs in generatorsegment_set:
                timepoints_list.append(t)
                index_name.append(str(g) + "-" + str(gs))
                results_offer.append(
                    format_6f(instance.gensegmentoffer[t, g, gs].value)
                )
                total_dispatch.append(format_6f(instance.segmentdispatch[t, g, gs]()))
                total_emissions.append(format_6f(instance.CO2_emissions[t, g, gs]()))
                max_dual.append(format_6f(instance.gensegmentmaxdual[t, g, gs].value))
                min_dual.append(format_6f(instance.gensegmentmindual[t, g, gs].value))
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
        "MaxDual",
        "MinDual",
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
                np.asarray(max_dual),
                np.asarray(min_dual),
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


def export_zonal_price(instance, timepoints_set, zones_set, results_directory, is_MPEC):

    results_prices = []
    index_name = []
    timepoints_list = []
    voltage_angle_list = []
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
            load.append(format_6f(instance.gross_load[t, z]))

    load_payment = [float(a) * float(b) for a, b in zip(results_prices, load)]
    col_names = ["hour", "LMP", "VoltageAngle", "Load", "LoadPayment"]
    df = pd.DataFrame(
        data=np.column_stack(
            (
                np.asarray(timepoints_list),
                np.asarray(results_prices),
                np.asarray(voltage_angle_list),
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
    instance, timepoints_set, generators_set, results_directory
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
    for g in generators_set:
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
    # storage_charge = []
    # storage_discharge = []
    soc = []
    storage_offer = []
    storage_max_dual = []
    storage_min_dual = []
    node = []
    lmp = []

    for t in timepoints_set:
        for s in storage_set:
            index_name.append(s)
            results_time.append(t)
            # storage_charge.append(format_6f(instance.charge[t,s].value))
            # storage_discharge.append(format_6f(instance.discharge[t,s].value))
            storage_dispatch.append(format_6f(instance.storagedispatch[t, s].value))
            soc.append(format_6f(instance.soc[t, s].value))
            storage_offer.append(format_6f(instance.storageoffer[t, s].value))
            storage_max_dual.append(format_6f(instance.storagemaxdual[t, s].value))
            storage_min_dual.append(format_6f(instance.storagemindual[t, s].value))
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
        "dispatch",
        "soc",
        "offer",
        "maxdual",
        "mindual",
        "node",
        "lmp",
        "profit",
    ]
    df = pd.DataFrame(
        data=np.column_stack(
            (
                np.asarray(results_time),
                np.asarray(storage_dispatch),
                np.asarray(soc),
                np.asarray(storage_offer),
                np.asarray(storage_max_dual),
                np.asarray(storage_min_dual),
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
=======
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
    """
    Retrieves the relevant sets over which it will loop, then call functions to export different result categories.
    If an exception is encountered, log the error traceback. If not in debug mode, exit. If in debug mode,
    open an interactive Python session that will make it possible to try to correct the error without having to re-run
    problem; quitting the interactive session will resume running the next export function, not exit.
    :param instance: the problem instance
    :param results: the results
    :param inputs_directory: directory of case inputs
    :param results_directory: directory to export results files to
    :param prod_cost_inputs_directory: directory that contains production cost model inputs
    :param debug_mode:
    :return:
    """

    print ("Exporting results... ")

    # First, load solution
    load_solution(instance, results)

    # Get sets
    # Sort the sets to return a predictable format of results files
    timepoints_set = sorted(instance.TIMEPOINTS)
    generators_set = instance.GENERATORS #don't sort these so order is preserved for future cases
    transmission_lines_set = instance.TRANSMISSION_LINE
    zones_set = sorted(instance.ZONES)
    generatorsegment_set = sorted(instance.GENERATORSEGMENTS)
    storage_set = sorted(instance.STORAGE)
    
    # Call various export functions, throw debug errors if problem and desired
    
    # Export generator dispatch
    try:
        export_generator_dispatch(instance, timepoints_set, generators_set, zones_set, results_directory)
    except Exception as err:
        msg = "ERROR exporting generator dispatch! Check export_generator_dispatch()."
        handle_exception(msg, debug_mode)
    
    # Export segmented generator dispatch
    try:
        export_generator_segment_dispatch(instance, timepoints_set, generators_set, zones_set, generatorsegment_set, results_directory)
    except Exception as err:
        msg = "ERROR exporting segmented generator dispatch! Check export_generator_segment_dispatch()."
        handle_exception(msg, debug_mode)
    
    try:
        export_generator_segment_offer(instance, timepoints_set, generators_set, zones_set, generatorsegment_set, results_directory, is_MPEC)
    except Exception as err:
        msg = "ERROR exporting segmented generator offers! Check export_generator_segment_offer()."
        handle_exception(msg, debug_mode)
    
    #export tx lines
    try:
        export_lines(instance, timepoints_set, transmission_lines_set, results_directory, is_MPEC)
    except Exception as err:
        msg = "ERROR exporting transmission lines! Check export_lines()."
        handle_exception(msg, debug_mode)
    
    #export zonal prices
    try:
        export_zonal_price(instance, timepoints_set, zones_set, results_directory, is_MPEC)
    except Exception as err:
        msg = "ERROR exporting zonal prices! Check export_zonal_price()."
        handle_exception(msg, debug_mode)

    #export storage
    try:
        export_storage(instance, timepoints_set, storage_set, results_directory, is_MPEC)
    except Exception as err:
        msg = "ERROR exporting storage! Check export_storage()."
        handle_exception(msg, debug_mode)
    #export VRE
    try:
        export_VREs(instance, results_directory)
    except Exception as err:
        msg = "ERROR exporting VRE results! Check export_VREs()."
        handle_exception(msg, debug_mode)
        
    #return call
    return None

# formatting functions
# return value rounded to six decimal places
def format_2f(input_data):
    """
    :param input_data: The data to format
    :return:
    """
    if input_data is None:
        formatted_data = None
    else:
        formatted_data = '{:0.6f}'.format(input_data)
        # if formatted_data is negative but rounds to zero, it will be printed as a negative zero
        # this gets rid of the negative zero
        if formatted_data == '0.00' or formatted_data == '-0.00':
            formatted_data = 0
    return formatted_data

#debugging call, if desired
def handle_exception(message, debug):
    """
    How to handle exceptions. First print a custom message and the traceback.
    If in debug mode, open the Python debugger (PDB), else exit. To execute multi-line statements in PDB, type this
    to launch an interactive Python session with all the local variables available.
    This function is used by export_results() to handle exceptions.
    :param message:
    :param debug:
    :return:
    """
    print (message)
    print(traceback.format_exc())
    if debug:
        print ("""
           Launching Python debugger (PDB)...
           To execute multi-line statements in PDB, type this:
                import code; code.interact(local=vars())
           (The command launches an interactive Python session with all the local variables available.)
           To exit the interactive session and continue script execution, press Ctrl+D.
           """)
        tp, value, tb = sys.exc_info()
        pdb.post_mortem(tb)
    else:
        print ("Debug option not chosen, so exiting.")
        sys.exit()

# #################### Data Exports #################### #
def load_solution(instance, results):
    """
    Load results. This function is called by export_results().
    :param instance:
    :param results:
    :return:
    """
    instance.solutions.load_from(results)
    
def export_generator_dispatch(instance, timepoints_set, generators_set, zones_set, results_directory):

    results_dispatch = []
    index_name = []
    #for z in zones_set:   
    for g in generators_set:
        index_name.append(str(g))#z+"-"+
        for t in timepoints_set:
            results_dispatch.append(format_2f(instance.dispatch[t,g]()))
    results_dispatch_np = np.reshape(results_dispatch, (int(len(results_dispatch)/len(timepoints_set)), int(len(timepoints_set))))
    df = pd.DataFrame(results_dispatch_np, index=pd.Index(index_name))
    df.to_csv(os.path.join(results_directory,"generator_dispatch.csv"))
    
def export_generator_segment_dispatch(instance, timepoints_set, generators_set, zones_set, generatorsegment_set, results_directory):
    
    results_dispatch = []
    index_name = []
    
    for g in generators_set:
        for gs in generatorsegment_set:
            index_name.append(str(g)+"-" +str(gs))
            for t in timepoints_set:
                    results_dispatch.append(format_2f(instance.segmentdispatch[t,g,gs].value))
    results_dispatch_np = np.reshape(results_dispatch, (int(len(results_dispatch)/len(timepoints_set)), int(len(timepoints_set))))
    df = pd.DataFrame(results_dispatch_np, index=pd.Index(index_name))
    df.to_csv(os.path.join(results_directory,"generator_segment_dispatch.csv"))
    
def export_generator_segment_offer(instance,timepoints_set,generators_set,zones_set,generatorsegment_set,results_directory,is_MPEC):
    
    results_offer = []
    index_name = []
    timepoints_list = []
    total_dispatch = []
    total_emissions = []
    max_dual = []
    min_dual = []
    zone_names = []
    lmp = []
    marginal_cost = []
    previous_offer = []
    
    for t in timepoints_set:
        for g in generators_set:
            for gs in generatorsegment_set:
                timepoints_list.append(t)
                index_name.append(str(g)+"-" +str(gs))
                results_offer.append(format_2f(instance.gensegmentoffer[t,g,gs].value))
                total_dispatch.append(format_2f(instance.segmentdispatch[t,g,gs]()))
                total_emissions.append(format_2f(instance.CO2_emissions[t,g,gs]()))
                max_dual.append(format_2f(instance.gensegmentmaxdual[t,g,gs].value))
                min_dual.append(format_2f(instance.gensegmentmindual[t,g,gs].value))
                marginal_cost.append(format_2f(instance.generator_marginal_cost[t,g,gs]))
                previous_offer.append(format_2f(instance.previous_offer[t,g,gs]))
                zone_names.append(instance.zonelabel[g])
                if is_MPEC:
                    lmp.append(format_2f(instance.zonalprice[t,instance.zonelabel[g]].value))
                else:
                    lmp.append(format_2f(instance.dual[instance.LoadConstraint[t,instance.zonelabel[g]]]))
        #for z in zone_list:
        #    for gs in generatorsegment_set:
                #zone_names.append(z)
        #        if is_MPEC:
        #            lmp.append(format_2f(instance.zonalprice[t,z].value))
        #        else:
        #            lmp.append(format_2f(instance.dual[instance.LoadConstraint[t,z]]))
    #results_dispatch_np = np.reshape(results_dispatch, (int(len(results_dispatch)/len(timepoints_set)), int(len(timepoints_set))))
    profit = [(float(a_i) - float(b_i))*float(c_i) for a_i, b_i, c_i in zip(lmp, marginal_cost, total_dispatch)]
    col_names = ['hour','SegmentOffer','SegmentDispatch','SegmentEmissions',
                 'MaxDual','MinDual','Zone','LMP','MarginalCost','Profit','PreviousOffer']
    df = pd.DataFrame(data = np.column_stack((np.asarray(timepoints_list),np.asarray(results_offer),
                                              np.asarray(total_dispatch),np.asarray(total_emissions),
                                              np.asarray(max_dual),
                                              np.asarray(min_dual),np.asarray(zone_names),
                                              np.asarray(lmp),np.asarray(marginal_cost),
                                              np.asarray(profit),np.asarray(previous_offer))),
                      columns=col_names,
                      index=pd.Index(index_name))
    df.to_csv(os.path.join(results_directory,"generator_segment_offer.csv"))
    
def export_zonal_price(instance, timepoints_set, zones_set, results_directory, is_MPEC):
    
    results_prices = []
    index_name = []
    timepoints_list = []
    voltage_angle_list = []
    load = []
    
    for z in zones_set:
        for t in timepoints_set:
            index_name.append(z)
            if is_MPEC:
                results_prices.append(format_2f(instance.zonalprice[t,z].value))
            else:
                results_prices.append(format_2f(instance.dual[instance.LoadConstraint[t,z]]))
            
            timepoints_list.append(t)        
            voltage_angle_list.append(format_2f(instance.voltage_angle[t,z].value))
            load.append(format_2f(instance.gross_load[t,z]))
            
    load_payment = [float(a)*float(b) for a,b in zip(results_prices,load)]
    col_names = ['hour','LMP','VoltageAngle','Load','LoadPayment']
    df = pd.DataFrame(data=np.column_stack((np.asarray(timepoints_list), np.asarray(results_prices),
                                            np.asarray(voltage_angle_list),np.asarray(load),
                                            np.asarray(load_payment))),
                       columns=col_names,
                       index=pd.Index(index_name))
    
    df.to_csv(os.path.join(results_directory,"zonal_prices.csv"))
    
def export_lines(instance, timepoints_set, transmission_lines_set, results_directory, is_MPEC):
    
    transmission_duals_from = []
    transmission_duals_to = []
    results_transmission_line_flow = []
    dc_opf_dual = []
    index_name = []
    for line in transmission_lines_set:
        for t in timepoints_set:
            index_name.append(line+"-"+str(t))
            if is_MPEC:
                transmission_duals_from.append(format_2f(instance.transmissionmindual[t,line].value))
                transmission_duals_to.append(format_2f(instance.transmissionmaxdual[t,line].value))
            else:
                transmission_duals_from.append(format_2f(instance.dual[instance.TxFromConstraint[t,line]]))
                transmission_duals_to.append(format_2f(instance.dual[instance.TxToConstraint[t,line]]))
            results_transmission_line_flow.append(format_2f(instance.transmit_power_MW[t,line].value))
            #dc_opf_dual.append(format_2f(instance.dual[instance.DCOPFConstraint[t,line]]))
    col_names = ['flow (MW)','congestion price from ($/MW)','congestion price to ($/MW)']#['congestion price ($/MW)','flow (MW)', 'OPF Dual']
    #df = pd.DataFrame(data=np.column_stack((np.asarray(transmission_duals),np.asarray(results_transmission_line_flow),np.asarray(dc_opf_dual))),
    #                  columns=col_names,index=pd.Index(index_name))
    df = pd.DataFrame(data=np.column_stack((np.asarray(results_transmission_line_flow),np.asarray(transmission_duals_from),
                                            np.asarray(transmission_duals_to))),
                      columns=col_names,index=pd.Index(index_name))
    df.to_csv(os.path.join(results_directory,"tx_flows.csv"))
    

def export_generator_commits_reserves(instance, timepoints_set, generators_set, results_directory):
    
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
    for g in generators_set:
        for t in timepoints_set:
            index_name.append(str(g)+","+str(t))
            results_gens.append(g)
            results_time.append(t)
            results_commitment.append(instance.commitment[t,g].value)
            results_starts.append(instance.startup[t,g].value)
            results_shuts.append(instance.shutdown[t,g].value)
            
            if t==1 and instance.commitinit[g]==instance.commitment[t,g].value:
                results_hourson.append(instance.commitinit[g] * instance.upinit[g] + instance.commitment[t,g].value)
                results_hoursoff.append((1-instance.commitinit[g]) * instance.downinit[g] + (1-instance.commitment[t,g].value))
            elif instance.startup[t,g].value==1 or instance.shutdown[t,g].value==1:
                results_hourson.append(instance.commitment[t,g].value)
                results_hoursoff.append((1-instance.commitment[t,g].value))
            else:
                results_hourson.append(results_hourson[-1]+instance.commitment[t,g].value)
                results_hoursoff.append(results_hoursoff[-1]+(1-instance.commitment[t,g].value))
            
            results_primarysynchreserves.append(format_2f(instance.synchreserves[t,g].value))
            results_primarynonsynchreserves.append(format_2f(instance.nonsynchreserves[t,g].value))
            results_allreserves.append(format_2f(instance.synchreserves[t,g].value + instance.nonsynchreserves[t,g].value))
            results_secondaryreserves.append(format_2f(instance.secondaryreserves[t,g].value))
    
    col_names = ['Gen_Index','timepoint','Committed','Started','Shut','TimeOn','TimeOff',
                 'Total Held as Primary Synch Reserves (MW)', 'Total Held as Primary NonSynch Reserves (MW)',
                 'Total Held as Primary Reserves (MW)', 'Total Held as Secondary Reserves (MW)']
    df = pd.DataFrame(data=np.column_stack((np.asarray(results_gens), np.asarray(results_time), np.asarray(results_commitment),
                                            np.asarray(results_starts), np.asarray(results_shuts),np.asarray(results_hourson),
                                            np.asarray(results_hoursoff),
                                            np.asarray(results_primarysynchreserves), np.asarray(results_primarynonsynchreserves),
                                            np.asarray(results_allreserves), np.asarray(results_secondaryreserves))),
                      columns=col_names,index=pd.Index(index_name))
    df.to_csv(os.path.join(results_directory,"generator_commits_reserves.csv"), index=False)
    
def export_reserve_segment_commits(instance, timepoints_set, ordc_segments_set, results_directory):
    
    results_synch_segments = []
    results_nonsynch_segments = []
    results_secondary_segments = []
    index_name = []
    for s in ordc_segments_set:
        for t in timepoints_set:
            index_name.append(str(s)+","+str(t))
            results_synch_segments.append(format_2f(instance.segmentreserves[t,s].value))
            results_nonsynch_segments.append(format_2f(instance.nonsynchsegmentreserves[t,s].value))
            results_secondary_segments.append(format_2f(instance.secondarysegmentreserves[t,s].value))
    col_names = ['MW on primary synch reserve segment', 'MW on primary nonsynch reserve segment',
                 'MW on secondary reserve segment']
    df = pd.DataFrame(data=np.column_stack((np.asarray(results_synch_segments), np.asarray(results_nonsynch_segments),
                                            np.asarray(results_secondary_segments))),
                      columns=col_names,index=pd.Index(index_name))
    df.to_csv(os.path.join(results_directory,"reserve_segment_commit.csv"))
    
def export_storage(instance, timepoints_set, storage_set, results_directory, is_MPEC):
    
    index_name = []
    results_time = []
    storage_dispatch = []
    #storage_charge = []
    #storage_discharge = []
    soc = []
    storage_offer = []
    storage_max_dual = []
    storage_min_dual = []
    node = []
    lmp = []
    
    for t in timepoints_set:
        for s in storage_set:
            index_name.append(s)
            results_time.append(t)
            #storage_charge.append(format_2f(instance.charge[t,s].value))
            #storage_discharge.append(format_2f(instance.discharge[t,s].value))
            storage_dispatch.append(format_2f(instance.storagedispatch[t,s].value))
            soc.append(format_2f(instance.soc[t,s].value))
            storage_offer.append(format_2f(instance.storageoffer[t,s].value))
            storage_max_dual.append(format_2f(instance.storagemaxdual[t,s].value))
            storage_min_dual.append(format_2f(instance.storagemindual[t,s].value))
            node.append(instance.storage_zone_label[s])
            if is_MPEC:
                lmp.append(format_2f(instance.zonalprice[t,instance.storage_zone_label[s]].value))
            else:
                lmp.append(format_2f(instance.dual[instance.LoadConstraint[t,instance.storage_zone_label[s]]]))
                
    profit = [float(c)*float(price) for c,price in zip(storage_dispatch,lmp)]
    col_names = ['time','dispatch','soc','offer','maxdual','mindual','node','lmp','profit']
    df = pd.DataFrame(data=np.column_stack((np.asarray(results_time), np.asarray(storage_dispatch),
                                            np.asarray(soc),
                                            np.asarray(storage_offer),np.asarray(storage_max_dual),
                                            np.asarray(storage_min_dual), np.asarray(node),
                                            np.asarray(lmp), np.asarray(profit))),
                      columns=col_names,index=pd.Index(index_name))
    
    df.to_csv(os.path.join(results_directory,"storage_dispatch.csv"))
    
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
            results_wind.append(instance.windgen[t,z].value)
            results_solar.append(instance.solargen[t,z].value)
            results_curtailment.append(instance.curtailment[t,z].value)
            
    VRE = pd.DataFrame({"timepoint": tmps, "zone": zones, "wind":results_wind, "solar": results_solar, "curtailment":results_curtailment})
    VRE.to_csv(os.path.join(results_directory,"renewable_generation.csv"), index=False)
>>>>>>> 33b8c0e3277c1b0395cc8f618f5b8608e292a7f1
