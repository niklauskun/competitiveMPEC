# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:55:18 2020

@author: Luke
"""


# general imports
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
import warnings
import matplotlib.pyplot as plt
from collections import OrderedDict
from pyutilib.services import TempfileManager
from pyomo.environ import Suffix, TransformationFactory, Set
from pyomo.core.base.sets import OrderedSimpleSet
from pyomo.gdp import bigm
from pyomo.opt import SolverFactory

import input_competitive_test
import input_competitive_RT
import model_competitive_test
import write_results_competitive


def create_scenario_list(start_str, end_str, is_RT, tmps, total_tmps):
    """ creates list of scenarios from input dates

    Arguments:
        start_str {str} -- beginning day date as string in mm-dd-yy format
        end_str {str} -- ending day date as string in mm-dd-yy format (exclusive)

    Raises:
        Exception: end date must be strictly after start date

    Returns:
        [list] -- info about case
    """
    assert total_tmps % tmps == 0
    start = datetime.datetime.strptime(start_str, "%m-%d-%Y")
    end = datetime.datetime.strptime(end_str, "%m-%d-%Y")
    if start >= end:
        raise Exception("end date must be after start date")
    date_generated = [
        start + datetime.timedelta(days=x) for x in range(0, (end - start).days)
    ]

    date_folders = [date.strftime("%m.%d.%Y") for date in date_generated]
    tmps_range = [i for i in range(1, int(total_tmps / tmps) + 1)]
    if is_RT:
        return [(d, False, "", 1, i) for i in tmps_range for d in date_folders]
    else:
        return [(d, False, "", 1, 1) for d in date_folders]


def update_offers(dir_str):
    """ updates competitive offers of generators. Really only relevant for EPEC with multiple iterations

    Arguments:
        dir_str {class(DirStructure)} -- file directory structure

    Returns:
        [list] -- list of bids (offers) by generator segment
    """
    df = pd.read_csv(
        join(dir_str.INPUTS_DIRECTORY, "generator_segment_marginalcost.csv")
    )
    df.prev_offer = df.marginal_cost
    df.to_csv(
        join(dir_str.INPUTS_DIRECTORY, "generator_segment_marginalcost.csv"),
        index=False,
    )  # overwrite
    overwritten_offers = [0] * len(df.prev_offer)
    return overwritten_offers


class CreateAndRunScenario(object):
    def __init__(
        self,
        dir_str,
        load_init,
        is_MPEC,
        is_RT,
        mitigate_storage_offers,
        genco_index,
        overwritten_offers,
        *args,
        **kwargs
    ):
        self.dir_str = dir_str
        self.scenario_inputs_directory = dir_str.INPUTS_DIRECTORY
        self.scenario_results_directory = dir_str.RESULTS_DIRECTORY
        self.scenario_logs_directory = dir_str.LOGS_DIRECTORY
        self.load_init = load_init
        self.mitigate_storage_offers = mitigate_storage_offers
        self.is_MPEC = is_MPEC
        self.is_RT = is_RT
        self.genco_index = genco_index
        self.overwritten_offers = overwritten_offers

        self.args = args
        # unpack kwargs, which should be only CPLEX params and a warmstart_flag
        self.cplex_params = {}
        for k, v in kwargs.items():
            self.cplex_params[k] = v

    def create_problem_instance(self):
        """ instantiates and loads data into a pyomo model
        if it's a MPEC, the model is also reformulated using the "BigM" method
        Pyomo says this is deprecated, but it seems to work ok. Still a little concerning

        Returns:
            [pyomo.core.base.PyomoModel.ConcreteModel] -- a Pyomo model instance
        """

        # Get model, load data, and solve
        print("Reading model...")
        self.model = model_competitive_test.dispatch_model
        print("...model read.")

        print("creating competitive generators file...")
        pd.DataFrame(
            data=[self.genco_index], columns=["genco"], index=pd.Index(["index"])
        ).to_csv(os.path.join(self.scenario_inputs_directory, "case_index.csv"))
        print("...competitive generators recorded.")

        print("Loading data...")
        storageclass = StorageOfferMitigation(
            self.dir_str, self.is_RT, mitigation_flag=False, suppress_print=True
        )
        storageclass.write_SPP_mitigated_offers()
        if self.is_RT:
            self.data = input_competitive_RT.scenario_inputs(
                self.scenario_inputs_directory
            )
            print(".. real-time data read.")
        else:
            self.data = input_competitive_test.scenario_inputs(
                self.scenario_inputs_directory
            )
            print(".. day-ahead data read.")

        print("Compiling instance...")
        instance = self.model.create_instance(self.data)
        print("...instance created.")

        print("Creating Offer Mitigation (model will solve as dispatch if TRUE)...")
        if self.mitigate_storage_offers:
            self.instance = instance
            self.instance.dual = Suffix(direction=Suffix.IMPORT)
            solution_pre = self.solve("LP")
            write_results_competitive.export_results(
                instance,
                solution_pre,
                self.scenario_results_directory,
                False,
                debug_mode=1,
            )
        storageclass = StorageOfferMitigation(
            self.dir_str, self.is_RT, mitigation_flag=self.mitigate_storage_offers
        )
        storageclass.write_SPP_mitigated_offers()
        print("...storage offer mitigation file created")

        if self.is_MPEC:
            print("Converting model to MPEC...")
            # transformed = model.transform("mpec.simple_nonlinear")
            xfrm = TransformationFactory("mpec.simple_disjunction")
            xfrm.apply_to(instance)
            xfrm2 = TransformationFactory("gdp.bigm")
            xfrm2.apply_to(instance)
            print("...converted")

        return instance

    def solve(self, case_type, mip_iter=1, warmstart_flag=False):
        """ solve pyomo model(s). Some recursion happens to both handle errors and 
        model relaxations (e.g., after fixing storage commitment, presolve for cases with many competitive generators)

        Arguments:
            case_type {str} -- either "MIP" or "LP". "MIP" is for competitive cases, "LP" is cost-min dispatch

        Keyword Arguments:
            mip_iter {int} -- iteration of pre (1) and full solve (>1) for competitive cases (default: {1})
            warmstart_flag {bool} -- whether to warmstart full model with presolved solution (default: {False})

        Returns:
            <class 'pyomo.opt.results.results_.SolverResults'> -- pyomo model solution
        """
        for a in self.args:
            constraint = getattr(self.instance, a)
            constraint.deactivate()  # deactivate *args constraints passed for deactivation

        if case_type == "MIP" and mip_iter == 1:
            print(
                "NOTE: initial MIP solve with only storage competitive to get feasible solution"
            )
            self.instance.TotalCost.deactivate()  # deactivates the simple objective
            self.instance.TotalCost2.deactivate()
            self.instance.GeneratorProfit.deactivate()
            self.instance.GeneratorProfitDualPre.activate()
            self.instance.GeneratorProfitDual.deactivate()

        elif case_type == "MIP" and mip_iter > 1:
            print(
                "NOTE: resolving with all competitive generators using storage-only solution to warmstart"
            )
            self.instance.GeneratorProfitDualPre.deactivate()
            self.instance.GeneratorProfitDual.activate()

        elif case_type == "LP":
            self.instance.TotalCost2.activate()
            self.instance.TotalCost.deactivate()  # switch objective to exclude start-up and no-load costs
            self.instance.GeneratorProfit.deactivate()
            self.instance.GeneratorProfitDualPre.deactivate()
            self.instance.GeneratorProfitDual.deactivate()
            # instance.PminConstraint.deactivate()

        # ### Solve ### #
        if self.executable != "":
            print("using user-defined executable to call " + self.solver_name)
            solver = SolverFactory(self.solver_name, executable=self.executable)
        else:
            solver = SolverFactory(self.solver_name)

        print("Solving...")

        # if mip_iter > 1:
        for k, v in self.cplex_params.items():
            solver.options[k] = v  # update solver options for warm-started solve
        # to keep human-readable files for debugging, set keepfiles = True

        try:
            solution = solver.solve(
                self.instance, tee=True, warmstart=warmstart_flag
            )  # , keepfiles=False

        # solution = solver.solve(instance, tee=True, keepfiles=False, options={'optimalitytarget':1e-5})
        except PermissionError:
            print("Yuck, a permission error")
            for file in glob.glob("*.log"):
                print("removing log files due to Permission Error")
                file_path = open(file)
                file_path.close()
                time.sleep(1)
                os.remove(file)
            return self.solve(
                case_type, mip_iter=mip_iter, warmstart_flag=warmstart_flag
            )

        if case_type == "MIP" and mip_iter == 1:
            try:
                warmstart_flag = self.cplex_params.pop("warmstart_flag")
            except KeyError:
                print(
                    "NOTE: no warmstart specified for MIP iteration, default behavior is true"
                )
                warmstart_flag = True
            assert (type(warmstart_flag)) == bool  # make sure this is a boolean
            self.solve(
                case_type, mip_iter=2, warmstart_flag=warmstart_flag
            )  # run second warm-started iteration for MIP

        return solution

    def run_scenario(self, solver_name="cplex", executable=""):
        """ runs the input model scenario

        Keyword Arguments:
            solver_name {str} -- name of your solver (default: {'cplex'})
            executable {str} -- path of your CPLEX. Default assumes solver on your path (default: {""})
        """
        self.executable = executable
        self.solver_name = solver_name
        if self.load_init:
            self.scenario_createinputs_directory = os.path.join(
                self.dir_str.INIT_DIRECTORY
            )
        else:
            self.scenario_createinputs_directory = None

        # Write logs to this directory
        TempfileManager.tempdir = self.scenario_logs_directory

        # Create problem instance
        self.instance = self.create_problem_instance()

        # Create a 'dual' suffix component on the instance, so the solver plugin will know which suffixes to collect
        self.instance.dual = Suffix(direction=Suffix.IMPORT)

        if self.is_MPEC:
            self.solution_type = "MIP"
            self.solution = self.solve(self.solution_type)

        else:
            self.solution_type = "LP"
            self.solution = self.solve(
                self.solution_type
            )  # solve LP, storage dispatch now linearized
            # self.instance.storagebool.fix()  # relaxes to lp after mip solve if needed
            # self.solution = self.solve(self.solution_type)

        # export results to csvs
        write_results_competitive.export_results(
            self.instance,
            self.solution,
            self.scenario_results_directory,
            self.is_MPEC,
            debug_mode=1,
        )

    def format_solution_for_plots(self):
        """ reformats pyomo model instance vars and params into lists for plotting

        Returns:
            [dict] -- dictionary with formatted case results for plotting
        """
        results = {}  # create dict for storing results
        results["tmps"] = [t for t in self.instance.TIMEPOINTS]

        results["dispatch"] = [
            self.instance.gd[t, g]()
            for t in self.instance.TIMEPOINTS
            for g in self.instance.GENERATORS
        ]

        results["starts"] = [
            self.instance.gup[t, g].value
            for t in self.instance.TIMEPOINTS
            for g in self.instance.GENERATORS
        ]
        results["shuts"] = [
            self.instance.gsd[t, g].value
            for t in self.instance.TIMEPOINTS
            for g in self.instance.GENERATORS
        ]

        results["wind"], results["solar"], results["curtailment"], results["lmps"] = (
            [],
            [],
            [],
            [],
        )
        for t in self.instance.TIMEPOINTS:
            for z in self.instance.ZONES:
                results["wind"].append(self.instance.windgen[t, z].value)
                results["solar"].append(self.instance.solargen[t, z].value)
                results["curtailment"].append(self.instance.curtailment[t, z].value)
                if self.solution_type == "LP":
                    results["lmps"].append(
                        self.instance.dual[self.instance.LoadConstraint[t, z]]
                    )
                else:
                    results["lmps"].append(self.instance.zonalprice[t, z].value)

        return results  # dict only

    def diagnostic_plots(self):
        """ creates two plots. (1) Generation by fuel. (2) LMP by bus
        There are a few hardcoded assumptions that will break if generator types are renamed 
        """
        results_dict = self.format_solution_for_plots()
        lmp_duals_np = np.reshape(
            results_dict["lmps"],
            (
                int(len(results_dict["tmps"])),
                int(len(results_dict["lmps"]) / len(results_dict["tmps"])),
            ),
        )
        # read in the gen and zone types so aggregation can be done for plots
        gens = pd.read_csv(
            join(self.scenario_inputs_directory, "generators_descriptive.csv")
        )
        zones = pd.read_csv(join(self.scenario_inputs_directory, "zones.csv"))

        plot_df = pd.DataFrame(
            {
                "Dispatch": results_dict["dispatch"],
                "FuelID": list(gens["Category"].values) * (len(results_dict["tmps"])),
                "Hours": [
                    i
                    for i in list(range(1, len(results_dict["tmps"]) + 1))
                    for z in range(len(list(gens["Category"].values)))
                ],
            }
        )
        plot_df_grouped = (
            plot_df.groupby(["FuelID", "Hours"]).sum().reset_index().set_index("Hours")
        )
        fig, ax = plt.subplots()  # figsize=(9, 6)
        df_pivot = plot_df_grouped.pivot(columns="FuelID", values="Dispatch")

        col_name_colors = OrderedDict()
        col_name_colors["Nuclear"] = "purple"
        col_name_colors["Hydro"] = "blue"
        col_name_colors["Coal"] = "k"
        col_name_colors["Gas CC"] = "orange"
        col_name_colors["Gas CT"] = "sienna"
        col_name_colors["Oil CT"] = "g"
        col_name_colors["Oil ST"] = "g"
        col_name_colors["Wind"] = "cyan"
        col_name_colors["CSP"] = "red"
        col_name_colors["Solar PV"] = "yellow"
        col_name_colors["Solar RTPV"] = "yellow"
        col_name_colors["Storage"] = "slategray"
        col_name_colors["Sync_Cond"] = "k"

        df_pivot = df_pivot.reindex(columns=[k for k in col_name_colors.keys()])
        df_pivot = df_pivot.abs()  # get rid of below zero rounding errors
        # df_pivot.to_csv('checkpivot.csv'), write to check if needed

        df_pivot.plot.area(ax=ax, color=[c for c in col_name_colors.values()])
        plt.show()

        # Your x and y axis
        x = range(1, len(results_dict["tmps"]) + 1)
        # y is made above

        # and finally, plot the energy LMP dual
        lmp_palette = ["r", "b", "m", "k", "g", "y"] * 16
        legend_label = []
        for z in range(len(zones["zone"])):
            plt.plot(x, lmp_duals_np[:, z], color=lmp_palette[z])
            legend_label.append("Zone " + str(zones["zone"][z]))
        plt.ylabel("LMP ($/MWh)")
        plt.xlabel("Hour")
        plt.show()


def create_default_prices_df(case_directory, is_RT):
    if is_RT:
        df = pd.read_csv(
            os.path.join(case_directory.INPUTS_DIRECTORY, "timepoints_zonal_rt.csv")
        )
    else:
        df = pd.read_csv(
            os.path.join(case_directory.INPUTS_DIRECTORY, "timepoints_zonal.csv")
        )
    df = df[["timepoint", "zone"]].sort_values("zone").set_index("zone")
    df["LMP"] = [0] * len(df.index)
    df.reset_index(inplace=True)
    df.columns = ["zone", "hour", "LMP"]
    return df


class StorageOfferMitigation(object):
    def __init__(
        self, case_directory, is_RT, mitigation_flag=True, suppress_print=False
    ):
        self.case_directory = case_directory
        self.mitigation_flag = mitigation_flag
        if is_RT:
            self.storage_df = pd.read_csv(
                os.path.join(
                    case_directory.INPUTS_DIRECTORY, "storage_resources_rt.csv"
                )
            )
        else:
            self.storage_df = pd.read_csv(
                os.path.join(case_directory.INPUTS_DIRECTORY, "storage_resources.csv")
            )  # storage_df
        try:
            self.prices_df = pd.read_csv(
                os.path.join(case_directory.RESULTS_DIRECTORY, "zonal_prices.csv")
            )  # prices_df
        except FileNotFoundError:
            if not suppress_print:
                print("NOTE: storage offers will not be mitigated")
            self.prices_df = create_default_prices_df(case_directory, is_RT)
            self.mitigation_flag = False
        self.prices_df.columns = ["zone"] + list(self.prices_df.columns[1:])
        self.storage_prices = pd.merge(
            self.storage_df,
            self.prices_df[["zone", "hour", "LMP"]],
            how="left",
            left_on=["StorageZoneLabel"],
            right_on=["zone"],
        )
        self.RTE = round(
            self.storage_df["ChargeEff"].mean()
            / self.storage_df["DischargeEff"].mean(),
            3,
        )

    def period_type(self, i, df):
        if i == len(df.index) - 1:
            return "last"  # last same as prvs
        elif i == 0:
            return "+"
        elif not np.isnan(df.at[i + 1, "min"]):
            return "+"
        elif not np.isnan(df.at[i + 1, "max"]):
            return "-"
        elif not np.isnan(df.at[i + 1, "absmax"]):
            return "-"
        elif not np.isnan(df.at[i + 1, "absmin"]):
            return "+"
        else:
            return self.period_type(i - 1, df)

    def write_SPP_mitigated_offers(self):
        # print(self.storage_prices)
        storage_list = []
        for esr in self.storage_prices["Storage_Index"].unique():
            subset_storage_df = (
                self.storage_prices[(self.storage_prices.Storage_Index == esr)]
                .copy()
                .reset_index()
            )
            # Find local peaks
            subset_storage_df["min"] = subset_storage_df.LMP[
                (
                    (subset_storage_df.LMP.shift(1) > subset_storage_df.LMP)
                    & (subset_storage_df.LMP.shift(-1) > subset_storage_df.LMP)
                )
            ]
            subset_storage_df["max"] = subset_storage_df.LMP[
                (subset_storage_df.LMP.shift(1) < subset_storage_df.LMP)
                & (subset_storage_df.LMP.shift(-1) < subset_storage_df.LMP)
            ]
            # absolute peaks?
            subset_storage_df["absmin"] = subset_storage_df.LMP[
                subset_storage_df.LMP == subset_storage_df.LMP.min()
            ]
            subset_storage_df["absmax"] = subset_storage_df.LMP[
                subset_storage_df.LMP == subset_storage_df.LMP.max()
            ]
            # categorize whether approaching peak or trough
            # based off SPP's "DYNAMIC OPPORTUNITY COST MITIGATED ENERGY OFFER FRAMEWORK FOR ELECTRIC STORAGE RESOURCES"
            test_l = []
            for i in subset_storage_df.index:
                test_l.append(self.period_type(i, subset_storage_df))
            subset_storage_df["flag"] = test_l

            charge_list = []
            discharge_list = []
            for i in subset_storage_df.index:
                if subset_storage_df.at[i, "flag"] == "-":
                    discharge_list.append(subset_storage_df.at[i + 1, "LMP"])
                    charge_list.append(subset_storage_df.at[i + 1, "LMP"] * self.RTE)
                elif subset_storage_df.at[i, "flag"] == "+":
                    discharge_list.append(subset_storage_df.at[i + 1, "LMP"] / self.RTE)
                    charge_list.append(subset_storage_df.at[i + 1, "LMP"])
                elif subset_storage_df.at[i, "flag"] == "last":
                    charge_list.append(0)
                    discharge_list.append(subset_storage_df.at[i, "LMP"] / self.RTE)
            subset_storage_df["ChargeMaxOffer"] = charge_list
            subset_storage_df["DischargeMaxOffer"] = discharge_list
            # print(
            #    subset_storage_df[
            #        ["Storage_Index", "ChargeMaxOffer", "DischargeMaxOffer"]
            #    ]
            # )
            storage_list.append(
                subset_storage_df[
                    ["hour", "Storage_Index", "ChargeMaxOffer", "DischargeMaxOffer"]
                ]
            )
        storage_df = pd.concat(storage_list, axis=0)
        storage_df.sort_values("hour", inplace=True)
        storage_df.columns = [
            "timepoint",
            "Storage_Index",
            "ChargeMaxOffer",
            "DischargeMaxOffer",
        ]
        if not self.mitigation_flag:
            storage_df.ChargeMaxOffer = [5000 for i in storage_df.ChargeMaxOffer]
            storage_df.DischargeMaxOffer = [5000 for i in storage_df.DischargeMaxOffer]
        storage_df.to_csv(
            os.path.join(self.case_directory.INPUTS_DIRECTORY, "storage_offers.csv"),
            index=False,
        )
        return storage_df.reset_index()


def write_timepoint_subset(directory, is_RT, tmps, slicer):
    if not is_RT:
        case_dict = {}
        existing_df = pd.read_csv(
            join(directory.INPUTS_DIRECTORY, "timepoints_index.csv")
        )
        case_dict["timepoint"] = existing_df.timepoint
        case_dict["first_timepoint"] = [min(case_dict["timepoint"])] * len(
            case_dict["timepoint"]
        )
    # if not RT, just copy the DA file with a new name
    else:
        case_dict = {}
        existing_tmps = pd.read_csv(
            join(directory.INPUTS_DIRECTORY, "timepoints_index_rt.csv")
        )
        existing_tmps_list = list(existing_tmps.timepoint)
        assert max(existing_tmps_list) % tmps == 0
        case_dict["timepoint"] = existing_tmps_list[tmps * (slicer - 1) : tmps * slicer]
        case_dict["first_timepoint"] = [min(case_dict["timepoint"])] * len(
            case_dict["timepoint"]
        )
    df = pd.DataFrame.from_dict(case_dict)
    df.set_index("timepoint", inplace=True)
    df.to_csv(join(directory.INPUTS_DIRECTORY, "timepoints_index_subset_rt.csv"))


def create_case_suffix(directory, RT, rt_tmps, n_iter):
    if not RT:
        return "_DA"
    else:
        case_string = (
            "_" + str((n_iter - 1) * rt_tmps + 1) + "_" + str((n_iter) * rt_tmps)
        )
        return "_RT" + case_string


def write_DA_bids(directory,RT):
    directory = directory
    storage_list = []
    storage_list2 = []
    if RT:
        try:
            bid_df = pd.read_csv(
                os.path.join(directory.RESULTS_DIRECTORY, "storage_dispatch.csv")
            )
        except FileNotFoundError:
            print("NOTE: storage offer not exist")
        bid_df.rename(columns={'Unnamed: 0':'Storage_Index'}, inplace = True)
        for esr in bid_df["Storage_Index"].unique():
            subset_bid_df = (
                bid_df[(bid_df.Storage_Index == esr)]
                .copy()
                .reset_index()
            )
            storage_list.append(
                subset_bid_df[
                    ["time", "Storage_Index", "discharge_offer", "charge_offer"]
                ]
            )
            storage_tmp = pd.concat(storage_list, axis=0)
            storage_tmp2 = pd.DataFrame()
            for i in range(len(storage_tmp)):
                a=storage_tmp.loc[i]
                d=pd.DataFrame(a).T
                storage_tmp2=storage_tmp2.append([d]*12)
            for j in range(len(storage_tmp2)):
                storage_tmp2.iloc[j,0] = j + 1
            storage_list2.append(
                storage_tmp2[
                    ["time", "Storage_Index", "discharge_offer", "charge_offer"]
                ]
            )
        storage_df = pd.concat(storage_list2, axis=0)
        storage_df.sort_values("time", inplace=True)
        storage_df.columns = [
            "timepoint",
            "Storage_Index",
            "discharge_offer",
            "charge_offer",
        ]
        storage_df.to_csv(
            os.path.join(directory.INPUTS_DIRECTORY, "storage_offers_DA.csv"),
            index=False,
        )
        return storage_df.reset_index()