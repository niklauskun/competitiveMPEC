# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:48:54 2020

@author: Luke
"""

# general imports
import os
from os.path import join
import pandas as pd
import numpy as np
import math
import time
import sys
import datetime

# import other scripts
import input_competitive_test  # DataPortal for loading data from csvs into Pyomo model
import model_competitive_test  # the actual Pyomo model formulation
import write_results_competitive  # writes model outputs to csvs

# import utility functions
# these mostly help set up and do model runs
from utility_functions import (
    update_offers,
    create_scenario_list,
    CreateAndRunScenario,
)

start_time = time.time()
cwd = os.getcwd()

### GENERAL INPUTS ###
case_folder = "Desktop\\competitiveMPECNik\\test"  # andWind309

start_date = "01-01-2019"  # use this string format
end_date = "02-05-2019"  # end date is exclusive
MPEC = True  # if you wish to run as MPEC, if false runs as min cost dispatch LP
EPEC, iters = False, 9  # if EPEC and max iterations if True.
show_plots = False  # if True show plot of gen by fuel and bus LMPs after each case

### OPTIONAL SOLVER INPUTS ###
executable_path = ""  # if you wish to specify cplex.exe path
solver_name = "cplex"  # only change if you wish to use a solver other than cplex
solver_kwargs = {
    "parallel": -1,
    "mip_tolerances_mipgap": 0.01,
    "dettimelimit": 300000,
}  # note if you use a non-cplex solver, you may have to change format of solver kwargs
#    "warmstart_flag": True,
### OPTIONAL MODEL MODIFYING INPUTS ###
# for now, I'll just include ability here to deactivate constraints if you don't want the model to use them
deactivated_constraint_args = []  # list of constraint names to deactivate
# an example that won't affect problem much is "OfferCapConstraint"
# "OneCycleConstraint"

### END INPUTS ###

# Directory structure, using existing files rather than creating case structure for now
class DirStructure(object):
    """
    Create directory and file structure.
    """

    def __init__(self, code_directory, case_folder, load_init, load_dir):
        self.DIRECTORY = code_directory
        # self.DIRECTORY = os.path.join(self.CODE_DIRECTORY, "..")
        self.CASE_DIRECTORY = os.path.join(self.DIRECTORY, case_folder)
        self.INPUTS_DIRECTORY = os.path.join(
            self.CASE_DIRECTORY, scenario_name, "inputs"
        )
        self.RESULTS_DIRECTORY = os.path.join(
            self.CASE_DIRECTORY, scenario_name, "results"
        )
        self.LOGS_DIRECTORY = os.path.join(self.DIRECTORY, "logs")
        if load_init:
            self.INIT_DIRECTORY = os.path.join(self.CASE_DIRECTORY, load_dir, "results")

    def make_directories(self):
        if not os.path.exists(self.RESULTS_DIRECTORY):
            os.mkdir(self.RESULTS_DIRECTORY)
        if not os.path.exists(self.LOGS_DIRECTORY):
            os.mkdir(self.LOGS_DIRECTORY)


# Logging
class Logger(object):
    """
    The print statement will call the write() method of any object you assign to sys.stdout,
    so assign the terminal (stdout) and a log file as output destinations.
    """

    def __init__(self, directory_structure):
        self.terminal = sys.stdout
        self.log_file_path = os.path.join(
            directory_structure.LOGS_DIRECTORY,
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_"
            + str(scenario_name)
            + ".log",
        )
        self.log_file = open(self.log_file_path, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


scenario_list = create_scenario_list(start_date, end_date)  # create scenario list
### RUN MODEL ###
for counter, s in enumerate(scenario_list):
    # initialize scenario data in the tuple
    if EPEC:
        print("EPEC not currently enabled, so exiting")
        break
    scenario_name, load_init, load_dir, genco_index = (
        s[0],
        s[1],  # this and the next one are only needed for initializing a case based on
        s[2],  # a previous day. Don't worry about this for now
        s[3],  # this is only needed for EPEC
    )

    # run the case, as usual
    code_directory = cwd
    dir_str = DirStructure(code_directory, case_folder, load_init, load_dir)
    dir_str.make_directories()
    logger = Logger(dir_str)
    log_file = logger.log_file_path
    print(
        "Running scenario "
        + str(counter + 1)
        + " of "
        + str(len(scenario_list))
        + " ("
        + str(scenario_name)
        + ")..."
    )
    stdout = sys.stdout
    sys.stdout = logger

    # updates generator offers for iterable scenarios
    overwritten_offers = update_offers(dir_str)

    # create and run scenario (this is the big one)
    scenario = CreateAndRunScenario(
        dir_str,
        load_init,
        MPEC,
        genco_index,
        overwritten_offers,
        *deactivated_constraint_args,
        **solver_kwargs
    )
    scenario.run_scenario(
        solver_name=solver_name, executable=executable_path
    )  # can pass executable if desired

    sys.stdout = stdout  # return to original

    # show diagnostic plots if you wanted to
    if show_plots:
        scenario.diagnostic_plots()

    end_time = time.time() - start_time
    print("time elapsed during run is " + str(round(end_time, 2)) + " seconds")

"""
old EPEC code, will update when I have time
count_iters = 0
if EPEC:
    for i in range(iters):
        count_iters += 1
        print("Running EPEC iter " + str(count_iters) + " of " + str(iters) + "...")
        # reformulate previous offer of generators for non-initial iters
        if count_iters > 1:

            df = pd.read_csv(
                join(dir_str.INPUTS_DIRECTORY, "generator_segment_marginalcost.csv")
            )
            break_flag = True
            for v in range(len(scenario_results[-1])):
                if abs(scenario_results[-1][v] - df.prev_offer[v]) > 0.1:
                    break_flag = False
            if break_flag:
                print("reached equilibrium, exiting run with solution in hand...")
                df.prev_offer = scenario_results[-1]
                # do comparison, eventually, that could cause loop to terminate
                df.to_csv(
                    join(
                        dir_str.INPUTS_DIRECTORY, "generator_segment_marginalcost.csv"
                    ),
                    index=False,
                )  # overwrite
                break
            df.prev_offer = scenario_results[-1]
            # do comparison, eventually, that could cause loop to terminate
            df.to_csv(
                join(dir_str.INPUTS_DIRECTORY, "generator_segment_marginalcost.csv"),
                index=False,
            )  # overwrite
            # overwritten_offers = scenario_results[-1]

        else:  # initialization always sets prev offer to mcos
            df = pd.read_csv(
                join(dir_str.INPUTS_DIRECTORY, "generator_segment_marginalcost.csv")
            )
            df.prev_offer = df.marginal_cost
            df.to_csv(
                join(dir_str.INPUTS_DIRECTORY, "generator_segment_marginalcost.csv"),
                index=False,
            )  # overwrite
            overwritten_offers = [0] * len(df.prev_offer)
            # do overwrite

        count_case = 0
        for s in scenario_list:
            count_case += 1
            # initialize scenario data in the tuple
            scenario_name = s[0]  # for now
            load_init = s[1]
            load_dir = s[2]
            genco_index = s[3]

            # run the case, as usual
            code_directory = cwd
            dir_str = DirStructure(code_directory, case_folder, load_init, load_dir)
            dir_str.make_directories()
            logger = Logger(dir_str)
            log_file = logger.log_file_path
            print(
                "Running scenario "
                + str(count_case)
                + " of "
                + str(len(scenario_list))
                + "..."
            )
            print("Running scenario " + str(scenario_name) + "...")
            stdout = sys.stdout
            sys.stdout = logger

            scenario_results = run_scenario(
                dir_str, load_init, MPEC, genco_index, overwritten_offers
            )
            overwritten_offers = scenario_results[-1]

            sys.stdout = stdout  # return to original

            # run diagnostic plots
            diag_plots(scenario_results, dir_str)

            end_time = time.time() - start_time
            print("time elapsed during run is " + str(end_time) + " seconds")
"""
