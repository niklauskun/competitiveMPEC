<<<<<<< HEAD
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
import input_competitive  # DataPortal for loading data from csvs into Pyomo model
import model_competitive  # the actual Pyomo model formulation
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
case_folder = "Desktop\\luke_mpec\\competitiveMPEC-master\\scripttest"  # andWind309

start_date = "01-05-2019"  # use this string format
end_date = "01-06-2019"  # end date is exclusive
MPEC = True  # if you wish to run as MPEC, if false runs as min cost dispatch LP
EPEC, iters = False, 9  # if EPEC and max iterations if True.
show_plots = False  # if True show plot of gen by fuel and bus LMPs after each case

### OPTIONAL SOLVER INPUTS ###
executable_path = ""  # if you wish to specify CPLEX path
solver_name = "cplex"  # only change if you wish to use a solver other than cplex
solver_kwargs = {
    "mip_tolerances_mipgap": 0.01,
    "dettimelimit": 300000,
    "parallel": -1,
    "warmstart_flag": False,
}  # note if you use a non-cplex solver, you may have to change format of solver kwargs

### OPTIONAL MODEL MODIFYING INPUTS ###
# for now, I'll just include ability here to deactivate constraints if you don't want the model to use them
deactivated_constraint_args = []  # list of constraint names to deactivate
# an example that won't affect problem much is "OfferCapConstraint"

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
=======
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:48:54 2020

@author: Luke
"""

#general imports
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
from pyutilib.services import TempfileManager
from pyomo.environ import *
from pyomo.mpec import *
from pyomo.gdp import bigm
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

#import other scripts
import input_competitive
import model_competitive
import write_results_competitive
from plotting_competitive import diag_plots


start_time = time.time()
cwd = os.getcwd()

case_folder = "CompetitiveStorageBus309andWind303"#"CompetitiveStorageBus303"#competitive300CCs#
#scenario_list = [("5bus",False,"",1),("5bus",False,"",2),("5bus",False,"",3),("5bus",False,"",4),("5bus",False,"",5)] #("5bus",False,"",1),

#create scenario list from dates to match folder naming convention

start = datetime.datetime.strptime("06-06-2019", "%m-%d-%Y")
end = datetime.datetime.strptime("06-08-2019", "%m-%d-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
date_folders = [date.strftime("%m.%d.%Y") for date in date_generated]
scenario_list = [(d,False,"",1) for d in date_folders]

#("1.3.2019.1",False,"",1),
#scenario_list = [("01.10.2019",False,"",1)]
MPEC = True
EPEC = False
iters = 9 #limit number of EPEC iters

#if not EPEC and MPEC:
#    assert(len(scenario_list)==1) #do this bc mpec will only solve for a single case anyway

# Allow user to specify solver path if needed (default assumes solver on path)
executable=""

#Directory structure, using existing files rather than creating case structure for now
class DirStructure(object):
    """
    Create directory and file structure.
    """
    def __init__(self, code_directory, case_folder, load_init, load_dir):
        self.DIRECTORY = code_directory
        #self.DIRECTORY = os.path.join(self.CODE_DIRECTORY, "..")
        self.CASE_DIRECTORY = os.path.join(self.DIRECTORY, case_folder)
        self.INPUTS_DIRECTORY = os.path.join(self.CASE_DIRECTORY, scenario_name, "inputs")
        self.RESULTS_DIRECTORY = os.path.join(self.CASE_DIRECTORY, scenario_name, "results")
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
        self.log_file_path = os.path.join(directory_structure.LOGS_DIRECTORY,
                                          datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_" +
                                          str(scenario_name) + ".log")
        self.log_file = open(self.log_file_path, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

            
def create_problem_instance(scenario_inputs_directory, load_init, scenario_from_directory, is_MPEC, genco_index):
    """
    Load model formulation and data, and create problem instance.
    """
    # Get model, load data, and solve
    print ("Reading model...")
    model = model_competitive.dispatch_model
    print ("...model read.")
    
    if load_init:
        print("Creating initial conditions data file...")
        create_init.create_init_file(scenario_from_directory, scenario_inputs_directory, 24)
        print("...initial conditions created.")
        
    print("creating competitive generators file...")
    pd.DataFrame(data=[genco_index], columns=["genco"],index=pd.Index(['index'])).to_csv(os.path.join(scenario_inputs_directory,"case_index.csv"))
    print("...competitive generators recorded.")

    print ("Loading data...")
    data = input_competitive.scenario_inputs(scenario_inputs_directory)
    print ("..data read.")
    
    #model.pprint()
    print ("Compiling instance...")
    instance = model.create_instance(data)
    print ("...instance created.")
    
    
    
    if is_MPEC:
        print("Converting model to MPEC...")
        #transformed = model.transform("mpec.simple_nonlinear")
        xfrm = TransformationFactory('mpec.simple_disjunction')
        xfrm.apply_to(instance)
        xfrm2 = TransformationFactory('gdp.bigm')
        xfrm2.apply_to(instance)
        print("...converted")

    return instance

def solve(instance, case_type):
    """
    Select solver for the problem instance
    Run instance of model
    """
    #Choose active/inactive objective
    #set_names
    #print(instance.get_num())#LoadConstraint.set_names('LoadConstraint')
    #print(instance.LoadConstraint.pprint())
    if case_type == "MIP":
        instance.TotalCost.deactivate() #deactivates the simple objective
        instance.TotalCost2.deactivate()
        instance.GeneratorProfit.deactivate()
        instance.GeneratorProfitDual.activate()
        
    elif case_type == "LP":
        instance.TotalCost2.activate()
        instance.TotalCost.deactivate() #switch objective to exclude start-up and no-load costs
        instance.GeneratorProfit.deactivate()
        instance.GeneratorProfitDual.deactivate()
        
        
        #instance.PminConstraint.deactivate()
    # ### Solve ### #
    if executable != "":
        solver = SolverFactory("cplex", executable=executable)
        #solver.options['mip_tolerances_absmipgap'] = 0.2 #sets mip optimality gap, which is 1e-06 by default
        #solver.options['mip_tolerances_mipgap'] = 0.01
        #solver.options['parallel'] = -1 #opportunistic
        #solver.options['dettimelimit'] = 1000000
    else:
        solver = SolverFactory("cplex") 
        #solver = SolverFactory("gurobi")
        #solver.options['mip_tolerances_absmipgap'] = 0.1
        #solver.options['optimalitytarget']=3
        
        solver.options['mip_tolerances_mipgap'] = 0.02#raise this to help solve faster
        solver.options['parallel'] = -1 #opportunistic
        
        #solver.options['conflict_algorithm']=6#conflict_display=2
        #solver.options['set_results_stream']=1
        #solver.options['feasopt_tolerance']=10000000
        #solver.options['feasopt_mode']=1
        #solver.options['set_results_stream']=1
        #solver.options['preprocessing_presolve']='n'
        #solver.options['conflict_algorithm']=6
        #solver.options['conflict_display']=2
        #solver.options['simplex_tolerances_feasibility']=.00000001
        #solver.options['lpmethod']=1
    #print('feasopt...')
    #instance.feasopt.all_constraints()
    print ("Solving...")
    
    # to keep human-readable files for debugging, set keepfiles = True
    
    try:
        solution = solver.solve(instance, tee=True, keepfiles=False)

        #solution = solver.solve(instance, tee=True, keepfiles=False, options={'optimalitytarget':1e-5})
    except PermissionError:
        print("Yuck, a permission error")
        for file in glob.glob("*.log"):
            print("removing log files due to Permission Error")
            file_path = open(file)
            file_path.close()
            time.sleep(1)
            os.remove(file)
        return solve(instance, case_type)
    
    return solution

def load_solution(instance, results):
    """
    Load results.
    """
    instance.solutions.load_from(results)

def run_scenario(directory_structure, load_init, is_MPEC, genco_index, overwritten_offers):
    """
    Run a scenario.
    """

    # Directories
    scenario_inputs_directory = os.path.join(directory_structure.INPUTS_DIRECTORY)
    scenario_results_directory = os.path.join(directory_structure.RESULTS_DIRECTORY) #results not needed yet
    scenario_logs_directory = os.path.join(directory_structure.LOGS_DIRECTORY)
    if load_init:
        scenario_createinputs_directory = os.path.join(directory_structure.INIT_DIRECTORY)
    else:
        scenario_createinputs_directory = None
    
    # Write logs to this directory
    TempfileManager.tempdir = scenario_logs_directory

    #
    # Create problem instance
    instance = create_problem_instance(scenario_inputs_directory, load_init, scenario_createinputs_directory, is_MPEC, genco_index)
            
    
    # Create a 'dual' suffix component on the instance, so the solver plugin will know which suffixes to collect
    instance.dual = Suffix(direction=Suffix.IMPORT)
    
    if is_MPEC:
        solution_type="MIP"
        solution = solve(instance,solution_type)
    else:
        solution_type = "LP" 
        solution = solve(instance,solution_type) #solve MIP with commitment
        instance.storagebool.fix() #relaxes to lp after mip solve if needed
        solution = solve(instance,solution_type)
    '''
    print ("Done running MIP, relaxing to LP to obtain duals...")
     
    #fix binary variables to relax to LP
    instance.commitment.fix()
    instance.startup.fix() 
    instance.shutdown.fix()
    instance.preprocess()   
    instance.dual = Suffix(direction=Suffix.IMPORT) 
    solution = solve(instance,"LP") 
    '''
    ###

    # export results to csv
    write_results_competitive.export_results(instance, solution, scenario_results_directory, is_MPEC, debug_mode=1)
    
    
    #load up the instance that was just solved
    load_solution(instance, solution)
    
    #this first part does the overwrite of iterated generator offers for the EPEC
    #overwritten_offers = []
    counter = 0
    #print(genco_index)
    for t in instance.TIMEPOINTS:
        for gs in instance.GENERATORSEGMENTS:
            for g in instance.GENERATORS:    
                if abs(instance.gensegmentoffer[t,g,gs].value - overwritten_offers[counter]) > .1 and genco_index==instance.genco_index[g] and instance.dispatch[t,g]()>0.1:
                    #overwritten_offers.append(instance.gensegmentoffer[t,g,gs].value)
                    #print(genco_index, instance.genco_index[g])
                    #print(abs(instance.gensegmentoffer[t,g,gs].value - overwritten_offers[counter]))
                    overwritten_offers[counter] = instance.gensegmentoffer[t,g,gs].value
                elif overwritten_offers[counter] < instance.generator_marginal_cost[t,g,gs]:
                    overwritten_offers[counter] = instance.generator_marginal_cost[t,g,gs]
                counter+=1
                #else:
                #    overwritten_offers.append(instance.previous_offer[t,g,gs])
    
    #instance.solutions.load_from(solution)
    #write it to an array
    #eventually this should be converted to real results writing, 
    #but for not it's just a single result
    #so OK to do this
    # THE REST OF THIS LOOP IS ONLY NEEDED FOR PLOTTING RESULTS
    results_dispatch = []
    results_starts = []
    results_shuts = []
    tmps = []
    zone_stamp = []
    results_wind = []
    results_solar = []
    results_curtailment = []
    price_duals = []
    

    for t in instance.TIMEPOINTS:
        tmps.append(instance.TIMEPOINTS[t])
        
        for z in instance.ZONES:
            
            results_wind.append(instance.windgen[t,z].value)
            results_solar.append(instance.solargen[t,z].value)
            results_curtailment.append(instance.curtailment[t,z].value)
            if solution_type == "LP":
                price_duals.append(instance.dual[instance.LoadConstraint[t,z]])
            else:
                price_duals.append(instance.zonalprice[t,z].value)
            
            #for g in instance.GENERATORS:
                #results_dispatch.append(instance.dispatch[t,g]())
                #for gs in instance.GENERATORSEGMENTS:
                    #if t==1:
                        #print('offer')
                        #print([t,g,z,gs])
                        #print(instance.gensegmentoffer[t,g,z,gs].value)
                        #print(instance.zonalprice[t,z].value)
                        #print(-instance.gensegmentmaxdual[t,g,z,gs].value)
                        #print(instance.gensegmentmindual[t,g,z,gs].value)
                        #print('and the dispatch')
                        #print(instance.segmentdispatch[t,g,z,gs].value)
                        
                        #if instance.gensegmentmaxdual[t,g,z,gs].value>0:
                        #    print('and one of these should bind')
                        #    print(instance.generatorsegmentlength[gs]*instance.capacity[g,z]*instance.scheduledavailable[t,g]-instance.segmentdispatch[t,g,z,gs].value)
                        #    print(instance.gensegmentmaxdual[t,g,z,gs].value)
        for g in instance.GENERATORS:
            results_starts.append(instance.startup[t,g].value)
            results_shuts.append(instance.shutdown[t,g].value)
            results_dispatch.append(instance.dispatch[t,g]())
            
    zones = pd.read_csv(join(dir_str.INPUTS_DIRECTORY, 'zones.csv'))
    for z in zones['zone']:
        zone_stamp.append(z)
    
    #export objective function value
    #objective_list.append(value(instance.TotalCost2))
    #objective_index.append('ComparisonObjectiveValue')
    #df = pd.DataFrame(objective_list, index=pd.Index(objective_index))
    #df.to_csv(os.path.join(scenario_results_directory,"comparison_objective_function_value.csv"))
    
    ### ###
    
    return (results_dispatch, len(tmps), results_wind, results_solar, results_curtailment, results_starts,\
            results_shuts, price_duals, overwritten_offers)

### RUN MODEL ###

count_iters = 0
if EPEC:
    for i in range(iters):
        count_iters+=1
        print("Running EPEC iter " + str(count_iters) + " of " + str(iters) + "...")
        #reformulate previous offer of generators for non-initial iters
        if count_iters>1:
            
            df = pd.read_csv(join(dir_str.INPUTS_DIRECTORY, 'generator_segment_marginalcost.csv'))
            break_flag = True
            for v in range(len(scenario_results[-1])):
                if abs(scenario_results[-1][v]-df.prev_offer[v])>0.1:
                    break_flag=False
            if break_flag:
                print('reached equilibrium, exiting run with solution in hand...')
                df.prev_offer = scenario_results[-1]
                #do comparison, eventually, that could cause loop to terminate
                df.to_csv(join(dir_str.INPUTS_DIRECTORY, 'generator_segment_marginalcost.csv'),index=False) #overwrite
                break
            df.prev_offer = scenario_results[-1]
            #do comparison, eventually, that could cause loop to terminate
            df.to_csv(join(dir_str.INPUTS_DIRECTORY, 'generator_segment_marginalcost.csv'),index=False) #overwrite
            #overwritten_offers = scenario_results[-1]
            
        else: #initialization always sets prev offer to mcos
            df = pd.read_csv(join(dir_str.INPUTS_DIRECTORY, 'generator_segment_marginalcost.csv'))
            df.prev_offer = df.marginal_cost
            df.to_csv(join(dir_str.INPUTS_DIRECTORY, 'generator_segment_marginalcost.csv'),index=False) #overwrite
            overwritten_offers = [0]*len(df.prev_offer)
            #do overwrite
        
        count_case = 0
        for s in scenario_list:
            count_case+=1
            #initialize scenario data in the tuple
            scenario_name = s[0] #for now
            load_init = s[1]
            load_dir = s[2]
            genco_index = s[3]
            
            #run the case, as usual
            code_directory = cwd
            dir_str = DirStructure(code_directory, case_folder, load_init, load_dir)
            dir_str.make_directories()
            logger = Logger(dir_str)
            log_file = logger.log_file_path
            print("Running scenario " + str(count_case) + " of " + str(len(scenario_list)) + "...")
            print ("Running scenario " + str(scenario_name) + "...")
            stdout = sys.stdout
            sys.stdout = logger
            
            scenario_results = run_scenario(dir_str, load_init, MPEC, genco_index, overwritten_offers)
            overwritten_offers = scenario_results[-1]
            
            sys.stdout = stdout #return to original
                
            # run diagnostic plots
            diag_plots(scenario_results, dir_str)
        
            end_time = time.time() - start_time
            print ("time elapsed during run is " + str(end_time) + " seconds")
else:
    count_case = 0
    for s in scenario_list:
        count_case+=1
        #initialize scenario data in the tuple
        scenario_name = s[0] #for now
        load_init = s[1]
        load_dir = s[2]
        genco_index = s[3]
        
        #run the case, as usual
        code_directory = cwd
        dir_str = DirStructure(code_directory, case_folder, load_init, load_dir)
        dir_str.make_directories()
        logger = Logger(dir_str)
        log_file = logger.log_file_path
        print("Running scenario " + str(count_case) + " of " + str(len(scenario_list)) + "...")
        print ("Running scenario " + str(scenario_name) + "...")
        stdout = sys.stdout
        sys.stdout = logger
        
        df = pd.read_csv(join(dir_str.INPUTS_DIRECTORY, 'generator_segment_marginalcost.csv'))
        df.prev_offer = df.marginal_cost
        df.to_csv(join(dir_str.INPUTS_DIRECTORY, 'generator_segment_marginalcost.csv'),index=False) #overwrite
        overwritten_offers = [0]*len(df.prev_offer)
        
        scenario_results = run_scenario(dir_str, load_init, MPEC, genco_index, overwritten_offers)
        
        sys.stdout = stdout #return to original
        
        # run diagnostic plots
        diag_plots(scenario_results, dir_str)
    
        end_time = time.time() - start_time
        print ("time elapsed during run is " + str(end_time) + " seconds")
>>>>>>> 33b8c0e3277c1b0395cc8f618f5b8608e292a7f1
