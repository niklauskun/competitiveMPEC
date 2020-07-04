# load libraries
# these should all be fairly common but let me know if there are issues.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
import math
import time
import datetime

# load functions from case_creation_functions script
# I will list them all to be as explicit as possible instead of import *
from case_creation_functions import CreateRTSCase
from case_creation_functions import DirStructure
from case_creation_functions import LoadNRELData
from case_creation_functions import write_RTS_case

# create a file structure object, "f"
f = DirStructure(
    "C:\\Users\\wenmi\\Desktop",
    RTS_folder="RTS-GMLC-master",
    MPEC_folder="competitiveMPEC",
)  # the first arg should be the local directory you put NREL-RTS in
# the second should be what you named the NREL-RTS folder (what I've input is default behavior)
# the third is whatever you named the folder where you put the MPEC code
f.make_directories()

# load input data, including constants
# this ends up taking the format of a dictionary
data_class = LoadNRELData(f)
kw_dict = data_class.load_nrel_data()
kw_dict = data_class.define_constants(kw_dict)
# data_class.add_unit()  # [323, 301, 301], ["322_HYDRO_1", "303_WIND_1", "313_STORAGE_1"]

# inputs for running
start = datetime.datetime.strptime("01-01-2019", "%m-%d-%Y")  # day case starts on
end = datetime.datetime.strptime(
    "01-08-2019", "%m-%d-%Y"
)  # day case ends on. Generally this can be 01-01-2020.
folder_out = "test"  # name of folder to write the case to

# optional inputs for running
# these define differences between cases
# I will input default behavior if you decide not to use these,
# but they're important for creating different cases
optional_args = {
    "gentypes_included": [
        "CT",
        "STEAM",
        "CC",
        "NUCLEAR",
        "HYDRO",
        "RTPV",
        "WIND",
        "PV",
        "CSP",
    ],
    "owned_gens": ["303_WIND_1"],
    "owned_storage": ["313_STORAGE_1"],
    "hybrid_gens": [],
    "hybrid_storage": [],
    "retained_buses": [
        a for a in range(301, 326)
    ],  # [a for a in range(301, 326)] to use only area 3 buses
    "storage_bus": 303,
    "storage_capacity_scalar": 1,
    "storage_duration_scalar": 1,
    "tx_capacity_scalar": 1,
    "battery_roundtrip_efficiency": 0.85,
    "start_cost_scalar": 0,
    "no_load_cost_scalar": 0,
    "pmin_scalar": 0,
}

write_RTS_case(kw_dict, start, end, f, folder_out, **optional_args)
