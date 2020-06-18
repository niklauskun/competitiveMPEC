# imports
import os
import pandas as pd
import math
import datetime
import warnings


# directory structure for inputs/outputs
class DirStructure(object):
    """Creates directory structure for inputs (from NREL-RTS) and 
    outputs (which will be inputs for Pyomo model)"""

    def __init__(
        self,
        code_directory,
        RTS_folder="RTS-GMLC-master",
        MPEC_folder="competitiveMPEC-master",
        results_folder="",
        results_case="",
    ):
        """initializes directory structure

        Arguments:
            code_directory {str} -- name of directory where NREL-RTS and your desired output folder are

        Keyword Arguments:
            RTS_folder {str} -- [name of NREL RTS folder you forked from their repo] (default: {'RTS-GMLC-master'})
            MPEC_folder {str} -- [name of folder with MPEC code] (default: {'competitiveMPEC-master'})
            results_folder {str} -- [folder to write cases to] (default: {''})
            results_case {str} -- [name of individual case] (default: {''})"""
        self.BASE_DIRECTORY = code_directory

        self.RTS_GMLC_DIRECTORY = os.path.join(
            self.BASE_DIRECTORY, RTS_folder, "RTS_Data"
        )  # code_directory
        self.FORMATTED_INPUTS_DIRECTORY = os.path.join(
            self.RTS_GMLC_DIRECTORY, "FormattedData"
        )
        self.SOURCE_INPUTS_DIRECTORY = os.path.join(
            self.RTS_GMLC_DIRECTORY, "SourceData"
        )
        self.TIMESERIES_INPUTS_DIRECTORY = os.path.join(
            self.RTS_GMLC_DIRECTORY, "timeseries_data_files"
        )
        self.HEATRATE_INPUTS_DIRECTORY = os.path.join(
            self.SOURCE_INPUTS_DIRECTORY, "HR_data"
        )

        self.MPEC_DIRECTORY = os.path.join(self.BASE_DIRECTORY, MPEC_folder)
        self.CASE_DIRECTORY = os.path.join(self.MPEC_DIRECTORY, results_folder)
        self.RESULTS_DIRECTORY = os.path.join(
            self.MPEC_DIRECTORY, results_folder + "\\" + results_case
        )
        self.RESULTS_INPUTS_DIRECTORY = os.path.join(self.RESULTS_DIRECTORY, "inputs")

    def make_directories(self):
        """create any directories that don't already exist (applies only to output directories)
        """
        if not os.path.exists(self.CASE_DIRECTORY):
            os.mkdir(self.CASE_DIRECTORY)
        if not os.path.exists(self.RESULTS_DIRECTORY):
            os.mkdir(self.RESULTS_DIRECTORY)
        if not os.path.exists(self.RESULTS_INPUTS_DIRECTORY):
            os.mkdir(self.RESULTS_INPUTS_DIRECTORY)


# class for loading NREL RTS case data
class LoadNRELData(object):
    def __init__(self, f):
        self.f = f
        self.nrel_dict = {}

    def load_nrel_data(self):
        """
        Arguments:
            f {class(DirStructure)} -- a folder directory for the case. Needs pointer to NREL data
        Returns:
            [dict] -- dictionary containing dataframes with loaded NREL data
        """
        ## NREL SourceData folder imports ##
        self.nrel_dict["branch_data"] = pd.read_csv(
            os.path.join(self.f.SOURCE_INPUTS_DIRECTORY, "branch.csv")
        )
        self.nrel_dict["bus_data"] = pd.read_csv(
            os.path.join(self.f.SOURCE_INPUTS_DIRECTORY, "bus.csv")
        )
        self.nrel_dict["dcbranch_data"] = pd.read_csv(
            os.path.join(self.f.SOURCE_INPUTS_DIRECTORY, "dc_branch.csv")
        )
        self.nrel_dict["gen_data"] = pd.read_csv(
            os.path.join(self.f.SOURCE_INPUTS_DIRECTORY, "gen.csv")
        )
        self.nrel_dict["reserves_data"] = pd.read_csv(
            os.path.join(self.f.SOURCE_INPUTS_DIRECTORY, "reserves.csv")
        )
        self.nrel_dict["simulationobjects_data"] = pd.read_csv(
            os.path.join(self.f.SOURCE_INPUTS_DIRECTORY, "simulation_objects.csv")
        )
        self.nrel_dict["storage_data"] = pd.read_csv(
            os.path.join(self.f.SOURCE_INPUTS_DIRECTORY, "storage.csv")
        )
        self.nrel_dict["timeseriespointers_data"] = pd.read_csv(
            os.path.join(self.f.SOURCE_INPUTS_DIRECTORY, "timeseries_pointers.csv")
        )

        ## NREL TimeSeries folder imports ##
        # you'll note the basic difference here is these are just params, like load, that will be indexed by time
        # for now I pull the day-ahead ("DA") time-series files, since the model is only at hourly resolution.

        # generation
        self.nrel_dict["hydro_data"] = pd.read_csv(
            os.path.join(
                self.f.TIMESERIES_INPUTS_DIRECTORY, "Hydro\\DAY_AHEAD_hydro.csv"
            )
        )
        self.nrel_dict["load_data"] = pd.read_csv(
            os.path.join(
                self.f.TIMESERIES_INPUTS_DIRECTORY, "Load\\DAY_AHEAD_regional_Load.csv"
            )
        )
        self.nrel_dict["pv_data"] = pd.read_csv(
            os.path.join(self.f.TIMESERIES_INPUTS_DIRECTORY, "PV\\DAY_AHEAD_pv.csv")
        )
        self.nrel_dict["rtpv_data"] = pd.read_csv(
            os.path.join(self.f.TIMESERIES_INPUTS_DIRECTORY, "RTPV\\DAY_AHEAD_rtpv.csv")
        )
        self.nrel_dict["csp_data"] = pd.read_csv(
            os.path.join(
                self.f.TIMESERIES_INPUTS_DIRECTORY, "CSP\\DAY_AHEAD_Natural_Inflow.csv"
            )
        )
        self.nrel_dict["wind_data"] = pd.read_csv(
            os.path.join(self.f.TIMESERIES_INPUTS_DIRECTORY, "Wind\\DAY_AHEAD_Wind.csv")
        )

        # reserves
        self.nrel_dict["reg_up_data"] = pd.read_csv(
            os.path.join(
                self.f.TIMESERIES_INPUTS_DIRECTORY,
                "Reserves\\DAY_AHEAD_regional_Reg_Up.csv",
            )
        )
        self.nrel_dict["reg_down_data"] = pd.read_csv(
            os.path.join(
                self.f.TIMESERIES_INPUTS_DIRECTORY,
                "Reserves\\DAY_AHEAD_regional_Reg_Down.csv",
            )
        )
        self.nrel_dict["flex_up_data"] = pd.read_csv(
            os.path.join(
                self.f.TIMESERIES_INPUTS_DIRECTORY,
                "Reserves\\DAY_AHEAD_regional_Flex_Up.csv",
            )
        )
        self.nrel_dict["flex_down_data"] = pd.read_csv(
            os.path.join(
                self.f.TIMESERIES_INPUTS_DIRECTORY,
                "Reserves\\DAY_AHEAD_regional_Flex_Down.csv",
            )
        )
        return self.nrel_dict

    def define_constants(self, input_dict):
        """defines constants to be used in case with input data

        Arguments:
            input_dict {dict} -- dictionary of input data, must already exist to run this method

        Returns:
            input_dict {dict} -- dictionary of input data updated with constants to be used in the case
        """
        input_dict["hours"] = 24
        input_dict["lb_to_tonne"] = 0.000453592
        input_dict["baseMVA"] = 100
        input_dict["km_per_mile"] = 1.60934
        return input_dict


# Class for creating case (this is the big, complicated part)
class CreateRTSCase(object):
    def __init__(self, gentypes, directory, hour_begin, **kwargs):
        self.gentypes = gentypes
        self.directory = directory
        self.hour_begin = hour_begin
        # data loads as kwargs
        # constants are also kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.hour_end = hour_begin + self.hours

    def __getattr__(self, name):
        # suppresses warning about **kwargs not existing
        warnings.warn('No member "%s" contained in settings config.' % name)
        return ""

    def dict_to_csv(self, filename, mydict, index=["Gen_Index"], owned_gens=False):
        df = pd.DataFrame.from_dict(mydict)
        df.set_index(index, inplace=True)
        if owned_gens:
            for gen in self.owned_gen_list:
                df.at[
                    gen, "GencoIndex"
                ] = 1  # overwrite to make owned by competitive agent when applicable
        df.to_csv(
            os.path.join(self.directory.RESULTS_INPUTS_DIRECTORY, filename + ".csv")
        )

    def df_to_csv(self, filename, mydict):
        df = pd.DataFrame(mydict, index=[0])
        df.set_index(["Storage_Index"], inplace=True)
        df.to_csv(
            os.path.join(self.directory.RESULTS_INPUTS_DIRECTORY, filename + ".csv")
        )

    def create_index(self):
        case_index_df = pd.DataFrame({"genco": [1], "": ["index"]})
        case_index_df.set_index([""], inplace=True)
        case_index_df.to_csv(
            os.path.join(self.directory.RESULTS_INPUTS_DIRECTORY, "case_index.csv")
        )

    def generators(self, filename, owned_gen_list=[], retained_bus_list=[]):

        self.generators_dict = {}
        self.owned_gen_list = owned_gen_list
        index_list = [
            "Gen_Index",
            "Capacity",
            "Fuel_Cost",
            "Pmin",
            "start_cost",
            "Can_Spin",
            "Can_NonSpin",
            "Min_Up",
            "Min_Down",
            "No_Load_Cost",
            "Ramp_Rate",
            "tonneCO2perMWh",
            "CO2price",
            "CO2dollarsperMWh",
            "ZoneLabel",
            "GencoIndex",
        ]
        if retained_bus_list == []:
            pass  # print("default")
        else:
            self.gen_data = self.gen_data[
                self.gen_data["Bus ID"].isin(retained_bus_list)
            ].copy()
        self.retained_bus_list = retained_bus_list

        self.generators_dict[index_list[0]] = self.gen_data[
            self.gen_data["Unit Type"].isin(self.gentypes)
        ]["GEN UID"].values
        self.generators_dict[index_list[1]] = self.gen_data[
            self.gen_data["Unit Type"].isin(self.gentypes)
        ]["PMax MW"].values
        self.gen_data.loc[:, "$/MWH"] = (
            self.gen_data.loc[:, "Fuel Price $/MMBTU"]
            * self.gen_data.loc[:, "HR_incr_3"]
            / 1000
        )
        self.generators_dict[index_list[2]] = self.gen_data[
            self.gen_data["Unit Type"].isin(self.gentypes)
        ]["$/MWH"].values
        self.generators_dict[index_list[3]] = [0] * len(
            self.generators_dict[index_list[0]]
        )
        #self.generators_dict[index_list[3]] = self.gen_data[
        #    self.gen_data["Unit Type"].isin(self.gentypes)
        #]["PMin MW"].values
        self.generators_dict[index_list[4]] = [0] * len(
            self.generators_dict[index_list[0]]
        )
        self.generators_dict[index_list[5]] = [1] * len(
            self.generators_dict[index_list[0]]
        )
        self.generators_dict[index_list[6]] = [1] * len(
            self.generators_dict[index_list[0]]
        )
        self.generators_dict[index_list[7]] = [1] * len(
            self.generators_dict[index_list[0]]
        )
        self.generators_dict[index_list[8]] = [1] * len(
            self.generators_dict[index_list[0]]
        )
        self.generators_dict[index_list[9]] = [0] * len(
            self.generators_dict[index_list[0]]
        )
        # ramp rates
        self.generators_dict[index_list[10]] = (
            60
            * self.gen_data[self.gen_data["Unit Type"].isin(self.gentypes)][
                "Ramp Rate MW/Min"
            ].values
        )

        self.gen_data.loc[:, "CO2/MWH"] = (
            self.gen_data.loc[:, "Emissions CO2 Lbs/MMBTU"]
            * self.lb_to_tonne
            * self.gen_data.loc[:, "HR_incr_3"]
            / 1000
        )
        self.generators_dict[index_list[11]] = self.gen_data[
            self.gen_data["Unit Type"].isin(self.gentypes)
        ]["CO2/MWH"].values
        self.generators_dict[index_list[12]] = [0] * len(
            self.generators_dict[index_list[0]]
        )
        self.generators_dict[index_list[13]] = [
            a * b
            for a, b in zip(
                self.generators_dict[index_list[11]],
                self.generators_dict[index_list[12]],
            )
        ]
        self.generators_dict[index_list[14]] = self.gen_data[
            self.gen_data["Unit Type"].isin(self.gentypes)
        ]["Bus ID"].values
        self.generators_dict[index_list[15]] = [2] * len(
            self.generators_dict[index_list[0]]
        )
        # for gen in owned_gen_list:

        self.dict_to_csv(filename, self.generators_dict, owned_gens=True)

    def generators_descriptive(self, filename):
        d = {}
        index_list = [
            "Gen_Index",
            "Name",
            "Zone",
            "Category",
            "Capacity",
            "Fuel_Cost",
            "Can_Spin",
            "UTILUNIT",
        ]
        d[index_list[0]] = self.generators_dict["Gen_Index"]
        d[index_list[1]] = self.generators_dict["Gen_Index"]
        d[index_list[2]] = self.generators_dict["ZoneLabel"]
        d[index_list[3]] = self.gen_data[
            self.gen_data["Unit Type"].isin(self.gentypes)
        ]["Category"].values
        d[index_list[4]] = self.generators_dict["Capacity"]
        d[index_list[5]] = self.generators_dict["Fuel_Cost"]
        d[index_list[6]] = self.generators_dict["Can_Spin"]
        d[index_list[7]] = ["NA"] * len(self.generators_dict[index_list[0]])

        self.dict_to_csv(filename, d)

    def storage(
        self, filename, capacity_scalar=1, duration_scalar=1, busID=313, RTEff=1.0
    ):
        storage_dict = {}
        index_list = [
            "Storage_Index",
            "Discharge",
            "Charge",
            "SOCMax",
            "DischargeEff",
            "ChargeEff",
            "StorageZoneLabel",
            "StorageIndex",
        ]
        # size_scalar*
        storage_dict[index_list[0]] = (
            self.gen_data[self.gen_data["Unit Type"] == "STORAGE"]
            .reset_index()
            .at[0, "GEN UID"]
        )
        storage_dict[index_list[1]] = (
            capacity_scalar
            * self.gen_data[self.gen_data["Unit Type"] == "STORAGE"]
            .reset_index()
            .at[0, "PMax MW"]
        )
        storage_dict[index_list[2]] = (
            capacity_scalar
            * self.gen_data[self.gen_data["Unit Type"] == "STORAGE"]
            .reset_index()
            .at[0, "PMax MW"]
        )
        storage_dict[index_list[3]] = (
            duration_scalar * self.storage_data.at[1, "Max Volume GWh"] * 1000
        )
        storage_dict[index_list[4]] = 1.0 / RTEff ** 0.5
        storage_dict[index_list[5]] = RTEff ** 0.5
        if busID == 0:
            storage_dict[index_list[6]] = (
                self.gen_data[self.gen_data["Unit Type"] == "STORAGE"]
                .reset_index()
                .at[0, "Bus ID"]
            )
        else:
            storage_dict[index_list[6]] = int(busID)
        storage_dict[index_list[7]] = 2

        self.df_to_csv(filename, storage_dict)

    def init_gens(self, filename):
        d = {}
        index_list = ["Gen_Index", "commit_init", "time_up_init", "time_down_init"]

        d[index_list[0]] = self.generators_dict["Gen_Index"]
        d[index_list[1]] = [1] * len(d[index_list[0]])
        d[index_list[2]] = [200] * len(d[index_list[0]])
        d[index_list[3]] = [0] * len(d[index_list[0]])
        self.dict_to_csv(filename, d)

    def scheduled_gens(self, filename):
        scheduled_dict = {}
        index_list = ["timepoint", "Gen_Index", "available", "Capacity", "Fuel_Cost"]

        scheduled_dict[index_list[0]] = [
            e
            for e in list(range(1, self.hours + 1))
            for i in range(len(self.generators_dict["Gen_Index"]))
        ]
        scheduled_dict[index_list[1]] = (
            list(self.generators_dict["Gen_Index"]) * self.hours
        )
        scheduled_dict[index_list[2]] = [1] * len(scheduled_dict[index_list[1]])

        gen_cap_list = []
        for h in range(self.hour_begin, self.hour_end):
            for gen, capacity in zip(
                self.generators_dict["Gen_Index"], self.generators_dict["Capacity"],
            ):
                if gen in self.hydro_data.columns:
                    gen_cap_list.append(self.hydro_data.at[h, gen])
                elif gen in self.pv_data.columns:
                    gen_cap_list.append(self.pv_data.at[h, gen])
                elif gen in self.rtpv_data.columns:
                    gen_cap_list.append(self.rtpv_data.at[h, gen])
                elif gen in self.csp_data.columns:
                    gen_cap_list.append(self.csp_data.at[h, gen])
                elif gen in self.wind_data.columns:
                    gen_cap_list.append(self.wind_data.at[h, gen])
                else:
                    gen_cap_list.append(capacity)
        scheduled_dict[index_list[3]] = gen_cap_list
        scheduled_dict[index_list[4]] = (
            list(self.generators_dict["Fuel_Cost"]) * self.hours
        )
        self.dict_to_csv(filename, scheduled_dict, index="timepoint")

    def timepoints(self, filename):
        self.timepoint_dict = {}
        index_list = [
            "timepoint",
            "reference_bus",
            "reg_up_mw",
            "reg_down_mw",
            "flex_up_mw",
            "flex_down_mw",
        ]

        self.timepoint_dict[index_list[0]] = list(range(1, self.hours + 1))

        if self.retained_bus_list == []:
            self.timepoint_dict[index_list[1]] = [self.bus_data.at[12, "Bus ID"]] * len(
                self.timepoint_dict[index_list[0]]
            )
        else:
            self.timepoint_dict[index_list[1]] = [self.retained_bus_list[0]] * len(
                self.timepoint_dict[index_list[0]]
            )
        # reformatting for timepoints
        period_list = []
        for h in range(self.hour_begin, self.hour_end):
            if h == self.hour_begin:
                month = self.hydro_data.at[h, "Month"]
                day = self.hydro_data.at[h, "Day"]
            period_list.append(str(self.hydro_data.at[h, "Period"]))
        self.timepoint_dict[index_list[2]] = list(
            self.reg_up_data[
                (self.reg_up_data["Month"] == month) & (self.reg_up_data["Day"] == day)
            ][period_list].values.ravel()
        )
        self.timepoint_dict[index_list[3]] = list(
            self.reg_down_data[
                (self.reg_down_data["Month"] == month)
                & (self.reg_down_data["Day"] == day)
            ][period_list].values.ravel()
        )
        self.timepoint_dict[index_list[4]] = list(
            self.flex_up_data[
                (self.flex_up_data["Month"] == month)
                & (self.flex_up_data["Day"] == day)
            ][period_list].values.ravel()
        )
        self.timepoint_dict[index_list[5]] = list(
            self.flex_down_data[
                (self.flex_down_data["Month"] == month)
                & (self.flex_down_data["Day"] == day)
            ][period_list].values.ravel()
        )
        self.dict_to_csv(filename, self.timepoint_dict, index="timepoint")

    def zones(self, filename):
        self.zone_dict = {}
        index_list = [
            "zone",
            "wind_cap",
            "solar_cap",
            "voltage_angle_max",
            "voltage_angle_min",
        ]

        if self.retained_bus_list == []:
            pass  # print("default")
        else:
            self.bus_data = self.bus_data[
                self.bus_data["Bus ID"].isin(self.retained_bus_list)
            ]

        self.zone_dict[index_list[0]] = self.bus_data.loc[:, "Bus ID"].values
        self.zone_dict[index_list[1]] = [0] * len(
            self.zone_dict[index_list[0]]
        )  # eventually replace this
        self.zone_dict[index_list[2]] = [0] * len(
            self.zone_dict[index_list[0]]
        )  # eventually replace this
        self.zone_dict[index_list[3]] = [180] * len(
            self.zone_dict[index_list[0]]
        )  # math.pi/3
        self.zone_dict[index_list[4]] = [-180] * len(
            self.zone_dict[index_list[0]]
        )  # math.pi/3
        self.dict_to_csv(filename, self.zone_dict, index="zone")

    def zonal_loads(self, filename):
        bus_df = self.bus_data[["Bus ID", "MW Load", "Area"]]
        bus_df_load = pd.merge(
            bus_df,
            bus_df.groupby("Area").sum()[["MW Load"]].reset_index(),
            how="left",
            left_on="Area",
            right_on="Area",
        )
        bus_df_load["Frac Load"] = bus_df_load["MW Load_x"] / bus_df_load["MW Load_y"]
        hourly_df = pd.concat([bus_df_load] * self.hours, ignore_index=True)
        hourly_df["timepoint"] = [
            e for e in self.timepoint_dict["timepoint"] for i in self.zone_dict["zone"]
        ]
        # print(load_data)
        load_short = (
            self.load_data.loc[self.hour_begin : self.hour_end - 1]
            .set_index(["Month", "Day", "Period"])
            .reset_index()
        )  # sets unique index
        hourly_df["zonal_load"] = [
            load_short.at[t - 1, str(a)]
            for t, a in zip(hourly_df["timepoint"].values, hourly_df["Area"].values)
        ]
        hourly_df["bus load"] = hourly_df["zonal_load"] * hourly_df["Frac Load"]

        time_zone_dict = {}
        index_list = ["timepoint", "zone", "gross_load", "wind_cf", "solar_cf"]

        time_zone_dict[index_list[0]] = [
            e for e in self.timepoint_dict["timepoint"] for i in self.zone_dict["zone"]
        ]
        time_zone_dict[index_list[1]] = [
            i for e in self.timepoint_dict["timepoint"] for i in self.zone_dict["zone"]
        ]
        time_zone_dict[index_list[2]] = hourly_df[
            "bus load"
        ].values  # hourly_df['MW Load_x'].values
        time_zone_dict[index_list[3]] = [0] * len(time_zone_dict[index_list[0]])
        time_zone_dict[index_list[4]] = [0] * len(time_zone_dict[index_list[0]])

        self.dict_to_csv(filename, time_zone_dict, index="timepoint")

    def tx_lines(self, filename):

        branch_data_update = pd.merge(
            self.branch_data,
            self.bus_data[["Bus ID", "BaseKV"]],
            how="left",
            left_on="From Bus",
            right_on="Bus ID",
        )
        branch_data_update["puconversion"] = (
            branch_data_update.BaseKV ** 2 / self.baseMVA
        )  # V^2/P=R, kv^2 and MV will cancel units
        branch_data_update["Lengthkm"] = branch_data_update["Length"] * self.km_per_mile
        for i in branch_data_update.index:
            if branch_data_update.at[i, "B"] == 0:
                branch_data_update.at[i, "puconversion"] = branch_data_update.at[
                    i, "Tr Ratio"
                ]
                branch_data_update.at[i, "Lengthkm"] = 1
            else:
                branch_data_update.at[i, "puconversion"] = 1
        branch_data_update["x_final"] = (
            branch_data_update.X * branch_data_update.puconversion
        )  # *branch_data_update.Lengthkm
        branch_data_update["r_final"] = (
            branch_data_update.R * branch_data_update.puconversion
        )  # *branch_data_update.Lengthkm
        self.branch_data_final = (
            branch_data_update  # [branch_data_update['Lengthkm']!=0].copy()
        )

        self.branch_data_final["Susceptance"] = [
            1 / x * 100 * math.pi / 180
            for x in self.branch_data_final["x_final"].values
        ]
        self.tx_dict = {}
        index_list = ["transmission_line", "susceptance"]

        if self.retained_bus_list == []:
            pass  # print("default")
        else:
            self.branch_data_final = self.branch_data_final[
                self.branch_data_final["From Bus"].isin(self.retained_bus_list)
            ]
            self.branch_data_final = self.branch_data_final[
                self.branch_data_final["To Bus"].isin(self.retained_bus_list)
            ]
        self.tx_dict[index_list[0]] = self.branch_data_final["UID"].values
        self.tx_dict[index_list[1]] = self.branch_data_final["Susceptance"].values
        self.dict_to_csv(filename, self.tx_dict, index="transmission_line")

    def tx_lines_hourly(self, filename, flow_multiplier=1):
        tx_hourly_dict = {}
        index_list = [
            "timepoint",
            "transmission_line",
            "transmission_from",
            "transmission_to",
            "min_flow",
            "max_flow",
            "hurdle_rate",
        ]

        tx_hourly_dict[index_list[0]] = [
            t
            for t in self.timepoint_dict["timepoint"]
            for line in self.tx_dict["transmission_line"]
        ]
        tx_hourly_dict[index_list[1]] = [
            line
            for t in self.timepoint_dict["timepoint"]
            for line in self.tx_dict["transmission_line"]
        ]
        tx_hourly_dict[index_list[2]] = [
            t_from
            for t in self.timepoint_dict["timepoint"]
            for t_from in self.branch_data_final["From Bus"].values
        ]
        tx_hourly_dict[index_list[3]] = [
            t_to
            for t in self.timepoint_dict["timepoint"]
            for t_to in self.branch_data_final["To Bus"].values
        ]
        tx_hourly_dict[index_list[4]] = [
            -flow_multiplier * flow
            for t in self.timepoint_dict["timepoint"]
            for flow in self.branch_data_final["Cont Rating"].values
        ]
        tx_hourly_dict[index_list[5]] = [
            flow_multiplier * flow
            for t in self.timepoint_dict["timepoint"]
            for flow in self.branch_data_final["Cont Rating"].values
        ]
        tx_hourly_dict[index_list[6]] = [0] * len(tx_hourly_dict[index_list[0]])
        self.dict_to_csv(
            filename, tx_hourly_dict, index=["timepoint", "transmission_line"]
        )

    def gs(self):
        self.gs_list = [0, 1, 2, 3]
        gs_df = pd.DataFrame(
            {"generator_segment": self.gs_list, "length": [0.4, 0.2, 0.2, 0.2]}
        )
        gs_df.set_index(["generator_segment"], inplace=True)
        gs_df.to_csv(
            os.path.join(
                self.directory.RESULTS_INPUTS_DIRECTORY, "generator_segments.csv"
            )
        )

    def gs_seg(self, CO2price=0):
        gs_seg_dict = {}
        index_list = [
            "time",
            "Gen_Index",
            "generator_segment",
            "segment_length",
            "marginal_cost",
            "prev_offer",
            "marginal_CO2",
            "CO2damage",
        ]

        gs_seg_dict[index_list[0]] = [
            t
            for t in self.timepoint_dict["timepoint"]
            for gen in self.generators_dict["Gen_Index"]
            for gs in self.gs_list
        ]
        gs_seg_dict[index_list[1]] = [
            gen
            for t in self.timepoint_dict["timepoint"]
            for gen in self.generators_dict["Gen_Index"]
            for gs in self.gs_list
        ]
        gs_seg_dict[index_list[2]] = [
            gs
            for t in self.timepoint_dict["timepoint"]
            for gen in self.generators_dict["Gen_Index"]
            for gs in self.gs_list
        ]

        mcos_list = []
        emissions_list = []
        seg_length_list = []
        for gen, gs in zip(gs_seg_dict[index_list[1]], gs_seg_dict[index_list[2]]):
            EmissionsRate = (
                self.gen_data.set_index("GEN UID").at[gen, "Emissions CO2 Lbs/MMBTU"]
                * self.lb_to_tonne
            )
            if gs == 0:
                seg_length_list.append(
                    self.gen_data.set_index("GEN UID").at[gen, "Output_pct_" + str(gs)]
                )
                if (
                    self.gen_data.set_index("GEN UID").at[gen, "HR_incr_" + str(1)]
                    / 1000
                    != 0
                ):
                    HeatRate = (
                        self.gen_data.set_index("GEN UID").at[gen, "HR_incr_" + str(1)]
                        / 1000
                    )  # convexify offer curve
                else:
                    HeatRate = (
                        self.gen_data.set_index("GEN UID").at[gen, "HR_avg_" + str(gs)]
                        / 1000
                    )
                mcos_list.append(
                    HeatRate
                    * self.gen_data.set_index("GEN UID").at[gen, "Fuel Price $/MMBTU"]
                )
                emissions_list.append(HeatRate * EmissionsRate)
            else:
                seg_length_list.append(
                    max(
                        0,
                        self.gen_data.set_index("GEN UID").at[
                            gen, "Output_pct_" + str(gs)
                        ]
                        - self.gen_data.set_index("GEN UID").at[
                            gen, "Output_pct_" + str(gs - 1)
                        ],
                    )
                )
                if (
                    self.gen_data.set_index("GEN UID").at[gen, "HR_incr_" + str(gs)]
                    / 1000
                    == 0
                    and self.gen_data.set_index("GEN UID").at[gen, "HR_avg_" + str(0)]
                    / 1000
                    != 0
                ):
                    HeatRate = (
                        self.gen_data.set_index("GEN UID").at[gen, "HR_avg_" + str(0)]
                        / 1000
                    )
                else:
                    HeatRate = (
                        self.gen_data.set_index("GEN UID").at[gen, "HR_incr_" + str(gs)]
                        / 1000
                    )
                mcos_list.append(
                    HeatRate
                    * self.gen_data.set_index("GEN UID").at[gen, "Fuel Price $/MMBTU"]
                )
                emissions_list.append(HeatRate * EmissionsRate)

        gs_seg_dict[index_list[3]] = seg_length_list
        gs_seg_dict[index_list[4]] = mcos_list
        gs_seg_dict[index_list[5]] = gs_seg_dict[index_list[4]]
        gs_seg_dict[index_list[6]] = emissions_list
        gs_seg_dict[index_list[7]] = [CO2price] * len(emissions_list)

        for i in range(len(gs_seg_dict[index_list[4]])):
            gs_seg_dict[index_list[4]][i] += (
                gs_seg_dict[index_list[6]][i] * gs_seg_dict[index_list[7]][i]
            )

        gs_seg_df = pd.DataFrame.from_dict(gs_seg_dict)
        gs_seg_df.set_index(["time", "Gen_Index"], inplace=True)
        # overwrite 0's on segment 1 for renewable generators
        gs_seg_df.loc[
            (gs_seg_df.generator_segment == 0) & (gs_seg_df.segment_length == 0),
            "segment_length",
        ] = 1
        gs_seg_df.to_csv(
            os.path.join(
                self.directory.RESULTS_INPUTS_DIRECTORY,
                "generator_segment_marginalcost.csv",
            )
        )
        # print(gs_seg_df)
        return gs_seg_df


def write_RTS_case(kw_dict, start, end, dir_structure, case_folder, **kwargs):
    zero_day = datetime.datetime.strptime("01-01-2019", "%m-%d-%Y")
    # some checks
    assert (
        start - zero_day
    ).days >= 0.0  # just to check you start after the first day for which data is available
    assert (start - zero_day).days <= 365.0  # limits to a year of files
    range_start = (start - zero_day).days * 24  # scales to hours with *24

    date_generated = [
        start + datetime.timedelta(days=x) for x in range(0, (end - start).days)
    ]
    date_folders = [date.strftime("%m.%d.%Y") for date in date_generated]

    try:
        gt = kwargs["gentypes_included"]
    except KeyError:
        print("NOTE: no gentype_included, included gentypes based on default behavior")
        gt = [
            "CT",
            "STEAM",
            "CC",
            "NUCLEAR",
            "HYDRO",
            "SYNC_COND",
            "RTPV",
            "WIND",
            "PV",
            "CSP",
        ]
    try:
        owned_gens = kwargs["owned_gens"]  #'309_WIND_1'
    except KeyError:
        print("NOTE: no owned_gens, default behavior is for agent to only own storage")
        owned_gens = []
    try:
        retained_bus = kwargs["retained_buses"]  # [a for a in range(301, 326)]
    except KeyError:
        print("NOTE: no retained_buses, default behavior is to use all buses")
        retained_bus = []
    try:
        storage_bus = kwargs["storage_bus"]
    except KeyError:
        print("NOTE: no storage_bus, default behavior puts storage at bus 313")
        storage_bus = 313
    try:
        capacity_scalar = kwargs["storage_capacity_scalar"]
        duration_scalar = kwargs["storage_duration_scalar"]
    except KeyError:
        print("NOTE: no capacity and duration scalars, default is 50MW/150MWh battery")
        capacity_scalar = 1
        duration_scalar = 1
    try:
        flow_multiplier = kwargs["tx_capacity_scalar"]
    except KeyError:
        print("NOTE: no scaling of transmission line capacity, default inputs used")
        flow_multiplier = 1
    try:
        RTEfficiency = kwargs["battery_roundtrip_efficiency"]
    except KeyError:
        print("NOTE: no roundtrip efficiency input for battery, default is RTE=1")
        RTEfficiency = 1
    # loop creation of folder for each day between start and end
    for case_name, begin_hour in zip(
        date_folders,
        [i for i in range(range_start, range_start + len(date_folders) * 24, 24)],
    ):
        f = DirStructure(
            dir_structure.BASE_DIRECTORY,
            RTS_folder=dir_structure.RTS_GMLC_DIRECTORY,
            MPEC_folder=dir_structure.MPEC_DIRECTORY,
            results_folder=case_folder,
            results_case=case_name,
        )
        f.make_directories()
        print("creating case for " + str(case_name))

        case = CreateRTSCase(gt, f, begin_hour, **kw_dict)

        case.create_index()

        case.generators(
            "generators", owned_gen_list=owned_gens, retained_bus_list=retained_bus
        )
        case.generators_descriptive("generators_descriptive")
        case.storage(
            "storage_resources",
            capacity_scalar=capacity_scalar,
            duration_scalar=duration_scalar,
            busID=storage_bus,
            RTEff=RTEfficiency,
        )  # size_scalar=0
        case.init_gens("initialize_generators")
        case.scheduled_gens("generators_scheduled_availability")
        case.timepoints("timepoints_index")
        case.zones("zones")
        case.zonal_loads("timepoints_zonal")
        case.tx_lines("transmission_lines")
        case.tx_lines_hourly(
            "transmission_lines_hourly", flow_multiplier=flow_multiplier
        )
        case.gs()

        # the last file, gs_seg, takes awhile to create but also isn't indexed by time, so I create it only once and then
        # copy that first csv into all other folders when they are created
        try:
            a.to_csv(
                os.path.join(
                    case.directory.RESULTS_INPUTS_DIRECTORY,
                    "generator_segment_marginalcost.csv",
                )
            )
        except NameError:
            print("for first day, create gs_seg (takes a bit)")
            a = case.gs_seg(CO2price=0)

    print("...completed creating all cases!")
    return None
