import numpy as np
import glob
import pdb
import os
import sys
import datetime as dt

wdir = os.getcwd() + "/"
path_tools = os.path.dirname(wdir[:-1]) + "/tools/"
sys.path.insert(0, path_tools)

from import_data import import_MOSAiC_Radiosondes_PS122_Level3_merged_txt, import_MOSAiC_Radiosondes_PS122_Level2_tab
from data_tools import *


def save_MOSAiC_Radiosondes_PS122_Level2_as_nc(
    export_file,
    rs_dict,
    attribute_info):

    """
    Saves single MOSAiC Polarstern Level 2 Radiosonde to a netCDF4 file.

    Parameters:
    -----------
    export_file : str
        Path and filename to which the file is to be saved to.
    rs_dict : dict
        Dictionary that contains the radiosonde information.
    attribute_info : dict
        Dictionary that contains global attributes found in the .tab header.
    """

    RS_DS = xr.Dataset({'Latitude':     (['time'], rs_dict['Latitude'],
                                        {'units': "deg N"}),
                        'Longitude':    (['time'], rs_dict['Longitude'],
                                        {'units': "deg E"}),
                        'Altitude':     (['time'], rs_dict['Altitude'],
                                        {'description': "Altitude",
                                        'units': "m"}),
                        'h_geom':       (['time'], rs_dict['h_geom'],
                                        {'description': "Geometric Height",
                                        'units': "m"}),
                        'ETIM':         (['time'], rs_dict['ETIM'],
                                        {'description': "Elapsed time since sonde start"}),
                        'P':            (['time'], rs_dict['P'],
                                        {'description': "hPa",
                                        'units': "deg"}),
                        'T':            (['time'], rs_dict['T'],
                                        {'description': "Temperature",
                                        'units': "deg C"}),
                        'RH':           (['time'], rs_dict['RH'],
                                        {'description': "Relative humidity",
                                        'units': "percent"}),
                        'wdir':         (['time'], rs_dict['wdir'],
                                        {'description': "Wind direction",
                                        'units': "deg"}),
                        'wspeed':       (['time'], rs_dict['wspeed'],
                                        {'description': "Wind speed",
                                        'units': "m s^-1"}),
                        'q':            (['time'], rs_dict['q'],
                                        {'description': "Specific humidity",
                                        'conversion': "Saturation water vapour pressure based on Hyland and Wexler, 1983.",
                                        'units': "kg kg^-1"}),
                        'rho_v':        (['time'], rs_dict['rho_v'],
                                        {'description': "Absolute humidity",
                                        'conversion': "Saturation water vapour pressure based on Hyland and Wexler, 1983.",
                                        'units': "kg m^-3"}),
                        'IWV':          ([], rs_dict['IWV'],
                                        {'description': "Integrated Water Vapour",
                                        'calculation': ("Integration of (specific humidity x pressure). " +
                                                        "Humidity conversion based on Hyland and Wexler, 1983. "),
                                        'further_comment': ("IWV computation function checks if pressure truely " +
                                                            "decreases with increasing time since sonde start."),
                                        'units': "kg m^-2"})},
                        coords =        {'time': (['time'], rs_dict['time_sec'],
                                        {'description': "Time stamp or seconds since 1970-01-01 00:00:00 UTC",
                                        'units': "seconds since 1970-01-01 00:00:00 UTC"})})

    # Set global attributes:
    for attt in attribute_info:
        if (":" in attt[0]) & (len(attt) > 1):
            RS_DS.attrs[attt[0].replace(":","")] = attt[1]
    RS_DS.attrs['Author_of_netCDF'] = "Andreas Walbroel, a.walbroel@uni-koeln.de"

    # encode time:
    RS_DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
    RS_DS['time'].encoding['dtype'] = 'double'

    RS_DS.to_netcdf(export_file, mode='w', format="NETCDF4")
    RS_DS = RS_DS.close()


def save_MOSAiC_Radiosondes_PS122_Level3_merged_as_nc(
    export_file,
    rs_dict,
    attribute_info):

    """
    Saves single MOSAiC Polarstern Level 3 (merged) radiosonde to a netCDF4 file.

    Parameters:
    -----------
    export_file : str
        Path and filename to which the file is to be saved to.
    rs_dict : dict
        Dictionary that contains the radiosonde information.
    attribute_info : dict
        Dictionary that contains global attributes found in the .tab header.
    """

    RS_DS = xr.Dataset({'Latitude':     (['time'], rs_dict['lat'],
                                        {'units': "deg N"}),
                        'Longitude':    (['time'], rs_dict['lon'],
                                        {'units': "deg E"}),
                        'Altitude':     (['time'], rs_dict['height'],
                                        {'description': "Height above mean sea level",
                                        'units': "m"}),
                        'P':            (['time'], rs_dict['pres'],
                                        {'description': "Air pressure",
                                        'units': "hPa"}),
                        'T':            (['time'], rs_dict['temp'],
                                        {'description': "Air temperature",
                                        'units': "deg C"}),
                        'RH':           (['time'], rs_dict['rh'],
                                        {'description': "Relative humidity",
                                        'units': "percent"}),
                        'wdir':         (['time'], rs_dict['wdir'],
                                        {'description': "Wind direction",
                                        'units': "deg"}),
                        'wspeed':       (['time'], rs_dict['wspeed'],
                                        {'description': "Wind speed",
                                        'units': "m s-1"}),
                        'q':            (['time'], rs_dict['q'],
                                        {'description': "Specific humidity",
                                        'conversion': "Saturation water vapour pressure based on Hyland and Wexler, 1983.",
                                        'units': "kg kg-1"}),
                        'rho_v':        (['time'], rs_dict['rho_v'],
                                        {'description': "Absolute humidity",
                                        'conversion': "Saturation water vapour pressure based on Hyland and Wexler, 1983.",
                                        'units': "kg m-3"}),
                        'IWV':          ([], rs_dict['IWV'],
                                        {'description': "Integrated Water Vapour",
                                        'units': "kg m-2"}),
                        'T_flag':       (['time'], rs_dict['temp_flag'],
                                        {'description': "Temperature flag, see comment",
                                        'comment': ", ".join(attribute_info['flag_comments'])}),
                        'RH_flag':      (['time'], rs_dict['rh_flag'],
                                        {'description': "Relative humidity flag, see comment",
                                        'comment': ", ".join(attribute_info['flag_comments'])})},
                        coords =        {'time': (['time'], rs_dict['time_sec'])})

    # Set global attributes:
    RS_DS.attrs['Citation'] = ("Dahlke, Sandro; Shupe, Matthew D; Cox, Christopher J; Brooks, Ian M; Blomquist, Byron; " +
                        "Persson, P Ola G (2023): Extended radiosonde profiles 2019/09-2020/10 during MOSAiC Legs " +
                        "PS122/1 - PS122/5. PANGAEA, https://doi.org/10.1594/PANGAEA.961881")
    RS_DS.attrs['Abstract'] = ("During the MOSAiC expedition 2019-2020 atmospheric thermodynamic profile measurements have been " +
                        "conducted from a meteorological (Met) Tower on the sea ice, as well as via collocated radiosondes that " +
                        "were launched approximately every six hours from aboard Polarstern. While the radiosondes lack the " +
                        "lowermost 10 m above the sea ice, the Met Tower profile can be used to fill this gap (observations at " +
                        "0, 2, 6 and 10 meters). This is a blended data product that merges the Met Tower profile (data version 3.4, " +
                        "doi:10.18739/A2PV6B83F) in the minute of the radiosonde launch with the radiosonde profile aloft (data " +
                        "version 3, doi:10.1594/PANGAEA.943870). Parameters included are temperature (T), relative humidity (RH), " +
                        "wind speed and -direction, and air pressure. The aim of this product is two-fold: (1) To provide comprehensive " +
                        "atmospheric profiles for each radiosonde launch, that additionally retain the lowermost meters of the " +
                        "atmospheric boundary layer above the sea ice and (2) to remove potential unrealistic T/RH values from the " +
                        "radiosonde profiles that can emerge in the lowermost 100 m due to the influence of the ship on the measurement. " +
                        "Examples for the latter are occasional warm anomalies due to the heat island effect of the ship, or elevated, " +
                        "vertically confined peaks that can arise from the ship's exhaust plume. The potential effect of the exhaust " +
                        "plume on the T profile is estimated by comparing the radiosonde at 30 m height to the concurring Polarstern " +
                        "meteorological observation (doi:10.1594/PANGAEA.935263 - doi:10.1594/PANGAEA.935267). Given the geometrical " +
                        "constellation of the Polarstern observation towards the bow of the ship and the sounding launch platform at the " +
                        "aft of the ship, and depending on the wind direction relative to the ship, it can be assumed that at least one of " +
                        "the T measurements is less impacted from the ship exhaust than the other, and is retained. In a next step, the " +
                        "10 - 30 m height segment in T and RH is filled with a linear interpolation between the Met Tower at 10 m and the " +
                        "radiosonde observation at 30 m. When identified, remaining T/RH peaks in the lowermost 100 m of the profile are " +
                        "removed and filled with a linear interpolation from below to above the peak. T/RH flags are provided to indicate " +
                        "where the profiles have been manipulated from the original data, and to indicate the reason for missing data in " +
                        "the profile. Compared to the original profiles, this blended product adds value and quality control in the " +
                        "lowest 100 m, which makes it better suitable, for example, for boundary layer analyses.")
    RS_DS.attrs['Author'] = attribute_info['data_author']
    RS_DS.attrs['launch_time'] = str(attribute_info['launch_time'])
    RS_DS.attrs['data_based_on'] = ", ".join(attribute_info['data_based_on'])
    RS_DS.attrs['Author_of_netCDF'] = "Andreas Walbroel, a.walbroel@uni-koeln.de"


    # encode time:
    RS_DS['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
    RS_DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
    RS_DS['time'].encoding['dtype'] = 'double'

    RS_DS.to_netcdf(export_file, mode='w', format="NETCDF4")
    RS_DS = RS_DS.close()


"""
    Convert PANGAEA .tab files to netcdf.
"""

path_data_base = os.path.abspath(wdir + "../..") + "/data/"

if len(sys.argv) == 1:
    rs_version = 'level_2'
elif (len(sys.argv) == 2) and (sys.argv[1] in ['level_2', 'level_3']):
    rs_version = sys.argv[1]


# radiosonde data:
if rs_version == 'level_2':
    path_radiosondes = path_data_base + "MOSAiC_radiosondes/"

    radiosonde_files = sorted(glob.glob(path_radiosondes + "*.tab"))
    for rs_file in radiosonde_files:
        print(rs_file)
        rs_dict, rs_att_info = import_MOSAiC_Radiosondes_PS122_Level2_tab(rs_file)

        # Save each radiosonde in a single file:
        n_sondes = len(rs_dict.keys())
        for k in range(n_sondes):
            launch_time_str = dt.datetime.strftime(rs_dict[str(k)]['time'][0], "%Y%m%d_%H%M%SZ")
            export_file = path_radiosondes + "PS122_mosaic_radiosonde_level2_" + launch_time_str + ".nc"
            save_MOSAiC_Radiosondes_PS122_Level2_as_nc(export_file, rs_dict[str(k)], rs_att_info)

elif rs_version == 'level_3':
    path_radiosondes = path_data_base + "MOSAiC_radiosondes_level_3/raw/"
    path_export = path_data_base + "MOSAiC_radiosondes_level_3/"

    radiosonde_files = sorted(glob.glob(path_radiosondes + "*.txt"))
    for rs_file in radiosonde_files:
        print(rs_file)
        rs_dict, rs_att_info = import_MOSAiC_Radiosondes_PS122_Level3_merged_txt(rs_file)

        # Save each radiosonde in a single file:
        launch_time_str = str(rs_att_info['launch_time']).replace("T", "_").replace(":", "").replace("-", "")
        export_file = path_export + "PS122_mosaic_radiosonde_level3_" + launch_time_str + "Z.nc"
        save_MOSAiC_Radiosondes_PS122_Level3_merged_as_nc(export_file, rs_dict, rs_att_info)