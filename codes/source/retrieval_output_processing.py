import sys
import os
import glob
import datetime as dt
import pdb

wdir = os.getcwd() + "/"

import numpy as np
import xarray as xr


"""
    Script to add error estimates based on the evaluation with ERA5 data performed in 
    retrieval_evaluation.py to the retrieval output files created with NN_retrieval.py.
    - import error data
    - loop over retrieval output files and include error estimates
    - filter EEZ times
    - export processed files
"""


# Paths
path_data_base = os.path.abspath(wdir + "../..") + "/data/"
path_data = {'nn_syn_mosaic': path_data_base + "retrieval_output/mosaic/",
            'error_stats': path_data_base + "retrieval_evaluation/"}
path_output = path_data_base + "retrieval_output/mosaic/edited/"


# settings:
set_dict = {'err_file_transl': {'prw': 'iwv', 'temp': 'temp', 'temp_bl': 'temp_bl', 'q': 'q'},      # to address the correct error stat files
            'var_transl': {'prw': 'prw', 'temp': 'temp', 'temp_bl': 'temp', 'q': 'q'},  # to get the variable names of the output file right
            'std_name_dict': {'prw': 'atmosphere_mass_content_of_water_vapor',      # to translate to CF conventions standard_name
                                'temp': 'air_temperature', 
                                'temp_bl': 'air_temperature', 
                                'q': 'specific_humidity'}
            }
set_dict['path_output'] = path_output

# create output path if not existing:
os.makedirs(os.path.dirname(set_dict['path_output']), exist_ok=True)


# unit conversion dictionary:
set_dict['u_c_d'] = {'q': [0.0, 0.001],     # from g kg-1 to kg kg-1
                                'temp': [0.0, 1.0],     # from K to K
                                'temp_bl': [0.0, 1.0],} # from K to K


# EEZ periods during MOSAiC:
EEZ_periods = {'range0': [np.datetime64("2020-06-03T20:36:00"), np.datetime64("2020-06-08T20:00:00")],
                'range1': [np.datetime64("2020-10-02T04:00:00"), np.datetime64("2020-10-02T20:00:00")],
                'range2': [np.datetime64("2020-10-03T03:15:00"), np.datetime64("2020-10-04T17:00:00")]}


# find retrieval output files for each retrieved quantity:
ret_vars = ['prw', 'temp', 'temp_bl', 'q']
rounding_val = {'temp': 0.1, 'temp_bl': 0.1, 'q': 0.002}    # for rounding; in K, K, g kg-1
for ret_var in ret_vars:
    files = sorted(glob.glob(path_data['nn_syn_mosaic'] + f"MOSAiC_uoc_hatpro_lhumpro-243-340_l2_{ret_var}_*.nc"))


    # find respective error stat data:
    err_ret_var = set_dict['err_file_transl'][ret_var]
    err_file = glob.glob(path_data['error_stats'] + f"NN_syn_ret_ERA5_eval_data_error_stats_{err_ret_var}.nc")
    if len(err_file) == 1:
        ERR_DS = xr.open_dataset(err_file[0])
        ERR_DS = ERR_DS.load()
    else:
        pdb.set_trace() # debug


    # loop over files and open the file to add the error estimates:
    var_ret_var = set_dict['var_transl'][ret_var]
    std_name = set_dict['std_name_dict'][ret_var]
    for file in files:
        DS = xr.open_dataset(file)


        # create the error estimate variable:
        if ret_var == 'prw':
            # create a variable containing the correct error estimates for the respective IWV bins
            prw_rmse = xr.DataArray(np.full((len(DS.prw),), -9999.0), dims=DS.prw.dims, coords=DS.prw.coords)

            # loop over IWV bins:
            for i_bin, bin in enumerate(ERR_DS.bins):
                rmse_bins_mean_val = np.ceil((ERR_DS.rmse_bins_mean[i_bin].values+1e-9)*10.)*0.1
                prw_rmse = xr.where((DS.prw >= ERR_DS.bins_bnds[i_bin,0].values) & (DS.prw < ERR_DS.bins_bnds[i_bin,1].values),
                                    rmse_bins_mean_val, prw_rmse)

            # set minimum value to 0.1 kg m-2 to be conservative regarding error estimates:
            prw_rmse[(prw_rmse < 0.1) & (prw_rmse >= 0.0)] = 0.1

            # save to DS:
            DS[f'{var_ret_var}_rmse'] = xr.DataArray(prw_rmse.astype(np.float32))

            # attributes:
            DS[f'{var_ret_var}_rmse'].attrs['long_name'] = f"Estimated uncertainty of {var_ret_var} given as root mean squared error"
            DS[f'{var_ret_var}_rmse'].attrs['standard_name'] = f"{std_name} standard_error"
            DS[f'{var_ret_var}_rmse'].attrs['units'] = DS[var_ret_var].attrs['units']
            DS[f'{var_ret_var}_rmse'].attrs['comment'] = "Root mean squared error has been computed over 4 years of ERA5 data and rounded up to the next 0.1."


        else:
            rmse_mean_val = np.ceil((ERR_DS.rmse_tot_mean.values+1e-9)/rounding_val[ret_var])*rounding_val[ret_var]

            # convert units and save to dataset:
            rmse_mean_val = (rmse_mean_val + set_dict['u_c_d'][ret_var][0])*set_dict['u_c_d'][ret_var][1]
            DS[f'{var_ret_var}_rmse'] = xr.DataArray(rmse_mean_val.astype(np.float32), dims=['height'])

            # attributes:
            DS[f'{var_ret_var}_rmse'].attrs['long_name'] = f"Estimated uncertainty of {var_ret_var} given as root mean squared error"
            DS[f'{var_ret_var}_rmse'].attrs['standard_name'] = f"{std_name} standard_error"
            DS[f'{var_ret_var}_rmse'].attrs['units'] = DS[var_ret_var].attrs['units']
            DS[f'{var_ret_var}_rmse'].attrs['comment'] = ("Root mean squared error has been computed over 4 years of ERA5 data and rounded up" +
                                                        f" to the next {str(rounding_val[ret_var]*set_dict['u_c_d'][ret_var][1])}. Errors may vary over the seasons.")

        # add _FillValue attribute:
        DS[f'{var_ret_var}_rmse'].encoding["_FillValue"] = float(-9999.)


        # find out if time is within an EEZ:
        ds_time = DS.time.values
        outside_eez = np.full((len(ds_time),), True)
        for EEZ_range in EEZ_periods.keys():
            outside_eez[(ds_time >= EEZ_periods[EEZ_range][0]) &
                        (ds_time <= EEZ_periods[EEZ_range][1])] = False
        if np.all(~outside_eez):    # all times within EEZ
            continue
        if ds_time[0] < np.datetime64("2019-10-21T23:30:00"):
            continue                # don't process data before first successful calibration
        DS = DS.sel(time=outside_eez)       # remove times within EEZs


        # update history attribute:
        DS.attrs['history'] += f"; {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}, added error estimates and excluded times within EEZs with retrieval_output_processing.py"


        # encode time:
        DS['time'] = DS.time.values.astype("datetime64[s]").astype(np.float64)
        DS['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
        DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
        DS['time'].encoding['dtype'] = 'double'


        # Make sure that _FillValue is not added to certain variables:
        exclude_vars_fill_value = ['time', 'height']
        for kk in exclude_vars_fill_value:
            if kk in DS.variables:
                DS[kk].encoding["_FillValue"] = None


        # export:
        outfile = set_dict['path_output'] + os.path.basename(file)
        DS.to_netcdf(outfile, mode='w', format="NETCDF4")

        print(f"\rEdited {os.path.basename(file)}....", sep='', end='', flush=True)


        DS = DS.close()
        del DS

    print("")