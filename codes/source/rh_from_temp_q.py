import sys
import os
import glob
import datetime as dt
import pdb

wdir = os.getcwd() + "/"
path_tools = os.path.dirname(wdir) + "/tools/"
path_data_base = os.path.dirname(wdir[:-1]) + "/data/"

import numpy as np
import xarray as xr

sys.path.insert(0, path_tools)

from import_data import import_radiosonde_daterange
from met_tools import convert_spechum_to_relhum, e_sat


# constants (taken from met_tools.py):
R_d = 287.0597  # gas constant of dry air, in J kg-1 K-1
R_v = 461.5     # gas constant of water vapour, in J kg-1 K-1
M_dv = R_d / R_v # molar mass ratio , in ()
g = 9.80665     # gravitation acceleration, in m s^-2 (from https://doi.org/10.6028/NIST.SP.330-2019 )


def radiosonde_dict_to_xarray(sonde_dict):

    """
    Convert radiosonde dictionary imported with import_radiosonde_daterange to xarray dataset.

    Parameters:
    -----------
    sonde_dict : dictionary
        Dictionary containing radiosonde data imported with import_radiosonde_daterange.
    """

    DS = xr.Dataset(coords={'height': (['height'], sonde_dict['height'][0,:]),
                            'launch_time': (['launch_time'], sonde_dict['launch_time'].astype('datetime64[s]').astype('datetime64[ns]'))})
    DS['temp'] = xr.DataArray(sonde_dict['temp'], dims=['launch_time', 'height'])
    DS['rh'] = xr.DataArray(sonde_dict['rh'], dims=['launch_time', 'height'])
    DS['q'] = xr.DataArray(sonde_dict['q'], dims=['launch_time', 'height'], attrs={'units': "kg kg-1"})
    DS['rho_v'] = xr.DataArray(sonde_dict['q'], dims=['launch_time', 'height'], attrs={'units': "kg m-3"})
    DS['pres'] = xr.DataArray(sonde_dict['pres'], dims=['launch_time', 'height'], attrs={'units': "Pa"})
    DS['lat'] = xr.DataArray(sonde_dict['lat'], dims=['launch_time'], attrs={'units': "deg north"})
    DS['lon'] = xr.DataArray(sonde_dict['lon'], dims=['launch_time'], attrs={'units': "deg east"})
    DS['iwv'] = xr.DataArray(sonde_dict['iwv'], dims=['launch_time'], attrs={'units': "kg m-2"})

    return DS


def compute_RH_error(
    q, 
    temp,
    pres,
    d_q,
    d_T,
    d_p):

    """
    Compute error of relative humidity, propagated from measurement errors of temperature, specific
    humidity and pressure. 

    Parameters:
    -----------
    q : array of floats
        Specifc humidity in kg kg-1.
    temp : array of floats
        Air temperature in K.
    pres : array of floats
        Air pressure in Pa.
    d_q : array of floats
        Specific humidity uncertainty / error in kg kg-1.
    d_T : array of floats
        Air temperature uncertainty / error in K.
    d_p : array of floats
        Air pressure uncertainty / error in Pa.
    """

    # saturation water vapour pressure:
    e_s = e_sat(temp)

    # coefficients of Hyland and Wexler's sat. w.v. pressure equation (needed for derivative):
    a = 0.65459673e+01
    b = -0.58002206e+04
    c = 0.13914993e+01
    d = -0.48640239e-01
    eps = 0.41764768e-04
    f = -0.14452093e-07

    B = np.exp(b/temp + c + d*temp + eps*temp**2 + f*temp**3)
    d_e_s = a*temp**(a-1) * B + temp**(a)*B * (-b/temp**2 + d + 2*eps*temp + 3*f*temp**2)   # derivative of e_s

    # separate derivatives of the rh->spec_hum conversion: term_1 = drh/dp * d_p; term_2 = drh/dq * d_q; 
    # term_3 = dq/dtemp * d_T
    term_1 = 1 / (M_dv * (1/q + 1/M_dv - 1)) * (1/e_s) * d_p
    term_2 = (pres / e_s) * ((M_dv/(q**2)) * d_q) / ((M_dv * (1/q + 1/M_dv - 1))**2)
    term_3 = (pres / (M_dv*(1/q + 1/M_dv - 1))) * (-1) * e_s**(-2) * d_e_s * d_T

    delta_rh = np.sqrt(term_1**2 + term_2**2 + term_3**2)

    return delta_rh


"""
    Script to compute relative humidity from retrieved specific humidity and temperature profiles,
    as well as radiosonde level 3 air pressure data for the MOSAiC expedition. The relative 
    humidity data will be saved as netCDF files similar to those of the retrieved q and temp
    profiles.
    - import radiosonde and retrieved MWR data
    - loop over MOSAiC days: interpolate data to zenith MWR obs time grid
"""


# Paths
path_data = {'nn_syn_mosaic': path_data_base + "retrieval_output/mosaic/",
            'radiosondes': path_data_base + "MOSAiC_radiosondes_level_3/"}
path_output = path_data_base + "retrieval_output/mosaic/"


# settings:
set_dict = {'date_0': "2019-09-20",     # lower limit of dates for import of mosaic data (default: "2019-09-20")
            'date_1': "2020-10-12",     # upper limit of dates (default: "2020-10-12")
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


# import radiosonde data for the full MOSAiC period:
print("Importing radiosonde data....")
sonde_dict = import_radiosonde_daterange(path_data['radiosondes'], set_dict['date_0'], set_dict['date_1'], 
                                        s_version='level_3', remove_failed=True)
sonde_dict['launch_time_npdt'] = sonde_dict['launch_time'].astype('datetime64[s]')

# put radiosonde data into xarray dataset:
RS_DS_all = radiosonde_dict_to_xarray(sonde_dict)


# loop over dates:
date_range = np.arange(np.datetime64(set_dict['date_0']+"T00:00:00"),
                        np.datetime64(set_dict['date_1']+"T00:00:00")+np.timedelta64(1,"D"),
                        np.timedelta64(1, "D"))
files_q = sorted(glob.glob(path_data['nn_syn_mosaic'] + "MOSAiC_uoc_hatpro_lhumpro-243-340_l2_q_v00_*.nc"))
files_temp = sorted(glob.glob(path_data['nn_syn_mosaic'] + "MOSAiC_uoc_hatpro_lhumpro-243-340_l2_temp_v00_*.nc"))
for date in date_range:
    date_str = str(date)[:10]
    date_str_short = date_str.replace("-", "")

    # import temperature and humidity profile data:
    file_q = [file for file in files_q if date_str_short in file]
    if len(file_q) == 1:
        NN_DS = xr.open_dataset(file_q[0])
    else:
        continue        # no data -> skip

    file_temp = [file for file in files_temp if date_str_short in file]
    if len(file_temp) == 1:
        NN_DS_temp = xr.open_dataset(file_temp[0])

    # merge temp and q data:
    NN_DS['temp'] = NN_DS_temp.temp
    NN_DS['temp_rmse'] = NN_DS_temp.temp_rmse
    NN_DS_temp = NN_DS_temp.close()
    del NN_DS_temp

    # limit height to 8000 m:
    NN_DS = NN_DS.sel(height=slice(0,8000))

    # repair flags:
    NN_DS['flag_h'][np.isnan(NN_DS['flag_h'])] = 0.
    NN_DS['flag_m'][np.isnan(NN_DS['flag_m'])] = 0.

    # it may occur that the whole day is flagged. If so, skip this file:
    flag_mask = (NN_DS.flag_m == 0) & ((NN_DS.flag_h == 0) | (NN_DS.flag_h == 32))
    if not np.any(flag_mask):
        n_samp_real = 0
        continue
        pdb.set_trace()


    # limit radiosonde data to +/- 1 day around date:
    date_minus = date + np.timedelta64(-1, "D")
    date_plus = date + np.timedelta64(2, "D")   # +2 days, because T00:00:00
    RS_DS = RS_DS_all.sel(launch_time=slice(date_minus, date_plus))

    if len(RS_DS.launch_time) == 0: continue    # no radiosonde data


    # ret sonde to retrieval grid and fill the pressure data near the surface:
    rs_pres = RS_DS.pres
    rs_temp = RS_DS.temp
    rs_hgt = RS_DS.height.values
    RS_DS = RS_DS.interp(coords={'height': NN_DS.height.values})

    # use barometric height formula to get pressure at the surface: find lowest nonnan values
    lowest_nonnan = np.where((~np.isnan(rs_pres.values)) & (~np.isnan(rs_temp.values)))
    diff_lnn = np.diff(lowest_nonnan[1])        # if difference of the height dim is < 0, that's then the 
                                                # first index of the next radiosonde launch
    lowest_nonnan_idx = np.concatenate((np.array([0]), np.where(diff_lnn < 0)[0]+1))
    lowest_nonnan = (lowest_nonnan[0][lowest_nonnan_idx], lowest_nonnan[1][lowest_nonnan_idx])
    pres_ref = rs_pres.values[lowest_nonnan]    # lowest nonnan pres in Pa
    temp_ref = rs_temp.values[lowest_nonnan]    # lowest nonnan temp in K
    hgt_ref = rs_hgt[lowest_nonnan[1]]          # lowest nonnan height in m
    pres_sfc = pres_ref / (np.exp(-g*hgt_ref / (R_d*temp_ref)))     # extrapolated surface pressure in Pa
    RS_DS['pres'][:,0] = pres_sfc


    # interpolate radiosonde pressure on NN_DS time grid and compute relative humidity:
    rs_pres = RS_DS.pres.interp(launch_time=NN_DS.time)
    NN_DS['rh'] = xr.DataArray(convert_spechum_to_relhum(NN_DS.temp.values, rs_pres.values, 
                                NN_DS.q.values).astype(np.float32), dims=NN_DS.q.dims)
    NN_DS['rh'] = NN_DS.rh.where(NN_DS.rh < 2.0, other=float(-9999.))       # mask where relative humidity is too high

    # set attributes:
    NN_DS['rh'].attrs = {'units': '1',
                        'standard_name': 'relative_humidity',
                        'long_name': "relative humidity with respect to water",
                        'ancillary_variables': "rh_rmse",
                        'valid_min': np.array([0.0]).astype(np.float32)[0],
                        'valid_max': np.array([2.0]).astype(np.float32)[0],
                        'comment': ("Relative humidity was computed from retrieved temperature and specific humidity profiles, " +
                                    "as well as level 3 radiosonde air pressure data, which has been interpolated to the retrieval " +
                                    "grid and the retrieval time steps. " +
                                    "Note that the height grid does not reflect the true vertical resolution of the retrieved " +
                                    "profile. The vertical resolution of microwave radiometer retrievals is usually much " +
                                    "lower and rather on the order of kilometers.")
                        }
    NN_DS['rh'].encoding['_FillValue'] = float(-9999.)


    # also compute relative humidity uncertainty:
    rh_err = compute_RH_error(NN_DS.q.sel(time=flag_mask).mean('time').values, 
                                NN_DS.temp.sel(time=flag_mask).mean('time').values, 
                                rs_pres.sel(time=flag_mask).mean('time').values,
                                NN_DS.q_rmse.values, NN_DS.temp_rmse.values, np.full(NN_DS.height.shape, 100.0))
    NN_DS['rh_rmse'] = xr.DataArray(rh_err.astype(np.float32), dims=['height'])
    NN_DS['rh_rmse'].attrs = {'long_name': "Relative humidity uncertainty",
                                'standard_name': "relative_humidity standard error",
                                'units': "1",
                                 'comment': ("Computed from error propagation of this day's mean " +
                                            "temperature, specific humidity and radiosonde air pressure.")
                                }
    NN_DS['rh_rmse'].encoding["_FillValue"] = float(-9999.)     # add _FillValue attribute

    # to export the dataset, exclude the variables that already exist as standalone files:
    NN_DS = NN_DS.drop(['temp', 'temp_rmse', 'q', 'q_rmse'])


    # update flag:
    NN_DS['flag_h'].attrs['comment'] = NN_DS['flag_h'].attrs['comment'].replace("retrieved quantity valid range: [0.0, 0.06] in 1;", 
                                                                "retrieved quantity valid range: [0.0, 2.0] in 1;")
    NN_DS['flag_m'].attrs['comment'] = NN_DS['flag_m'].attrs['comment'].replace("retrieved quantity valid range: [0.0, 0.06] in 1;", 
                                                                "retrieved quantity valid range: [0.0, 2.0] in 1;")

    # set the 1024 flag if relative humidity thresholds are exceeded and if that flag has not been set:
    NN_DS['flag_h'] = NN_DS.flag_h.where(((np.all((NN_DS.rh >= 0.) & (NN_DS.rh <= 2.0), axis=1)) | (NN_DS.flag_h >= 1024)), other=NN_DS.flag_h+1024).astype(np.short)
    NN_DS['flag_m'] = NN_DS.flag_m.where(((np.all((NN_DS.rh >= 0.) & (NN_DS.rh <= 2.0), axis=1)) | (NN_DS.flag_m >= 1024)), other=NN_DS.flag_m+1024).astype(np.short)
    NN_DS['flag_h'].encoding["_FillValue"] = np.array([0]).astype(np.short)[0]
    NN_DS['flag_m'].encoding["_FillValue"] = np.array([0]).astype(np.short)[0]

    # update other attributes:
    NN_DS.attrs['title'] = ("Relative humidity (rh) computed from radiosonde air pressure, retrieved temperature (temp) and " + 
                            "specific humidity (q) during the MOSAiC expedition")
    NN_DS.attrs['source'] += " and level 3 radiosondes"
    NN_DS.attrs['dependencies'] += ", extended radiosonde profiles: https://doi.org/10.1594/PANGAEA.961881"
    NN_DS.attrs['history'] += f"; {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}, computed relative humidity with rh_from_temp_q.py"

    exclude_attrs = ['retrieval_net_architecture', 'retrieval_batch_size', 'retrieval_epochs', 
                                'retrieval_learning_rate', 'retrieval_activation_function', 'retrieval_feature_range', 
                                'retrieval_rng_seed', 'retrieval_kernel_initializer', 'retrieval_optimizer', 
                                'retrieval_dropout', 'retrieval_batch_normalization', 'retrieval_callbacks', 
                                'input_vector', 'output_vector']
    for attr in exclude_attrs:
        del NN_DS.attrs[attr]


    # encode time:
    NN_DS['time'] = NN_DS.time.values.astype("datetime64[s]").astype(np.float64)
    NN_DS['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
    NN_DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
    NN_DS['time'].encoding['dtype'] = 'double'


    # Make sure that _FillValue is not added to certain variables:
    exclude_vars_fill_value = ['time', 'height']
    for kk in exclude_vars_fill_value:
        if kk in NN_DS.variables:
            NN_DS[kk].encoding["_FillValue"] = None


    # export:
    outfile = set_dict['path_output'] + os.path.basename(file_q[0]).replace("_q_", "_rh_")
    NN_DS.to_netcdf(outfile, mode='w', format="NETCDF4")

    print(f"\rEdited {os.path.basename(file_q[0]).replace('_q_', '_rh_')}....", sep='', end='', flush=True)
    NN_DS = NN_DS.close()
    del NN_DS