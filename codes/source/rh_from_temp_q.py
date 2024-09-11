import sys
import os
import glob
import datetime as dt
import pdb

wdir = os.getcwd() + "/"
path_tools = os.path.dirname(wdir[:-1]) + "/tools/"
path_data_base = os.path.abspath(wdir + "../..") + "/data/"

import numpy as np
import xarray as xr

sys.path.insert(0, path_tools)

from import_data import import_hatpro_level1b_daterange_pangaea
from met_tools import convert_spechum_to_relhum, e_sat, virtual_temp


# constants (taken from met_tools.py):
R_d = 287.0597  # gas constant of dry air, in J kg-1 K-1
R_v = 461.5     # gas constant of water vapour, in J kg-1 K-1
M_dv = R_d / R_v # molar mass ratio , in ()
g = 9.80665     # gravitation acceleration, in m s^-2 (from https://doi.org/10.6028/NIST.SP.330-2019 )


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
    Script to compute relative humidity from retrieved specific humidity and temperature profiles.
    The relative humidity data will be saved as netCDF files similar to those of the retrieved q 
    and temp profiles..
    - import retrieved MWR data
    - loop over MOSAiC days: get surface pressure from HATPRO's weather station, apply running mean 
        for smoothing and interpolate to MWR obs time grid
"""


# Paths
path_data = {'nn_syn_mosaic': path_data_base + "retrieval_output/mosaic/",
            'tb_hat': path_data_base + "hatpro_l1/"}
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


    # for pressure estimation, get surface pressure observations from HATPRO:
    hat_dict = import_hatpro_level1b_daterange_pangaea(path_data['tb_hat'], date_str, date_str)

    # identify time duplicates (xarray dataset coords not permitted to have any):
    hat_dupno = np.where(~(np.diff(hat_dict['time']) == 0))[0]

    # fill gaps, then reduce to non-time-duplicates and forward it to the pressure computation:
    HAT_pres = xr.DataArray(hat_dict['pa'][hat_dupno], dims=['time'], 
                            coords={'time': (['time'], hat_dict['time'][hat_dupno].astype('datetime64[s]'))})
    HAT_pres[HAT_pres < 0] = np.nan     # fill fill values with nan
    HAT_pres = HAT_pres.interpolate_na(dim='time', method='linear')
    HAT_pres = HAT_pres.ffill(dim='time')

    # apply smoothing to correct measurement errors: 60 min running mean ; then get it onto retrieval time grid:
    HAT_pres_DF = HAT_pres.to_dataframe(name='pres')
    HAT_pres = HAT_pres_DF.rolling("60min", center=True).mean().to_xarray().pres
    HAT_pres = HAT_pres.interp(time=NN_DS.time)

    # compute pressure for each height level using the hypsometric equation: loop over height grid:
    hgt = NN_DS.height.values
    n_hgt = len(hgt)
    temp_v = virtual_temp(NN_DS.temp.values, NN_DS.q.values)    # virtual temperature
    pres = np.full(NN_DS.temp.values.shape, np.nan)
    for k in range(n_hgt):
        if k == 0:
            pres[:,k] = HAT_pres
        else:
            # layer-average virtual temperature:
            temp_v_avg = 0.5*(temp_v[:,k-1] + temp_v[:,k])

            # hypsometric height formula:
            pres[:,k] = pres[:,k-1] * np.exp((hgt[k-1] - hgt[k])*g / (R_d*temp_v_avg))

    pres = xr.DataArray(pres, dims=NN_DS.q.dims, coords=NN_DS.q.coords)     # to xarray data array


    # compute relative humidity:
    NN_DS['rh'] = xr.DataArray(convert_spechum_to_relhum(NN_DS.temp.values, pres.values, 
                                NN_DS.q.values).astype(np.float32), dims=NN_DS.q.dims)
    NN_DS['rh'] = NN_DS.rh.where(NN_DS.rh < 2.0, other=float(-9999.))       # mask where relative humidity is too high
    

    # flag strange relative humidity values (when temperature estimates were bad:
    NN_DS['rh'] = NN_DS.rh.where((NN_DS.temp > 180.0) & (NN_DS.temp < 330.0), other=float(-9999.))

    # set attributes:
    NN_DS['rh'].attrs = {'units': '1',
                        'standard_name': 'relative_humidity',
                        'long_name': "relative humidity with respect to water",
                        'ancillary_variables': "rh_rmse",
                        'valid_min': np.array([0.0]).astype(np.float32)[0],
                        'valid_max': np.array([2.0]).astype(np.float32)[0],
                        'comment': ("Relative humidity was computed from retrieved temperature and specific humidity profiles " +
                                    "using the hypsometric equation and HATPRO's surface air pressure data. " +
                                    "Note that the height grid does not reflect the true vertical resolution of the retrieved " +
                                    "profile. The vertical resolution of microwave radiometer retrievals is usually much " +
                                    "lower and rather on the order of kilometers.")
                        }
    NN_DS['rh'].encoding['_FillValue'] = float(-9999.)


    # also compute relative humidity uncertainty:
    rh_err = compute_RH_error(NN_DS.q.sel(time=flag_mask).mean('time').values, 
                                NN_DS.temp.sel(time=flag_mask).mean('time').values, 
                                pres.sel(time=flag_mask).mean('time').values,
                                NN_DS.q_rmse.values, NN_DS.temp_rmse.values, np.full(NN_DS.height.shape, 100.0))
    NN_DS['rh_rmse'] = xr.DataArray(rh_err.astype(np.float32), dims=['height'])
    NN_DS['rh_rmse'].attrs = {'long_name': "Relative humidity uncertainty",
                                'standard_name': "relative_humidity standard error",
                                'units': "1",
                                 'comment': ("Computed from error propagation of this day's mean " +
                                            "temperature, specific humidity and air pressure.")
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
    NN_DS.attrs['title'] = ("Relative humidity (rh) computed with the hypsometric equation, surface air pressure from HATPRO, retrieved temperature (temp) and " +  
                            "specific humidity (q) during the MOSAiC expedition")
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