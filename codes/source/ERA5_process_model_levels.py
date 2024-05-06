import os
import sys
import glob
import datetime as dt
import pdb
import gc

wdir = os.getcwd() + "/"
path_tools = os.path.dirname(wdir[:-1]) + "/tools/"
path_data_base = os.path.dirname(wdir[:-10]) + "/data/"
sys.path.insert(0, path_tools)

import numpy as np
import xarray as xr

from met_tools import Z_from_GP


R_d = 287.0597


def load_par_data():
    a_half = np.array([0.000000, 2.000365, 3.102241, 4.666084, 6.827977, 9.746966, 13.605424, 18.608931, 24.985718,
                        32.985710, 42.879242, 54.955463, 69.520576, 86.895882, 107.415741, 131.425507, 159.279404,
                        191.338562, 227.968948, 269.539581,316.420746,368.982361,427.592499,492.616028,564.413452,
                        643.339905,729.744141,823.967834,926.344910,1037.201172,1156.853638,1285.610352,1423.770142,
                        1571.622925,1729.448975,1897.519287,2076.095947,2265.431641,2465.770508,2677.348145,
                        2900.391357,3135.119385,3381.743652,3640.468262,3911.490479,4194.930664,4490.817383,
                        4799.149414,5119.895020,5452.990723,5798.344727,6156.074219,6526.946777,6911.870605,
                        7311.869141,7727.412109,8159.354004,8608.525391,9076.400391,9562.682617,10065.978516,
                        10584.631836,11116.662109,11660.067383,12211.547852,12766.873047,13324.668945,13881.331055,
                        14432.139648,14975.615234,15508.256836,16026.115234,16527.322266,17008.789063,17467.613281,
                        17901.621094,18308.433594,18685.718750,19031.289063,19343.511719,19620.042969,19859.390625,
                        20059.931641,20219.664063,20337.863281,20412.308594,20442.078125,20425.718750,20361.816406,
                        20249.511719,20087.085938,19874.025391,19608.572266,19290.226563,18917.460938,18489.707031,
                        18006.925781,17471.839844,16888.687500,16262.046875,15596.695313,14898.453125,14173.324219,
                        13427.769531,12668.257813,11901.339844,11133.304688,10370.175781,9617.515625,8880.453125,
                        8163.375000,7470.343750,6804.421875,6168.531250,5564.382813,4993.796875,4457.375000,
                        3955.960938,3489.234375,3057.265625,2659.140625,2294.242188,1961.500000,1659.476563,
                        1387.546875,1143.250000,926.507813,734.992188,568.062500,424.414063,302.476563,202.484375,
                        122.101563,62.781250,22.835938,3.757813,0.000000,0.000000])
    b_half = np.array([0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
                        0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
                        0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
                        0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
                        0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
                        0.000000,0.000000,0.000000,0.000000,0.000000,0.000007,0.000024,0.000059,0.000112,0.000199,
                        0.000340,0.000562,0.000890,0.001353,0.001992,0.002857,0.003971,0.005378,0.007133,0.009261,
                        0.011806,0.014816,0.018318,0.022355,0.026964,0.032176,0.038026,0.044548,0.051773,0.059728,
                        0.068448,0.077958,0.088286,0.099462,0.111505,0.124448,0.138313,0.153125,0.168910,0.185689,
                        0.203491,0.222333,0.242244,0.263242,0.285354,0.308598,0.332939,0.358254,0.384363,0.411125,
                        0.438391,0.466003,0.493800,0.521619,0.549301,0.576692,0.603648,0.630036,0.655736,0.680643,
                        0.704669,0.727739,0.749797,0.770798,0.790717,0.809536,0.827256,0.843881,0.859432,0.873929,
                        0.887408,0.899900,0.911448,0.922096,0.931881,0.940860,0.949064,0.956550,0.963352,0.969513,
                        0.975078,0.980072,0.984542,0.988500,0.991984,0.995003,0.997630,1.000000])
    a_model = 0.5*(a_half[1:] + a_half[:-1])
    b_model = 0.5*(b_half[1:] + b_half[:-1])

    # save to dataset:
    DS = xr.Dataset(coords={'level': (['level'], np.arange(1, 138, dtype=np.int32),
                                        {'long_name': "model level", 
                                        'standard_name': "atmosphere_hybrid_sigma_pressure_coordinate",
                                        'short_name': "lev", 'formula': "p = a + b*ps",
                                        'order': "near top of atmosphere to near surface",
                                        'units': "1"}),
                            'half_level': (['half_level'], np.arange(1, 139, dtype=np.int32),
                                        {'long_name': "half level", 
                                        'alternate_name': "interface",
                                        'short_name': "hlev", 'formula': "p = a + b*ps",
                                        'order': "top of atmosphere to surface",
                                        'units': "1"})})
    DS['a_model'] = xr.DataArray(a_model, dims=['level'], attrs={'long_name': "model level a coefficient",
                                    'short_name': "a_model", 'units': "Pa"})
    DS['b_model'] = xr.DataArray(b_model, dims=['level'], attrs={'long_name': "model level b coefficient",
                                    'short_name': "b_model", 'units': "1"})
    DS['a_half'] = xr.DataArray(a_half, dims=['half_level'], attrs={'long_name': "half level a coefficient",
                                    'alternate_name': "interface a coefficient", 'short_name': "a_half",
                                    'units': "Pa", 
                                    'source': "https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions"})
    DS['b_half'] = xr.DataArray(b_half, dims=['half_level'], attrs={'long_name': "half level b coefficient",
                                    'alternate_name': "interface b coefficient", 'short_name': "b_half",
                                    'units': "1", 
                                    'source': "https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions"})

    return DS


"""
    ERA5 data on model levels (downloaded via CDSAPI following
    https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5 ) will be processed here. 
    Pressure on model leves will be computed (requires parameters a and b on model levels and half
    levels). The a and b parameters on model and half levels have been obtained through ERA5 data 
    downloaded from https://rda.ucar.edu/datasets/ds633.6/dataaccess/ or extracted from 
    https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions . The computation of 
    pressure on model and levels is based on
    https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height .
    - import ERA5 data and parameters a and b
    - compute pressure levels
    - export
"""


# Paths:
path_data = path_data_base + "ERA5_data/"
path_output = path_data_base + "ERA5_data/"


# settings:
set_dict = {'path_output': path_output,
            'levlim': 90,       # number of levels to remain in the ERA5 dataset (seen from the surface);
                                # example: levlim=90 means that the lowest 90 model levels and 91 half 
                                # levels (incl surface) remain
            }


# create output path if not existing:
outpath_dir = os.path.dirname(set_dict['path_output'])
if not os.path.exists(outpath_dir):
    os.makedirs(outpath_dir)


# import pressure level computation parameters:
PAR_DS = load_par_data()


# find files; identify which files belong to the same time stamp
files = sorted(glob.glob(path_data + "*.nc"))
files = [file for file in files if 'model_lev' in file]

# loop over identified dates and import files:
for file in files:
        
    print(f"Processing {file}:")

    try:
        DS_all = xr.open_dataset(file)
    except:
        pdb.set_trace()
        print(f"Could not open {file}.... skipping to next file....")
        continue

    # split data set by dates:
    dates = np.unique(DS_all.time.dt.date)
    for date in dates:
        DS = DS_all.sel(time=date.strftime("%Y-%m-%d"))

        # convert longitudes from 0-360 deg to -180-+180 deg:
        DS['longitude'] = xr.where(DS.longitude > 180.0, DS.longitude - 360.0, DS.longitude)


        # convert natural log of surface pressure to normal surface pressure:
        DS['lnsp'] = DS.lnsp.sel(level=1)       # it's actually on level 137, which is closest to earth's surface
        DS['z'] = DS.z.sel(level=1)
        DS['pres_sfc'] = np.exp(DS.lnsp)


        # compute pressure levels:
        # DS['pres'] could also be computed by:
        # 0.5*(DS.pres_half.isel(half_level=slice(1,None)).values + DS.pres_half.isel(half_level=slice(None,-1)).values)
        DS['pres_half'] = PAR_DS.a_half + PAR_DS.b_half*DS.pres_sfc
        DS['pres_half'].attrs = {'long_name': "Air pressure on half levels", 'units': "Pa"}
        DS['pres'] = PAR_DS.a_model + PAR_DS.b_model*DS.pres_sfc        # identical to using the line commented out above
        DS['pres'].attrs = {'standard_name': 'air_pressure', 'long_name': "Air pressure on model levels", 
                            'units': "Pa"}

        # rename some variables:
        DS = DS.rename({'z': 'z_sfc', 't': 'temp'})
        DS['Z_sfc'] = xr.DataArray(Z_from_GP(DS.z_sfc.values), dims=DS.z_sfc.dims, 
                                    attrs={'standard_name': "surface_geopotential_height", 'long_name': "Surface geopotential height",
                                            'units': "m"})
        DS['z_sfc'].attrs['standard_name'] = "surface_geopotential"
        DS['z_sfc'].attrs['long_name'] = "Surface geopotential"


        # also compute geopotential height:
        # loop over model levels:
        print("Computing geopotential height for model levels....")
        n_lev = len(DS.level)           # number of model levels (or better 'layers')
        n_hlev = len(DS.half_level)     # number of half model levels (or better 'levels')
        z_h = DS.z_sfc.values           # start with surface geopotential in m**2 s**-2
        z_f_save = np.zeros(DS.pres.shape)          # height is on axis=0; geopotential in m**2 s**-2
        z_h_save = np.zeros(DS.pres_half.shape)     # height is on axis=0
        z_h_save[-1,...] = z_h                      # surface == index -1
        for i_lev in sorted(range(len(DS.level)), reverse=True):

            # compute virtual temperature:
            t_v = DS.temp.values[:,i_lev,:,:] * (1. + 0.609133 * DS.q.values[:,i_lev,:,:])

            # the pressure at the current level, indicated by the index i_lev, is actually the mean 
            # pressure of the layer [pres_hlev_plus, pres_hlev] where pres_hlev_plus is closer to earth's sfc
            # and pres_hlev is further aloft.
            pres_hlev, pres_hlev_plus = DS.pres_half.values[i_lev,...], DS.pres_half.values[i_lev+1,...]


            if i_lev == 0:  # top of atmosphere
                dlog_p = np.log(pres_hlev_plus / 0.1)
                alpha = np.log(2)
            else:
                dlog_p = np.log(pres_hlev_plus / pres_hlev)
                alpha = 1. - ((pres_hlev / (pres_hlev_plus - pres_hlev)) * dlog_p)

            t_v = t_v * R_d

            # z_f is the geopotential of this full level. Integrate from previous (lower) half-level z_h to the
            # full level:
            z_f = z_h + (t_v * alpha)

            # z_h is the geopotential of 'half-levels' integrate z_h to next half level:
            z_h = z_h + (t_v * dlog_p)


            # save computed geopotential:
            z_h_save[i_lev,...] = z_h
            z_f_save[i_lev,...] = z_f


        # Convert geopotential to geopotential height and save to dataset:
        DS['Z_half'] = xr.DataArray(Z_from_GP(z_h_save), dims=DS.pres_half.dims, 
                                    attrs={'standard_name': "geopotential_height", 'long_name': "Geopotential height",
                                            'units': "m"})
        DS['Z'] = xr.DataArray(Z_from_GP(z_f_save), dims=DS.pres.dims, 
                                    attrs={'standard_name': "geopotential_height", 'long_name': "Geopotential height",
                                            'units': "m"})


        # flip height axis and limit to certain model level number:
        DS = DS.reindex(level=DS.level[::-1], half_level=DS.half_level[::-1])
        DS = DS.isel(level=slice(None,set_dict['levlim']), half_level=slice(None,set_dict['levlim']+1))


        # reorder some arrays so that the height axis is axis -1 and time is axis 0:
        for dv in ['temp', 'q', 'pres', 'Z']:
            DS[dv] = DS[dv].transpose('time', 'latitude', 'longitude', 'level')
        for dv in ['pres_half', 'Z_half']:
            DS[dv] = DS[dv].transpose('time', 'latitude', 'longitude', 'half_level')


        # some encoding of the dataset:
        for kk in DS.variables:
            DS[kk].encoding["_FillValue"] = None
        
        # add some attributes:
        DS.attrs['processing'] = ("Dataset was processed with ERA5_process_model_levels.py by Andreas Walbroel (a.walbroel@uni-koeln.de) " +
                                    "to compute pressure and geopotential height from model levels.")


        # encode time:
        DS['time'] = DS.time.values.astype("datetime64[s]").astype(np.float64)
        DS['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
        DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
        DS['time'].encoding['dtype'] = 'double'


        # export:
        nc_filename = os.path.basename(file)
        DS.to_netcdf(path_output + nc_filename, mode='w', format="NETCDF4")
        DS = DS.close()
        del DS

        print(f"Saved {path_output + nc_filename}")