import numpy as np
import netCDF4 as nc
import datetime as dt
# import pandas as pd
# import xarray as xr
import copy
import pdb
import os
import glob
import sys
import warnings
import csv
from met_tools import *
from data_tools import *


def import_mirac_level2a(
    filename,
    keys='basic',
    minute_avg=False):

    """
    Importing MiRAC-P level 2a (integrated quantities, e.g. IWV, LWP).

    Parameters:
    -----------
    filename : str
        Path and filename of mwr data (level2a).
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (keys='all') or import basic keys
        that the author considers most important (keys='basic')
        or leave this argument out.
    minute_avg : bool
        If True: averages over one minute are computed and returned instead of False when all
        data points are returned (more outliers, higher memory usage).
    """

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        keys = ['time', 'lat', 'lon', 'zsl', 'azi', 'ele', 'flag']
        if 'clwvi_' in filename:
            for add_key in ['clwvi']: keys.append(add_key)
        if 'prw_' in filename:
            for add_key in ['prw']: keys.append(add_key)

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in level 2a file." % key)
        mwr_dict[key] = np.asarray(file_nc.variables[key])


    if 'time' in keys:  # avoid nasty digita after decimal point
        mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
        time_shape_old = mwr_dict['time'].shape

        if minute_avg:
            # start the timer at the first time, when seconds is 00 (e.g. 09:31:00):
            time0 = mwr_dict['time'][0]     # start time in sec since 1970-01-01...
            dt_time0 = dt.datetime.utcfromtimestamp(mwr_dict['time'][0])
            dt_time0_Y = dt_time0.year
            dt_time0_M = dt_time0.month
            dt_time0_D = dt_time0.day
            dt_time0_s = dt_time0.second
            dt_time0_m = dt_time0.minute
            dt_time0_h = dt_time0.hour
            if dt_time0_s != 0:     # then the array mwr_dict['time'] does not start at second 0
                start_time = datetime_to_epochtime(dt.datetime(dt_time0_Y, dt_time0_M,
                                                    dt_time0_D, dt_time0_h, dt_time0_m+1, 0))
            else:
                start_time = time0

            if np.abs(start_time - time0) >= 60:
                print("Start time is far off the first time point in this file.")
                pdb.set_trace()
            # compute minute average
            n_minutes = int(np.ceil((mwr_dict['time'][-1] - start_time)/60))    # number of minutes
            min_time_idx_save = 0       # saves the last min_time_index value to speed up computation
            for min_count in range(n_minutes):
                # find time_idx when time is in the correct minute:
                # slower version:
                # # # # min_time_idx = np.argwhere((mwr_dict['time'] >= (start_time + min_count*60)) & 
                                # # # # (mwr_dict['time'] < (start_time + (min_count+1)*60))).flatten()
                # faster version:
                min_time_idx = np.argwhere((mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] >= (start_time + min_count*60)) & 
                                (mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] < (start_time + (min_count+1)*60))).flatten()

                # it may occur that no measurement exists in a certain minute-range. Then
                # we cannot compute the average but simply set that minute to nan.
                if len(min_time_idx) == 0:
                    for key in keys:
                        if key == 'time':
                            mwr_dict['time'][min_count] = start_time + min_count*60
                        elif mwr_dict[key].shape == time_shape_old and key != 'flag':
                            mwr_dict[key][min_count] = np.nan
                        elif mwr_dict[key].shape == time_shape_old and key == 'flag':
                            mwr_dict[key][min_count] = 99       # np.nan not possible because int is required
                else:
                    min_time_idx = min_time_idx + min_time_idx_save     # also belonging to the 'faster version'
                    min_time_idx_save = min_time_idx[-1]                # also belonging to the 'faster version'
                    for key in keys:
                        if key == 'time':
                            mwr_dict['time'][min_count] = start_time + min_count*60
                        elif mwr_dict[key].shape == time_shape_old and key != 'flag':
                            if min_time_idx[-1] < len(mwr_dict['time']):
                                mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx])
                            else:
                                mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx[0]:])
                        elif mwr_dict[key].shape == time_shape_old and key == 'flag':
                            # find out how many entries show flag > 0. Then if it exceeds a threshold
                            # the whole minute is flagged. If threshold not exceeded, minute is not
                            # flagged!
                            if min_time_idx[-1] < len(mwr_dict['time']):
                                if np.count_nonzero(mwr_dict[key][min_time_idx]) > len(min_time_idx)/10:    
                                    # then there are too many flags set... so flag the whole minute:
                                    mwr_dict[key][min_count] = 99
                                else:
                                    mwr_dict[key][min_count] = 0
                            else:
                                if np.count_nonzero(mwr_dict[key][min_time_idx[0]:]) > len(min_time_idx)/10:
                                    # then there are too many flags set... so flag the whole minute:
                                    mwr_dict[key][min_count] = 99
                                else:
                                    mwr_dict[key][min_count] = 0

            # truncate time arrays to reduce memory usage!
            for key in keys:
                if mwr_dict[key].shape == time_shape_old:
                    mwr_dict[key] = mwr_dict[key][:n_minutes]

    else:
        if minute_avg:
            raise KeyError("'time' must be included in the list of keys that will be imported for minute averages.")

    return mwr_dict


def import_hatpro_level1b(
    filename,
    keys='basic'):

    """
    Importing HATPRO level 1b (zenith TBs in K). Can also be used to import
    MiRAC-P level 1b (zenith TBs in K) data if it was processed with mwr_pro.

    Parameters:
    -----------
    filename : str
        Path and filename of mwr data (level2b).
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (specify keys='all') or import basic keys
        that the author considers most important (specify keys='basic')
        or leave this argument out.).
    """

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        keys = ['time', 'freq_sb', 'flag', 'tb']

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key '%s'. Key not found in level 1 file." % key)
        mwr_dict[key] = np.asarray(file_nc.variables[key])


    if 'time' in keys:  # avoid nasty digita after decimal point
        mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)

    return mwr_dict


def import_hatpro_level2a(
    filename,
    keys='basic',
    minute_avg=False):

    """
    Importing HATPRO level 2a (integrated quantities, e.g. IWV, LWP).

    Parameters:
    -----------
    filename : str
        Path and filename of mwr data (level2a).
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (keys='all') or import basic keys
        that the author considers most important (keys='basic')
        or leave this argument out.
    minute_avg : bool
        If True: averages over one minute are computed and returned instead of False when all
        data points are returned (more outliers, higher memory usage).
    """

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        keys = ['time', 'lat', 'lon', 'zsl', 'flag']
        if 'clwvi_' in filename:
            for add_key in ['clwvi', 'clwvi_err', 'clwvi_offset']: keys.append(add_key)
        if 'prw_' in filename:
            for add_key in ['prw', 'prw_err', 'prw_offset']: keys.append(add_key)

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in level 2a file." % key)
        mwr_dict[key] = np.asarray(file_nc.variables[key])


    if 'time' in keys:  # avoid nasty digita after decimal point
        mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
        time_shape_old = mwr_dict['time'].shape

        if minute_avg:
            # start the timer at the first time, when seconds is 00 (e.g. 09:31:00):
            time0 = mwr_dict['time'][0]     # start time in sec since 1970-01-01...
            dt_time0 = dt.datetime.utcfromtimestamp(mwr_dict['time'][0])
            dt_time0_Y = dt_time0.year
            dt_time0_M = dt_time0.month
            dt_time0_D = dt_time0.day
            dt_time0_s = dt_time0.second
            dt_time0_m = dt_time0.minute
            dt_time0_h = dt_time0.hour
            if dt_time0_s != 0:     # then the array mwr_dict['time'] does not start at second 0
                start_time = datetime_to_epochtime(dt.datetime(dt_time0_Y, dt_time0_M, dt_time0_D,
                                                    dt_time0_h, dt_time0_m+1, 0))
            else:
                start_time = time0

            if np.abs(start_time - time0) >= 60:
                print("Start time is far off the first time point in this file.")
                pdb.set_trace()
            # compute minute average
            n_minutes = int(np.ceil((mwr_dict['time'][-1] - start_time)/60))    # number of minutes
            min_time_idx_save = 0       # saves the last min_time_index value to speed up computation
            for min_count in range(n_minutes):
                # find time_idx when time is in the correct minute:
                # slower version:
                # # # # min_time_idx = np.argwhere((mwr_dict['time'] >= (start_time + min_count*60)) & 
                                # # # # (mwr_dict['time'] < (start_time + (min_count+1)*60))).flatten()
                # faster version:
                min_time_idx = np.argwhere((mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] >= (start_time + min_count*60)) & 
                                (mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] < (start_time + (min_count+1)*60))).flatten()

                # it may occur that no measurement exists in a certain minute-range. Then
                # we cannot compute the average but simply set that minute to nan.
                if len(min_time_idx) == 0:
                    for key in keys:
                        if key == 'time':
                            mwr_dict['time'][min_count] = start_time + min_count*60
                        elif mwr_dict[key].shape == time_shape_old and key != 'flag':
                            mwr_dict[key][min_count] = np.nan
                        elif mwr_dict[key].shape == time_shape_old and key == 'flag':
                            mwr_dict[key][min_count] = 99       # np.nan not possible because int is required
                else:
                    min_time_idx = min_time_idx + min_time_idx_save     # also belonging to the 'faster version'
                    min_time_idx_save = min_time_idx[-1]                # also belonging to the 'faster version'
                    for key in keys:
                        if key == 'time':
                            mwr_dict['time'][min_count] = start_time + min_count*60
                        elif mwr_dict[key].shape == time_shape_old and key != 'flag':
                            if min_time_idx[-1] < len(mwr_dict['time']):
                                mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx])
                            else:
                                mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx[0]:])
                        elif mwr_dict[key].shape == time_shape_old and key == 'flag':
                            # find out how many entries show flag > 0. Then if it exceeds a threshold
                            # the whole minute is flagged. If threshold not exceeded, minute is not
                            # flagged!
                            if min_time_idx[-1] < len(mwr_dict['time']):
                                if np.count_nonzero(mwr_dict[key][min_time_idx]) > len(min_time_idx)/10:    
                                    # then there are too many flags set... so flag the whole minute:
                                    mwr_dict[key][min_count] = 99
                                else:
                                    mwr_dict[key][min_count] = 0
                            else:
                                if np.count_nonzero(mwr_dict[key][min_time_idx[0]:]) > len(min_time_idx)/10:
                                    # then there are too many flags set... so flag the whole minute:
                                    mwr_dict[key][min_count] = 99
                                else:
                                    mwr_dict[key][min_count] = 0

            # truncate time arrays to reduce memory usage!
            for key in keys:
                if mwr_dict[key].shape == time_shape_old:
                    mwr_dict[key] = mwr_dict[key][:n_minutes]

    else:
        if minute_avg:
            raise KeyError("'time' must be included in the list of keys that will be imported for minute averages.")

    return mwr_dict


def import_hatpro_level2b(
    filename,
    keys='basic',
    minute_avg=False):

    """
    Importing HATPRO level 2b (zenith profiles, temperature or humidity 
    (in K or kg m^-3, respectively).

    Parameters:
    -----------
    filename : str
        Path and filename of mwr data (level2b).
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (specify keys='all') or import basic keys
        that the author considers most important (specify keys='basic')
        or leave this argument out.
    minute_avg : bool
        If True: averages over one minute are computed and returned. False: all
        data points are returned (more outliers, higher memory usage but may result in
        long computation time).
    """

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        keys = ['time', 'lat', 'lon', 'zsl', 'height', 'flag']
        if 'hua_' in filename:
            for add_key in ['hua']: keys.append(add_key)
        if 'ta_' in filename:
            for add_key in ['ta']: keys.append(add_key)

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key '%s'. Key not found in level 2b file." % key)
        mwr_dict[key] = np.asarray(file_nc.variables[key])


    if 'time' in keys:  # avoid nasty digita after decimal point
        mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
        time_shape_old = mwr_dict['time'].shape

        if minute_avg:
            # start the timer at the first time, when seconds is 00 (e.g. 09:31:00):
            time0 = mwr_dict['time'][0]     # start time in sec since 1970-01-01...
            dt_time0 = dt.datetime.utcfromtimestamp(mwr_dict['time'][0])
            dt_time0_Y = dt_time0.year
            dt_time0_M = dt_time0.month
            dt_time0_D = dt_time0.day
            dt_time0_s = dt_time0.second
            dt_time0_m = dt_time0.minute
            dt_time0_h = dt_time0.hour
            if dt_time0_s != 0:     # then the array mwr_dict['time'] does not start at second 0
                start_time = datetime_to_epochtime(dt.datetime(dt_time0_Y, dt_time0_M,
                                dt_time0_D, dt_time0_h, dt_time0_m+1, 0))
            else:
                start_time = time0

            if np.abs(start_time - time0) >= 60:
                print("Start time is far off the first time point in this file.")
                pdb.set_trace()
            # compute minute average
            n_minutes = int(np.ceil((mwr_dict['time'][-1] - start_time)/60))    # number of minutes
            min_time_idx_save = 0       # saves the last min_time_index value to speed up computation
            for min_count in range(n_minutes):
                # find time_idx when time is in the correct minute:
                # slower version:
                # # # # min_time_idx = np.argwhere((mwr_dict['time'] >= (start_time + min_count*60)) & 
                                # # # # (mwr_dict['time'] < (start_time + (min_count+1)*60))).flatten()
                # faster version:
                min_time_idx = np.argwhere((mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] >= (start_time + min_count*60)) & 
                                (mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] < (start_time + (min_count+1)*60))).flatten()

                # it may occur that no measurement exists in a certain minute-range. Then
                # we cannot compute the average but simply set that minute to nan.
                if len(min_time_idx) == 0:
                    for key in keys:
                        if key == 'time':
                            mwr_dict['time'][min_count] = start_time + min_count*60
                        elif mwr_dict[key].shape == time_shape_old and key != 'flag':
                            mwr_dict[key][min_count] = np.nan
                        elif mwr_dict[key].shape == time_shape_old and key == 'flag':
                            mwr_dict[key][min_count] = 99       # np.nan not possible because int is required
                else:
                    min_time_idx = min_time_idx + min_time_idx_save     # also belonging to the 'faster version'
                    min_time_idx_save = min_time_idx[-1]                # also belonging to the 'faster version'
                    for key in keys:
                        if key == 'time':
                            mwr_dict['time'][min_count] = start_time + min_count*60
                        elif mwr_dict[key].shape == time_shape_old and key != 'flag':
                            if min_time_idx[-1] < len(mwr_dict['time']):
                                mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx])
                            else:
                                mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx[0]:])
                        elif mwr_dict[key].shape == time_shape_old and key == 'flag':
                            # find out how many entries show flag > 0. Then if it exceeds a threshold
                            # the whole minute is flagged. If threshold not exceeded, minute is not
                            # flagged!
                            if min_time_idx[-1] < len(mwr_dict['time']):
                                if np.count_nonzero(mwr_dict[key][min_time_idx]) > len(min_time_idx)/10:    
                                    # then there are too many flags set... so flag the whole minute:
                                    mwr_dict[key][min_count] = 99
                                else:
                                    mwr_dict[key][min_count] = 0
                            else:
                                if np.count_nonzero(mwr_dict[key][min_time_idx[0]:]) > len(min_time_idx)/10:
                                    # then there are too many flags set... so flag the whole minute:
                                    mwr_dict[key][min_count] = 99
                                else:
                                    mwr_dict[key][min_count] = 0

            # truncate time arrays to reduce memory usage!
            for key in keys:
                if mwr_dict[key].shape == time_shape_old:
                    mwr_dict[key] = mwr_dict[key][:n_minutes]

    else:
        if minute_avg:
            raise KeyError("'time' must be included in the list of keys that will be imported for minute averages.")


    return mwr_dict


def import_hatpro_level2c(
    filename,
    keys='basic'):

    """
    Importing HATPRO level 2c (boundary layer profiles, temperature (or humidity)
    (in K or kg m^-3, respectively).

    Parameters:
    -----------
    filename : str
        Path and filename of mwr data (level2c).
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (specify keys='all') or import basic keys
        that the author considers most important (specify keys='basic')
        or leave this argument out.
    """

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        keys = ['time', 'lat', 'lon', 'zsl', 'height', 'flag']
        if 'hua_' in filename:
            for add_key in ['hua']: keys.append(add_key)
        if 'ta_' in filename:
            for add_key in ['ta']: keys.append(add_key)

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key '%s'. Key not found in level 2c file." % key)
        mwr_dict[key] = np.asarray(file_nc.variables[key])


    if 'time' in keys:  # avoid nasty digita after decimal point
        mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
        time_shape_old = mwr_dict['time'].shape

    return mwr_dict


def import_hatpro_level1b_daterange_pangaea(
    path_data,
    date_start,
    date_end=None):

    """
    Runs through all days between a start and an end date. It concats the level 1b TB time
    series of each day so that you'll have one dictionary, whose 'TB' will contain the TB
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Path of level 1 (brightness temperature, TB) data. This directory contains daily files
        as netCDF.
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str or None
        If date_start is str: Marks the last day of the desired period. To be specified in 
        yyyy-mm-dd (e.g. 2021-01-14)!
    """

    def cut_vars(DS):
        DS = DS.drop_vars(['freq_sb', 'freq_shift', 'tb_absolute_accuracy', 'tb_cov'])
        return DS


    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # Identify files in the date range: First, load all into a list, then check which ones 
    # suit the daterange:
    mwr_dict = dict()
    sub_str = "_v01_"
    l_sub_str = len(sub_str)
    files = sorted(glob.glob(path_data + "ioppol_tro_mwr00_l1_tb_v01_*.nc"))


    if type(date_start) == type(""):
        # extract day, month and year from start date:
        date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start and file_date <= date_end:
                files_filtered.append(file)
    else:
        # extract day, month and year from date_start:
        date_list = [dt.datetime.strptime(ddd, "%Y-%m-%d") for ddd in date_start]

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date in date_list:
                files_filtered.append(file)


    # load data:
    DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested', preprocess=cut_vars)
    interesting_vars = ['time', 'flag', 'ta', 'pa', 'hur', 'tb', 'tb_bias_estimate', 'freq_sb', 'freq_shift',
                        'tb_absolute_accuracy', 'tb_cov']
    for vava in interesting_vars:
        if vava not in ['freq_sb', 'freq_shift', 'tb_absolute_accuracy', 'tb_cov']:
            mwr_dict[vava] = DS[vava].values.astype(np.float64)

    mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.
    mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
    DS.close()

    DS = xr.open_dataset(files_filtered[0], decode_times=False)
    mwr_dict['freq_sb'] = DS.freq_sb.values.astype(np.float32)
    mwr_dict['freq_shift'] = DS.freq_shift.values.astype(np.float32)
    mwr_dict['tb_absolute_accuracy'] = DS.tb_absolute_accuracy.values.astype(np.float32)
    mwr_dict['tb_cov'] = DS.tb_cov.values.astype(np.float32)

    DS.close()
    del DS

    return mwr_dict


def import_hatpro_level1c_daterange_pangaea(
    path_data,
    date_start,
    date_end=None):

    """
    Runs through all days between a start and an end date. It concats the level 1c TB time
    series of each day so that you'll have one dictionary, whose 'TB' will contain the TB
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Path of level 1 (brightness temperature, TB) data. This directory contains daily files
        as netCDF.
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str or None
        If date_start is str: Marks the last day of the desired period. To be specified in 
        yyyy-mm-dd (e.g. 2021-01-14)!
    """

    def cut_vars(DS):
        DS = DS.drop_vars(['freq_sb', 'freq_shift', 'tb_absolute_accuracy', 'tb_cov', 'ele'])
        return DS


    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # Identify files in the date range: First, load all into a list, then check which ones 
    # suit the daterange:
    mwr_dict = dict()
    sub_str = "_v01_"
    l_sub_str = len(sub_str)
    files = sorted(glob.glob(path_data + "ioppol_tro_mwrBL00_l1_tb_v01_*.nc"))


    if type(date_start) == type(""):
        # extract day, month and year from start date:
        date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start and file_date <= date_end:
                files_filtered.append(file)
    else:
        # extract day, month and year from date_start:
        date_list = [dt.datetime.strptime(ddd, "%Y-%m-%d") for ddd in date_start]

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date in date_list:
                files_filtered.append(file)


    # load data:
    DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested', preprocess=cut_vars)
    interesting_vars = ['time', 'flag', 'tb', 'freq_sb', 'ele']
    for vava in interesting_vars:
        if vava in DS.variables:
            mwr_dict[vava] = DS[vava].values.astype(np.float64)

    mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.
    mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
    DS.close()

    DS = xr.open_dataset(files_filtered[0], decode_times=False)
    mwr_dict['freq_sb'] = DS.freq_sb.values.astype(np.float32)
    mwr_dict['ele'] = DS.ele.values.astype(np.float32)

    DS.close()
    del DS

    return mwr_dict


def import_hatpro_level2a_daterange_pangaea(
    path_data,
    date_start,
    date_end=None,
    which_retrieval='both'):

    """
    Runs through all days between a start and an end date. It concats the level 2a data time
    series of each day so that you'll have one dictionary, whose e.g. 'IWV' will contain the IWV
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Path of level 2a data. 
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
        integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. 'both' will 
        load both. Default: 'both'
    """

    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")

    else:
        if which_retrieval not in ['prw', 'iwv', 'clwvi', 'lwp', 'both']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")

        else:
                if which_retrieval == 'iwv':
                    which_retrieval = ['prw']
                elif which_retrieval == 'lwp':
                    which_retrieval = ['clwvi']
                elif which_retrieval == 'both':
                    which_retrieval = ['prw', 'clwvi']
                else:
                    raise ValueError("Argument '" + which_retrieval + "' not recognised. Please use one of the following options: " +
                        "'iwv' or 'prw' will load the " +
                        "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                        "'both' will load both. Default: 'both'")
                    

    # Identify files in the date range: First, load all into a list, then check
    # which ones suit the daterange:
    mwr_dict = dict()
    sub_str = "_v01_"
    l_sub_str = len(sub_str)
    if 'prw' in which_retrieval:
        files = sorted(glob.glob(path_data + "ioppol_tro_mwr00_l2_prw_v01_*.nc"))

        if type(date_start) == type(""):
            # extract day, month and year from start date:
            date_start_dt = dt.datetime.strptime(date_start, "%Y-%m-%d")
            date_end_dt = dt.datetime.strptime(date_end, "%Y-%m-%d")

            # run through list: identify where date is written and check if within date range:
            files_filtered = list()
            for file in files:
                ww = file.find(sub_str) + l_sub_str
                if file.find(sub_str) == -1: continue
                file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
                if file_date >= date_start_dt and file_date <= date_end_dt:
                    files_filtered.append(file)
        else:
            # extract day, month and year from date_start:
            date_list = [dt.datetime.strptime(ddd, "%Y-%m-%d") for ddd in date_start]

            # run through list: identify where date is written and check if within date range:
            files_filtered = list()
            for file in files:
                ww = file.find(sub_str) + l_sub_str
                if file.find(sub_str) == -1: continue
                file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
                if file_date in date_list:
                    files_filtered.append(file)


        # laod data:
        DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested')
        interesting_vars = ['time', 'flag', 'lat', 'lon', 'zsl', 'prw', 'prw_offset', 'prw_err']
        for vava in interesting_vars: 
            if vava != 'prw_err':
                mwr_dict[vava] = DS[vava].values.astype(np.float64)
            else:
                mwr_dict[vava] = DS[vava][0,:].values.astype(np.float64)
        mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.
        mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
        DS.close()

    if 'clwvi' in which_retrieval:
        files = sorted(glob.glob(path_data + "ioppol_tro_mwr00_l2_clwvi_v01_*.nc"))

        if type(date_start) == type(""):
            # extract day, month and year from start date:
            date_start_dt = dt.datetime.strptime(date_start, "%Y-%m-%d")
            date_end_dt = dt.datetime.strptime(date_end, "%Y-%m-%d")

            # run through list: identify where date is written and check if within date range:
            files_filtered = list()
            for file in files:
                ww = file.find(sub_str) + l_sub_str
                if file.find(sub_str) == -1: continue
                file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
                if file_date >= date_start_dt and file_date <= date_end_dt:
                    files_filtered.append(file)
        else:
            # extract day, month and year from date_start:
            date_list = [dt.datetime.strptime(ddd, "%Y-%m-%d") for ddd in date_start]

            # run through list: identify where date is written and check if within date range:
            files_filtered = list()
            for file in files:
                ww = file.find(sub_str) + l_sub_str
                if file.find(sub_str) == -1: continue
                file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
                if file_date in date_list:
                    files_filtered.append(file)


        # load data:
        DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested')
        if mwr_dict:
            interesting_vars = ['flag', 'clwvi', 'clwvi_offset_zeroing', 'clwvi_err']
            for vava in interesting_vars:
                if vava != 'clwvi_err':
                    mwr_dict[vava] = DS[vava].values.astype(np.float64)
                else:
                    mwr_dict[vava] = DS[vava][0,:].values.astype(np.float64)
            mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.

        else:
            interesting_vars = ['time', 'flag', 'lat', 'lon', 'zsl', 'clwvi', 'clwvi_offset_zeroing', 'clwvi_err']
            for vava in interesting_vars:
                if vava != 'clwvi_err':
                    mwr_dict[vava] = DS[vava].values.astype(np.float64)
                else:
                    mwr_dict[vava] = DS[vava][0,:].values.astype(np.float64)
            mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.
            mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
        DS.close()

    return mwr_dict


def import_hatpro_level2b_daterange_pangaea(
    path_data,
    date_start,
    date_end=None,
    which_retrieval='both',
    around_radiosondes=True,
    path_radiosondes="",
    s_version='level_2',
    mwr_avg=0,
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the level 2b data time
    series of each day so that you'll have one dictionary, whose e.g. 'ta' will contain the
    temperature profile for the entire date range period with samples around the radiosonde
    launch times or alternatively 4 samples per day at fixed times: 05, 11, 17 and 23 UTC.

    Parameters:
    -----------
    path_data : str
        Base path of level 2b data. This directory contains subfolders representing the year, which,
        in turn, contain months, which contain day subfolders. Example path_data:
        "/data/obs/campaigns/mosaic/hatpro/l2/"
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case. The
        date list must be sorted in ascending order!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'ta' or 'hus' will load either the
        temperature or the specific humidity profile. 'both' will load both. Default: 'both'
    around_radiosondes : bool, optional
        If True, data will be limited to the time around radiosonde launches. If False, something else
        (e.g. around 4 times a day) might be done. Default: True
    path_radiosondes : str, optional
        Path to radiosonde data (Level 2). Default: ""
    s_version : str, optional
        Specifies the radiosonde version that is to be imported. Must be 'level_2' to work properly.
        Other versions have not been implemeted because they are considered to be inferior to level_2
        radiosondes.
    mwr_avg : int, optional
        If > 0, an average over mwr_avg seconds will be performed from sample_time to sample_time + 
        mwr_avg seconds. If == 0, no averaging will be performed.
    verbose : int, optional
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")


    if mwr_avg < 0:
        raise ValueError("mwr_avg must be an int >= 0.")
    elif type(mwr_avg) != type(1):
        raise TypeError("mwr_avg must be int.")

    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
        raise TypeError("Argument 'which_retrieval' must be a string. Options: 'ta' or 'hus' will load either the " +
            "temperature or the absolute humidity profile. 'both' will load both. Default: 'both'")

    elif which_retrieval not in ['ta', 'hus', 'both']:
        raise ValueError("Argument 'which_retrieval' must be one of the following options: 'ta' or 'hus' will load either the " +
            "temperature or the absolute humidity profile. 'both' will load both. Default: 'both'")

    else:
        which_retrieval_dict = {'ta': ['ta'],
                                'hus': ['hus'],
                                'both': ['ta', 'hus']}
        level2b_dataID_dict = {'ta': ['ta'],
                                'hus': ['hua'],
                                'both': ['ta', 'hua']}
        level2b_dataID = level2b_dataID_dict[which_retrieval]           # to find correct file names
        which_retrieval = which_retrieval_dict[which_retrieval]


    # extract day, month and year from start date:
    date_list = []
    if type(date_start) == type([]): 
        date_list = copy.deepcopy(date_start)
        date_start = date_start[0]
        date_list = [dt.datetime.strptime(dl, "%Y-%m-%d").date() for dl in date_list]
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1
    n_ret = 1           # inquired from level 2b data, number of available elevation angles in retrieval
    n_hgt = 43          # inquired from level 2b data, number of vertical retrieval levels (height levels)

    # basic variables that should always be imported:
    mwr_time_keys = ['time', 'flag', 'lat', 'lon', 'zsl']               # keys with time as coordinate
    mwr_height_keys = ['height']                            # keys with height as coordinate

    # Create an array that includes the radiosonde launch times:
    if around_radiosondes:
        if not path_radiosondes:
            raise ValueError("If 'around_radiosondes' is True, the path to the radiosonde level 2 data ('path_radiosondes') " +
                                "must be given.")

        if s_version != 'level_2':
            raise ValueError("Radiosonde version 's_version' must be 'level_2' if around_radiosondes is True because " +
                                "for this version, the launch time is directly read from the filename. This has not " +
                                "been implemeted for other radiosonde versions ('mossonde', 'psYYMMDDwHH') because these " +
                                "are considered to be inferior.")
        else:
            add_files = sorted(glob.glob(path_radiosondes + "*.nc"))        # filenames only; filter path
            add_files = [os.path.basename(a_f) for a_f in add_files]
            
            # identify launch time:
            n_samp = len(add_files)     # number of radiosondes
            launch_times = np.full((n_samp,), dt.datetime(1970,1,1))
            kk = 0
            if date_list:   # then only consider dates within date_list
                for a_f in add_files:
                    ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                    # only save those that are in the considered period
                    if ltt.date() in date_list:
                        launch_times[kk] = ltt
                        kk += 1
            else:
                for a_f in add_files:
                    ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                    # only save those that are in the considered period
                    if ltt >= date_start and ltt < (date_end + dt.timedelta(days=1)):
                        launch_times[kk] = ltt
                        kk += 1
            
            # truncate launch_times and convert to sec since 1970-01-01 00:00:00 UTC:
            launch_times = launch_times[:kk]
            sample_times = datetime_to_epochtime(launch_times)
            n_samp_tot = len(sample_times)

    else:
        # max number of samples: n_days*4
        sample_times = [5, 11, 17, 23]      # UTC on each day
        n_samp = len(sample_times)
        n_samp_tot = n_days*n_samp

    # mwr_master_dict (output) will contain all desired variables on specific axes:
    # e.g. level 2b has got a time axis (according to pl_mk_nds.pro) for flag,
    # azimuth, elevation angles and the data.
    mwr_master_dict = dict()

    # save import keys for each retrieval option in a dict:
    import_keys = dict()
    mwr_time_height_keys = []
    for l2b_ID in level2b_dataID: mwr_time_height_keys.append(l2b_ID)

    if 'ta' in which_retrieval:
        mwr_master_dict['ta_err'] = np.full((n_hgt, n_ret), np.nan)

        # define the keys that will be imported via import_hatpro_level2b:
        import_keys['ta'] = (mwr_time_keys + mwr_height_keys +
                        ['ta', 'ta_err'])

    if 'hus' in which_retrieval:
        # here, we can only import and concat absolute humidity (hua) because
        # the conversion requires temperature and pressure
        mwr_master_dict['hua_err'] = np.full((n_hgt, n_ret), np.nan)

        # define the keys that will be imported via import_hatpro_level2b:
        import_keys['hua'] = (mwr_time_keys + mwr_height_keys +
                        ['hua', 'hua_err'])

    for mthk in mwr_time_height_keys: mwr_master_dict[mthk] = np.full((n_samp_tot, n_hgt), np.nan)
    for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_samp_tot,), np.nan)
    for mhk in mwr_height_keys: mwr_master_dict[mhk] = np.full((n_hgt,), np.nan)


    # first list all available files and then reduce them to the specific date range and sampling:
    # list of v01 files:
    hatpro_level2_nc = sorted(glob.glob(path_data + "*_v01_*.nc"))
    if len(hatpro_level2_nc) == 0:
        if verbose >= 2:
            raise RuntimeError("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))


    # cycle through all years, all months and days:
    time_index = 0  # this index (for lvl 2b) will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly.
    sample_time_tolerance = 900     # sample time tolerance in seconds: mwr time must be within this
                                    # +/- tolerance of a sample_time to be accepted


    if not date_list:
        date_list = (date_start + dt.timedelta(days=n) for n in range(n_days))
    else:
        date_list = [dt.datetime(dl_i.year, dl_i.month, dl_i.day) for dl_i in date_list]
    for now_date in date_list:

        if verbose >= 1: print("Working on HATPRO Level 2b, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day
        now_date_str = now_date.strftime("%Y%m%d")


        # specify sample times as time: sec since 1970-01-01 00:00:00 UTC:
        if around_radiosondes:
            now_date_date = now_date.date()
            sample_mask = np.full((n_samp_tot,), False)
            for kk, l_t in enumerate(launch_times):
                sample_mask[kk] = l_t.date() == now_date_date

            sample_times_t = sample_times[sample_mask]

        else:
            sample_times_t = np.asarray([datetime_to_epochtime(dt.datetime(yyyy, mm, dd, st, 0, 0)) for st in sample_times])


        # identify level 2b files:
        # also save the dataID into the list to access the correct keys to be imported (import_keys)
        # later on.
        hatpro_level2b_nc = []
        for lvl2_nc in hatpro_level2_nc:
            for dataID in level2b_dataID:
                # must avoid including the boundary layer scan
                if (dataID + '_' in lvl2_nc) and ('BL00_' not in lvl2_nc) and (now_date_str in lvl2_nc):
                    hatpro_level2b_nc.append([lvl2_nc, dataID])

        if len(hatpro_level2b_nc) == 0: continue


        # load one retrieved variable after another from current day and save it into the mwr_master_dict
        for lvl2_nc in hatpro_level2b_nc:
            mwr_dict = import_hatpro_level2b(lvl2_nc[0], import_keys[lvl2_nc[1]])

            # it may occur that the whole day is flagged. If so, skip this file:
            if not np.any(mwr_dict['flag'] == 0):
                n_samp_real = 0
                continue

            # remove values where flag > 0:
            for mthk in mwr_time_height_keys:
                if mthk in lvl2_nc[1]:
                    mwr_dict[mthk] = mwr_dict[mthk][(mwr_dict['flag'] == 0) | (mwr_dict['flag'] == 32),:]
            for mtkab in mwr_time_keys:
                if mtkab != 'flag':
                    mwr_dict[mtkab] = mwr_dict[mtkab][(mwr_dict['flag'] == 0) | (mwr_dict['flag'] == 32)]
            mwr_dict['flag'] = mwr_dict['flag'][(mwr_dict['flag'] == 0) | (mwr_dict['flag'] == 32)]


            # find the time slice where the mwr time is closest to the sample_times.
            # The identified index must be within 15 minutes, otherwise it will be discarded
            # Furthermore, it needs to be respected, that the flag value must be 0 for that case.
            if mwr_avg == 0:
                sample_idx = []
                for st in sample_times_t:
                    idx = np.argmin(np.abs(mwr_dict['time'] - st))
                    if np.abs(mwr_dict['time'][idx] - st) < sample_time_tolerance:
                        sample_idx.append(idx)
                sample_idx = np.asarray(sample_idx)
                n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

            else:
                sample_idx = []
                for st in sample_times_t:
                    idx = np.where((mwr_dict['time'] >= st) & (mwr_dict['time'] <= st + mwr_avg))[0]
                    if len(idx) > 0:    # then an overlap has been found
                        sample_idx.append(idx)
                n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

            if n_samp_real == 0: continue

            # save to mwr_master_dict
            for mwr_key in mwr_dict.keys():
                mwr_key_shape = mwr_dict[mwr_key].shape

                if (mwr_key_shape == mwr_dict['time'].shape) and (mwr_key in mwr_time_keys):    # then the variable is on time axis:
                    if mwr_avg > 0:             # these values won't be averaged because they don't contain "data"
                        sample_idx_idx = [sii[0] for sii in sample_idx]
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real] = mwr_dict[mwr_key][sample_idx_idx]
                    
                    else:
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real] = mwr_dict[mwr_key][sample_idx]

                elif mwr_key == 'hua_err' or mwr_key == 'ta_err':   # these variables are n_hgt x n_ret arrays
                    mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

                elif mwr_key in mwr_height_keys:    # handled after the for loop
                    continue

                elif mwr_key in mwr_time_height_keys:
                    if mwr_avg > 0:
                        for k, sii in enumerate(sample_idx):
                            mwr_master_dict[mwr_key][time_index+k:time_index+k + 1,:] = np.nanmean(mwr_dict[mwr_key][sii,:], axis=0)
                    else:
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real,:] = mwr_dict[mwr_key][sample_idx,:]

                else:
                    raise RuntimeError("Something went wrong in the " +
                        "import_hatpro_level2b_daterange routine. Unexpected MWR variable dimension for " + mwr_key + ".")


        time_index = time_index + n_samp_real

    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:
        # save non height dependent variables to master dict:
        for mwr_key in mwr_height_keys: mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        time_height_shape_old = mwr_master_dict[mwr_time_height_keys[0]].shape
        for mwr_key in mwr_master_dict.keys():
            shape_new = mwr_master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]
            elif shape_new == time_height_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1, :]

    return mwr_master_dict


def import_hatpro_level2c_daterange_pangaea(
    path_data,
    date_start,
    date_end,
    which_retrieval='both',
    around_radiosondes=True,
    path_radiosondes="",
    s_version='level_2',
    mwr_avg=0,
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the level 2c data time
    series of each day so that you'll have one dictionary, whose e.g. 'ta' will contain the
    temperature profile for the entire date range period with samples around the radiosonde
    launch times or alternatively 4 samples per day at fixed times: 05, 11, 17 and 23 UTC.

    Parameters:
    -----------
    path_data : str
        Base path of level 2c data. This directory contains subfolders representing the year, which,
        in turn, contain months, which contain day subfolders. Example path_data:
        "/data/obs/campaigns/mosaic/hatpro/l2/"
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case. The
        date list must be sorted in ascending order!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'ta' will load the temperature 
        profile (boundary layer scan). 'both' will also load temperature only because humidity profile
        boundary layer scan does not exist. Default: 'both'
    around_radiosondes : bool, optional
        If True, data will be limited to the time around radiosonde launches. If False, something else
        (e.g. around 4 times a day) might be done. Default: True
    path_radiosondes : str, optional
        Path to radiosonde data (Level 2). Default: ""
    s_version : str
        Specifies the radiosonde version that is to be imported. Must be 'level_2' to work properly.
        Other versions have not been implemeted because they are considered to be inferior to level_2
        radiosondes.
    mwr_avg : int, optional
        If > 0, an average over mwr_avg seconds will be performed from sample_time - mwr_avg to 
        sample_time + mwr_avg seconds. If == 0, no averaging will be performed.
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")

    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
        raise TypeError("Argument 'which_retrieval' must be a string. Options: 'ta' will load the temperature " +
        "profile (boundary layer scan). 'both' will also load temperature only because humidity profile" +
        "boundary layer scan does not exist. Default: 'both'")

    elif which_retrieval not in ['ta', 'hus', 'both']:
        raise ValueError("Argument 'which_retrieval' must be one of the following options: 'ta' will load the temperature " +
        "profile (boundary layer scan). 'both' will also load temperature only because humidity profile" +
        "boundary layer scan does not exist. Default: 'both'")

    else:
        which_retrieval_dict = {'ta': ['ta'],
                                'both': ['ta']}
        level2c_dataID_dict = {'ta': ['ta'],
                                'both': ['ta']}
        level2c_dataID = level2c_dataID_dict[which_retrieval]
        which_retrieval = which_retrieval_dict[which_retrieval]

    if mwr_avg < 0:
        raise ValueError("mwr_avg must be an int >= 0.")
    elif type(mwr_avg) != type(1):
        raise TypeError("mwr_avg must be int.")

    # check if around_radiosondes is the right type:
    if not isinstance(around_radiosondes, bool):
        raise TypeError("Argument 'around_radiosondes' must be either True or False (boolean type).")

    # extract day, month and year from start date:
    date_list = []
    if type(date_start) == type([]): 
        date_list = copy.deepcopy(date_start)
        date_start = date_start[0]
        date_list = [dt.datetime.strptime(dl, "%Y-%m-%d").date() for dl in date_list]
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1
    n_ret = 1           # inquired from level 2c data, number of available elevation angles in retrieval
    n_hgt = 43          # inquired from level 2c data, number of vertical retrieval levels (height levels)

    # basic variables that should always be imported:
    mwr_time_keys = ['time', 'flag', 'lat', 'lon', 'zsl']               # keys with time as coordinate
    mwr_height_keys = ['height']                        # keys with height as coordinate

    # Create an array that includes the radiosonde launch times:
    if around_radiosondes:
        if not path_radiosondes:
            raise ValueError("If 'around_radiosondes' is True, the path to the radiosonde level 2 data ('pathradiosondes') " +
                                "must be given.")

        if s_version != 'level_2':
            raise ValueError("Radiosonde version 's_version' must be 'level_2' if around_radiosondes is True because " +
                                "for this version, the launch time is directly read from the filename. This has not " +
                                "been implemeted for other radiosonde versions ('mossonde', 'psYYMMDDwHH') because these " +
                                "are considered to be inferior.")
        else:
            add_files = sorted(glob.glob(path_radiosondes + "*.nc"))        # filenames only; filter path
            add_files = [os.path.basename(a_f) for a_f in add_files]
            
            # identify launch time:
            n_samp = len(add_files)     # number of radiosondes
            launch_times = np.full((n_samp,), dt.datetime(1970,1,1))
            kk = 0
            for a_f in add_files:
                ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                # only save those that are in the considered period
                if ltt >= date_start and ltt < (date_end + dt.timedelta(days=1)):
                    launch_times[kk] = ltt
                    kk += 1
            
            # truncate launch_times and convert to sec since 1970-01-01 00:00:00 UTC:
            launch_times = launch_times[:kk]
            sample_times = datetime_to_epochtime(launch_times)
            n_samp_tot = len(sample_times)

    else:
        # max number of samples: n_days*4
        sample_times = [5, 11, 17, 23]      # UTC on each day
        n_samp = len(sample_times)
        n_samp_tot = n_days*n_samp

    # mwr_master_dict (output) will contain all desired variables on specific axes:
    # e.g. level 2c has got a time axis (according to pl_mk_nds.pro) for flag,
    # and the data.
    mwr_master_dict = dict()

    # save import keys for each retrieval option in a dict:
    import_keys = dict()
    mwr_time_height_keys = []
    for l2b_ID in level2c_dataID: mwr_time_height_keys.append(l2b_ID)

    if 'ta' in which_retrieval:
        mwr_master_dict['ta_err'] = np.full((n_hgt,), np.nan)

        # define the keys that will be imported via import_hatpro_level2b:
        import_keys['ta'] = (mwr_time_keys + mwr_height_keys +
                        ['ta', 'ta_err'])

    for mthk in mwr_time_height_keys: mwr_master_dict[mthk] = np.full((n_samp_tot, n_hgt), np.nan)
    for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_samp_tot,), np.nan)
    for mhk in mwr_height_keys: mwr_master_dict[mhk] = np.full((n_hgt,), np.nan)


    # first list all available files and then reduce them to the specific date range and sampling:
    # list of v01 files:
    hatpro_level2_nc = sorted(glob.glob(path_data + "*_v01_*.nc"))
    if len(hatpro_level2_nc) == 0:
        if verbose >= 2:
            raise RuntimeError("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))


    # cycle through all years, all months and days:
    time_index = 0  # this index (for lvl 2c) will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly.
    sample_time_tolerance = 1800        # sample time tolerance in seconds: mwr time must be within this
                                        # +/- tolerance of a sample_time to be accepted


    if not date_list:
        date_list = (date_start + dt.timedelta(days=n) for n in range(n_days))
    else:
        date_list = [dt.datetime(dl_i.year, dl_i.month, dl_i.day) for dl_i in date_list]
    for now_date in date_list:

        if verbose >= 1: print("Working on HATPRO Level 2c, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day
        now_date_str = now_date.strftime("%Y%m%d")

        # specify sample times as time: sec since 1970-01-01 00:00:00 UTC:
        if around_radiosondes:
            now_date_date = now_date.date()
            sample_mask = np.full((n_samp_tot,), False)
            for kk, l_t in enumerate(launch_times):
                sample_mask[kk] = l_t.date() == now_date_date

            sample_times_t = sample_times[sample_mask]

        else:
            sample_times_t = np.asarray([datetime_to_epochtime(dt.datetime(yyyy, mm, dd, st, 0, 0)) for st in sample_times])


        # identify level 2c files:
        # also save the dataID into the list to access the correct keys to be imported (import_keys)
        # later on.
        hatpro_level2c_nc = []
        for lvl2_nc in hatpro_level2_nc:
            for dataID in level2c_dataID:
                # must include the boundary layer scan
                if (dataID + '_' in lvl2_nc) and ('BL00_' in lvl2_nc) and (now_date_str in lvl2_nc):
                    hatpro_level2c_nc.append([lvl2_nc, dataID])

        if len(hatpro_level2c_nc) == 0: continue


        # load one retrieved variable after another from current day and save it into the mwr_master_dict
        for lvl2_nc in hatpro_level2c_nc:
            mwr_dict = import_hatpro_level2c(lvl2_nc[0], import_keys[lvl2_nc[1]])

            # it may occur that the whole day is flagged. If so, skip this file:
            if not np.any(mwr_dict['flag'] == 0):
                n_samp_real = 0
                continue

            # remove values where flag > 0:
            for mthk in mwr_time_height_keys: mwr_dict[mthk] = mwr_dict[mthk][mwr_dict['flag'] == 0,:]
            for mtkab in mwr_time_keys:
                if mtkab != 'flag':
                    mwr_dict[mtkab] = mwr_dict[mtkab][mwr_dict['flag'] == 0]
            mwr_dict['flag'] = mwr_dict['flag'][mwr_dict['flag'] == 0]


            # # # update the flag by taking the manually detected outliers into account:
            # # # (not needed if v01 or later is used)
            # mwr_dict['flag'] = outliers_per_eye(mwr_dict['flag'], mwr_dict['time'], instrument='hatpro')


            # find the time slice where the mwr time is closest to the sample_times.
            # The identified index must be within 15 minutes, otherwise it will be discarded
            # Furthermore, it needs to be respected, that the flag value must be 0 for that case.
            if mwr_avg == 0:
                sample_idx = []
                for st in sample_times_t:
                    idx = np.argmin(np.abs(mwr_dict['time'] - st))
                    if np.abs(mwr_dict['time'][idx] - st) < sample_time_tolerance:
                        sample_idx.append(idx)
                sample_idx = np.asarray(sample_idx)
                n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

            else:
                sample_idx = []
                for st in sample_times_t:
                    idx = np.where((mwr_dict['time'] >= st - mwr_avg) & (mwr_dict['time'] <= st + mwr_avg))[0]
                    if len(idx) > 0:    # then an overlap has been found
                        sample_idx.append(idx)
                n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

            if n_samp_real == 0: continue


            # save to mwr_master_dict
            for mwr_key in mwr_dict.keys():
                mwr_key_shape = mwr_dict[mwr_key].shape

                if (mwr_key_shape == mwr_dict['time'].shape) and (mwr_key in mwr_time_keys):    # then the variable is on time axis:
                    if mwr_avg > 0:             # these values won't be averaged because they don't contain "data"
                        sample_idx_idx = [sii[0] for sii in sample_idx]
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real] = mwr_dict[mwr_key][sample_idx_idx]
                    
                    else:
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real] = mwr_dict[mwr_key][sample_idx]

                elif mwr_key == 'ta_err':   # these variables are n_hgt x n_ret arrays
                    mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

                elif mwr_key in mwr_height_keys: # handled after the for loop
                    continue

                elif mwr_key in mwr_time_height_keys:
                    if mwr_avg > 0:
                        for k, sii in enumerate(sample_idx):
                            mwr_master_dict[mwr_key][time_index+k:time_index+k + 1,:] = np.nanmean(mwr_dict[mwr_key][sii,:], axis=0)
                    else:
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real,:] = mwr_dict[mwr_key][sample_idx,:]

                else:
                    raise RuntimeError("Something went wrong in the " +
                        "import_hatpro_level2c_daterange routine. Unexpected MWR variable dimension for %s."%mwr_key)


        time_index = time_index + n_samp_real

    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:
        # save non time dependent variables in master dict
        for mwr_key in mwr_height_keys: mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        time_height_shape_old = mwr_master_dict[mwr_time_height_keys[0]].shape
        for mwr_key in mwr_master_dict.keys():
            shape_new = mwr_master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]
            elif shape_new == time_height_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1, :]

    return mwr_master_dict


def import_mirac_level1b_daterange_pangaea(
    path_data,
    date_start,
    date_end=None):

    """
    Runs through all days between a start and an end date. It concats the level 1b TB time
    series of each day so that you'll have one dictionary, whose 'TB' will contain the TB
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Path of level 1 (brightness temperature, TB) data. This directory contains daily files
        as netCDF.
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str or None
        If date_start is str: Marks the last day of the desired period. To be specified in 
        yyyy-mm-dd (e.g. 2021-01-14)!
    """

    def cut_vars(DS):
        DS = DS.drop_vars(['freq_sb', 'freq_shift', 'tb_absolute_accuracy', 'tb_cov'])
        return DS


    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # Identify files in the date range: First, load all into a list, then check which ones 
    # suit the daterange:
    mwr_dict = dict()
    sub_str = "_v01_"
    l_sub_str = len(sub_str)
    files = sorted(glob.glob(path_data + "MOSAiC_uoc_lhumpro-243-340_l1_tb_v01_*.nc"))


    # extract day, month and year from start date:
    if type(date_start) == type(""):
        date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start and file_date <= date_end:
                files_filtered.append(file)
    else:
        # extract day, month and year from date_start:
        date_list = [dt.datetime.strptime(ddd, "%Y-%m-%d") for ddd in date_start]

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date in date_list:
                files_filtered.append(file)


    # load data:
    DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested', preprocess=cut_vars)
    interesting_vars = ['time', 'flag', 'ta', 'pa', 'hur', 'tb', 'tb_bias_estimate', 'freq_sb', 'freq_shift',
                        'tb_absolute_accuracy', 'tb_cov']
    for vava in interesting_vars:
        if vava not in ['freq_sb', 'freq_shift', 'tb_absolute_accuracy', 'tb_cov']:
            mwr_dict[vava] = DS[vava].values.astype(np.float64)

    mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.
    mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
    DS = DS.close()

    DS = xr.open_dataset(files_filtered[0], decode_times=False)
    mwr_dict['freq_sb'] = DS.freq_sb.values.astype(np.float32)
    mwr_dict['freq_shift'] = DS.freq_shift.values.astype(np.float32)
    mwr_dict['tb_absolute_accuracy'] = DS.tb_absolute_accuracy.values.astype(np.float32)
    mwr_dict['tb_cov'] = DS.tb_cov.values.astype(np.float32)

    DS = DS.close()
    del DS

    return mwr_dict


def import_mirac_level2a_daterange_pangaea(
    path_data,
    date_start,
    date_end=None,
    which_retrieval='both'):

    """
    Runs through all days between a start and an end date. It concats the level 2a data time
    series of each day so that you'll have one dictionary, whose e.g. 'IWV' will contain the IWV
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Path of level 2a data. 
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
        integrated water vapour. 'both' will also load integrated water vapour only. Default: 'both'
    """

    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'both' will also load integrated water vapour only. Default: 'both'")

    else:
        if which_retrieval not in ['prw', 'iwv', 'both']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'both' will also load integrated water vapour only. Default: 'both'")

        else:
                if which_retrieval == 'iwv':
                    which_retrieval = ['prw']
                elif which_retrieval == 'both':
                    which_retrieval = ['prw']
                    

    # Identify files in the date range: First, load all into a list, then check
    # which ones suit the daterange:
    mwr_dict = dict()
    sub_str = "_v01_"
    l_sub_str = len(sub_str)
    files = sorted(glob.glob(path_data + "MOSAiC_uoc_lhumpro-243-340_l2_prw_v01_*.nc"))

    if type(date_start) == type(""):
        # extract day, month and year from start date:
        date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start and file_date <= date_end:
                files_filtered.append(file)
    else:
        # extract day, month and year from date_start:
        date_list = [dt.datetime.strptime(ddd, "%Y-%m-%d") for ddd in date_start]

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date in date_list:
                files_filtered.append(file)


    # laod data:
    DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested')
    interesting_vars = ['time', 'flag', 'lat', 'lon', 'zsl', 'prw']
    for vava in interesting_vars: mwr_dict[vava] = DS[vava].values.astype(np.float64)
    mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.
    mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
    DS.close()

    return mwr_dict


def import_single_PS122_mosaic_radiosonde_level2(
    filename,
    keys='all',
    height_grid=np.array([]),
    ip_type='lin',
    verbose=0):

    """
    Imports single level 2 radiosonde data created with PANGAEA_tab_to_nc.py 
    ('PS122_mosaic_radiosonde_level2_yyyymmdd_hhmmssZ.nc'). Converts to SI units
    and interpolates to a height grid with 5 m resolution from 0 to 15000 m. 

    Parameters:
    -----------
    filename : str
        Name (including path) of radiosonde data file.
    keys : list of str or str, optional
        This describes which variable(s) will be loaded. Specifying 'all' will import all variables.
        Specifying 'basic' will load the variables the author consideres most useful for his current
        analysis.
        Default: 'all'
    height_grid : array of floats or None
        If not None, height_grid contains a 1D array of floats indicating the new height grid to 
        which the radiosonde data is interpolated to.
    ip_type : str
        String indicating the interpolation type. Option: 'lin' for linear interpolation using 
        np.interp; 'avg' for using interp_w_avg, which averages data over layers centered on the
        height_grid.
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    """
        Loaded values are imported in the following units:
        T: in deg C, will be converted to K
        P: in hPa, will be converted to Pa
        RH: in %, will be converted to [0-1]
        Altitude: in m
        q: in kg kg^-1 (water vapor specific humidity)
        time: in sec since 1970-01-01 00:00:00 UTC
    """

    file_nc = nc.Dataset(filename)

    if (not isinstance(keys, str)) and (not isinstance(keys, list)):
        raise TypeError("Argument 'key' must be a list of strings or 'all'.")

    if keys == 'all':
        keys = file_nc.variables.keys()
    elif keys == 'basic':
        keys = ['time', 'T', 'P', 'RH', 'q', 'Altitude']

    sonde_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in radiosonde file." % key)

        sonde_dict[key] = np.asarray(file_nc.variables[key])
        if key != "IWV" and len(sonde_dict[key]) == 0: # 'and': second condition only evaluated if first condition True
            return None

        if key in ['Latitude', 'Longitude']:    # only interested in the first lat, lon position
            sonde_dict[key] = sonde_dict[key][0]
        if key == 'IWV':
            sonde_dict[key] = np.float64(sonde_dict[key])

    # convert units:
    if 'RH' in keys:    # from percent to [0, 1]
        sonde_dict['RH'] = sonde_dict['RH']*0.01
    if 'T' in keys:     # from deg C to K
        sonde_dict['T'] = sonde_dict['T'] + 273.15
    if 'P' in keys:     # from hPa to Pa
        sonde_dict['P'] = sonde_dict['P']*100
    if 'time' in keys:  # from int64 to float64
        sonde_dict['time'] = np.float64(sonde_dict['time'])
        sonde_dict['launch_time'] = sonde_dict['time'][0]

    # interpolate to new height grid:
    if len(height_grid) == 0:
        height_grid = np.arange(0,15001,5)      # in m
    keys = [*keys]      # converts dict_keys to a list
    for key in keys:
        if sonde_dict[key].shape == sonde_dict['time'].shape:
            if key not in ['time', 'Latitude', 'Longitude', 'ETIM', 'Altitude']:
                sonde_dict[key + "_ip"] = np.interp(height_grid, sonde_dict['Altitude'], sonde_dict[key], right=np.nan)
            elif key == 'Altitude':
                sonde_dict[key + "_ip"] = height_grid


    # Renaming variables: ['Lat', 'Lon', 'p', 'T', 'RH', 'GeopHgt', 'qv', 'time', ...]
    renaming = {'T': 'temp',    'P': 'pres',    'RH': 'rh',
                'Altitude': 'height', 'h_geom': 'height_geom',
                'Latitude': 'lat',  'Longitude': 'lon',
                'T_ip': 'temp_ip', 'P_ip': 'pres_ip', 'RH_ip': 'rh_ip',
                'Altitude_ip': 'height_ip', 'h_geom_ip': 'height_geom_ip',
                'IWV': 'iwv'}
    for ren_key in renaming.keys():
        if ren_key in sonde_dict.keys():
            sonde_dict[renaming[ren_key]] = sonde_dict[ren_key]

    # height check: how high does the data reach:
    sonde_dict['height_check'] = sonde_dict['height'][-1]

    return sonde_dict


def import_single_PS122_mosaic_radiosonde_level3(
    filename,
    keys='all',
    height_grid=np.array([]),
    ip_type='lin',
    verbose=0):

    """
    Imports single level 3 radiosonde data (merged with MetTower) created with PANGAEA_tab_to_nc.py
    ('PS122_mosaic_radiosonde_level3_yyyymmdd_hhmmssZ.nc'). Converts to SI units and interpolates 
    to a height grid with 5 m resolution from 0 to 15000 m. 

    Parameters:
    -----------
    filename : str
        Name (including path) of radiosonde data file.
    keys : list of str or str, optional
        This describes which variable(s) will be loaded. Specifying 'all' will import all variables.
        Specifying 'basic' will load the variables the author consideres most useful for his current
        analysis.
        Default: 'all'
    height_grid : array of floats or None
        If not None, height_grid contains a 1D array of floats indicating the new height grid to 
        which the radiosonde data is interpolated to.
    ip_type : str
        String indicating the interpolation type. Option: 'lin' for linear interpolation using 
        np.interp; 'avg' for using interp_w_avg, which averages data over layers centered on the
        height_grid.
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    """
        Loaded values are imported in the following units:
        T: in deg C, will be converted to K
        P: in hPa, will be converted to Pa
        RH: in %, will be converted to [0-1]
        Altitude: in m
        q: in kg kg^-1 (water vapor specific humidity)
        time: in sec since 1970-01-01 00:00:00 UTC
    """

    DS = xr.open_dataset(filename)

    if (not isinstance(keys, str)) and (not isinstance(keys, list)):
        raise TypeError("Argument 'key' must be a list of strings or 'all'.")

    if keys == 'all':
        keys = [*DS.variables]
    elif keys == 'basic':
        keys = ['time', 'T', 'P', 'RH', 'q', 'Altitude']

    sonde_dict = dict()
    for key in keys:
        if not key in DS.variables:
            raise KeyError(f"I have no memory of this key: {key}. Key not found in radiosonde file.")

        sonde_dict[key] = DS[key].values
        if key != "IWV" and len(sonde_dict[key]) == 0: # 'and': second condition only evaluated if first condition True
            return None

        if key in ['Latitude', 'Longitude']:    # only interested in the first lat, lon position
            sonde_dict[key] = sonde_dict[key][np.where(~np.isnan(sonde_dict[key]))[0][0]]
        if key == 'IWV':
            sonde_dict[key] = np.float64(sonde_dict[key])

    # convert units:
    if 'T' in keys:     # from deg C to K
        sonde_dict['T'] = sonde_dict['T'] + 273.15
    if 'RH' in keys:    # from percent to [0, 1]
        sonde_dict['RH'] = sonde_dict['RH']*0.01
    if 'P' in keys:     # from hPa to Pa
        sonde_dict['P'] = sonde_dict['P']*100
    if 'time' in keys:  # from int64 to float64
        sonde_dict['time'] = sonde_dict['time'].astype('datetime64[s]').astype('float64')
        sonde_dict['launch_time'] = np.datetime64(DS.launch_time).astype("datetime64[s]").astype('float64')


    # interpolate to new height grid:
    if len(height_grid) == 0:
        height_grid = np.arange(0,15001,5)      # in m
    keys = [*keys]      # converts dict_keys to a list
    if ip_type == 'lin':
        for key in keys:
            if sonde_dict[key].shape == sonde_dict['time'].shape:
                if key not in ['time', 'Latitude', 'Longitude', 'Altitude']:
                    sonde_dict[key + "_ip"] = np.interp(height_grid, sonde_dict['Altitude'], sonde_dict[key], right=np.nan)
                elif key == 'Altitude':
                    sonde_dict[key + "_ip"] = height_grid
    elif ip_type == 'avg':
        for key in keys:
            if sonde_dict[key].shape == sonde_dict['time'].shape:
                if key not in ['time', 'Latitude', 'Longitude', 'Altitude']:
                    sonde_dict[key + "_ip"] = interp_w_avg(sonde_dict['Altitude'], sonde_dict[key], height_grid)
                elif key == 'Altitude':
                    sonde_dict[key + "_ip"] = height_grid


    # Renaming variables: ['Lat', 'Lon', 'p', 'T', 'RH', 'GeopHgt', 'qv', 'time', ...]
    renaming = {'T': 'temp', 'P': 'pres', 'RH': 'rh',
                'Altitude': 'height',
                'Latitude': 'lat',  'Longitude': 'lon',
                'T_ip': 'temp_ip', 'P_ip': 'pres_ip', 'RH_ip': 'rh_ip',
                'Altitude_ip': 'height_ip',
                'IWV': 'iwv'}
    for ren_key in renaming.keys():
        if ren_key in sonde_dict.keys():
            sonde_dict[renaming[ren_key]] = sonde_dict[ren_key]

    # height check: how high does the data reach:
    sonde_dict['height_check'] = sonde_dict['height'][-1]

    DS = DS.close()

    return sonde_dict


def import_single_NYA_RS_radiosonde(
    filename,
    keys='all',
    height_grid=np.array([]),
    verbose=0):

    """
    Imports single NYA-RS radiosonde data for Ny Alesund. Converts to SI units
    and interpolates to a height grid with 5 m resolution from 0 to 15000 m. 

    Parameters:
    -----------
    filename : str
        Name (including path) of radiosonde data file.
    keys : list of str or str, optional
        This describes which variable(s) will be loaded. Specifying 'all' will import all variables.
        Specifying 'basic' will load the variables the author consideres most useful for his current
        analysis.
        Default: 'all'
    height_grid : array of floats or None
        If not None, height_grid contains a 1D array of floats indicating the new height grid to 
        which the radiosonde data is interpolated to.
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    """
        Loaded values are imported in the following units:
        T: in K
        P: in hPa, will be converted to Pa
        RH: in [0-1]
        Altitude: in m
        time: will be converted to sec since 1970-01-01 00:00:00 UTC
    """

    file_nc = nc.Dataset(filename)

    if (not isinstance(keys, str)) and (not isinstance(keys, list)):
        raise TypeError("Argument 'key' must be a list of strings or 'all'.")

    if keys == 'all':
        keys = file_nc.variables.keys()
    elif keys == 'basic':
        keys = ['time', 'temp', 'press', 'rh', 'alt']

    sonde_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in radiosonde file." % key)

        sonde_dict[key] = np.asarray(file_nc.variables[key])
        if key != "IWV" and len(sonde_dict[key]) == 0: # 'and': second condition only evaluated if first condition True
            return None

        if key in ['lat', 'lon']:   # only interested in the first lat, lon position
            sonde_dict[key] = sonde_dict[key][0]

    # convert units:
    if 'P' in keys:     # from hPa to Pa
        sonde_dict['P'] = sonde_dict['P']*100
    if 'time' in keys:  # from int64 to float64
        time_unit = file_nc.variables['time'].units
        time_offset = (dt.datetime.strptime(time_unit[-19:], "%Y-%m-%dT%H:%M:%S") - dt.datetime(1970,1,1)).total_seconds()
        sonde_dict['time'] = np.float64(sonde_dict['time']) + time_offset
        sonde_dict['launch_time'] = sonde_dict['time'][0]

    # interpolate to new height grid:
    if len(height_grid) == 0:
        height_grid = np.arange(0,15001,5)      # in m
    keys = [*keys]      # converts dict_keys to a list
    for key in keys:
        if sonde_dict[key].shape == sonde_dict['time'].shape:
            if key not in ['time', 'lat', 'lon', 'alt']:
                sonde_dict[key + "_ip"] = np.interp(height_grid, sonde_dict['alt'], sonde_dict[key])
            elif key == 'alt':
                sonde_dict[key + "_ip"] = height_grid


    # Renaming variables to a standard convention
    renaming = {'press': 'pres', 'alt': 'height', 'press_ip': 'pres_ip', 'alt_ip': 'height_ip'}
    for ren_key in renaming.keys():
        if ren_key in sonde_dict.keys():
            sonde_dict[renaming[ren_key]] = sonde_dict[ren_key]

    return sonde_dict


def import_radiosonde_daterange(
    path_data,
    date_start,
    date_end,
    s_version='level_2',
    with_wind=False,
    remove_failed=False,
    extend_height_grid=False,
    height_grid=np.array([]),
    ip_type='lin',
    verbose=0):

    """
    Imports radiosonde data (several versions supported, see below) and concatenates the files 
    into time series x height. E.g. temperature profile will have the dimension: n_sondes x n_height

    Parameters:
    -----------
    path_data : str
        Path of radiosonde data.
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    s_version : str, optional
        Specifies the radiosonde version that is to be imported. Possible options: 'mossonde',
        'psYYMMDDwHH', 'level_2', 'nya-rs', 'GRUAN', 'level_3'. Default: 'level_2' (published by 
        Marion Maturilli)
    with_wind : bool, optional
        This describes if wind measurements are included (True) or not (False). Does not work with
        s_version='psYYMMDDwHH'. Default: False
    remove_failed : bool, optional
        If True, failed sondes with unrealistic IWV values will be removed (currently only implmented
        for s_version in ['level_2', 'GRUAN', 'level_3']). It also includes "height_check" to avoid 
        sondes that burst before reaching > 10000 m.
    extend_height_grid : bool
        If True, the new height grid, to which the radiosonde data is interpolated to is 0, 10, 20, ...
        25000 m. If False, it's 0, 5, 10, 15, ..., 15000 m.
    height_grid : array of floats or None
        If not None, height_grid contains a 1D array of floats indicating the new height grid to 
        which the radiosonde data is interpolated to.
    ip_type : str
        String indicating the interpolation type. Option: 'lin' for linear interpolation using 
        np.interp; 'avg' for using interp_w_avg, which averages data over layers centered on the
        height_grid. 'avg' is only available for s_version in ['level_2', 'level_3', 'GRUAN'].
    verbose : int, optional
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    def time_prematurely_bursted_sondes():

        """
        This little function merely returns time stamps of MOSAiC radiosondes, whose
        burst altitude was <= 10000 m. (Or other errors occurred.)
        """

        failed_sondes_dt = np.array([dt.datetime(2019, 10, 7, 11, 0),
                            dt.datetime(2019, 10, 15, 23, 0),
                            dt.datetime(2019, 11, 4, 11, 0),
                            dt.datetime(2019, 11, 17, 17, 0),
                            dt.datetime(2019, 12, 17, 5, 0),
                            dt.datetime(2019, 12, 24, 11, 0),
                            dt.datetime(2020, 1, 13, 11, 0),
                            dt.datetime(2020, 2, 1, 11, 0),
                            dt.datetime(2020, 2, 6, 23, 0),
                            dt.datetime(2020, 3, 9, 23, 0),
                            dt.datetime(2020, 3, 9, 11, 0), # unrealistic temperature and humidity values at the surface
                            dt.datetime(2020, 3, 11, 17, 0),
                            dt.datetime(2020, 3, 29, 5, 0),
                            dt.datetime(2020, 5, 14, 17, 0),
                            dt.datetime(2020, 6, 14, 17, 0),
                            dt.datetime(2020, 6, 19, 11, 0),
                            dt.datetime(2020, 9, 27, 9, 0)])

        reftime = dt.datetime(1970,1,1)
        failed_sondes_t = np.asarray([datetime_to_epochtime(fst) for fst in failed_sondes_dt])
        failed_sondes_t = np.asarray([(fst - reftime).total_seconds() for fst in failed_sondes_dt])
        
        return failed_sondes_t, failed_sondes_dt

    if not isinstance(s_version, str): raise TypeError("s_version in import_radiosonde_daterange must be a string.")

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    if s_version == 'mossonde':
        all_radiosondes_nc = sorted(glob.glob(path_data + "mossonde-curM1" + "*.nc"))

        # inquire the number of radiosonde files (date and time of launch is in filename):
        # And fill a list which will include the relevant radiosonde files.
        radiosondes_nc = []
        for rs_nc in all_radiosondes_nc:
            rs_date = rs_nc[-16:-8]     # date of radiosonde from filename
            yyyy = int(rs_date[:4])
            mm = int(rs_date[4:6])
            dd = int(rs_date[6:])
            rs_date_dt = dt.datetime(yyyy,mm,dd)
            if rs_date_dt >= date_start and rs_date_dt <= date_end:
                radiosondes_nc.append(rs_nc)

    elif s_version == 'psYYMMDDwHH':
        all_radiosondes_nc = sorted(glob.glob(path_data + "ps*.w*.nc"))[:-1]    # exclude last file because it's something about Ozone

        # inquire the number of radiosonde files (date and time of launch is in filename):
        # And fill a list which will include the relevant radiosonde files.
        radiosondes_nc = []
        for rs_nc in all_radiosondes_nc:
            rs_date = rs_nc[-13:-3]     # date of radiosonde from filename
            yyyy = 2000 + int(rs_date[:2])
            mm = int(rs_date[2:4])
            dd = int(rs_date[4:6])
            rs_date_dt = dt.datetime(yyyy,mm,dd)
            if rs_date_dt >= date_start and rs_date_dt <= date_end:
                radiosondes_nc.append(rs_nc)

    elif s_version == 'nya-rs':
        all_radiosondes_nc = sorted(glob.glob(path_data + "NYA-RS_*.nc"))

        # inquire the number of radiosonde files (date and time of launch is in filename):
        # And fill a list which will include the relevant radiosonde files.
        radiosondes_nc = []
        for rs_nc in all_radiosondes_nc:
            rs_date = rs_nc[-15:-3]     # date of radiosonde from filename
            yyyy = int(rs_date[:4])
            mm = int(rs_date[4:6])
            dd = int(rs_date[6:8])
            rs_date_dt = dt.datetime(yyyy,mm,dd)
            if rs_date_dt >= date_start and rs_date_dt <= date_end:
                radiosondes_nc.append(rs_nc)

    elif s_version == 'level_2':
        all_radiosondes_nc = sorted(glob.glob(path_data + "PS122_mosaic_radiosonde_level2*.nc"))

        # inquire the number of radiosonde files (date and time of launch is in filename):
        # And fill a list which will include the relevant radiosonde files.
        radiosondes_nc = []
        for rs_nc in all_radiosondes_nc:
            rs_date = rs_nc[-19:-3]     # date of radiosonde from filename
            yyyy = int(rs_date[:4])
            mm = int(rs_date[4:6])
            dd = int(rs_date[6:8])
            rs_date_dt = dt.datetime(yyyy,mm,dd)
            if rs_date_dt >= date_start and rs_date_dt <= date_end:
                radiosondes_nc.append(rs_nc)

    elif s_version == 'GRUAN':
        all_radiosondes_nc = sorted(glob.glob(path_data + "PS122_mosaic_radiosonde_level3_GRUAN*.nc"))

        # inquire the number of radiosonde files (date and time of launch is in filename):
        # And fill a list which will include the relevant radiosonde files.
        radiosondes_nc = []
        for rs_nc in all_radiosondes_nc:
            rs_date = rs_nc[-19:-3]     # date of radiosonde from filename
            yyyy = int(rs_date[:4])
            mm = int(rs_date[4:6])
            dd = int(rs_date[6:8])
            rs_date_dt = dt.datetime(yyyy,mm,dd)
            if rs_date_dt >= date_start and rs_date_dt <= date_end:
                radiosondes_nc.append(rs_nc)

    elif s_version == 'level_3':
        all_radiosondes_nc = sorted(glob.glob(path_data + "PS122_mosaic_radiosonde_level3*.nc"))

        # inquire the number of radiosonde files (date and time of launch is in filename):
        # And fill a list which will include the relevant radiosonde files.
        radiosondes_nc = []
        for rs_nc in all_radiosondes_nc:
            rs_date = rs_nc[-19:-3]     # date of radiosonde from filename
            yyyy = int(rs_date[:4])
            mm = int(rs_date[4:6])
            dd = int(rs_date[6:8])
            rs_date_dt = dt.datetime(yyyy,mm,dd)
            if rs_date_dt >= date_start and rs_date_dt <= date_end:
                radiosondes_nc.append(rs_nc)


    # number of sondes:
    n_sondes = len(radiosondes_nc)

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days

    # basic variables that should always be imported:
    if s_version == 'mossonde':
        geoinfo_keys = ['lat', 'lon', 'alt', 'launch_time']
        time_height_keys = ['pres', 'temp', 'rh', 'height', 'rho_v', 'q']       # keys with time and height as coordinate
        if with_wind: time_height_keys = time_height_keys + ['wspeed', 'wdir']
    elif s_version == 'psYYMMDDwHH':
        geoinfo_keys = ['lat', 'lon', 'launch_time']
        time_height_keys = ['pres', 'temp', 'rh', 'height', 'rho_v', 'q']
        if with_wind:
            print("No direct wind calculation for s_version='%s'."%s_version)
    elif s_version == 'nya-rs':
        geoinfo_keys = ['lat', 'lon', 'launch_time']
        time_height_keys = ['pres', 'temp', 'rh', 'height']
        if with_wind: time_height_keys = time_height_keys + ['wspeed', 'wdir']
    elif s_version in ['level_2', 'GRUAN', 'level_3']:
        geoinfo_keys = ['lat', 'lon', 'launch_time', 'iwv']
        time_height_keys = ['pres', 'temp', 'rh', 'height', 'rho_v', 'q']
        if with_wind: time_height_keys = time_height_keys + ['wspeed', 'wdir']
    else:
        raise ValueError("s_version in import_radiosonde_daterange must be 'mossonde', 'psYYMMDDwHH', 'nya-rs', " +
                            "'level_2', 'GRUAN' or 'level_3'.")
    all_keys = geoinfo_keys + time_height_keys


    # sonde_master_dict (output) will contain all desired variables on specific axes:
    # Time axis (one sonde = 1 timestamp) = axis 0; height axis = axis 1
    if len(height_grid) == 0:
        if extend_height_grid:
            new_height_grid = np.arange(0,25001,10)
        else:
            new_height_grid = np.arange(0,15001,5)
    else:
        new_height_grid = height_grid
    n_height = len(new_height_grid) # length of the interpolated height grid
    sonde_master_dict = dict()
    for gk in geoinfo_keys: sonde_master_dict[gk] = np.full((n_sondes,), np.nan)
    for thk in time_height_keys: sonde_master_dict[thk] = np.full((n_sondes, n_height), np.nan)


    if s_version == 'mossonde':
        all_keys_import = geoinfo_keys + time_height_keys + ['time', 'geopheight']  # 'time' required to create 'launch_time'
        all_keys_import.remove('launch_time')       # because this key is not saved in the radiosonde files
        all_keys_import.remove('rho_v')             # because this key is not saved in the radiosonde files
        all_keys_import.remove('q')                 # because this key is not saved in the radiosonde files
        all_keys_import.remove('height')                    # because this key is not saved in the radiosonde files
        if with_wind: all_keys_import = all_keys_import + ['wspeed', 'wdir']

        # cycle through all relevant sonde files:
        for rs_idx, rs_nc in enumerate(radiosondes_nc):

            if verbose >= 1:
                # rs_date = rs_nc[-16:-8]
                # print("Working on Radiosonde, ", 
                    # dt.datetime(int(rs_date[:4]), int(rs_date[4:6]), int(rs_date[6:])))
                print("Working on Radiosonde, " + rs_nc)

            sonde_dict = import_single_mossonde_curM1(rs_nc, keys=all_keys_import)

            # save to sonde_master_dict:
            for key in all_keys:
                if key in geoinfo_keys:
                    sonde_master_dict[key][rs_idx] = sonde_dict[key]

                elif key in time_height_keys:
                    sonde_master_dict[key][rs_idx, :] = sonde_dict[key + "_ip"]     # must use the interpolated versions!

                else:
                    raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with import_single_mossonde_curM1")

    elif s_version == 'psYYMMDDwHH':
        all_keys_import = ['Lat', 'Lon', 'p', 'T', 'RH', 'GeopHgt', 'qv', 'time']   # 'time' required to create 'launch_time'


        # cycle through all relevant sonde files:
        for rs_idx, rs_nc in enumerate(radiosondes_nc):

            if verbose >= 1: 
                # rs_date = rs_nc[-16:-8]
                print("Working on Radiosonde, " + rs_nc)

            sonde_dict = import_single_psYYMMDD_wHH_sonde(rs_nc, keys=all_keys_import)
            if not sonde_dict:  # then the imported sonde file appears to be empty
                continue

            else:
                # save to sonde_master_dict:
                for key in all_keys:
                    if key in geoinfo_keys:
                        sonde_master_dict[key][rs_idx] = sonde_dict[key]

                    elif key in time_height_keys:
                        sonde_master_dict[key][rs_idx, :] = sonde_dict[key + "_ip"]     # must use the interpolated versions!

                    else:
                        raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with import_single_mossonde_curM1")

        # As there are empty files among the current psYYMMDD.wHH sondes, they have to be filtered out:
        not_corrupted_sondes = ~np.isnan(sonde_master_dict['launch_time'])
        # not_corrupted_sondes_idx = np.where(~np.isnan(sonde_master_dict['launch_time']))[0]
        for key in sonde_master_dict.keys():
            if key in geoinfo_keys:
                sonde_master_dict[key] = sonde_master_dict[key][not_corrupted_sondes]
            else:
                sonde_master_dict[key] = sonde_master_dict[key][not_corrupted_sondes,:]

    elif s_version == 'nya-rs':
        all_keys_import = ['lat', 'lon', 'press', 'temp', 'rh', 'alt', 'time']
        if with_wind: all_keys_import = all_keys_import + ['wdir', 'wspeed']


        # cycle through all relevant sonde files:
        for rs_idx, rs_nc in enumerate(radiosondes_nc):
            
            if verbose >= 1:
                # rs_date = rs_nc[-19:-3]
                print("\rWorking on Radiosonde, " + rs_nc, end="")

            sonde_dict = import_single_NYA_RS_radiosonde(rs_nc, keys=all_keys_import)
            
            # save to sonde_master_dict:
            for key in all_keys:
                if key in geoinfo_keys:
                    sonde_master_dict[key][rs_idx] = sonde_dict[key]

                elif key in time_height_keys:
                    sonde_master_dict[key][rs_idx, :] = sonde_dict[key + "_ip"]     # must use the interpolated versions!

                else:
                    raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with " +
                                    "import_single_NYA_RS_radiosonde")

    elif s_version == 'level_2':
        all_keys_import = ['Latitude', 'Longitude', 'P', 'T', 'RH', 'Altitude', 'rho_v', 'q', 'time', 'IWV']
        if with_wind: all_keys_import = all_keys_import + ['wdir', 'wspeed']

        if remove_failed:
            failed_sondes_t, failed_sondes_dt = time_prematurely_bursted_sondes()       # load times of failed sondes


        # cycle through all relevant sonde files:
        rs_idx = 0
        for rs_nc in radiosondes_nc:
            
            if verbose >= 1:
                # rs_date = rs_nc[-19:-3]
                print("Working on Radiosonde, " + rs_nc, end='\r')

            sonde_dict = import_single_PS122_mosaic_radiosonde_level2(rs_nc, keys=all_keys_import, height_grid=new_height_grid,
                                                                        ip_type=ip_type)
            if (remove_failed and ((sonde_dict['iwv'] == 0.0) or (np.isnan(sonde_dict['iwv'])) or
                (sonde_dict['height_check'] < 10000) or (np.any(np.abs(sonde_dict['launch_time'] - failed_sondes_t) < 7200)))):
                continue
            
            # save to sonde_master_dict:
            for key in all_keys:
                if key in geoinfo_keys:
                    sonde_master_dict[key][rs_idx] = sonde_dict[key]

                elif key in time_height_keys:
                    sonde_master_dict[key][rs_idx, :] = sonde_dict[key + "_ip"]     # must use the interpolated versions!

                else:
                    raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with " +
                                    "import_single_PS122_mosaic_radiosonde_level2")

            rs_idx += 1

        # Truncate number of sondes:
        if remove_failed and (rs_idx < n_sondes):
            for key in geoinfo_keys: sonde_master_dict[key] = sonde_master_dict[key][:rs_idx]
            for key in time_height_keys: sonde_master_dict[key] = sonde_master_dict[key][:rs_idx,:]

    elif s_version == 'GRUAN':
        all_keys_import = ['Latitude', 'Longitude', 'P', 'T', 'RH', 'h_geop', 'q', 'rho_v', 'time', 'IWV']
        if with_wind: all_keys_import = all_keys_import + ['wdir', 'wspeed']

        if remove_failed:
            failed_sondes_t, failed_sondes_dt = time_prematurely_bursted_sondes()       # load times of failed sondes


        # cycle through all relevant sonde files:
        rs_idx = 0
        for rs_nc in radiosondes_nc:
            
            if verbose >= 1:
                # rs_date = rs_nc[-19:-3]
                print("Working on Radiosonde, " + rs_nc, end='\r')

            sonde_dict = import_single_PS122_mosaic_radiosonde_GRUAN(rs_nc, keys=all_keys_import, height_grid=new_height_grid,
                                                                    ip_type=ip_type)
            if (remove_failed and ((sonde_dict['iwv'] == 0.0) or (np.isnan(sonde_dict['iwv'])) or
                (sonde_dict['height_check'] < 10000) or (np.any(np.abs(sonde_dict['launch_time'] - failed_sondes_t) < 7200)))):
                continue
            
            # save to sonde_master_dict:
            for key in all_keys:
                if key in geoinfo_keys:
                    sonde_master_dict[key][rs_idx] = sonde_dict[key]

                elif key in time_height_keys:
                    sonde_master_dict[key][rs_idx,:] = sonde_dict[key + "_ip"]      # must use the interpolated versions!

                else:
                    raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with " +
                                    "import_single_PS122_mosaic_radiosonde_GRUAN")

            rs_idx += 1     # only counts non-failed sondes


        # Truncate number of sondes:
        if remove_failed and (rs_idx < n_sondes):
            for key in geoinfo_keys: sonde_master_dict[key] = sonde_master_dict[key][:rs_idx]
            for key in time_height_keys: sonde_master_dict[key] = sonde_master_dict[key][:rs_idx,:]

    elif s_version == 'level_3':
        all_keys_import = ['Latitude', 'Longitude', 'P', 'T', 'RH', 'Altitude', 'rho_v', 'q', 'time', 'IWV']
        if with_wind: all_keys_import += ['wdir', 'wspeed']

        if remove_failed:
            failed_sondes_t, failed_sondes_dt = time_prematurely_bursted_sondes()       # load times of failed sondes


        # cycle through all relevant sonde files:
        rs_idx = 0
        for rs_nc in radiosondes_nc:
            
            if verbose >= 1:
                # rs_date = rs_nc[-19:-3]
                print("Working on Radiosonde, " + rs_nc, end='\r')

            sonde_dict = import_single_PS122_mosaic_radiosonde_level3(rs_nc, keys=all_keys_import, height_grid=new_height_grid,
                                                                    ip_type=ip_type)

            if (remove_failed and ((sonde_dict['iwv'] == 0.0) or (np.isnan(sonde_dict['iwv'])) or
                (sonde_dict['height_check'] < 10000) or (np.any(np.abs(sonde_dict['launch_time'] - failed_sondes_t) < 7200)))):
                continue
            
            # save to sonde_master_dict:
            for key in all_keys:
                if key in geoinfo_keys:
                    sonde_master_dict[key][rs_idx] = sonde_dict[key]

                elif key in time_height_keys:
                    sonde_master_dict[key][rs_idx, :] = sonde_dict[key + "_ip"]     # must use the interpolated versions!

                else:
                    raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with " +
                                    "import_single_PS122_mosaic_radiosonde_level3")

            rs_idx += 1

        # Truncate number of sondes:
        if remove_failed and (rs_idx < n_sondes):
            for key in geoinfo_keys: sonde_master_dict[key] = sonde_master_dict[key][:rs_idx]
            for key in time_height_keys: sonde_master_dict[key] = sonde_master_dict[key][:rs_idx,:]


    if verbose >= 1: print("")

    return sonde_master_dict


def import_PS_mastertrack_tab(filename):

    """
    Imports Polarstern master track data during MOSAiC published on PANGAEA. Time
    will be given in seconds since 1970-01-01 00:00:00 UTC and datetime. It also
    returns global attributes in the .tab file so that the information can be
    forwarded to the netcdf version of the master tracks.

    Leg 1, Version 2:
    Rex, Markus (2020): Links to master tracks in different resolutions of POLARSTERN
    cruise PS122/1, Tromsø - Arctic Ocean, 2019-09-20 - 2019-12-13 (Version 2). Alfred
    Wegener Institute, Helmholtz Centre for Polar and Marine Research, Bremerhaven, 
    PANGAEA, https://doi.org/10.1594/PANGAEA.924668

    Leg 2, Version 2:
    Haas, Christian (2020): Links to master tracks in different resolutions of POLARSTERN
    cruise PS122/2, Arctic Ocean - Arctic Ocean, 2019-12-13 - 2020-02-24 (Version 2).
    Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research,
    Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.924674

    Leg 3, Version 2:
    Kanzow, Torsten (2020): Links to master tracks in different resolutions of POLARSTERN
    cruise PS122/3, Arctic Ocean - Longyearbyen, 2020-02-24 - 2020-06-04 (Version 2).
    Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, 
    Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.924681

    Leg 4:
    Rex, Markus (2021): Master tracks in different resolutions of POLARSTERN cruise
    PS122/4, Longyearbyen - Arctic Ocean, 2020-06-04 - 2020-08-12. Alfred Wegener 
    Institute, Helmholtz Centre for Polar and Marine Research, Bremerhaven, PANGAEA,
    https://doi.org/10.1594/PANGAEA.926829

    Leg 5:
    Rex, Markus (2021): Master tracks in different resolutions of POLARSTERN cruise
    PS122/5, Arctic Ocean - Bremerhaven, 2020-08-12 - 2020-10-12. Alfred Wegener
    Institute, Helmholtz Centre for Polar and Marine Research, Bremerhaven, PANGAEA,
    https://doi.org/10.1594/PANGAEA.926910

    Parameters:
    -----------
    filename : str
        Filename + path of the Polarstern Track data (.tab) downloaded from the DOI
        given above.
    """

    n_prel = 20000      # just a preliminary assumption of the amount of data entries
    reftime = dt.datetime(1970,1,1)
    pstrack_dict = {'time_sec': np.full((n_prel,), np.nan),     # in seconds since 1970-01-01 00:00:00 UTC
                    'time': np.full((n_prel,), reftime),        # datetime object
                    'Latitude': np.full((n_prel,), np.nan),     # in deg N
                    'Longitude': np.full((n_prel,), np.nan),    # in deg E
                    'Speed': np.full((n_prel,), np.nan),        # in knots
                    'Course': np.full((n_prel,), np.nan)}       # in deg

    f_handler = open(filename, 'r')
    list_of_lines = list()

    # identify header size and save global attributes:
    attribute_info = list()
    for k, line in enumerate(f_handler):
        attribute_info.append(line.strip().split("\t")) # split by tabs
        if line.strip() == "*/":
            break
    attribute_info = attribute_info[1:-1]   # first and last entry are "*/"

    m = 0       # used as index to save the entries into pstrack_dict
    for k, line in enumerate(f_handler):
        if k > 0:       # skip header
            current_line = line.strip().split()     # split by tabs

            # convert time stamp to seconds since 1970-01-01 00:00:00 UTC:
            pstrack_dict['time_sec'][m] = datetime_to_epochtime(dt.datetime.strptime(current_line[0], "%Y-%m-%dT%H:%M"))

            # extract other info:
            pstrack_dict['Latitude'][m] = float(current_line[1])
            pstrack_dict['Longitude'][m] = float(current_line[2])
            pstrack_dict['Speed'][m] = float(current_line[3])
            pstrack_dict['Course'][m] = float(current_line[4])

            m = m + 1

    # truncate redundant lines:
    last_nonnan = np.where(~np.isnan(pstrack_dict['time_sec']))[0][-1] + 1      # + 1 because of python indexing
    for key in pstrack_dict.keys(): pstrack_dict[key] = pstrack_dict[key][:last_nonnan]

    # time to datetime:
    pstrack_dict['time'] = np.asarray([dt.datetime.utcfromtimestamp(tt) for tt in pstrack_dict['time_sec']])

    return pstrack_dict, attribute_info


def import_MOSAiC_Radiosondes_PS122_Level2_tab(filename):

    """
    Imports level 2 radiosonde data launched from Polarstern
    during the MOSAiC campaign. Time will be given in seconds since 1970-01-01 00:00:00 UTC
    and datetime. Furthermore, the Integrated Water Vapour will be computed
    using the saturation water vapour pressure according to Hyland and Wexler 1983.

    Maturilli, Marion; Holdridge, Donna J; Dahlke, Sandro; Graeser, Jürgen;
    Sommerfeld, Anja; Jaiser, Ralf; Deckelmann, Holger; Schulz, Alexander 
    (2021): Initial radiosonde data from 2019-10 to 2020-09 during project 
    MOSAiC. Alfred Wegener Institute, Helmholtz Centre for Polar and Marine 
    Research, Bremerhaven, PANGAEA, https://doi.pangaea.de/10.1594/PANGAEA.928656 
    (DOI registration in progress)

    Parameters:
    -----------
    filename : str
        Filename + path of the Level 2 radiosonde data (.tab) downloaded from the DOI
        given above.
    """

    n_sonde_prel = 3000     # just a preliminary assumption of the amount of radiosondes
    n_data_per_sonde = 12000    # assumption of max. time points per sonde
    reftime = dt.datetime(1970,1,1)
    # the radiosonde dict will be structured as follows:
    # rs_dict['0'] contains all data from the first radiosonde: rs_dict['0']['T'] contains temperature
    # rs_dict['1'] : second radiosonde, ...
    # this structure allows to have different time dimensions for each radiosonde
    rs_dict = dict()
    for k in range(n_sonde_prel):
        rs_dict[str(k)] = {'time': np.full((n_data_per_sonde,), reftime),       # datetime object
                            'time_sec': np.full((n_data_per_sonde,), np.nan),   # in seconds since 1970-01-01 00:00:00 UTC
                            'Latitude': np.full((n_data_per_sonde,), np.nan),   # in deg N
                            'Longitude': np.full((n_data_per_sonde,), np.nan),  # in deg E
                            'Altitude': np.full((n_data_per_sonde,), np.nan),   # in m
                            'h_geom': np.full((n_data_per_sonde,), np.nan),     # geometric height in m
                            'ETIM': np.full((n_data_per_sonde,), np.nan),       # elapsed time in seconds since sonde start
                            'P': np.full((n_data_per_sonde,), np.nan),          # in hPa
                            'T': np.full((n_data_per_sonde,), np.nan),          # in deg C
                            'RH': np.full((n_data_per_sonde,), np.nan),         # in percent
                            'wdir': np.full((n_data_per_sonde,), np.nan),       # in deg
                            'wspeed': np.full((n_data_per_sonde,), np.nan)}     # in m s^-1


    f_handler = open(filename, 'r')

    # identify header size and save global attributes:
    attribute_info = list()
    for k, line in enumerate(f_handler):
        if line.strip().split("\t")[0] in ['Citation:', 'In:', 'Abstract:', 'Keyword(s):']:
            attribute_info.append(line.strip().split("\t")) # split by tabs
        if line.strip() == "*/":
            break


    m = -1      # used as index to save the entries into rs_dict; will increase for each new radiosonde
    mm = 0      # runs though all time points of one radiosonde and is reset to 0 for each new radiosonde
    precursor_event = ''
    for k, line in enumerate(f_handler):
        if k > 0:       # skip header
            current_line = line.strip().split("\t")     # split by tabs
            current_event = current_line[0]         # marks the radiosonde launch

            if current_event != precursor_event:    # then a new sonde is found in the current_line
                m = m + 1
                mm = 0

            # convert time stamp to seconds since 1970-01-01 00:00:00 UTC:
            rs_dict[str(m)]['time'][mm] = dt.datetime.strptime(current_line[1], "%Y-%m-%dT%H:%M:%S")
            rs_dict[str(m)]['time_sec'][mm] = datetime_to_epochtime(rs_dict[str(m)]['time'][mm])

            # extract other info:
            try:
                rs_dict[str(m)]['Latitude'][mm] = float(current_line[2])
                rs_dict[str(m)]['Longitude'][mm] = float(current_line[3])
                rs_dict[str(m)]['Altitude'][mm] = float(current_line[4])
                rs_dict[str(m)]['h_geom'][mm] = float(current_line[5])
                rs_dict[str(m)]['ETIM'][mm] = float(current_line[6])
                rs_dict[str(m)]['P'][mm] = float(current_line[7])
                rs_dict[str(m)]['T'][mm] = float(current_line[8])
                rs_dict[str(m)]['RH'][mm] = float(current_line[9])
                rs_dict[str(m)]['wdir'][mm] = float(current_line[10])
                rs_dict[str(m)]['wspeed'][mm] = float(current_line[11])

            except ValueError:      # then at least one measurement is missing:
                for ix, cr in enumerate(current_line):
                    if cr == '':
                        current_line[ix] = 'nan'
                try:
                    rs_dict[str(m)]['Latitude'][mm] = float(current_line[2])
                    rs_dict[str(m)]['Longitude'][mm] = float(current_line[3])
                    rs_dict[str(m)]['Altitude'][mm] = float(current_line[4])
                    rs_dict[str(m)]['h_geom'][mm] = float(current_line[5])
                    rs_dict[str(m)]['ETIM'][mm] = float(current_line[6])
                    rs_dict[str(m)]['P'][mm] = float(current_line[7])
                    rs_dict[str(m)]['T'][mm] = float(current_line[8])
                    rs_dict[str(m)]['RH'][mm] = float(current_line[9])
                    rs_dict[str(m)]['wdir'][mm] = float(current_line[10])
                    rs_dict[str(m)]['wspeed'][mm] = float(current_line[11])

                except IndexError:      # GPS connection lost
                    rs_dict[str(m)]['Latitude'][mm] = float('nan')
                    rs_dict[str(m)]['Longitude'][mm] = float('nan')
                    rs_dict[str(m)]['Altitude'][mm] = float(current_line[4])
                    rs_dict[str(m)]['h_geom'][mm] = float('nan')
                    rs_dict[str(m)]['ETIM'][mm] = float(current_line[6])
                    rs_dict[str(m)]['P'][mm] = float(current_line[7])
                    rs_dict[str(m)]['T'][mm] = float(current_line[8])
                    rs_dict[str(m)]['RH'][mm] = float(current_line[9])
                    rs_dict[str(m)]['wdir'][mm] = float('nan')
                    rs_dict[str(m)]['wspeed'][mm] = float('nan')

            mm = mm + 1
            precursor_event = current_event

    # truncate redundantly initialised sondes:
    for k in range(m+1, n_sonde_prel): del rs_dict[str(k)]
    
    # finally truncate unneccessary time dimension for each sonde and compute IWV:
    for k in range(m+1):
        last_nonnan = np.where(~np.isnan(rs_dict[str(k)]['time_sec']))[0][-1] + 1       # + 1 because of python indexing
        for key in rs_dict[str(k)].keys(): rs_dict[str(k)][key] = rs_dict[str(k)][key][:last_nonnan]
        rs_dict[str(k)]['q'] = np.asarray([convert_rh_to_spechum(t+273.15, p*100, rh/100) 
                                for t, p, rh in zip(rs_dict[str(k)]['T'], rs_dict[str(k)]['P'], rs_dict[str(k)]['RH'])])
        rs_dict[str(k)]['rho_v'] = np.asarray([convert_rh_to_abshum(t+273.15, rh/100) 
                                for t, rh in zip(rs_dict[str(k)]['T'], rs_dict[str(k)]['RH'])])
        rs_dict[str(k)]['IWV'] = compute_IWV_q(rs_dict[str(k)]['q'], rs_dict[str(k)]['P']*100)
    
    return rs_dict, attribute_info


def import_PS_mastertrack(
    filename,
    keys='all',
    return_DS=False):

    """
    Imports Polarstern master track data during MOSAiC published on PANGAEA. Time
    will be given in seconds since 1970-01-01 00:00:00 UTC and datetime. It also
    returns global attributes in the .tab file so that the information can be
    forwarded to the netcdf version of the master tracks.

    Parameters:
    -----------
    filename : str or list of str
        Filename + path of the Polarstern Track data (.nc).
    keys : list of str
        List of names of variables to be imported. 'all' will import all keys.
        Default: 'all'
    return_DS : bool
        If True, an xarray dataset will be returned (only if type(filename) == list).
    """

    if type(filename) == list:
        DS = xr.open_mfdataset(filename, combine='nested', concat_dim='time')
        if return_DS: return DS

    else:

        file_nc = nc.Dataset(filename)

        if keys == 'all':
            keys = file_nc.variables.keys()

        elif isinstance(keys, str) and (keys != 'all'):
            raise ValueError("Argument 'keys' must either be a string ('all') or a list of variable names.")

        ps_track_dict = dict()
        for key in keys:
            if not key in file_nc.variables.keys():
                raise KeyError("I have no memory of this key: '%s'. Key not found in file '%s'." %(key, filename))
            ps_track_dict[key] = np.asarray(file_nc.variables[key])

        return ps_track_dict


def import_cloudnet_product(
    filename,
    keys='basic'):

    """
    Importing CLOUDNET data (classification).

    Parameters:
    -----------
    filename : str
        Path and filename of data.
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (keys='all') or import basic keys
        that the author considers most important (keys='basic')
        or leave this argument out.
    """

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        keys = ['time', 'height', 'target_classification']

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    data_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in cloudnet file." % key)
        data_dict[key] = np.asarray(file_nc.variables[key])


    if 'time' in keys:  # avoid nasty digita after decimal point and convert to seconds since 1970-01-01 00:00:00 UTC
        time_units = file_nc.variables['time'].units
        reftime = dt.datetime.strptime(time_units[12:-6], '%Y-%m-%d %H:%M:%S')
        reftime_epoch = datetime_to_epochtime(reftime)
        data_dict['time'] = np.float64(data_dict['time'])   # for the conversion, I need greater precision
        data_dict['time'] = data_dict['time']*3600 + reftime_epoch

        data_dict['time'] = np.rint(data_dict['time']).astype(float)

    return data_dict


def import_cloudnet_product_daterange(
    path_data,
    date_start,
    date_end,
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the cloudnet product data time
    series of each day so that you'll have one dictionary for all data.

    Parameters:
    -----------
    path_data : str
        Base path of level 2a data. This directory contains subfolders representing the year which,
        in turn, contains the daily files. Example path_data:
        "/data/obs/campaigns/mosaic/cloudnet/products/classification/"
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1
    n_hgt = 595         # inquired from cloudnet data, number of model height levels

    # basic variables that should always be imported:
    time_keys = ['time']                # keys with time as coordinate
    height_keys = ['height']                            # keys with height as coordinate
    time_height_keys = ['target_classification']


    # master_dict (output) will contain all desired variables on specific axes:
    master_dict = dict()

    # save import keys in a list:
    import_keys = time_keys + height_keys + time_height_keys

    n_samp_tot = n_days*3000        # number of expected time dimension entries
    for mthk in time_height_keys: master_dict[mthk] = np.full((n_samp_tot, n_hgt), -99)
    for mtkab in time_keys: master_dict[mtkab] = np.full((n_samp_tot,), np.nan)
    for mhk in height_keys: master_dict[mhk] = np.full((n_hgt,), np.nan)

    # cycle through all years, all months and days:
    time_index = 0  # this index will be increased by the length of the time series of the 
                    # current day (now_date) to fill the master_dict time axis accordingly.
    for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

        if verbose >= 1: print("\rWorking on Cloudnet Product, ", now_date, end="")

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day

        day_path = path_data + str(yyyy) + "/"

        if not os.path.exists(os.path.dirname(day_path)):
            continue

        # cloudnet file for the current day:
        file_nc = sorted(glob.glob(day_path + "%4i%02i%02i*_classification.nc"%(yyyy,mm,dd)))

        if len(file_nc) == 0:
            if verbose >= 2:
                warnings.warn("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
            continue

        # select current day and import the data:
        file_nc = file_nc[0]
        data_dict = import_cloudnet_product(file_nc, import_keys)

        # save to master_dict
        for key in data_dict.keys():
            mwr_key_shape = data_dict[key].shape
            n_time = len(data_dict['time'])

            if mwr_key_shape == data_dict['time'].shape:    # then the variable is on time axis:
                master_dict[key][time_index:time_index + n_time] = data_dict[key]

            elif key in height_keys:    # will be handled after the loop
                continue

            elif key in time_height_keys:
                master_dict[key][time_index:time_index + n_time,:] = data_dict[key]

            else:
                raise RuntimeError("Something went wrong in the " +
                    "import_cloudnet_product_daterange routine. Unexpected variable dimension for " + key + ".")

        time_index = time_index + n_time


    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:
        # save height keys to master dict:
        for hkey in height_keys: master_dict[hkey] = data_dict[hkey]

        # truncate the master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(master_dict['time']))[-1][0]
        time_shape_old = master_dict['time'].shape
        time_height_shape_old = master_dict[time_height_keys[0]].shape
        for mwr_key in master_dict.keys():
            shape_new = master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                master_dict[mwr_key] = master_dict[mwr_key][:last_time_step+1]
            elif shape_new == time_height_shape_old:
                master_dict[mwr_key] = master_dict[mwr_key][:last_time_step+1, :]

    if verbose >= 1: print("")

    return master_dict


def import_MiRAC_outliers(filename):

    """
    Import and convert manually (per eye) detected outliers of MiRAC-P 
    to an array filled with datetime.

    Parameters:
    -----------
    filename : str
        Filename (including path) of the text file (.txt) that contains
        the outliers.
    """

    headersize = 1
    file_handler = open(filename, 'r')
    list_of_lines = list()

    for line in file_handler:
        current_line = line.strip().split('   ')    # split by 3 spaces
        list_of_lines.append(current_line)

    # delete header:
    list_of_lines = list_of_lines[headersize:]
    n_outliers = len(list_of_lines)         # number of outliers

    # read start and end time of an outlier from a line:
    list_outliers = []
    for ix, line in enumerate(list_of_lines):
        list_outliers.append([dt.datetime.strptime(line[0], "%Y %m %d %H %M"),
                            dt.datetime.strptime(line[1], "%Y %m %d %H %M")])

    return list_outliers


def import_MOSAiC_Radiosondes_PS122_Level3_tab(filename):

    """
    Imports level 3 radiosonde data launched from Polarstern
    during the MOSAiC campaign. Time will be given in seconds since 1970-01-01 00:00:00 UTC
    and datetime. Furthermore, the Integrated Water Vapour will be computed
    using the saturation water vapour pressure according to Hyland and Wexler 1983.

    Maturilli, Marion; Sommer, Michael; Holdridge, Donna J; Dahlke, Sandro; 
    Graeser, Jürgen; Sommerfeld, Anja; Jaiser, Ralf; Deckelmann, Holger; 
    Schulz, Alexander (2022): MOSAiC radiosonde data (level 3). PANGAEA, 
    https://doi.org/10.1594/PANGAEA.943870

    Parameters:
    -----------
    filename : str
        Filename + path of the Level 3 radiosonde data (.tab) downloaded from the DOI
        given above.
    """

    n_sonde_prel = 3000     # just a preliminary assumption of the amount of radiosondes
    n_data_per_sonde = 12000    # assumption of max. time points per sonde
    reftime = np.datetime64("1970-01-01T00:00:00")
    # the radiosonde dict will be structured as follows:
    # rs_dict['0'] contains all data from the first radiosonde: rs_dict['0']['T'] contains temperature
    # rs_dict['1'] : second radiosonde, ...
    # this structure allows to have different time dimensions for each radiosonde
    rs_dict = dict()
    for k in range(n_sonde_prel):
        rs_dict[str(k)] = {'time': np.full((n_data_per_sonde,), reftime),       # np datetime64
                            'time_sec': np.full((n_data_per_sonde,), np.nan),   # in seconds since 1970-01-01 00:00:00 UTC
                            'Latitude': np.full((n_data_per_sonde,), np.nan),   # in deg N
                            'Longitude': np.full((n_data_per_sonde,), np.nan),  # in deg E
                            'Altitude': np.full((n_data_per_sonde,), np.nan),   # gps altitude above WGS84 in m
                            'h_geop': np.full((n_data_per_sonde,), np.nan),     # geopotential height in m
                            'h_gps': np.full((n_data_per_sonde,), np.nan),      # geometric/GPS receiver height in m
                            'P': np.full((n_data_per_sonde,), np.nan),          # in hPa
                            'T': np.full((n_data_per_sonde,), np.nan),          # in K
                            'RH': np.full((n_data_per_sonde,), np.nan),         # in percent
                            'mixrat': np.full((n_data_per_sonde,), np.nan),     # in mg kg-1
                            'wdir': np.full((n_data_per_sonde,), np.nan),       # in deg
                            'wspeed': np.full((n_data_per_sonde,), np.nan),     # in m s^-1
                            'IWV': np.full((n_data_per_sonde,), np.nan)}        # in kg m-2


    f_handler = open(filename, 'r')

    # identify header size and save global attributes:
    attribute_info = list()
    for k, line in enumerate(f_handler):
        if line.strip().split("\t")[0] in ['Citation:', 'Project(s):', 'Abstract:', 'Keyword(s):']:
            attribute_info.append(line.strip().split("\t")) # split by tabs
        if line.strip() == "*/":
            break

    m = -1      # used as index to save the entries into rs_dict; will increase for each new radiosonde
    mm = 0      # runs though all time points of one radiosonde and is reset to 0 for each new radiosonde
    precursor_event = ''
    for k, line in enumerate(f_handler):
        if k == 0:
            headerline = line.strip().split("\t")

        if k > 0:       # skip header
            current_line = line.strip().split("\t")     # split by tabs
            current_event = current_line[0]         # marks the radiosonde launch

            if current_event != precursor_event:    # then a new sonde is found in the current_line
                m = m + 1
                mm = 0

            # convert time stamp to seconds since 1970-01-01 00:00:00 UTC:
            rs_dict[str(m)]['time'][mm] = np.datetime64(current_line[1])
            rs_dict[str(m)]['time_sec'][mm] = rs_dict[str(m)]['time'][mm].astype(np.float64)

            # extract other info:
            try:
                rs_dict[str(m)]['Latitude'][mm] = float(current_line[10])
                rs_dict[str(m)]['Longitude'][mm] = float(current_line[8])
                rs_dict[str(m)]['Altitude'][mm] = float(current_line[6])
                rs_dict[str(m)]['h_geop'][mm] = float(current_line[2])
                rs_dict[str(m)]['h_gps'][mm] = float(current_line[4])
                rs_dict[str(m)]['P'][mm] = float(current_line[12])
                rs_dict[str(m)]['T'][mm] = float(current_line[16])
                rs_dict[str(m)]['RH'][mm] = float(current_line[18])
                rs_dict[str(m)]['mixrat'][mm] = float(current_line[22])
                rs_dict[str(m)]['wdir'][mm] = float(current_line[30])
                rs_dict[str(m)]['wspeed'][mm] = float(current_line[32])
                try:
                    rs_dict[str(m)]['IWV'][mm] = float(current_line[41])
                except IndexError:  # sometimes, the final two columns just don't exist...whyever
                    rs_dict[str(m)]['IWV'][mm] = float('nan')

            except ValueError:      # then at least one measurement is missing:
                for ix, cr in enumerate(current_line):
                    if cr == '':
                        current_line[ix] = 'nan'
                try:
                    rs_dict[str(m)]['Latitude'][mm] = float(current_line[10])
                    rs_dict[str(m)]['Longitude'][mm] = float(current_line[8])
                    rs_dict[str(m)]['Altitude'][mm] = float(current_line[6])
                    rs_dict[str(m)]['h_geop'][mm] = float(current_line[2])
                    rs_dict[str(m)]['h_gps'][mm] = float(current_line[4])
                    rs_dict[str(m)]['P'][mm] = float(current_line[12])
                    rs_dict[str(m)]['T'][mm] = float(current_line[16])
                    rs_dict[str(m)]['RH'][mm] = float(current_line[18])
                    rs_dict[str(m)]['mixrat'][mm] = float(current_line[22])
                    rs_dict[str(m)]['wdir'][mm] = float(current_line[30])
                    rs_dict[str(m)]['wspeed'][mm] = float(current_line[32])
                    rs_dict[str(m)]['IWV'][mm] = float(current_line[41])

                except IndexError:      # GPS connection lost
                    rs_dict[str(m)]['Latitude'][mm] = float('nan')
                    rs_dict[str(m)]['Longitude'][mm] = float('nan')
                    rs_dict[str(m)]['Altitude'][mm] = float('nan')
                    rs_dict[str(m)]['h_geop'][mm] = float(current_line[6])
                    rs_dict[str(m)]['h_gps'][mm] = float('nan')
                    rs_dict[str(m)]['P'][mm] = float(current_line[12])
                    rs_dict[str(m)]['T'][mm] = float(current_line[16])
                    rs_dict[str(m)]['RH'][mm] = float(current_line[18])
                    rs_dict[str(m)]['mixrat'][mm] = float(current_line[22])
                    rs_dict[str(m)]['wdir'][mm] = float('nan')
                    rs_dict[str(m)]['wspeed'][mm] = float('nan')
                    rs_dict[str(m)]['IWV'][mm] = float('nan')

            mm = mm + 1
            precursor_event = current_event

    # truncate redundantly initialised sondes:
    for k in range(m+1, n_sonde_prel): del rs_dict[str(k)]
    
    # finally truncate unneccessary time dimension for each sonde and compute IWV:
    for k in range(m+1):
        last_nonnan = np.where(~np.isnan(rs_dict[str(k)]['time_sec']))[0][-1] + 1       # + 1 because of python indexing
        for key in rs_dict[str(k)].keys(): rs_dict[str(k)][key] = rs_dict[str(k)][key][:last_nonnan]
        rs_dict[str(k)]['q'] = np.asarray([convert_rh_to_spechum(t, p*100.0, rh/100.0) 
                                for t, p, rh in zip(rs_dict[str(k)]['T'], rs_dict[str(k)]['P'], rs_dict[str(k)]['RH'])])
        rs_dict[str(k)]['rho_v'] = np.asarray([convert_rh_to_abshum(t, rh/100.0) 
                                for t, rh in zip(rs_dict[str(k)]['T'], rs_dict[str(k)]['RH'])])

        rs_dict[str(k)]['IWV'] = rs_dict[str(k)]['IWV'][~np.isnan(rs_dict[str(k)]['IWV'])][-1]

    return rs_dict, attribute_info


def import_hatpro_mirac_level2a_daterange_pangaea(
    path_data,
    date_start,
    date_end=None,
    which_retrieval='both',
    data_version='v00'):

    """
    Imports the synergetic neural network retrieval output combining data from HATPRO and MiRAC-P
    for all days between a start and an end date or imports data for a certain list of dates. 
    Each day is concatenated in ascending order.

    Parameters:
    -----------
    path_data : str
        Path of the synergetic retrieval level 2a (IWV (prw), LWP (clwvi)) data. 
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
        integrated water vapour. 'clwvi' or 'lwp' will load the liquid water path. 'both' will 
        load integrated water vapour and liquid water path. Default: 'both'
    data_version : str, optional
        Indicated the version of the data as string. Example: "v00", "v01, "v02".
    """

    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'both' will also load integrated water vapour only. Default: 'both'")

    else:
        if which_retrieval not in ['prw', 'iwv', 'clwvi', 'lwp', 'both']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'clwvi' or 'lwp' will load the liquid water path. 'both' will load integrated " +
                "water vapour and liquid water path. Default: 'both'")

        else:
            if which_retrieval in ['iwv', 'prw']:
                which_retrieval = ['prw']
            elif which_retrieval in ['lwp', 'clwvi']:
                which_retrieval = ['clwvi']
            elif which_retrieval == 'both':
                which_retrieval = ['prw', 'clwvi']
            else:
                raise ValueError("Argument '" + which_retrieval + "' not recognized. Please use one of the following options: " +
                        "'iwv' or 'prw' will load the " +
                        "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                        "'both' will load both. Default: 'both'")
                    

    # Identify files in the date range: First, load all into a list, then check
    # which ones suit the daterange:
    sub_str = f"_{data_version}_"
    l_sub_str = len(sub_str)
    files = sorted(glob.glob(path_data + f"MOSAiC_uoc_hatpro_lhumpro-243-340_l2_*.nc"))

    if type(date_start) == type(""):
        # extract day, month and year from start date:
        date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start and file_date <= date_end:
                files_filtered.append(file)
    else:
        # extract day, month and year from date_start:
        date_list = [dt.datetime.strptime(ddd, "%Y-%m-%d") for ddd in date_start]

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date in date_list:
                files_filtered.append(file)


    # distinguish between the retrieved products:
    DS_ret = xr.Dataset()       # dummy dataset
    if 'prw' in which_retrieval:
        files_prw = [file for file in files_filtered if "_prw_" in file]

        # load data:
        if len(files_prw) > 0:
            DS_p = xr.open_mfdataset(files_prw, concat_dim='time', combine='nested', decode_times=False)
            DS_ret = DS_p

    if 'clwvi' in which_retrieval:
        files_clwvi = [file for file in files_filtered if "_clwvi_" in file]

        # load data:
        if len(files_clwvi) > 0:
            DS_c = xr.open_mfdataset(files_clwvi, concat_dim='time', combine='nested', decode_times=False)
            DS_ret = DS_c

    # if both are requested, merge both datasets by just adding the former:
    if ('prw' in which_retrieval) and ('clwvi' in which_retrieval):
        if len(files_prw) > 0: DS_ret['prw'] = DS_p['prw']


    # 'repair' some variables:
    if DS_ret:
        DS_ret['flag_h'][np.isnan(DS_ret['flag_h']).load()] = 0.
        DS_ret['flag_m'][np.isnan(DS_ret['flag_m']).load()] = 0.

    return DS_ret


def import_hatpro_mirac_level2b_daterange_pangaea(
    path_data,
    date_start,
    date_end=None,
    which_retrieval='q',
    data_version='v00',
    around_radiosondes=True,
    path_radiosondes="",
    s_version='level_2',
    mwr_avg=0,
    flag_data=True):

    """
    Imports the synergetic neural network retrieval output combining data from HATPRO and MiRAC-P
    for all days between a start and an end date or imports data for a certain list of dates. 
    Each day is concatenated in ascending order. Because of memory usage, only times around 
    radiosonde launch times or alternatively 4 samples per day at fixed times (05, 11, 17 and 23 UTC)
    are considered.

    Parameters:
    -----------
    path_data : str
        Path of the synergetic retrieval level 2b (temperature (temp), specific humidity (q)) data.
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'temp' or 'ta' will load the
        zenith temperature profile. 'q' will load the specific humidity profile. 'rh' will load the
        generated relative humidity profile product.
    data_version : str, optional
        Indicated the version of the data as string. Example: "v00"
    around_radiosondes : bool, optional
        If True, data will be limited to the time around radiosonde launches. If False, something else
        (e.g. around 4 times a day) might be done. Default: True
    path_radiosondes : str, optional
        Path to radiosonde data (Level 2). Default: ""
    s_version : str, optional
        Specifies the radiosonde version that is to be imported. Must be 'level_2' to work properly.
        Other versions have not been implemeted because they are considered to be inferior to level_2
        radiosondes.
    mwr_avg : int, optional
        If > 0, an average over mwr_avg seconds will be performed from sample_time to sample_time + 
        mwr_avg seconds. If == 0, no averaging will be performed.
    flag_data : bool, optional
        If True, data where flags indicate bad values are excluded (for HATPRO, flag = 0 and 
        flag = 32 are okay; for MiRAC-P, flag = 0 is okay). If False, no filtering will be applied.
    """

    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'temp' or 'ta' will load the " +
                "zenith temperature profile. 'q' will load humidity profiles. 'rh' will load the " +
                "generated relative humidity profile product.")

    else:
        if which_retrieval not in ['ta', 'temp', 'q', 'hus', 'rh']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'temp' or 'ta' will load the " +
                "zenith temperature profile. 'q' will load the specific humidity profile. 'rh' will load the " +
                "generated relative humidity profile product.")

        else:
            if which_retrieval in ['ta', 'temp']:
                which_retrieval = 'temp'
            elif which_retrieval in ['hus', 'q']:
                which_retrieval = 'q'
            elif which_retrieval == 'rh':
                which_retrieval = 'rh'
            elif which_retrieval not in ['ta', 'temp', 'q', 'rh']:
                raise ValueError("Argument '" + which_retrieval + "' not recognized")


    # extract dates from date_start (and date_end):
    date_list = []
    if type(date_start) == type([]):
        date_list = copy.deepcopy(date_start)
        date_start = date_start[0]
        date_end = date_list[-1]
        date_list = [dt.datetime.strptime(dl, "%Y-%m-%d").date() for dl in date_list]
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count number of days between start and end date as estimated max. array size:
    n_days = (date_end - date_start).days + 1
    n_hgt = {'q': 43, 'temp': 39, 'rh': 39}
    n_hgt = n_hgt[which_retrieval]


    # Create an array that includes the radiosonde launch times:
    if around_radiosondes:
        if not path_radiosondes:
            raise ValueError("If 'around_radiosondes' is True, the path to the radiosonde data ('path_radiosondes') " +
                                "must be given.")

        if s_version != 'level_2':
            raise ValueError("Radiosonde version 's_version' must be 'level_2' if around_radiosondes is True because " +
                                "for this version, the launch time is directly read from the filename. This has not " +
                                "been implemeted for other radiosonde versions ('mossonde', 'psYYMMDDwHH') because these " +
                                "are considered to be inferior.")
        else:
            add_files = sorted(glob.glob(path_radiosondes + "*.nc"))        # filenames only; filter path
            add_files = [os.path.basename(a_f) for a_f in add_files]
            
            # identify launch time:
            n_samp = len(add_files)     # number of radiosondes
            launch_times = np.full((n_samp,), dt.datetime(1970,1,1))
            kk = 0
            if date_list:   # then only consider dates within date_list
                for a_f in add_files:
                    ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                    # only save those that are in the considered period
                    if ltt.date() in date_list:
                        launch_times[kk] = ltt
                        kk += 1
            else:           # take all files between date_start and date_end
                for a_f in add_files:
                    ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                    # only save those that are in the considered period
                    if ltt >= date_start and ltt < (date_end + dt.timedelta(days=1)):
                        launch_times[kk] = ltt
                        kk += 1

            # truncate launch_times and convert to sec since 1970-01-01 00:00:00 UTC:
            launch_times = launch_times[:kk]
            sample_times = datetime_to_epochtime(launch_times)
            n_samp_tot = len(sample_times)

    else:
        # max number of samples: n_days*4
        sample_times = [5, 11, 17, 23]      # UTC on each day
        n_samp = len(sample_times)
        n_samp_tot = n_days*n_samp


    # Identify files in the date range: First, load all into a list, then check
    # which ones suit the daterange:
    sub_str = f"_{data_version}_"
    l_sub_str = len(sub_str)
    files = sorted(glob.glob(path_data + f"MOSAiC_uoc_hatpro_lhumpro-243-340_l2_*.nc"))


    if date_list:
        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date in date_list:
                files_filtered.append(file)

    else:
        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start and file_date <= date_end:
                files_filtered.append(file)
    if which_retrieval == 'temp':
        file_substr = "_temp_v"
    elif which_retrieval == 'q':
        file_substr = "_q_"
    elif which_retrieval == 'rh':
        file_substr = "_rh_"
    files_filtered = [file for file in files_filtered if file_substr in file]


    # cycle through all years, all months and days:
    time_idx = 0        # this index (for lvl 2b) will be increased by the length of the time
                        # series of the current day (now_date) to indicate if data has already been found
    sample_time_tolerance = 900     # sample time tolerance in seconds: mwr time must be within this
                                    # +/- tolerance of a sample_time to be accepted
    if not date_list:
        date_list = (date_start + dt.timedelta(days=n) for n in range(n_days))
    else:
        date_list = [dt.datetime(dl_i.year, dl_i.month, dl_i.day) for dl_i in date_list]

    for now_date in date_list:

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day
        now_date_str = now_date.strftime("%Y%m%d")

        print(f"\rImporting level 2b data for {now_date_str}.", end="")

        # specify sample times as time: sec since 1970-01-01 00:00:00 UTC:
        if around_radiosondes:
            now_date_date = now_date.date()
            sample_mask = np.full((n_samp_tot,), False)
            for kk, l_t in enumerate(launch_times):
                sample_mask[kk] = l_t.date() == now_date_date

            sample_times_t = sample_times[sample_mask]

        else:
            sample_times_t = np.asarray([datetime_to_epochtime(dt.datetime(yyyy, mm, dd, st, 0, 0)) for st in sample_times])


        # identify files for current date:
        files_now = []
        for file in files_filtered:
            if now_date_str in file:
                files_now.append(file)

        if len(files_now) == 0: continue
        if len(files_now) > 1: 
            print("\nUnexpected number of files.")
            pdb.set_trace() # unexpected. debug


        # load data:
        DS = xr.open_dataset(files_now[0], decode_times=False)

        # repair flags:
        DS['flag_h'][np.isnan(DS['flag_h'])] = 0.
        DS['flag_m'][np.isnan(DS['flag_m'])] = 0.

        # it may occur that the whole day is flagged. If so, skip this file:
        flag_mask = (DS.flag_m == 0) & ((DS.flag_h == 0) | (DS.flag_h == 32))
        if not np.any(flag_mask):
            n_samp_real = 0
            continue
            pdb.set_trace()

        # if desired, reduce data to good flags only:
        if flag_data: DS = DS.isel(time=flag_mask)


        # Option 1: find the time slice where the mwr time is closest to the sample_times.
        # The identified index must be within 15 minutes, otherwise it will be discarded
        # Furthermore, it needs to be respected, that the flag value must be 0 for that case.
        # Option 2: Average over launch_time:launch_time+mwr_avg. For this, find the respecitve
        # indices:
        if mwr_avg == 0:
            sample_idx = []
            for st in sample_times_t:
                idx = np.argmin(np.abs(DS['time'].values - st))
                if np.abs(DS['time'].values[idx] - st) < sample_time_tolerance:
                    sample_idx.append(idx)
            sample_idx = np.asarray(sample_idx)
            n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

        else:
            sample_idx = []
            for st in sample_times_t:
                idx = np.where((DS['time'].values >= st) & (DS['time'].values < st + mwr_avg))[0]
                if len(idx) > 0:    # then an overlap has been found
                    sample_idx.append(idx)
            n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

        if n_samp_real == 0: continue


        # select times at sample_idx and concatenate data (eventually combined with averaging):
        if mwr_avg == 0:
            if time_idx == 0:
                DS_ret = DS.isel(time=sample_idx)
            else:
                DS_ret = xr.concat((DS_ret, DS.isel(time=sample_idx)), dim='time')

        else:
            # reduce time around sample times (radiosonde launches) only and average
            # over time:
            ds_list = []
            for sii in sample_idx:
                DS_app = DS.isel(time=sii).mean('time').expand_dims('time', axis=0)

                # forward the highest flag value to DS_app:
                DS_app['flag_h'] = DS.isel(time=sii).flag_h.max('time').expand_dims('time', axis=0)
                DS_app['flag_m'] = DS.isel(time=sii).flag_m.max('time').expand_dims('time', axis=0)

                DS_app = DS_app.assign_coords({'time': np.array([DS.time[sii[0]]])})
                ds_list.append(DS_app)
            ds_list = xr.concat(ds_list, dim='time')

            # concatenate:
            if time_idx == 0:
                DS_ret = ds_list
            else:
                DS_ret = xr.concat((DS_ret, ds_list), dim='time')

                
        time_idx = time_idx + n_samp_real
        DS.close()
        del DS


    print("")
    if time_idx == 0:
        print("No files have been found....")
        DS_ret = None
    return DS_ret


def import_hatpro_mirac_level2c_daterange_pangaea(
    path_data,
    date_start,
    date_end=None,
    which_retrieval='temp',
    data_version='v00',
    around_radiosondes=True,
    path_radiosondes="",
    s_version='level_2',
    mwr_avg=0,
    flag_data=True):

    """
    Imports the synergetic neural network retrieval output for boundary layer temperature
    profiles for all days between a start and an end date or imports data for a certain list 
    of dates. Each day is concatenated in ascending order. Only times around radiosonde 
    launch times or alternatively 4 samples per day at fixed times (05, 11, 17 and 23 UTC)
    are considered.

    Parameters:
    -----------
    path_data : str
        Path of the synergetic retrieval level 2b (temperature (temp), specific humidity (q)) data.
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'temp' or 'ta' will both load the
        boundary layer scan based temperature profile.
    data_version : str, optional
        Indicated the version of the data as string. Example: "v00"
    around_radiosondes : bool, optional
        If True, data will be limited to the time around radiosonde launches. If False, something else
        (e.g. around 4 times a day) might be done. Default: True
    path_radiosondes : str, optional
        Path to radiosonde data (Level 2). Default: ""
    s_version : str, optional
        Specifies the radiosonde version that is to be imported. Must be 'level_2' to work properly.
        Other versions have not been implemeted because they are considered to be inferior to level_2
        radiosondes.
    mwr_avg : int, optional
        If > 0, an average over mwr_avg seconds will be performed from sample_time to sample_time + 
        mwr_avg seconds. If == 0, no averaging will be performed.
    flag_data : bool, optional
        If True, data where flags indicate bad values are excluded (for HATPRO, flag = 0 and 
        flag = 32 are okay; for MiRAC-P, flag = 0 is okay). If False, no filtering will be applied.
    """

    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'temp' or 'ta' will load the " +
                "zenith temperature profile. 'q' will load humidity profiles.")

    else:
        if which_retrieval not in ['ta', 'temp']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'temp' or 'ta' will load the " +
                "zenith temperature profile.")

        else:
            if which_retrieval in ['ta', 'temp']:
                which_retrieval = 'temp'


    # extract dates from date_start (and date_end):
    date_list = []
    if type(date_start) == type([]):
        date_list = copy.deepcopy(date_start)
        date_start = date_start[0]
        date_end = date_list[-1]
        date_list = [dt.datetime.strptime(dl, "%Y-%m-%d").date() for dl in date_list]
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count number of days between start and end date as estimated max. array size:
    n_days = (date_end - date_start).days + 1
    n_hgt = 39      # manually inquired in the netCDF files


    # Create an array that includes the radiosonde launch times:
    if around_radiosondes:
        if not path_radiosondes:
            raise ValueError("If 'around_radiosondes' is True, the path to the radiosonde data ('path_radiosondes') " +
                                "must be given.")

        if s_version != 'level_2':
            raise ValueError("Radiosonde version 's_version' must be 'level_2' if around_radiosondes is True because " +
                                "for this version, the launch time is directly read from the filename. This has not " +
                                "been implemeted for other radiosonde versions ('mossonde', 'psYYMMDDwHH') because these " +
                                "are considered to be inferior.")
        else:
            add_files = sorted(glob.glob(path_radiosondes + "*.nc"))        # filenames only; filter path
            add_files = [os.path.basename(a_f) for a_f in add_files]
            
            # identify launch time:
            n_samp = len(add_files)     # number of radiosondes
            launch_times = np.full((n_samp,), dt.datetime(1970,1,1))
            kk = 0
            if date_list:   # then only consider dates within date_list
                for a_f in add_files:
                    ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                    # only save those that are in the considered period
                    if ltt.date() in date_list:
                        launch_times[kk] = ltt
                        kk += 1
            else:           # take all files between date_start and date_end
                for a_f in add_files:
                    ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                    # only save those that are in the considered period
                    if ltt >= date_start and ltt < (date_end + dt.timedelta(days=1)):
                        launch_times[kk] = ltt
                        kk += 1

            # truncate launch_times and convert to sec since 1970-01-01 00:00:00 UTC:
            launch_times = launch_times[:kk]
            sample_times = datetime_to_epochtime(launch_times)
            n_samp_tot = len(sample_times)

    else:
        # max number of samples: n_days*4
        sample_times = [5, 11, 17, 23]      # UTC on each day
        n_samp = len(sample_times)
        n_samp_tot = n_days*n_samp
            

    # Identify files in the date range: First, load all into a list, then check
    # which ones suit the daterange:
    sub_str = f"_{data_version}_"
    l_sub_str = len(sub_str)
    files = sorted(glob.glob(path_data + f"MOSAiC_uoc_hatpro_lhumpro-243-340_l2_*.nc"))


    if date_list:
        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date in date_list:
                files_filtered.append(file)

    else:
        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start and file_date <= date_end:
                files_filtered.append(file)
    files_filtered = [file for file in files_filtered if "_temp_bl_" in file]


    # cycle through all years, all months and days:
    time_idx = 0        # this index (for lvl 2c) will be increased by the length of the time
                        # series of the current day (now_date) to indicate if data has already been found
    sample_time_tolerance = 1800        # sample time tolerance in seconds: mwr time must be within this
                                    # +/- tolerance of a sample_time to be accepted
    if not date_list:
        date_list = (date_start + dt.timedelta(days=n) for n in range(n_days))
    else:
        date_list = [dt.datetime(dl_i.year, dl_i.month, dl_i.day) for dl_i in date_list]

    for now_date in date_list:

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day
        now_date_str = now_date.strftime("%Y%m%d")

        print(f"\rImporting level 2c data for {now_date_str}.", end="")

        # specify sample times as time: sec since 1970-01-01 00:00:00 UTC:
        if around_radiosondes:
            now_date_date = now_date.date()
            sample_mask = np.full((n_samp_tot,), False)
            for kk, l_t in enumerate(launch_times):
                sample_mask[kk] = l_t.date() == now_date_date

            sample_times_t = sample_times[sample_mask]

        else:
            sample_times_t = np.asarray([datetime_to_epochtime(dt.datetime(yyyy, mm, dd, st, 0, 0)) for st in sample_times])


        # identify files for current date:
        files_now = []
        for file in files_filtered:
            if now_date_str in file:
                files_now.append(file)

        if len(files_now) == 0: continue
        if len(files_now) > 1: 
            print("\nUnexpected number of files.")
            pdb.set_trace() # unexpected. debug


        # load data:
        DS = xr.open_dataset(files_now[0], decode_times=False)

        # repair flags:
        DS['flag_h'][np.isnan(DS['flag_h'])] = 0.
        DS['flag_m'][np.isnan(DS['flag_m'])] = 0.

        # it may occur that the whole day is flagged. If so, skip this file:
        flag_mask = (DS.flag_m == 0) & ((DS.flag_h == 0) | (DS.flag_h == 32))
        if not np.any(flag_mask):
            n_samp_real = 0
            continue
            pdb.set_trace()

        # if desired, reduce data to good flags only:
        if flag_data: DS = DS.isel(time=flag_mask)


        # Option 1: find the time slice where the mwr time is closest to the sample_times.
        # The identified index must be within 15 minutes, otherwise it will be discarded
        # Furthermore, it needs to be respected, that the flag value must be 0 for that case.
        # Option 2: Average over launch_time-mwr_avg:launch_time+mwr_avg. For this, find the 
        # respecitve indices:
        if mwr_avg == 0:
            sample_idx = []
            for st in sample_times_t:
                idx = np.argmin(np.abs(DS['time'].values - st))
                if np.abs(DS['time'].values[idx] - st) < sample_time_tolerance:
                    sample_idx.append(idx)
            sample_idx = np.asarray(sample_idx)
            n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

        else:
            sample_idx = []
            for st in sample_times_t:
                idx = np.where((DS['time'].values >= st - mwr_avg) & (DS['time'].values <= st + mwr_avg))[0]
                if len(idx) > 0:    # then an overlap has been found
                    sample_idx.append(idx)
            n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

        if n_samp_real == 0: continue


        # select times at sample_idx and concatenate data (eventually combined with averaging):
        if mwr_avg == 0:
            if time_idx == 0:
                DS_ret = DS.isel(time=sample_idx)
            else:
                DS_ret = xr.concat((DS_ret, DS.isel(time=sample_idx)), dim='time')

        else:
            # reduce time around sample times (radiosonde launches) only and average
            # over time:
            ds_list = []
            for sii in sample_idx:
                DS_app = DS.isel(time=sii).mean('time').expand_dims('time', axis=0)

                # forward the highest flag value to DS_app:
                DS_app['flag_h'] = DS.isel(time=sii).flag_h.max('time').expand_dims('time', axis=0)
                DS_app['flag_m'] = DS.isel(time=sii).flag_m.max('time').expand_dims('time', axis=0)

                DS_app = DS_app.assign_coords({'time': np.array([DS.time[sii[0]]])})
                ds_list.append(DS_app)
            ds_list = xr.concat(ds_list, dim='time')

            # concatenate:
            if time_idx == 0:
                DS_ret = ds_list
            else:
                DS_ret = xr.concat((DS_ret, ds_list), dim='time')

                
        time_idx = time_idx + n_samp_real
        DS.close()
        del DS


    print("")
    if time_idx == 0:
        print("No files have been found....")
        DS_ret = None
    return DS_ret


def import_MOSAiC_Radiosondes_PS122_Level3_merged_txt(filename):

    """
    Imports level 3 radiosonde data that has been merged with MetTower data by Sandro Dahlke.
    Time will be given in seconds since 1970-01-01 00:00:00 UTC. Furthermore, the Integrated Water 
    Vapour, absolute humidity and specific humidity will be computed using the saturation water 
    vapour pressure according to Hyland and Wexler 1983.

    Dahlke, Sandro; Shupe, Matthew D; Cox, Christopher J; Brooks, Ian M; Blomquist, Byron; 
    Persson, P Ola G (2023): Extended radiosonde profiles 2019/09-2020/10 during MOSAiC 
    Legs PS122/1 - PS122/5. PANGAEA, https://doi.org/10.1594/PANGAEA.961881

    Parameters:
    -----------
    filename : str
        Filename + path of the Level 3 radiosonde data (.tab) downloaded from the DOI
        given above.
    """

    import re       # to search for dates

    n_data_per_sonde = 12000    # assumption of max. time points per sonde
    reftime = np.datetime64("1970-01-01T00:00:00")

    # initialize arrays:
    # the radiosonde dict will be structured as follows:
    # rs_dict contains all data from the first radiosonde: rs_dict['0']['T'] contains temperature
    # rs_dict['1'] : second radiosonde, ...
    # this structure allows to have different time dimensions for each radiosonde
    rs_dict = {'time': np.full((n_data_per_sonde,), reftime),       # np datetime64
                'time_sec': np.full((n_data_per_sonde,), np.nan),   # in seconds since 1970-01-01 00:00:00 UTC
                'lat': np.full((n_data_per_sonde,), np.nan),        # in deg N
                'lon': np.full((n_data_per_sonde,), np.nan),        # in deg E
                'height': np.full((n_data_per_sonde,), np.nan),     # altitude in m
                'pres': np.full((n_data_per_sonde,), np.nan),       # in hPa
                'temp': np.full((n_data_per_sonde,), np.nan),       # in K
                'rh': np.full((n_data_per_sonde,), np.nan),         # in percent
                'wdir': np.full((n_data_per_sonde,), np.nan),       # in deg
                'wspeed': np.full((n_data_per_sonde,), np.nan),     # in m s^-1
                'temp_flag': np.full((n_data_per_sonde,), np.nan),
                'rh_flag': np.full((n_data_per_sonde,), np.nan)}


    with open(filename, 'r', encoding='utf-8') as f_handler:

        # identify header size and save global attributes:
        attribute_info = {'launch_time': reftime, 'data_based_on': [],
                            'data_author': "", 'flag_comments': []}
        k_data_based_on = -100000000        # to save the line number of multi-line attributes
        k_flag = -100000000
        for k, line in enumerate(f_handler):
            line_str = line.strip()
            # pdb.set_trace()
            if "Radiosonde launch: " in line_str:
                match_str = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', line_str)     # finds the date with the format yyyy-mm-ddTHH:MM:SS
                attribute_info['launch_time'] = np.datetime64(match_str.group())
            elif "Data product based on:" in line_str:
                k_data_based_on = k
            elif "data prepared by" in line_str:
                attribute_info['data_author'] = line_str[len("data prepared by") + 1:].replace("...", " ")
            elif "FLAG:" in line_str:
                k_flag = k

            # the lines after k_flag and k_data_based_on contain some attributes:
            if k in [k_data_based_on+1, k_data_based_on+2, k_data_based_on+3]:
                attribute_info['data_based_on'] += [line_str]

            if k in [k_flag+1, k_flag+2, k_flag+3, k_flag+4, k_flag+5, k_flag+6]:
                attribute_info['flag_comments'] += [line_str]

            if line_str == "########ENDHEADER########":
                break

        mm = 0      # runs though all time points of one radiosonde and is reset to 0 for each new radiosonde
        for k, line in enumerate(f_handler):

            current_line = line.strip().split("\t")     # split by tabs

            # convert time stamp to seconds since 1970-01-01 00:00:00 UTC:
            rs_dict['time'][mm] = np.datetime64(current_line[0])
            rs_dict['time_sec'][mm] = rs_dict['time'][mm].astype(np.float64)

            # extract other info:
            try:
                rs_dict['lat'][mm] = float(current_line[3])
                rs_dict['lon'][mm] = float(current_line[4])
                rs_dict['height'][mm] = float(current_line[1])
                rs_dict['pres'][mm] = float(current_line[2])
                rs_dict['temp'][mm] = float(current_line[5])
                rs_dict['rh'][mm] = float(current_line[7])
                rs_dict['temp_flag'][mm] = float(current_line[6])
                rs_dict['rh_flag'][mm] = float(current_line[8])
                rs_dict['wdir'][mm] = float(current_line[10])
                rs_dict['wspeed'][mm] = float(current_line[9])

            except:     # then at least one measurement is missing:
                pdb.set_trace()

            mm += 1


        # convert fillvalues to nan:
        for key in rs_dict.keys():
            if key not in ['time', 'time_sec', 'temp_flag', 'rh_flag']:
                rs_dict[key][rs_dict[key] == -9999] = np.nan
        
        # finally truncate unneccessary time dimension, filter fillvalues and compute IWV, rho_v and q for each sonde:
        last_nonnan = np.where(~np.isnan(rs_dict['pres']))[0][-1] + 1       # + 1 because of python indexing
        for key in rs_dict.keys(): 
            rs_dict[key] = rs_dict[key][:last_nonnan]

        rs_dict['q'] = np.asarray([convert_rh_to_spechum(t+273.15, p*100.0, rh/100.0) 
                                for t, p, rh in zip(rs_dict['temp'], rs_dict['pres'], rs_dict['rh'])])
        rs_dict['rho_v'] = np.asarray([convert_rh_to_abshum(t+273.15, rh/100.0) 
                                for t, rh in zip(rs_dict['temp'], rs_dict['rh'])])


        # limit pressure and humidity to values from first non nan to last:
        non_nan_idx = np.where(~np.isnan(rs_dict['pres']))[0]
        pres = copy.deepcopy(rs_dict['pres'][non_nan_idx[0]:non_nan_idx[-1]+1])

        # repair pressure (fill gaps): Fill gaps in between when pressure measurements around the gap exist:
        n_nans = np.count_nonzero(np.isnan(pres))
        nan_idx = np.where(np.isnan(pres))[0]

        # restrict gap filling: if too many/large gaps, don't fill:
        if (n_nans / len(pres) < 0.1) and n_nans > 0:

            # fill gaps:
            nan_mask = np.isnan(pres)
            where_diff = np.where(np.diff(nan_mask))[0] # yields position of holes
            n_holes = round(len(where_diff) / 1.9999999999)


            # indices of bottom and top boundary of each hole:
            hole_boundaries = np.asarray([[where_diff[2*jj], where_diff[2*jj+1]+1] for jj in range(n_holes)])
                        
            # use the values of the hole boundaries as interpolation targets:
            pres_temp = copy.deepcopy(pres)

            # cycle through holes:
            for hole_b, hole_t in zip(hole_boundaries[:,0], hole_boundaries[:,1]):
                rpl_idx = np.arange(hole_b, hole_t + 1) # +1 because of python indexing
                bd_idx = np.array([rpl_idx[0], rpl_idx[-1]])
                bd_val = np.array([pres_temp[hole_b], pres_temp[hole_t]])

                bridge = np.interp(rpl_idx, bd_idx, bd_val)

                # fill the whole hole:
                pres[rpl_idx] = bridge

            # update value:
            rs_dict['pres'][non_nan_idx[0]:non_nan_idx[-1]+1] = pres


        # for IWV computation, neglect lowest measurements because also the MWRs are not directly on
        # the sea ice and limit until 20 km height (because sometimes, slightly negative q and rh 
        # occur at these high altitudes and because of some sondes containing many nans above):
        height_limit = np.where((rs_dict['height'] >= 15.0) & (rs_dict['height'] <= 20000.0))[0]    # index to cut pressure and humidity

        # if attribute_info['launch_time'] == np.datetime64("2020-09-13T07:55:58"): pdb.set_trace()
        rs_dict['IWV'] = compute_IWV_q(rs_dict['q'][height_limit], rs_dict['pres'][height_limit]*100, nan_threshold=0.1)

    return rs_dict, attribute_info
