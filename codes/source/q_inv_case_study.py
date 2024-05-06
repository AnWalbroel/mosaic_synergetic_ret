def run_q_inv_case_study(path_data, path_plots):

    """
    In this script, specific humidity profiles from MOSAiC observations (radiosondes, MWRs) will
    be compared to those from satellite observations (i.e., IASI) and reanalyses (i.e., ERA5) for
    certain case studies. Polarstern track data is not needed because the geo location of the 
    radiosonde launch is used.
    - import MOSAiC observations and cut to the case study time(s)
    - import reanalysis data and cut to case study time(s) and location(s)
    - import satellite data and cut to case study time(s) and location(s)
    - visualize

    Parameters:
    -----------
    path_data : dict
        Dictionary containing strings of the full paths of the synergetic retrieval output based on
        MOSAiC observations (key 'mwr_syn'), of the HATPRO-only retrievals (key 'mwr_hat'), MOSAiC
        level 2 radiosondes ('rs') and ERA5 on model levels ('era5_m').
    path_plots : str
        String indicating where to save the plots.
    """

    import os
    import sys
    import glob
    import datetime as dt
    import pdb
    import gc

    import numpy as np
    import xarray as xr
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    from import_data import (import_radiosonde_daterange, import_hatpro_mirac_level2b_daterange_pangaea,
                            import_hatpro_level2b_daterange_pangaea, )
    from met_tools import convert_abshum_to_spechum


    # font and marker sizes
    fs = 24
    fs_small = fs - 2
    fs_dwarf = fs_small - 2
    fs_micro = fs_dwarf - 2
    msize = 7.0

    # colours:
    c_S = (1,0.72,0.12)         # synergetic retrieval
    c_S_fade = (1,0.72,0.12,0.15)# synergetic ret, but less alpha
    c_H = (0.067,0.29,0.769)    # MWR_PRO (HATPRO)
    c_G = (0.2,0.2,0.2)         # for grey scale (i.e., profiles)
    c_G_fade = (0.2,0.2,0.2,0.2)
    c_I = (1,0.46,0.98)
    c_M2 = (0,0.616,0.678)      # for MERRA2
    c_M2_fade = (0,0.616,0.678,0.2)

    g = 9.80665                 # gravitational acceleration in m s**-2


    def spec_hum_all_data(
        syn_q, rs_q,
        hat_q, era5_m_q, 
        syn_z, rs_z,
        hat_z, era5_m_z,
        set_dict):

        """
        Plots the specific humidity profile of each non-null dataset for the selected case study
        as lines. Specific humidity is converted to g kg**-1.

        Parameters:
        -----------
        syn_q : array of floats
            Specific humidity (in g kg**-1) predicted by the synergetic Neural Network retrieval 
            (NN_retrieval.py).
        rs_q : array of floats
            Level 2 MOSAiC radiosonde specific humidity (in g kg**-1).
        hat_q : array of floats
            Specific humidity (in g kg**-1) predicted by HATPRO with MWR_PRO, as published on 
            https://doi.org/10.1594/PANGAEA.941389 .
        era5_m_q : array of floats
            Specific humidity (in g kg**-1) from ERA5 on model levels, processed with 
            ERA5_process_model_levels.py.
        syn_z : array of floats
            Height grid of the synergetic Neural Network retrieval in m.
        rs_z : array of floats
            Height grid of the vertically interpolated level 2 MOSAiC radiosonde data in m.
        hat_z : array of floats
            Height grid of the HATPRO MWR_PRO retrieval published on 
            https://doi.org/10.1594/PANGAEA.941389 in m.
        era5_m_z : array of floats
            Height grid of the ERA5 model level data (geopotential height) in m.
        set_dict : dict
            Dictionary containing additional information.
        """

        # create output path if not existing:
        plotpath_dir = os.path.dirname(set_dict['path_plots'])
        if not os.path.exists(plotpath_dir):
            os.makedirs(plotpath_dir)


        # date string for plot:
        date_str0 = dt.datetime.strptime(set_dict['cases'][0], "%Y-%m-%dT%H:%M").strftime("%d %B %Y, %H:%M UTC")
        date_str1 = dt.datetime.strptime(set_dict['cases'][1], "%Y-%m-%dT%H:%M").strftime("%d %B %Y, %H:%M UTC")


        f1 = plt.figure(figsize=(10,6.5))
        a1 = plt.subplot2grid((1,2), (0,0))
        a2 = plt.subplot2grid((1,2), (0,1))

        y_lim = np.array([0.0, 4000.0]) # in m
        max_q = np.ceil(np.nanmax(np.concatenate((era5_m_q.ravel(), hat_q.ravel(), rs_q.ravel(), 
                                                syn_q.ravel()))) * 2)/2
        x_lim = np.array([0.0, max_q])      # in g kg-1


        # plot profiles:
        for k, ax in enumerate([a1, a2]):

            ax.plot(era5_m_q[k,k,k,:], era5_m_z[k,k,k,:], color=c_G, linestyle='dashed', linewidth=1.5, label='ERA5')
            if set_dict['incl_hatpro']:
                ax.plot(hat_q[k,:], hat_z, color=c_H, linewidth=1.75, label='HATPRO')
            ax.plot(syn_q[k,:], syn_z, color=c_S, linewidth=1.75, label='Synergy')
            ax.plot(rs_q[k,:], rs_z, color=(0,0,0), linewidth=2, label='Radiosonde')


        # legends:
        lh, ll = a1.get_legend_handles_labels()
        a1.legend(lh, ll, loc='upper right', bbox_to_anchor=(0.98, 0.99), fontsize=fs_micro-4, framealpha=0.5)

        # axis lims:
        for ax in [a1, a2]:
            # ax.set_xlim(x_lim[0], x_lim[1])
            ax.set_xlim(x_lim[0],2.5)
            ax.set_ylim(y_lim[0], y_lim[1])
            ax.minorticks_on()

            # set ticks and tick labels and parameters; grid; labels:
            ax.tick_params(axis='both', labelsize=fs_micro-4)
            ax.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
            ax.set_xlabel("Specific humidity ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs_dwarf-4)


        # labels:
        a1.set_ylabel("Height (m)", fontsize=fs_dwarf)
        a1.set_title(f"a) {date_str0}", fontsize=fs_micro-4)
        a2.set_title(f"b) {date_str1}", fontsize=fs_micro-4)


        if set_dict['save_figures']:
            plotname = f"MOSAiC_mwr_sonde_reanalysis_q_prof_case_{set_dict['case'][:10]}"
            plotfile = plotpath_dir + "/" + plotname
            f1.savefig(plotfile + ".png", dpi=300, bbox_inches='tight')
            f1.savefig(plotfile + ".pdf", bbox_inches='tight')
            print(f"Saved {plotfile}....")
        else:
            plt.show()
            pdb.set_trace()

        plt.close()
        gc.collect()


    # settings:
    set_dict = {'save_figures': True,
                'path_plots': path_plots,
                'incl_hatpro': True,            # if True, HATPRO spec humidity profiles will be plotted as well; only possible if incl_merra2 is false
                'lw': 900,                      # time window (in sec) from sonde_launch_time to sonde_launch_time+lw for averaging
                }

    set_dict['cases'] = ['2019-11-20T05:05', '2019-11-20T10:49']    # case study radiosonde launch times in yyyy-mm-ddTHH:MM
    set_dict['case'] = set_dict['cases'][0]


    # import MOSAiC radiosondes:
    set_dict['date_0'] = set_dict['cases'][0][:10]
    set_dict['date_1'] = "2020-01-01"
    sonde_dict = import_radiosonde_daterange(path_data['rs'], set_dict['date_0'], set_dict['date_1'], s_version='level_2', remove_failed=True)

    # put into dataset:
    RS_DS_all = xr.Dataset(coords={'height': (['height'], sonde_dict['height'][0,:]),
                            'launch_time': (['launch_time'], sonde_dict['launch_time'].astype('datetime64[s]').astype('datetime64[ns]'))})
    RS_DS_all['temp'] = xr.DataArray(sonde_dict['temp'], dims=['launch_time', 'height'])
    RS_DS_all['rh'] = xr.DataArray(sonde_dict['rh'], dims=['launch_time', 'height'])
    RS_DS_all['q'] = xr.DataArray(sonde_dict['q'], dims=['launch_time', 'height'], attrs={'units': "kg kg**-1"})
    RS_DS_all['pres'] = xr.DataArray(sonde_dict['pres'], dims=['launch_time', 'height'], attrs={'units': "Pa"})
    RS_DS_all['lat'] = xr.DataArray(sonde_dict['lat'], dims=['launch_time'], attrs={'units': "deg north"})
    RS_DS_all['lon'] = xr.DataArray(sonde_dict['lon'], dims=['launch_time'], attrs={'units': "deg east"})
    RS_DS_all['iwv'] = xr.DataArray(sonde_dict['iwv'], dims=['launch_time'], attrs={'units': "kg m**-2"})


    # import retrieved MOSAiC MWR data:
    SYN_DS_all = import_hatpro_mirac_level2b_daterange_pangaea(path_data['mwr_syn'], set_dict['date_0'], set_dict['date_1'], 
                                                        which_retrieval='q', data_version='v00',
                                                        around_radiosondes=True, path_radiosondes=path_data['rs'], 
                                                        s_version='level_2', mwr_avg=set_dict['lw'])
    SYN_DS_all['time'] = SYN_DS_all.time.astype('datetime64[s]').astype('datetime64[ns]')


    # reduce dataset to current case study:
    RS_DS = RS_DS_all.sel(launch_time=set_dict['cases'], method='nearest')
    SYN_DS = SYN_DS_all.sel(time=set_dict['cases'], method='nearest')


    HAT_DS = xr.Dataset()
    if set_dict['incl_hatpro']:
        hatpro_dict = import_hatpro_level2b_daterange_pangaea(path_data['mwr_hat'], set_dict['date_0'], set_dict['date_1'],
                                                                which_retrieval='both', around_radiosondes=True,
                                                                path_radiosondes=path_data['rs'], s_version='level_2', 
                                                                mwr_avg=set_dict['lw'])                             # for case '2019-12-27T22:51', use mwr_avg=3600
        hatpro_dict['time_npdt'] = hatpro_dict['time'].astype("datetime64[s]")


        # filter bad flags:
        idx_ok = np.where((hatpro_dict['flag'] == 0) | (hatpro_dict['flag'] == 32))[0]
        for h_k in ['time', 'time_npdt', 'flag']: hatpro_dict[h_k] = hatpro_dict[h_k][idx_ok]
        for h_k in ['hua', 'ta']: hatpro_dict[h_k] = hatpro_dict[h_k][idx_ok,:]


        # compute avg and std over indices around sonde launches:
        hatson_idx = [np.argwhere((hatpro_dict['time_npdt'] >= lt) & 
                        (hatpro_dict['time_npdt'] < lt+np.timedelta64(set_dict['lw'], "s"))).flatten() for lt in RS_DS_all.launch_time.values]      # for case '2019-12-27T22:51', use (3600, "s")
        n_height_hatpro = len(hatpro_dict['height'])
        n_sondes = len(RS_DS_all.launch_time)
        for hat_ret_key in ['hua', 'ta']:
        # if which_retrieval_keys[predictand] in ['hua', 'ta']:
            hatpro_dict[f'{hat_ret_key}_mean_sonde'] = np.full((n_sondes,n_height_hatpro), np.nan)
            hatpro_dict[f'{hat_ret_key}_stddev_sonde'] = np.full((n_sondes,n_height_hatpro), np.nan)

            for k, hat in enumerate(hatson_idx):
                hatpro_dict[f'{hat_ret_key}_mean_sonde'][k,:] = np.nanmean(hatpro_dict[hat_ret_key][hat,:], axis=0)
                hatpro_dict[f'{hat_ret_key}_stddev_sonde'][k,:] = np.nanstd(hatpro_dict[hat_ret_key][hat,:], axis=0)

        # form dataset:
        HAT_DS_all = xr.Dataset(coords={'time': (['time'], RS_DS_all.launch_time.values),
                                    'height': (['height'], hatpro_dict['height'])})
        HAT_DS_all['temp_avg'] = xr.DataArray(hatpro_dict['ta_mean_sonde'], dims=['time', 'height'], attrs={'units': "K"})
        HAT_DS_all['rho_v_avg'] = xr.DataArray(hatpro_dict['hua_mean_sonde'], dims=['time', 'height'], attrs={'units': "kg m**-3"})


        # convert to specific humidity:
        # init array, then loop over sondes for 1D interpolation:
        sonde_dict["pres_ret"] = np.full((n_sondes, len(hatpro_dict['height'])), np.nan)
        for k in range(n_sondes):
            sonde_dict["pres_ret"][k,:] = np.interp(hatpro_dict['height'], sonde_dict['height'][k,:], sonde_dict['pres'][k,:])
        HAT_DS_all['q_avg'] = xr.DataArray(convert_abshum_to_spechum(HAT_DS_all.temp_avg.values, sonde_dict['pres_ret'], HAT_DS_all.rho_v_avg.values),
                                        dims=['time', 'height'], attrs={'units': "kg kg**-1"})


        # select case study time stamp:
        HAT_DS = HAT_DS_all.sel(time=set_dict['cases'], method='nearest')



        hat_file = sorted(glob.glob(path_data['mwr_hat'] + 
                                    f"ioppol_tro_mwr00_l2_hua_v01_{set_dict['case'][:10].replace('-','')}*.nc"))[0]
        hat_file_t = sorted(glob.glob(path_data['mwr_hat'] + 
                                    f"ioppol_tro_mwr00_l2_ta_v01_{set_dict['case'][:10].replace('-','')}*.nc"))[0]
        HAT_DS_all = xr.open_dataset(hat_file)
        HAT_DS_all_temp = xr.open_dataset(hat_file_t)
        HAT_DS_all['temp'] = HAT_DS_all_temp['ta']

        HAT_DS_all_temp = HAT_DS_all_temp.close()
        del HAT_DS_all_temp

        # remove values where flag > 0:
        HAT_DS_all['flag'][np.isnan(HAT_DS_all['flag'])] = 0.
        idx_ok = ((HAT_DS_all.flag == 0) | (HAT_DS_all.flag == 32))
        HAT_DS_all = HAT_DS_all.isel(time=idx_ok)


        # rename some variables and compute specific humidity:
        HAT_DS_all = HAT_DS_all.rename({'hua': 'rho_v'})

        # init array, then loop over sondes for 1D interpolation:
        n_sondes = len(RS_DS.launch_time)
        sonde_pres_ret = np.full((n_sondes, len(hatpro_dict['height'])), np.nan)
        for k in range(n_sondes):
            sonde_pres_ret[k,:] = np.interp(HAT_DS_all['height'].values, RS_DS['height'].values, RS_DS['pres'][k,:].values)
        sonde_pres_ret = xr.DataArray(sonde_pres_ret, dims=['launch_time', 'height'], 
                                        coords={'launch_time': RS_DS.launch_time, 'height': HAT_DS_all.height})

        # interpolate sonde_pres_ret to hatpro time axis:
        sonde_pres_ret_ip = sonde_pres_ret.interp(launch_time=HAT_DS_all.time)
        sonde_pres_ret_ip = sonde_pres_ret_ip.bfill(dim='time')
        sonde_pres_ret_ip = sonde_pres_ret_ip.ffill(dim='time')

        # compute spec humidity:
        HAT_DS_all['q_avg'] = xr.DataArray(convert_abshum_to_spechum(HAT_DS_all.temp.values, sonde_pres_ret_ip, HAT_DS_all.rho_v.values),
                                        dims=['time', 'height'], attrs={'units': "kg kg**-1"})


        # theoretically redundant selection if only one file has been searched for in hat_file:
        HAT_DS = HAT_DS_all
        del hatpro_dict

    del sonde_dict


    # import reanalyses data and select time closest to case study time; also select closest grid point:
    file_era5_m = sorted(glob.glob(path_data['era5_m'] + "MOSAiC_ERA5_model_levels_temp_q_*.nc"))
    file_era5_m = [file for file in file_era5_m if set_dict['case'][:10].replace("-","") in file]
    ERA5_M_DS = xr.Dataset()
    if len(file_era5_m) == 1:
        ERA5_M_DS_all = xr.open_dataset(file_era5_m[0])
        ERA5_M_DS = ERA5_M_DS_all.sel(time=set_dict['cases'], latitude=RS_DS.lat.values, 
                                    longitude=RS_DS.lon.values, method='nearest')


    # visualize:
    # convert specific humidity from kg kg**-1 to g kg**-1
    syn_q = SYN_DS.q.values*1000.
    rs_q = RS_DS.q.values*1000.
    hat_q = np.array([])
    if set_dict['incl_hatpro']: hat_q = (HAT_DS.q_avg.values*1000.).squeeze()
    era5_m_q = ERA5_M_DS.q.values*1000.


    # height grids of the datasets in m:
    syn_z = SYN_DS.height.values
    rs_z = RS_DS.height.values
    hat_z = np.array([])
    if set_dict['incl_hatpro']: hat_z = HAT_DS.height.values
    era5_m_z = ERA5_M_DS.Z.values


    spec_hum_all_data(syn_q, rs_q, hat_q, era5_m_q, syn_z, rs_z, hat_z, era5_m_z, set_dict)
