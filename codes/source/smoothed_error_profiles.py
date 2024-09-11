def run_smoothed_error_profiles(
    path_data, 
    path_plots):

    """
    In this script, retrieved specific humidity from MWRs will be compared to smoothed radiosonde 
    profiles. The smoothing is performed with the Averaging Kernel computed with info_content.py.
    - import MOSAiC obs and predictions for chosen predictand
    - process imported data
    - visualize

    Parameters:
    -----------
    path_data : dict
        Dictionary containing strings of the full paths of the synergetic retrieval data 
        (key 'nn_syn_mosaic'), HATPRO retrievals ('mwr_pro'), level 2 MOSAiC radiosondes ('radiosondes'),
        and the information content estimation output ('i_cont').
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

    from data_tools import compute_RMSE_profile
    from import_data import (import_hatpro_mirac_level2b_daterange_pangaea, import_hatpro_level2b_daterange_pangaea,
                            import_radiosonde_daterange)
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
    c_H_fade = (0.067,0.29,0.769,0.15)  # MWR_PRO (HATPRO)


    def filter_data_mosaic(
        DS,
        sonde_dict,
        HAT_DS=xr.Dataset()):

        """
        Postprocessing of MOSAiC MWR predictions and radiosondes. Potentially faulty values that 
        have been flagged will be removed. Additionally, times before the first successful 
        calibration and within Exclusive Economic Zones are removed.

        Parameters:
        -----------
        DS : xarray dataset
            Dataset containing predictions made with NN_retrieval.py.save_obs_predictions, imported 
            with import_hatpro_mirac_level2a_daterange_pangaea or 
            import_hatpro_mirac_level2b_daterange_pangaea or ...level2c....
        sonde_dict : dictionary
            Dictionary containing MOSAiC radiosonde level 2 data imported with 
            import_radiosonde_daterange.
        HAT_DS : xarray dataset
            Dataset containing predictions from HATPRO's MWR_PRO retrieval.
        """

        # Flag data using the HATPRO and MiRAC-P flags:
        if set_dict['pred'] not in ['temp', 'q', 'temp_bl', 'rh']: 
            idx_ok = np.where(((DS.flag_h == 0.0) | (DS.flag_h == 32.0)) & (DS.flag_m == 0.0))[0]
            DS = DS.isel(time=idx_ok)
        DS['time_npdt'] = DS.time.astype('datetime64[s]')


        # Remove values before the first MWR calibration:
        calibration_times_HATPRO = [np.datetime64("2019-10-19T06:00:00"), np.datetime64("2019-12-14T18:30:00"),
                                    np.datetime64("2020-03-01T11:00:00"), np.datetime64("2020-05-02T12:00:00"),
                                    np.datetime64("2020-07-06T09:33:00"), np.datetime64("2020-08-12T09:17:00")]
        calibration_times_MiRAC = [np.datetime64("2019-10-19T06:30:00"), np.datetime64("2019-10-22T05:40:00"),
                                    np.datetime64("2020-07-06T12:19:00"), np.datetime64("2020-08-12T09:37:00")]

        # find time index when both radiometers were successfully (!) calibrated for the first time:
        first_calib_HAT_MIR = np.array([calibration_times_HATPRO[0], calibration_times_MiRAC[1]]).max()
        DS = DS.sel(time=slice(first_calib_HAT_MIR.astype(np.float64),None))    # faster than isel for simple masks


        # Remove values within the Exclusive Economic Zones (EEZs):
        EEZ_periods_npdt = {'range0': [np.datetime64("2020-06-03T20:36:00"), np.datetime64("2020-06-08T20:00:00")],
                            'range1': [np.datetime64("2020-10-02T04:00:00"), np.datetime64("2020-10-02T20:00:00")],
                            'range2': [np.datetime64("2020-10-03T03:15:00"), np.datetime64("2020-10-04T17:00:00")]}

        # find when MWR or sonde time is outside of EEZ periods:
        outside_eez = dict()
        outside_eez['mwr'] = np.full((len(DS.time),), True)
        for EEZ_range in EEZ_periods_npdt.keys():
            outside_eez['mwr'][(DS.time_npdt >= EEZ_periods_npdt[EEZ_range][0]) &
                                    (DS.time_npdt <= EEZ_periods_npdt[EEZ_range][1])] = False

        # limit data to time outside of EEZs:
        DS = DS.isel(time=outside_eez['mwr'])


        # also limit radiosondes to times after the calibration:
        idx_calib_ok_sonde = np.where(sonde_dict['launch_time_npdt'] >= first_calib_HAT_MIR)[0]
        for sk in sonde_dict.keys(): sonde_dict[sk] = sonde_dict[sk][idx_calib_ok_sonde,...]
        set_dict['n_sondes'] = len(sonde_dict['launch_time'])

        # also filter EEZ periods:
        outside_eez['sonde'] = np.full((set_dict['n_sondes'],), True)
        for EEZ_range in EEZ_periods_npdt.keys():
            outside_eez['sonde'][(sonde_dict['launch_time_npdt'] >= EEZ_periods_npdt[EEZ_range][0]) & 
                                    (sonde_dict['launch_time_npdt'] <= EEZ_periods_npdt[EEZ_range][1])] = False
        for sk in sonde_dict.keys(): sonde_dict[sk] = sonde_dict[sk][outside_eez['sonde'],...]


        # if boundary layer dataset is loaded, repeat data filtering:
        if HAT_DS:
            HAT_DS['time_npdt'] = HAT_DS.time.astype('datetime64[s]')

            # filter bad flags:
            ok_idx = np.where((HAT_DS['flag'] == 0.0) | (HAT_DS['flag'] == 32.0))[0]
            HAT_DS = HAT_DS.isel(time=ok_idx)

            HAT_DS = HAT_DS.sel(time=slice(first_calib_HAT_MIR.astype(np.float64),None))
            outside_eez['hat'] = np.full((len(HAT_DS.time),), True)
            for EEZ_range in EEZ_periods_npdt.keys():
                outside_eez['hat'][(HAT_DS.time_npdt >= EEZ_periods_npdt[EEZ_range][0]) &
                                        (HAT_DS.time_npdt <= EEZ_periods_npdt[EEZ_range][1])] = False
            HAT_DS = HAT_DS.isel(time=outside_eez['hat'])

        return DS, sonde_dict, HAT_DS


    def conv_units(
        DS,
        set_dict,
        varname):

        """
        Convert units based on a dictionary which contains the offset as first and the factor as second
        value for desired predictands.

        Parameters:
        -----------
        DS : xarray dataset
            Dataset created with NN_retrieval.py containing predicted and reference data.
        set_dict : dict
            Dictionary containing additional information.
        varname : str
            String indicating the name of the variable of which to convert units.
        """
        
        u_c_d = set_dict['unit_conv_dict'][varname]
        DS[varname] = (DS[varname] + u_c_d[0])*u_c_d[1]

        return DS


    def sonde_to_ret_grid(
        sonde_dict,
        height_ret):

        """
        Interpolate profile information of radiosondes to the retrieval grid given in the prediction
        dataset.

        Parameters:
        -----------
        sonde_dict : dict
            Dictionary containing radiosonde observation data imported with 
            import_radiosonde_daterange.
        height_ret : array of floats
            Height grid of the Neural Network retrieval.
        """

        height_vars = [key for key in sonde_dict.keys() if len(sonde_dict[key].shape) == 2]

        # loop through height variables:
        n_sondes = len(sonde_dict['launch_time'])
        n_hgt_ret = len(height_ret)
        for var in height_vars:
            if var != 'height':

                # init array, then loop over sondes for 1D interpolation:
                sonde_dict[var + "_ret"] = np.full((n_sondes, n_hgt_ret), np.nan)
                for k in range(n_sondes):
                    sonde_dict[var + "_ret"][k,:] = np.interp(height_ret, sonde_dict['height'][k,:], sonde_dict[var][k,:])

        sonde_dict['height_ret'] = height_ret

        return sonde_dict


    def overlap_with_radiosondes(
        DS,
        sonde_dict,
        set_dict):

        """
        Find temporal overlap between microwave radiometer observations and radiosondes. Based on the
        variable, different overlap windows are selected (i.e., wider for boundary layer temperature
        profiles bc of reduced sampling frequency). Averages and standard deviations for the predicted 
        values around each radiosonde launch will be computed.

        Parameters:
        -----------
        DS : xarray dataset
            Dataset containing predictions made with NN_retrieval.py.save_obs_predictions, imported 
            with import_hatpro_mirac_level2a_daterange_pangaea or 
            import_hatpro_mirac_level2b_daterange_pangaea or ...level2c....
        sonde_dict : dictionary
            Dictionary containing MOSAiC radiosonde level 2 data imported with 
            import_radiosonde_daterange.
        set_dict : dict
            Dictionary containing additional information.
        """

        mwrson_idx = list()
        ds_time = DS.time_npdt.values
        last_overlap_idx = 0        # index to reduce searching time
        for lt in sonde_dict['launch_time_npdt']:
            overlap_idx = np.argwhere((ds_time >= lt) & (ds_time < lt+np.timedelta64(set_dict['lw'], "s"))).flatten() + last_overlap_idx
            mwrson_idx.append(overlap_idx)

            # remove times that were already checked (irrelevant for the upcoming radiosondes):
            if len(overlap_idx) > 0:
                ds_time = ds_time[(overlap_idx[-1]-last_overlap_idx):]
                last_overlap_idx = overlap_idx[-1]


        # compute average and std dev over the launch window for each radiosonde launch:
        if set_dict['pred'] in ['temp', 'q', 'rh']:

            for dv in ['temp', 'q', 'rh']:
                if dv in DS.data_vars:
                    DS[f"{dv}_avg"] = xr.DataArray(np.full((set_dict['n_sondes'], len(DS.height)), np.nan), dims=['sondes', 'height'])
                    DS[f"{dv}_std"] = xr.DataArray(np.full((set_dict['n_sondes'], len(DS.height)), np.nan), dims=['sondes', 'height'])
                    for k, msi in enumerate(mwrson_idx):
                        if len(msi) > 0:
                            DS[f"{dv}_avg"][k,:] = DS[f"{dv}"][msi,:].mean('time')
                            DS[f"{dv}_std"][k,:] = DS[f"{dv}"][msi,:].std('time')

        return DS


    def compute_error_stats(
        prediction, 
        predictand, 
        predictand_id,
        height=np.array([])):

        """
        Compute error statistics (Root Mean Squared Error (rmse), bias, Standard Deviation (stddev))
        between prediction and (test data) predictand. Height must be provided if prediction or respective
        predictand is a profile. The prediction_id describes the prediction and predictand and must also
        be forwarded to the function.

        Parameters:
        -----------
        prediction : array of floats
            Predicted variables also available in predictand, predicted by the Neural Network.
        predictand : array of floats
            Predictand data as array, used as evaluation reference. Likely equals the attribute 
            'output' of the predictand class object.
        predictand_id : str
            String indicating which output variable is forwarded to the function.
        height : array of floats
            Height array for respective predictand or predictand profiles (of i.e., temperature or 
            humidity). Can be a 1D or 2D array (latter must be of shape (n_training,n_height)).
        """

        error_dict = dict()

        # x: reference; y: prediction
        x_stuff = predictand
        y_stuff = prediction

        # Compute statistics:
        if predictand_id in ['temp', 'temp_bl', 'q', 'rh']:

            if len(height) == 0:
                raise ValueError("Please specify a height variable to estimate error statistics for profiles.")

            # filter for nans:
            no_nan_idx = np.where(~np.isnan(x_stuff[:,0]) & ~np.isnan(y_stuff[:,0]))[0]
            x_stuff = x_stuff[no_nan_idx,:]
            y_stuff = y_stuff[no_nan_idx,:]

            # Compute statistics for entire profile:
            error_dict['rmse_tot'] = compute_RMSE_profile(y_stuff, x_stuff, which_axis=0)
            error_dict['bias_tot'] = np.nanmean(y_stuff - x_stuff, axis=0)
            error_dict['stddev'] = compute_RMSE_profile(y_stuff - error_dict['bias_tot'], x_stuff, which_axis=0)

            # Don't only compute bias, stddev and rmse for entire profile, but also give summary for 
            # bottom, mid and top range (height related in this case):
            range_dict = {  'bot': [0., 1500.0],
                            'mid': [1500.0, 5000.0],
                            'top': [5000.0, 15000.0]}
            if height.ndim == 2:
                height = height[0,:]

            mask_range = dict()
            for range_id in range_dict.keys():
                mask_range[range_id] = ((height >= range_dict[range_id][0]) & (height < range_dict[range_id][1]))
                error_dict[f"rmse_{range_id}"] = np.nanmean(error_dict['rmse_tot'][mask_range[range_id]])
                error_dict[f"stddev_{range_id}"] = np.nanmean(error_dict['stddev'][mask_range[range_id]])
                error_dict[f"bias_{range_id}"] = np.nanmean(error_dict['bias_tot'][mask_range[range_id]])


            # compute relative errors (might be useful for q profile):
            x_mean = np.nanmean(x_stuff, axis=0)
            error_dict['rmse_tot_rel'] = error_dict['rmse_tot'] / x_mean
            error_dict['bias_tot_rel'] = error_dict['bias_tot'] / x_mean
            error_dict['stddev_rel'] = error_dict['stddev'] / x_mean

        return error_dict


    def bias_rmse_prof(
        sonde_dict,
        NN_MOS_DS,
        HAT_DS,
        ret_stats,
        set_dict,
        height):

        """
        Visualize the performance of the synergetic Neural Network prediction against a reference 
        predictand (evaluation data set or MOSAiC radiosondes). Here, profiles of bias and
        standard deviation between predictions and reference predictand will be plotted. 

        Parameters:
        -----------
        sonde_dict : dictionary
            Dictionary containing MOSAiC radiosonde level 2 data imported with 
            import_radiosonde_daterange.
        NN_MOS_DS : xarray dataset
            Dataset containing predictions made with NN_retrieval.py.save_obs_predictions, imported 
            with import_hatpro_mirac_level2a_daterange_pangaea or 
            import_hatpro_mirac_level2b_daterange_pangaea or ...level2c....
        HAT_DS : xarray dataset
            Dataset containing predictions from HATPRO's MWR_PRO retrieval.
        ret_stats : dict
            Dictionary containing error estimates (bias, rmse, standar deviation) for various data
            or height ranges for each retrieval type (keys of ret_stats). Each key represents the 
            output of compute_error_stats.
        set_dict : dict
            Dictionary containing additional information.
        height : array of floats or xarray DataArray
            Height grid of the predicted or reference predictand. Should be a 1D array (n_height,) or
            2D array (n_samples, n_height). In the latter case, the variation over the sample dimension
            is neglected.
        """

        # reduce unnecessary dimensions of height:
        if height.ndim == 2:
            height = height[0,:]


        # write out the important error stats: 
        STD_mos_op = ret_stats['mos']['stddev']     # std of synergy with respect to RS profile smoothed with synergy AK
        BIAS_mos_op = ret_stats['mos']['bias_tot']

        STD_hat = ret_stats['hat']['stddev']        # std of HATPRO with respect to RS profile smoothed with K-band AK
        BIAS_hat = ret_stats['hat']['bias_tot']

        STD_hat_syn = ret_stats['hat_syn']['stddev']        # std of HATPRO with respect to RS profile smoothed with synergy AK
        BIAS_hat_syn = ret_stats['hat_syn']['bias_tot']


        # dictionaries for adaptations:
        label_size_big = fs_dwarf
        label_size_small = fs_micro-4

        panel_id_dict = dict()      # dictionary containing further properties of the figure panel identifiers
        legend_pos = 'upper right'
        anchor_pos = (0.98, 1)
        bias_label = {'q': "$\mathrm{Bias}_{\mathrm{q}}$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)",
                        }
        std_label = {'q': "$\mathrm{RMSE}_{\mathrm{corr, q}}$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)",
                        }
        rel_bias_label = {'q': "Relative $\mathrm{bias}_{\mathrm{q}}$ (%)",
                        }
        rel_std_label = {'q': "Relative $\mathrm{RMSE}_{\mathrm{corr, q}}$ (%)",
                        }


        f1 = plt.figure(figsize=(15,6.5))
        ax_bias = plt.subplot2grid((1,3), (0,0))            # bias profile
        ax_std = plt.subplot2grid((1,3), (0,1))             # std dev profile
        ax_ex = plt.subplot2grid((1,3), (0,2))              # example

        y_lim = np.array([0.0, height.max()])
        x_lim_std = {'q': np.array([0.0, 0.45]),        # in g kg-1,
                    }
        x_lim_bias = {'q': np.array([-0.25, 0.5]),  # in g kg-1
                    }
        x_lim_ex = np.array([0.0, 1.0])         # in g kg-1


        # bias profiles:
        ax_bias.plot(np.full_like(height, 0.0), height, color=(0,0,0), linewidth=1.0)   # helper line
        ax_bias.plot(BIAS_hat, height, color=c_H, linewidth=2)
        ax_bias.plot(BIAS_mos_op, height, color=c_S, linewidth=2)
        ax_bias.plot(BIAS_hat_syn, height, color=c_H, linewidth=1.5, linestyle='dotted')

        # std dev profiles:
        ax_std.plot(STD_hat, height, color=c_H, linewidth=2, label='HATPRO*')
        ax_std.plot(STD_mos_op, height, color=c_S, linewidth=2, label='Synergy**')
        ax_std.plot(STD_hat_syn, height, color=c_H, linewidth=1.5, linestyle='dotted', label='HATPRO**')

        # example:
        example_time = np.datetime64("2019-12-27T10:50:04")
        ex = np.argmin(np.abs(sonde_dict['launch_time_npdt'] - example_time))
        ax_ex.plot(sonde_dict['q'][ex,:], sonde_dict['height'][ex,:], color=(0,0,0), linewidth=1.5, label='Radiosonde')
        ax_ex.plot(sonde_dict['q_sm_K'][ex,:], sonde_dict['height_ret'], color=(0,0,0), linewidth=1.5, linestyle='dotted',
                    label='Radiosonde, smoothed\nwith AK$_{\mathrm{K}}$')
        ax_ex.plot(sonde_dict['q_sm'][ex,:], sonde_dict['height_ret'], color=(0,0,0), linewidth=1.75, linestyle='dashed',
                    label='Radiosonde, smoothed\nwith AK$_{\mathrm{all}}$')
        ax_ex.plot(HAT_DS['q_avg'][ex,:], HAT_DS.height, color=c_H, linewidth=1.5, label='HATPRO')
        ax_ex.plot(NN_MOS_DS['q_avg'][ex,:], NN_MOS_DS.height, color=c_S, linewidth=1.75, label='Synergy')


        # add relative errors (e.g., for q profiles):
        rel_plots = False           # boolean indicator if relative plots are added
        # if set_dict['pred'] == 'q':
            # rel_plots = True
            # ax_bias2 = ax_bias.twiny()
            # ax_bias2.plot(ret_stats['hat']['bias_tot_rel']*100., height, color=c_H, linestyle='dashed', linewidth=1.5)
            # ax_bias2.plot(ret_stats['mos']['bias_tot_rel']*100., height, color=c_S, linestyle='dashed', linewidth=1.5)

            # ax_std2 = ax_std.twiny()
            # ax_std2.plot(ret_stats['hat']['stddev_rel']*100., height, color=c_H, linestyle='dashed', linewidth=1.5, label='HATPRO')
            # ax_std2.plot(ret_stats['mos']['stddev_rel']*100., height, color=c_S, linestyle='dashed', linewidth=1.5, label='Synergy')


        # legends:
        lh, ll = ax_std.get_legend_handles_labels()
        if rel_plots:
            # also adapt label font sizes:
            label_size_big = fs_dwarf-2
            panel_id_dict['y'] = 1.065

            leg1 = ax_bias.legend(lh, ll, loc=legend_pos, bbox_to_anchor=anchor_pos, 
                            fontsize=label_size_small, framealpha=0.5, title='Absolute errors', title_fontsize=label_size_small)

            # add second legend:
            lh, ll = ax_std2.get_legend_handles_labels()
            leg2 = ax_bias2.legend(lh, ll, loc=legend_pos, bbox_to_anchor=(anchor_pos[0], 0.74),
                            fontsize=label_size_small, framealpha=0.5, title='Relative errors', title_fontsize=label_size_small)

            # adapt axis properties:
            ax_bias2.set_xlim(-20.,20.)
            ax_std2.set_xlim(0., 60.)
            for ax in [ax_bias2, ax_std2]: ax.tick_params(axis='both', labelsize=label_size_small)
            ax_bias2.set_xlabel(rel_bias_label[set_dict['pred']], fontsize=label_size_big, labelpad=8)
            ax_std2.set_xlabel(rel_std_label[set_dict['pred']], fontsize=label_size_big, labelpad=8)

            # explaining text explaining the HATPRO dotted lines:
            ax_bias.text(0.01, 0.5, "* Error with respect to\nradiosonde profile\nsmoothed with AK$_{\mathrm{K}}$\n" +
                            "** Error with respect to\nradiosonde profile\nsmoothed with AK$_{\mathrm{all}}$", ha='left', 
                            va='center', fontsize=label_size_small-4, rotation=90, transform=ax_bias.transAxes)

        else:
            leg1 = ax_std.legend(lh, ll, loc=legend_pos, bbox_to_anchor=anchor_pos, 
                            fontsize=label_size_small, framealpha=0.5)

            # explaining text explaining the HATPRO dotted lines:
            ax_std.text(0.575, 0.79, "* Error with respect to\nradiosonde profile\nsmoothed with AK$_{\mathrm{K}}$\n" +
                            "** Error with respect to\nradiosonde profile\nsmoothed with AK$_{\mathrm{all}}$", ha='left', 
                            va='top', fontsize=label_size_small-4, transform=ax_std.transAxes)

        lh, ll = ax_ex.get_legend_handles_labels()
        leg3 = ax_ex.legend(lh, ll, loc=legend_pos, bbox_to_anchor=anchor_pos, fontsize=label_size_small, framealpha=0.5)


        # axis lims:
        ax_bias.set_xlim(x_lim_bias[set_dict['pred']])
        ax_std.set_xlim(x_lim_std[set_dict['pred']])
        ax_ex.set_xlim(x_lim_ex)


        for ax in [ax_bias, ax_std, ax_ex]:
            ax.set_ylim(bottom=y_lim[0], top=y_lim[1])

            ax.minorticks_on()
            ax.tick_params(axis='both', labelsize=label_size_small)
            ax.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

        # remove tick labels for y axis:
        ax_std.yaxis.set_ticklabels([])
        ax_ex.yaxis.set_ticklabels([])


        # labels:
        ax_bias.set_xlabel(bias_label[set_dict['pred']], fontsize=label_size_big)
        ax_std.set_xlabel(std_label[set_dict['pred']], fontsize=label_size_big)
        ax_ex.set_xlabel("Specific humidity ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=label_size_big)
        ax_bias.set_ylabel("Height (m)", fontsize=label_size_big)


        # figure labels:
        example_dt = dt.datetime.strptime(str(example_time), "%Y-%m-%dT%H:%M:%S")
        ax_bias.set_title("a)", loc='left', fontsize=label_size_big, **panel_id_dict)
        ax_std.set_title("b)", loc='left', fontsize=label_size_big, **panel_id_dict)
        ax_ex.set_title(f"c) {example_dt.strftime('%d %b %Y, %H:%M UTC')}", loc='left', fontsize=label_size_big, **panel_id_dict)

        f1.tight_layout()

        if set_dict['save_figures']:
            plotname = f"MOSAiC_nn_syn_mwr_pro_smoothed_{set_dict['pred']}_err_profs"
            plotfile = set_dict['path_plots'] + plotname
            f1.savefig(plotfile + ".png", dpi=300, bbox_inches='tight')
            f1.savefig(plotfile + ".pdf", bbox_inches='tight')

            print(f"Saved {plotfile}.pdf ....")
        else:
            plt.show()
            pdb.set_trace()

        plt.close()
        gc.collect()


    # settings:
    set_dict = {'pred_list': ['q'],         # available predictands; will be looped over
                'lw': 900,                  # time window (in sec) from radiosonde launch_time to launch_time+lw for averaging
                'date_0': "2019-09-20",     # lower limit of dates for import of mosaic data (default: "2019-09-20")
                'date_1': "2020-10-12",     # upper limit of dates (default: "2020-10-12")
                'data_version': {'temp': 'v00', 'temp_bl': 'v00',
                                'q': 'v00'},                    # indicates the version of the mosaic data to be used
                'i_cont_id': {'q': "472"},  # dictionary to identify the correct information content file
                'save_figures': True,       # if True, plot is saved to file. If False, it won't.
                }

    # set paths:
    set_dict['path_plots'] = path_plots

    # dictionary to convert units: [offset value, factor to multiply data with]: converted = (original+offset)*factor
    # keys have to be in set_dict['pred_list']!
    set_dict['unit_conv_dict'] = {'q': [0.0, 1000.]}        # from kg kg-1 to g kg-1

    set_dict['pred'] = 'q'
    print(f"Creating plots for {set_dict['pred']}....")

    # set alternative predictand name to address variables in the eval datasets:
    if set_dict['pred'] in ['q']:
        set_dict['p'] = set_dict['pred']        # to address variables in datasets


    # load MOSAiC observations (and predictions):
    if set_dict['pred'] in ['q']:
        NN_MOS_DS = import_hatpro_mirac_level2b_daterange_pangaea(path_data['nn_syn_mosaic'], set_dict['date_0'], set_dict['date_1'], 
                                                            which_retrieval=set_dict['pred'], data_version=set_dict['data_version'][set_dict['pred']],
                                                            around_radiosondes=True, path_radiosondes=path_data['radiosondes'], 
                                                            s_version='level_2', mwr_avg=set_dict['lw'])


    # load old retrievals:
    print("Importing single-instrument retrievals....")
    HAT_DS = xr.Dataset()

    if set_dict['pred'] in ['q']:
        which_retrieval = {'temp': 'ta', 'q': 'hus'}        # to translate between old and new names
        which_retrieval_keys = {'temp': 'ta', 'q': 'hua'}   # to translate between old and new names
        hatpro_dict = import_hatpro_level2b_daterange_pangaea(path_data['mwr_pro'], set_dict['date_0'], set_dict['date_1'],
                                                                which_retrieval='both', around_radiosondes=True,
                                                                path_radiosondes=path_data['radiosondes'], s_version='level_2', 
                                                                mwr_avg=set_dict['lw'])
        hatpro_dict['time_npdt'] = hatpro_dict['time'].astype("datetime64[s]")

        # put into dataset:
        HAT_DS = xr.Dataset(coords={'time': (['time'], hatpro_dict['time']),
                                    'height': (['height'], hatpro_dict['height'])})
        HAT_DS['rho_v'] = xr.DataArray(hatpro_dict['hua'], dims=['time', 'height'])
        HAT_DS['temp'] = xr.DataArray(hatpro_dict['ta'], dims=['time', 'height'])
        HAT_DS['flag'] = xr.DataArray(hatpro_dict['flag'], dims=['time'])

        del hatpro_dict


    # import radiosonde data
    print("Importing radiosonde data....")
    sonde_dict = import_radiosonde_daterange(path_data['radiosondes'], set_dict['date_0'], set_dict['date_1'], 
                                            s_version='level_2', remove_failed=True)
    sonde_dict['launch_time_npdt'] = sonde_dict['launch_time'].astype('datetime64[s]')
        

    # Flag MWR predictions using HATPRO and MiRAC-P flags. Remove values before first successful calibration.
    # Remove values within the Exclusive Economic Zones (EEZs):
    print("Filtering data....")
    NN_MOS_DS, sonde_dict, HAT_DS = filter_data_mosaic(NN_MOS_DS, sonde_dict, HAT_DS)
    set_dict['n_sondes'] = len(sonde_dict['launch_time'])


    # convert units: for HATPRO, first create a dummy specific humidity variable, which is absolute
    # humidity for now:
    if set_dict['p'] in set_dict['unit_conv_dict'].keys():
        NN_MOS_DS = conv_units(NN_MOS_DS, set_dict, set_dict['p'])
        HAT_DS['q'] = xr.DataArray(HAT_DS.rho_v.values, dims=['time', 'height'])    # actually abs. humidity in kg m-3

    sonde_dict['q'] *= 1000.0       # to g kg**-1


    # import Averaging Kernel data:
    file_i_cont = path_data['i_cont'] + f"NN_synergetic_ret_info_content_{set_dict['i_cont_id'][set_dict['pred']]}.nc"
    file_i_cont_k = path_data['i_cont'] + f"NN_synergetic_ret_info_content_481.nc"
    I_DS = xr.open_dataset(file_i_cont)
    I_DS_K = xr.open_dataset(file_i_cont_k)     # info content with K band frequencies
    AK_mean = I_DS.AK.mean('n_s')   # average Averaging Kernel over all samples
    AK_mean_K = I_DS_K.AK.mean('n_s')


    # interpolate sonde data to retrieval grid:
    if (set_dict['pred'] in ['q', 'temp', 'rh']) and ('height_ret' not in sonde_dict.keys()):
        sonde_dict = sonde_to_ret_grid(sonde_dict, NN_MOS_DS.height.values)


    # find overlap of synergetic ret with radiosonde times:
    print("Find overlaps with MOSAiC radiosondes....")
    NN_MOS_DS = overlap_with_radiosondes(NN_MOS_DS, sonde_dict, set_dict)

    # because the synergy combines flags from both MWRs, it has the min overlap with
    # radiosondes. Also reduce HATPRO data to that minimum
    intersct_overlap = ~np.isnan(NN_MOS_DS[f"{set_dict['p']}_avg"])
    if intersct_overlap.ndim > 1: intersct_overlap = intersct_overlap.min('height')

    if HAT_DS: 
        HAT_DS = overlap_with_radiosondes(HAT_DS, sonde_dict, set_dict)

        # limit to radiosondes that are included in all two-three (NN_MOS_DS, HAT_DS, MIR_DS) data sets:
        for dvv in ['q', 'temp']:
            if f"{dvv}_avg" in HAT_DS.data_vars:
                HAT_DS[f"{dvv}_avg"] = HAT_DS[f"{dvv}_avg"].where(intersct_overlap, other=np.nan)
                HAT_DS[f"{dvv}_std"] = HAT_DS[f"{dvv}_std"].where(intersct_overlap, other=np.nan)


    # Convert rho_v to specific humidity for HATPRO or compute relative humidity from temperature 
    # and humidity retrievals:
    if set_dict['pred'] == 'q':
        HAT_DS['q_avg'][:] = convert_abshum_to_spechum(HAT_DS.temp_avg.values, sonde_dict['pres_ret'], HAT_DS.q_avg.values)*1000.

        # limit HATPRO to 8000 m:
        HAT_DS = HAT_DS.sel(height=NN_MOS_DS.height)


    # smooth radiosonde profiles: using eq. 1 of Löhnert and Maier 2012, 10.5194/amt-5-1121-2012
    sonde_dict['q_sm'] = (NN_MOS_DS.q_avg.values.T + AK_mean.values @ (sonde_dict['q_ret'] - NN_MOS_DS.q_avg.values).T).T
    sonde_dict['q_sm_K'] = (HAT_DS.q_avg.values.T + AK_mean_K.values @ (sonde_dict['q_ret'] - HAT_DS.q_avg.values).T).T


    # compute error stats:
    ret_stats = dict()
    if set_dict['pred'] == 'q':
        ret_stats['mos'] = compute_error_stats(NN_MOS_DS[set_dict['p']+"_avg"].values, sonde_dict[set_dict['p'] + "_sm"],
                                                set_dict['pred'], NN_MOS_DS.height)
        ret_stats['hat'] = compute_error_stats(HAT_DS[set_dict['p']+"_avg"].values, sonde_dict[set_dict['p'] + "_sm_K"],
                                                set_dict['pred'], HAT_DS.height)
        ret_stats['hat_syn'] = compute_error_stats(HAT_DS[set_dict['p']+"_avg"].values, sonde_dict[set_dict['p'] + "_sm"],
                                                set_dict['pred'], HAT_DS.height)        # error using the rs profile smoothed with synergy


    # visualize errors:
    if set_dict['pred'] in ['q']:
        bias_rmse_prof(sonde_dict, NN_MOS_DS, HAT_DS, ret_stats, set_dict, height=NN_MOS_DS.height)
