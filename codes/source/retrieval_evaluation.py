def run_retrieval_evaluation(path_data, path_plots, path_output, var):

	"""
	This script combines the scripts eval_data_vis.py and mosaic_obs_eval.py for the evaluation of
	the synergetic retrieval with respect to the evaluation data set and MOSAiC radiosondes 
	(level 2). When computing errors (bias, RMSE, bias corrected RMSE, correlation coeff) with
	respect to the evaluation data set, all 20 Neural Networks are considered to estimate the
	spread among the different RNG initialized weights and selected training/validation data.
	- import evaluation data or MOSAiC obs and predictions for chosen predictand
	- process imported data
	- visualize

	Parameters:
	-----------
	path_data : dict
		Dictionary containing strings of the full paths of the synergetic retrieval output based
		on MOSAiC data and ERA5 evaluation data. It must also contain the path of the MOSAiC
		radiosondes as reference. The keys must therefore contain 'nn_syn_mosaic', 
		'nn_syn_eval_data' and 'radiosondes', respectively.
	path_plots : str
		String indicating where to save the plots.
	path_output : str
		String indicating where to save error estimates.
	var : str
		String indicating which variable to evaluate. Options: 'iwv', 'q', 'temp'
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

	from import_data import (import_hatpro_mirac_level2a_daterange_pangaea, import_hatpro_mirac_level2b_daterange_pangaea,
							import_hatpro_mirac_level2c_daterange_pangaea, import_radiosonde_daterange)
	from data_tools import compute_retrieval_statistics, compute_RMSE_profile


	# font and marker sizes
	fs = 24
	fs_small = fs - 2
	fs_dwarf = fs_small - 2
	fs_micro = fs_dwarf - 2
	msize = 7.0

	# colours:
	c_S = (1,0.72,0.12)			# synergetic retrieval
	c_S_fade = (1,0.72,0.12,0.15)# synergetic ret, but less alpha
	c_H = (0.067,0.29,0.769)	# MWR_PRO (HATPRO)
	c_M = (0,0.779,0.615)		# MiRAC-P retrieval
	c_G = (0.1,0.1,0.1)			# for grey scale (i.e., profiles)
	c_G_fade = (0.1,0.1,0.1,0.2)


	def unify_names_mos(
		DS, 
		set_dict):

		"""
		Renaming some variables to unify the naming within this script (i.e., IWV is called prw in DS).

		Parameters:
		-----------
		DS : xarray dataset
		Dataset containing predictions made with NN_retrieval.py.save_obs_predictions, imported 
			with import_hatpro_mirac_level2a_daterange_pangaea or ...level2b or ...level2c.
		set_dict : dict
			Dictionary containing additional information.
		"""

		for dv in DS.data_vars:
			for td_key in set_dict['translate_dict'].keys():
				sub_str_i = dv.find(td_key)
				if sub_str_i != -1:	# if True: match has been found
					DS = DS.rename({dv: dv.replace(td_key, set_dict['translate_dict'][td_key])})

		return DS


	def filter_data_mosaic(
		DS,
		sonde_dict,
		DS_BL=None):

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
		"""

		# Flag data using the HATPRO and MiRAC-P flags:
		if set_dict['pred'] not in ['temp', 'q', 'temp_bl']: 
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
		DS = DS.sel(time=slice(first_calib_HAT_MIR.astype(np.float64),None))	# faster than isel for simple masks


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
		if DS_BL:
			DS_BL['time_npdt'] = DS_BL.time.astype('datetime64[s]')
			DS_BL = DS_BL.sel(time=slice(first_calib_HAT_MIR.astype(np.float64),None))
			outside_eez['mwr_bl'] = np.full((len(DS_BL.time),), True)
			for EEZ_range in EEZ_periods_npdt.keys():
				outside_eez['mwr_bl'][(DS_BL.time_npdt >= EEZ_periods_npdt[EEZ_range][0]) &
										(DS_BL.time_npdt <= EEZ_periods_npdt[EEZ_range][1])] = False
			DS_BL = DS_BL.isel(time=outside_eez['mwr_bl'])

			return DS, sonde_dict, DS_BL

		else:
			return DS, sonde_dict


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
		
		u_c_d = set_dict['unit_conv_dict'][set_dict['pred']]
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
		set_dict,
		bl_mode=False):

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
		bl_mode : bool
			Boolean indicating if the data is boundary layer observations with reduced sampling 
			frequency. If True, boundary layer observations.
		"""

		if bl_mode:		# use a larger window for overlap search
			mwrson_idx = list()
			ds_time = DS.time_npdt.values
			last_overlap_idx = 0		# index to reduce searching time
			for lt in sonde_dict['launch_time_npdt']:
				overlap_idx = np.argwhere((ds_time >= lt - np.timedelta64(set_dict['lw']*2, "s")) & 
										(ds_time <= lt+np.timedelta64(set_dict['lw']*2, "s"))).flatten() + last_overlap_idx
				mwrson_idx.append(overlap_idx)

				# remove times that were already checked (irrelevant for the upcoming radiosondes):
				if len(overlap_idx) > 0:
					ds_time = ds_time[(overlap_idx[-1]-last_overlap_idx):]
					last_overlap_idx = overlap_idx[-1]

		else:
			mwrson_idx = list()
			ds_time = DS.time_npdt.values
			last_overlap_idx = 0		# index to reduce searching time
			for lt in sonde_dict['launch_time_npdt']:
				overlap_idx = np.argwhere((ds_time >= lt) & (ds_time < lt+np.timedelta64(set_dict['lw'], "s"))).flatten() + last_overlap_idx
				mwrson_idx.append(overlap_idx)

				# remove times that were already checked (irrelevant for the upcoming radiosondes):
				if len(overlap_idx) > 0:
					ds_time = ds_time[(overlap_idx[-1]-last_overlap_idx):]
					last_overlap_idx = overlap_idx[-1]

		if set_dict['pred'] == 'iwv':
			DS[f"{set_dict['p']}_avg"] = xr.DataArray(np.full((set_dict['n_sondes'],), np.nan), dims=['sondes'])
			DS[f"{set_dict['p']}_std"] = xr.DataArray(np.full((set_dict['n_sondes'],), np.nan), dims=['sondes'])
			for k, msi in enumerate(mwrson_idx):
				if len(msi) > 0:
					DS[f"{set_dict['p']}_avg"][k] = DS[f"{set_dict['p']}"][msi].mean('time')
					DS[f"{set_dict['p']}_std"][k] = DS[f"{set_dict['p']}"][msi].std('time')
			sonde_dict[set_dict['pred'] + "_ret"] = sonde_dict[set_dict['pred']]	# renamed for comparison

		elif set_dict['pred'] in ['temp', 'q']:
			DS[f"{set_dict['p']}_avg"] = xr.DataArray(np.full((set_dict['n_sondes'], len(DS.height)), np.nan), dims=['sondes', 'height'])
			DS[f"{set_dict['p']}_std"] = xr.DataArray(np.full((set_dict['n_sondes'], len(DS.height)), np.nan), dims=['sondes', 'height'])
			for k, msi in enumerate(mwrson_idx):
				if len(msi) > 0:
					DS[f"{set_dict['p']}_avg"][k,:] = DS[f"{set_dict['p']}"][msi,:].mean('time')
					DS[f"{set_dict['p']}_std"][k,:] = DS[f"{set_dict['p']}"][msi,:].std('time')

		return DS, sonde_dict


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

		# on x axis: reference; y axis: prediction
		x_stuff = predictand
		y_stuff = prediction

		# Compute statistics:
		if predictand_id in ['iwv', 'lwp']:
			# remove redundant dimension:
			x_stuff = x_stuff.squeeze()
			y_stuff = y_stuff.squeeze()
			stats_dict = compute_retrieval_statistics(x_stuff.squeeze(), y_stuff.squeeze(), compute_stddev=True)

			# For entire range:
			error_dict['N'] = stats_dict['N']
			error_dict['R'] = stats_dict['R']
			error_dict['rmse_tot'] = stats_dict['rmse']
			error_dict['stddev'] = stats_dict['stddev']
			error_dict['bias_tot'] = stats_dict['bias']

			# also compute rmse and bias for specific ranges only:
			# 'bias': np.nanmean(y_stuff - x_stuff),
			# 'rmse': np.sqrt(np.nanmean((x_stuff - y_stuff)**2)),
			range_dict = dict()
			if predictand_id == 'iwv':	# in mm
				range_dict['bot'] = [0,5]
				range_dict['mid'] = [5,10]
				range_dict['top'] = [10,100]
			elif predictand_id == 'lwp': # in kg m^-2
				range_dict['bot'] = (np.array([0,0.025]) + aux_i['unit_conv_dict'][predictand_id][0])*aux_i['unit_conv_dict'][predictand_id][1]
				range_dict['mid'] = (np.array([0.025,0.100]) + aux_i['unit_conv_dict'][predictand_id][0])*aux_i['unit_conv_dict'][predictand_id][1]
				range_dict['top'] = (np.array([0.100, 1e+06]) + aux_i['unit_conv_dict'][predictand_id][0])*aux_i['unit_conv_dict'][predictand_id][1]

			mask_range = dict()
			x_stuff_range = dict()
			y_stuff_range = dict()
			stats_dict_range = dict()
			for range_id in range_dict.keys():
				mask_range[range_id] = ((x_stuff >= range_dict[range_id][0]) & (x_stuff < range_dict[range_id][1]))
				x_stuff_range[range_id] = x_stuff[mask_range[range_id]]
				y_stuff_range[range_id] = y_stuff[mask_range[range_id]]

				# compute retrieval stats for the respective ranges:
				stats_dict_range[range_id] = compute_retrieval_statistics(x_stuff_range[range_id], y_stuff_range[range_id], compute_stddev=True)
				error_dict[f"rmse_{range_id}"] = stats_dict_range[range_id]['rmse']
				error_dict[f"stddev_{range_id}"] = stats_dict_range[range_id]['stddev']
				error_dict[f"bias_{range_id}"] = stats_dict_range[range_id]['bias']


		elif predictand_id in ['temp', 'temp_bl', 'q']:

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
			range_dict = {	'bot': [0., 1500.0],
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

		return error_dict


	def iwv_err_diff_comp(
		NN_MOS_DS, 
		sonde_dict, 
		NN_SYN_DS, 
		set_dict):

		"""
		Visualize the performance of the synergetic Neural Network prediction against a reference 
		predictand (evaluation data set or MOSAiC radiosondes). Here, a composit of RMSE and bias 
		against the reference IWV will be plotted.

		Parameters:
		-----------
		NN_MOS_DS : xarray dataset
			Dataset created with NN_retrieval.py containing predicted MOSAiC observation data. 
		sonde_dict : dict
			Dictionary containing the key 'iwv', which is a 1D array of floats containing the reference
			IWV in kg m**-2. 
		NN_SYN_DS : xarray dataset
			Dataset created with save_eval_data_predictions in NN_retrieval.py containing predicted and
			reference data. The prediction contains "_p" in the variable name.
		set_dict : dict
			Dictionary containing additional information.
		"""


		def compute_errors_bins(
			bins,
			prediction,
			reference):

			"""
			Compute the root mean squared error (RMSE) and bias (mean difference) between a prediction
			(i.e., a retrieval) and a reference for each bin in the array bins.

			Parameters:
			-----------
			bins : array of floats
				Bins over which to compute the error statistics. Must be a (n_bins,2) array where the
				second axis's first (last) value denotes the lower (upper) boundary of the bin.
			prediction : array of floats
				Predicted variable from a retrieval.
			reference : array of floats
				Reference predictand for evaluation (i.e., of test data, here, MOSAiC radiosondes) data as array.
			"""

			RMSE_bins = np.full((bins.shape[0],), np.nan)
			BIAS_bins = np.full((bins.shape[0],), np.nan)
			N_bins = np.zeros((bins.shape[0],))		# number of matches for each bin
			for ibi, bin in enumerate(bins):
				# find indices for the respective bin (based on the reference (==truth)):
				idx_bin = np.where((reference >= bin[0]) & (reference < bin[1]))[0]
				N_bins[ibi] = len(idx_bin)

				# compute errors:
				bias_ibi = np.nanmean(prediction[idx_bin] - reference[idx_bin])
				RMSE_bins[ibi] = np.sqrt(np.nanmean((prediction[idx_bin] - reference[idx_bin])**2))
				# RMSE_bins[ibi] = np.sqrt(np.nanmean((reference[idx_bin] - (prediction[idx_bin] - bias_ibi))**2))
				BIAS_bins[ibi] = bias_ibi

			return RMSE_bins, BIAS_bins, N_bins


		# create output path if not existing:
		plotpath_dir = os.path.dirname(set_dict['path_plots'] + f"{set_dict['pred']}/")
		if not os.path.exists(plotpath_dir):
			os.makedirs(plotpath_dir)


		# error diff composit: Generate bins and compute RMSE, Bias for each bin:
		val_max = 35.0
		val_bins = np.zeros((13,2))
		val_bins[:12,0] = np.arange(0.0,22.1,2.0)
		val_bins[:12,1] = np.arange(2.0,24.1,2.0)
		val_bins[12] = np.array([val_bins[11,1], val_max])
		n_bins = val_bins.shape[0]


		# IWV error composit plot:
		predictand_syn = NN_SYN_DS.iwv			# (n_sample,) array
		prediction_syn = NN_SYN_DS.iwv_p		# (n_rand, n_sample) array -> compute spread of prediction
		prediction_mos = NN_MOS_DS[set_dict['p']+"_avg"]
		predictand_mos = sonde_dict[set_dict['p']]


		# compute errors for each bin:
		RMSE_bins_mos, BIAS_bins_mos, N_bins_mos = compute_errors_bins(val_bins, prediction_mos.values, predictand_mos)

		# for synergetic ret, compute errors for each bin and each RNG seed to obtain uncertainties 
		# over the 20 rng runs (compute mean and stand dev of the errors over the 20 runs; also 
		# highlight the operational rng seed used for final retrieval):
		n_rand = len(prediction_syn.n_rand)	
		RMSE_bins_syn = xr.DataArray(np.full((n_rand, n_bins), np.nan), dims=['n_rand', 'n_bins'],
									coords={'n_rand': prediction_syn.n_rand})
		BIAS_bins_syn = xr.DataArray(np.full((n_rand, n_bins), np.nan), dims=['n_rand', 'n_bins'],
									coords={'n_rand': prediction_syn.n_rand})
		N_bins_syn = xr.DataArray(np.full((n_rand, n_bins), np.nan), dims=['n_rand', 'n_bins'],
									coords={'n_rand': prediction_syn.n_rand})
		for i_r, rng_seed in enumerate(prediction_syn.n_rand):
			RMSE_bins_syn[i_r,:], BIAS_bins_syn[i_r,:], N_bins_syn[i_r,:] = compute_errors_bins(val_bins,
																								prediction_syn.sel(n_rand=rng_seed).values,
																								predictand_syn.values)


		# compute average and spread (std dev or min/max) over rng seeds and highlight the
		# operational rng seed:
		def compute_mean_min_max_std_op_n_rand(data_array):
			da_mean = data_array.mean('n_rand').values
			da_min = data_array.min('n_rand').values
			da_max = data_array.max('n_rand').values
			da_std = data_array.std('n_rand').values
			da_op = data_array.sel(n_rand=set_dict['op_rng'][set_dict['pred']]).values
			return da_mean, da_min, da_max, da_std, da_op
		RMSE_bins_syn_mean, RMSE_bins_syn_min, RMSE_bins_syn_max, RMSE_bins_syn_std, RMSE_bins_syn_op = (
														compute_mean_min_max_std_op_n_rand(RMSE_bins_syn))
		BIAS_bins_syn_mean, BIAS_bins_syn_min, BIAS_bins_syn_max, BIAS_bins_syn_std, BIAS_bins_syn_op = (
														compute_mean_min_max_std_op_n_rand(BIAS_bins_syn))
		N_bins_syn = N_bins_syn.sel(n_rand=set_dict['op_rng'][set_dict['pred']]).values	# doesn't vary over RNGs


		# visualize:
		f1 = plt.figure(figsize=(11,7))
		a1 = plt.axes()

		# deactivate some spines:
		a1.spines[['right', 'top']].set_visible(False)

		ax_lims = np.asarray([0.0, 30.0])
		er_lims = np.asarray([-1.5, 1.5])

		# plotting:
		# thin lines indicating RELATIVE errors:
		rel_err_contours = np.array([1.0,2.0,5.0,10.0,20.0])
		rel_err_range = np.arange(0.0, val_max+0.0001, 0.01)
		rel_err_curves = np.zeros((len(rel_err_contours), len(rel_err_range)))
		for i_r, r_e_c in enumerate(rel_err_contours):
			rel_err_curves[i_r,:] = rel_err_range*r_e_c / 100.0
			a1.plot(rel_err_range, rel_err_curves[i_r,:], color=(0,0,0,0.5), linewidth=1, linestyle='dotted')
			a1.plot(rel_err_range, -1.0*rel_err_curves[i_r,:], color=(0,0,0,0.5), linewidth=1, linestyle='dotted')

			# add annotation (label) to rel error curve:
			rel_err_label_pos_x = er_lims[1] * 100. / r_e_c
			if rel_err_label_pos_x > val_max:
				a1.text(ax_lims[1], ax_lims[1]*r_e_c / 100., f"{int(r_e_c)} %",
					color=(0,0,0,0.5), ha='left', va='center', transform=a1.transData, fontsize=fs_micro-6)
			else:
				a1.text(rel_err_label_pos_x, er_lims[1], f"{int(r_e_c)} %", 
					color=(0,0,0,0.5), ha='left', va='bottom', transform=a1.transData, fontsize=fs_micro-6)


		val_bins_plot = (val_bins[:,1] - val_bins[:,0])*0.5 + val_bins[:,0]
		a1.plot(ax_lims, [0,0], color=(0,0,0))

		# actual data plots:
		rmse_syn_minmax = a1.fill_between(val_bins_plot, RMSE_bins_syn_min, RMSE_bins_syn_max, color=c_G_fade, label='Min-Max RMSE')
		rmse_syn_mean, = a1.plot(val_bins_plot, RMSE_bins_syn_mean, color=c_G, linewidth=1, label='Mean RMSE')
		rmse_syn_op, = a1.plot(val_bins_plot, RMSE_bins_syn_op, color=c_G, linewidth=2, label='Final RMSE')
		bias_syn_minmax = a1.fill_between(val_bins_plot, BIAS_bins_syn_min, BIAS_bins_syn_max, color=c_G_fade, label='Min-Max bias')
		bias_syn_mean, = a1.plot(val_bins_plot, BIAS_bins_syn_mean, color=c_G, linewidth=1, linestyle='dashed', label='Mean bias')
		bias_syn_op, = a1.plot(val_bins_plot, BIAS_bins_syn_op, color=c_G, linewidth=2, linestyle='dashed', label='Final bias')

		rmse_mos_plot, = a1.plot(val_bins_plot, RMSE_bins_mos, color=c_S, linewidth=2, label='Final RMSE')
		bias_mos_plot, = a1.plot(val_bins_plot, BIAS_bins_mos, color=c_S, linewidth=2, linestyle='dashed', label='Final bias')

		
		# Legends:
		try:
			leg_syn = a1.legend(handles=[rmse_syn_minmax, rmse_syn_mean, rmse_syn_op, bias_syn_minmax, bias_syn_mean, bias_syn_op], 
							loc='lower left', ncols=2, bbox_to_anchor=(0.01, 0.00), fontsize=fs_micro-4, 
							framealpha=0.5, title='ERA5', title_fontsize=fs_micro-4)
		except:
			leg_syn = a1.legend(handles=[rmse_syn_minmax, rmse_syn_mean, rmse_syn_op, bias_syn_minmax, bias_syn_mean, bias_syn_op], 
							loc='lower left', ncol=2, bbox_to_anchor=(0.01, 0.00), fontsize=fs_micro-4, 
							framealpha=0.5, title='ERA5', title_fontsize=fs_micro-4)
		a1.add_artist(leg_syn)
		leg_mos = a1.legend(handles=[rmse_mos_plot, bias_mos_plot], loc='lower right', bbox_to_anchor=(0.99, 0.00), fontsize=fs_micro-4,
							framealpha=0.5, title='MOSAiC', title_fontsize=fs_micro-4)


		# set axis limits:
		a1.set_ylim(bottom=er_lims[0], top=er_lims[1])
		a1.set_xlim(left=ax_lims[0], right=ax_lims[1])

		# set axis ticks, ticklabels and tick parameters:
		a1.minorticks_on()
		a1.tick_params(axis='both', labelsize=fs_micro-4)

		# grid:
		a1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# labels:
		a1.set_ylabel("IWV error ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs_dwarf)
		a1.set_xlabel("Reference IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs_dwarf)

		if set_dict['save_figures']:
			plotname = f"MOSAiC_nn_syn_mwr_eval_data_{set_dict['pred']}_err_diff_comp"
			plotfile = plotpath_dir + "/" + plotname
			f1.savefig(plotfile + ".png", dpi=300, bbox_inches='tight')
			f1.savefig(plotfile + ".pdf", bbox_inches='tight')

			print(f"Saved {plotfile}.pdf ....")
		else:
			plt.show()
			pdb.set_trace()

		plt.close()
		gc.collect()


		# Save error estimates to dict, so that error data can be written to file:
		ret_stat_dict = dict()
		ret_stat_dict['syn'] = {'rmse_bins_min': RMSE_bins_syn_min,
								'rmse_bins_max': RMSE_bins_syn_max,
								'rmse_bins_mean': RMSE_bins_syn_mean,
								'rmse_bins_op': RMSE_bins_syn_op,
								'bias_bins_min': BIAS_bins_syn_min,
								'bias_bins_max': BIAS_bins_syn_max,
								'bias_bins_mean': BIAS_bins_syn_mean,
								'bias_bins_op': BIAS_bins_syn_op}
		ret_stat_dict['mos'] = {'rmse_bins_op': RMSE_bins_mos,
								'bias_bins_op': BIAS_bins_mos}
		ret_stat_dict['aux'] = {'val_bins': val_bins, 'val_bins_centre': val_bins_plot}		# auxiliary information

		return ret_stat_dict


	def bias_rmse_prof(
		ret_stats,
		set_dict,
		height):

		"""
		Visualize the performance of the synergetic Neural Network prediction against a reference 
		predictand (evaluation data set or MOSAiC radiosondes). Here, profiles of bias and
		standard deviation between predictions and reference predictand will be plotted. 

		Parameters:
		-----------
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

		# create output path if not existing:
		plotpath_dir = os.path.dirname(set_dict['path_plots'] + f"{set_dict['pred']}/")
		if not os.path.exists(plotpath_dir):
			os.makedirs(plotpath_dir)

		# reduce unnecessary dimensions of height:
		if height.ndim == 2:
			height = height[0,:]


		# write out the important error stats: 
		# for syn: operational, min, max, mean exist
		STD_mos_op = ret_stats['mos']['stddev']
		BIAS_mos_op = ret_stats['mos']['bias_tot']


		STD_syn_op = ret_stats['syn']['stddev']
		BIAS_syn_op = ret_stats['syn']['bias_tot']

		STD_syn_mean = ret_stats['syn_20']['stddev_mean'].values
		BIAS_syn_mean = ret_stats['syn_20']['bias_tot_mean'].values

		STD_syn_min = ret_stats['syn_20']['stddev_min'].values
		BIAS_syn_min = ret_stats['syn_20']['bias_tot_min'].values

		STD_syn_max = ret_stats['syn_20']['stddev_max'].values
		BIAS_syn_max = ret_stats['syn_20']['bias_tot_max'].values


		# dictionaries for adaptations:
		legend_pos = 'upper right'
		anchor_pos = (0.98, 0.99)
		bias_label = "$\mathrm{Bias}_{\mathrm{q}}$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)"
		std_label = "$\mathrm{RMSE}_{\mathrm{corr, q}}$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)"


		f1 = plt.figure(figsize=(10,6.5))
		ax_bias = plt.subplot2grid((1,2), (0,0))			# bias profile
		ax_std = plt.subplot2grid((1,2), (0,1))				# std dev profile

		y_lim = np.array([0.0, height.max()])
		x_lim_std = np.array([0.0, 2.2])		# in g kg-1
		x_lim_bias = np.array([-0.25, 0.25])	# in g kg-1


		# bias profiles:
		ax_bias.plot(np.full_like(height, 0.0), height, color=(0,0,0), linewidth=1.0)	# helper line
		ax_bias.fill_betweenx(height, BIAS_syn_min, BIAS_syn_max, color=c_G_fade, label='Min-Max')
		ax_bias.plot(BIAS_syn_mean, height, color=c_G, linewidth=1, linestyle='dashed', label='Mean')
		ax_bias.plot(BIAS_syn_op, height, color=c_G, linewidth=2, label='Final')
		ax_bias.plot(BIAS_mos_op, height, color=c_S, linewidth=2)


		# std dev profiles:
		mean_syn_ref, = ax_std.plot(ret_stats['syn']['mean_ref'], height, color=c_G, linewidth=1.5, linestyle='dotted')
		mean_mos_ref, = ax_std.plot(ret_stats['mos']['mean_ref'], height, color=c_S, linewidth=2.0, linestyle='dotted')
		mean_mos_ret, = ax_std.plot(ret_stats['mos']['mean_ret'], height, color=c_H, linewidth=2.0, linestyle='dotted')
		rmse_corr_syn_min_max = ax_std.fill_betweenx(height, STD_syn_min, STD_syn_max, color=c_G_fade, label='Min-Max')
		rmse_corr_syn_mean, = ax_std.plot(STD_syn_mean, height, color=c_G, linewidth=1, linestyle='dashed', label='Mean')
		rmse_corr_syn_final, = ax_std.plot(STD_syn_op, height, color=c_G, linewidth=2, label='Final')
		dummy_syn_ref, = ax_std.plot([], [], color=c_G, linewidth=1.5, linestyle='dotted', label='Mean ERA5')
		rmse_corr_mos_final, = ax_std.plot(STD_mos_op, height, color=c_S, linewidth=2, label='Final')
		dummy_mos_ref, = ax_std.plot([], [], color=c_S, linewidth=2, linestyle='dotted', label="Mean RS")
		dummy_mos_ret, = ax_std.plot([], [], color=c_H, linewidth=2, linestyle='dotted', label="Mean MWR")


		# legends:
		leg_syn = ax_std.legend(handles=[rmse_corr_syn_min_max, rmse_corr_syn_mean, rmse_corr_syn_final, dummy_syn_ref],
								loc=legend_pos, bbox_to_anchor=anchor_pos, 
								fontsize=fs_micro-4, framealpha=0.5, title='ERA5', title_fontsize=fs_micro-4)
		ax_std.add_artist(leg_syn)
		
		leg_mos = ax_std.legend(handles=[rmse_corr_mos_final, dummy_mos_ref, dummy_mos_ret], loc=legend_pos, bbox_to_anchor=(anchor_pos[0], 0.68),
								# handletextpad=1.9,
								fontsize=fs_micro-4, framealpha=0.5, title='MOSAiC', title_fontsize=fs_micro-4)

		# axis lims:
		ax_bias.set_xlim(left=x_lim_bias[0], right=x_lim_bias[1])
		ax_std.set_xlim(left=x_lim_std[0], right=x_lim_std[1])


		for ax in [ax_bias, ax_std]:
			ax.set_ylim(bottom=y_lim[0], top=y_lim[1])

			ax.minorticks_on()
			ax.tick_params(axis='both', labelsize=fs_micro-4)
			ax.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# remove tick labels for y axis:
		ax_std.yaxis.set_ticklabels([])


		# labels:
		ax_bias.set_xlabel(bias_label, fontsize=fs_dwarf)
		ax_std.set_xlabel(std_label, fontsize=fs_dwarf)
		ax_bias.set_ylabel("Height (m)", fontsize=fs_dwarf)

		# figure labels:
		ax_bias.set_title("a)", loc='left', fontsize=fs_dwarf)
		ax_std.set_title("b)", loc='left', fontsize=fs_dwarf)

		# plt.tight_layout()

		if set_dict['save_figures']:
			plotname = f"MOSAiC_nn_syn_mwr_eval_data_{set_dict['pred']}_err_profs"
			plotfile = plotpath_dir + "/" + plotname
			f1.savefig(plotfile + ".png", dpi=300, bbox_inches='tight')
			f1.savefig(plotfile + ".pdf", bbox_inches='tight')

			print(f"Saved {plotfile}.pdf ....")
		else:
			plt.show()
			pdb.set_trace()

		plt.close()
		gc.collect()


	def bias_rmse_prof_temp(
		ret_stats,
		set_dict,
		height):

		"""
		Visualize the performance of the synergetic Neural Network prediction against a reference 
		predictand (evaluation data set or MOSAiC radiosondes).  Here, profiles of bias and standard 
		deviation between predictions and reference predictand will be plotted. 

		Parameters:
		-----------
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

		# create output path if not existing:
		plotpath_dir = os.path.dirname(set_dict['path_plots'] + f"{set_dict['pred']}/")
		if not os.path.exists(plotpath_dir):
			os.makedirs(plotpath_dir)

		# reduce unnecessary dimensions of height:
		if height.ndim == 2:
			height = height[0,:]


		# write out the important error stats: 
		# for syn: operational, min, max, mean exist
		STD_mos_op = ret_stats['mos']['stddev']
		BIAS_mos_op = ret_stats['mos']['bias_tot']
		STD_mos_bl_op = ret_stats['mos_bl']['stddev']
		BIAS_mos_bl_op = ret_stats['mos_bl']['bias_tot']


		STD_syn_op = ret_stats['syn']['stddev']
		BIAS_syn_op = ret_stats['syn']['bias_tot']
		STD_syn_bl_op = ret_stats['syn_bl']['stddev']
		BIAS_syn_bl_op = ret_stats['syn_bl']['bias_tot']

		STD_syn_mean = ret_stats['syn_20']['stddev_mean'].values
		BIAS_syn_mean = ret_stats['syn_20']['bias_tot_mean'].values
		STD_syn_bl_mean = ret_stats['syn_20_bl']['stddev_mean'].values
		BIAS_syn_bl_mean = ret_stats['syn_20_bl']['bias_tot_mean'].values

		STD_syn_min = ret_stats['syn_20']['stddev_min'].values
		BIAS_syn_min = ret_stats['syn_20']['bias_tot_min'].values
		STD_syn_bl_min = ret_stats['syn_20_bl']['stddev_min'].values
		BIAS_syn_bl_min = ret_stats['syn_20_bl']['bias_tot_min'].values

		STD_syn_max = ret_stats['syn_20']['stddev_max'].values
		BIAS_syn_max = ret_stats['syn_20']['bias_tot_max'].values
		STD_syn_bl_max = ret_stats['syn_20_bl']['stddev_max'].values
		BIAS_syn_bl_max = ret_stats['syn_20_bl']['bias_tot_max'].values


		# dictionaries for adaptations:
		legend_pos = 'upper right'
		anchor_pos = (0.98, 0.50)
		bias_label = "$\mathrm{Bias}_{\mathrm{T}}$ (K)"
		std_label = "$\mathrm{RMSE}_{\mathrm{corr, T}}$ (K)"


		f1 = plt.figure(figsize=(10,12))
		ax_bias = plt.subplot2grid((2,2), (0,0))			# bias profile
		ax_std = plt.subplot2grid((2,2), (0,1))				# std dev profile
		ax_bias_bl = plt.subplot2grid((2,2), (1,0))			# bias profile BL
		ax_std_bl = plt.subplot2grid((2,2), (1,1))				# std dev profile BL

		y_lim = np.array([0.0, height.max()])
		x_lim_std = np.array([0.0, 5.0])		# in K
		x_lim_bias = np.array([-1.2, 1.2])		# in K


		# bias profiles:
		ax_bias.plot(np.full_like(height, 0.0), height, color=(0,0,0), linewidth=1.0)	# helper line
		ax_bias.fill_betweenx(height, BIAS_syn_min, BIAS_syn_max, color=c_G_fade)
		ax_bias.plot(BIAS_syn_mean, height, color=c_G, linewidth=1, linestyle='dashed')
		ax_bias.plot(BIAS_syn_op, height, color=c_G, linewidth=2)
		ax_bias.plot(BIAS_mos_op, height, color=c_S, linewidth=2)


		# std dev profiles:
		rmse_corr_syn_min_max = ax_std.fill_betweenx(height, STD_syn_min, STD_syn_max, color=c_G_fade, label='Min-Max')
		rmse_corr_syn_mean, = ax_std.plot(STD_syn_mean, height, color=c_G, linewidth=1, linestyle='dashed', label='Mean')
		rmse_corr_syn_final, = ax_std.plot(STD_syn_op, height, color=c_G, linewidth=2, label='Final')
		rmse_corr_mos_final, = ax_std.plot(STD_mos_op, height, color=c_S, linewidth=2, label='Final')


		# BL
		ax_bias_bl.plot(np.full_like(height, 0.0), height, color=(0,0,0), linewidth=1.0)	# helper line
		ax_bias_bl.fill_betweenx(height, BIAS_syn_bl_min, BIAS_syn_bl_max, color=c_G_fade)
		ax_bias_bl.plot(BIAS_syn_bl_mean, height, color=c_G, linewidth=1, linestyle='dashed')
		ax_bias_bl.plot(BIAS_syn_bl_op, height, color=c_G, linewidth=2)
		ax_bias_bl.plot(BIAS_mos_bl_op, height, color=c_S, linewidth=2)

		ax_std_bl.fill_betweenx(height, STD_syn_bl_min, STD_syn_bl_max, color=c_G_fade)
		ax_std_bl.plot(STD_syn_bl_mean, height, color=c_G, linewidth=1, linestyle='dashed')
		ax_std_bl.plot(STD_syn_bl_op, height, color=c_G, linewidth=2)
		ax_std_bl.plot(STD_mos_bl_op, height, color=c_S, linewidth=2)


		# legends:

		leg_syn = ax_std.legend(handles=[rmse_corr_syn_min_max, rmse_corr_syn_mean, rmse_corr_syn_final],
								loc=legend_pos, bbox_to_anchor=anchor_pos, 
								fontsize=fs_micro-4, framealpha=0.5, title='ERA5', title_fontsize=fs_micro-4)
		ax_std.add_artist(leg_syn)
		
		leg_mos = ax_std.legend(handles=[rmse_corr_mos_final], loc=legend_pos, bbox_to_anchor=(anchor_pos[0], 0.25),
								# handletextpad=1.9,
								fontsize=fs_micro-4, framealpha=0.5, title='MOSAiC', title_fontsize=fs_micro-4)


		# axis lims:
		ax_bias.set_xlim(left=x_lim_bias[0], right=x_lim_bias[1])
		ax_std.set_xlim(left=x_lim_std[0], right=x_lim_std[1])

		ax_bias_bl.set_xlim(left=x_lim_bias[0], right=x_lim_bias[1])
		ax_std_bl.set_xlim(left=x_lim_std[0], right=x_lim_std[1])


		for ax in [ax_bias, ax_std, ax_bias_bl, ax_std_bl]:
			ax.set_ylim(bottom=y_lim[0], top=y_lim[1])

			ax.minorticks_on()
			ax.tick_params(axis='both', labelsize=fs_micro-4)
			ax.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# remove tick labels for y axis:
		ax_std.yaxis.set_ticklabels([])
		ax_std_bl.yaxis.set_ticklabels([])


		# labels:
		ax_bias.set_ylabel("Height (m)", fontsize=fs_dwarf)
		ax_std_bl.set_xlabel(std_label, fontsize=fs_dwarf)
		ax_bias_bl.set_xlabel(bias_label, fontsize=fs_dwarf)
		ax_bias_bl.set_ylabel("Height (m)", fontsize=fs_dwarf)

		# figure labels:
		ax_bias.set_title("a)", loc='left', fontsize=fs_dwarf)
		ax_std.set_title("b)", loc='left', fontsize=fs_dwarf)
		ax_bias_bl.set_title("c)", loc='left', fontsize=fs_dwarf)
		ax_std_bl.set_title("d)", loc='left', fontsize=fs_dwarf)

		plt.tight_layout()

		if set_dict['save_figures']:
			plotname = f"MOSAiC_nn_syn_mwr_eval_data_{set_dict['pred']}_err_profs"
			plotfile = plotpath_dir + "/" + plotname
			f1.savefig(plotfile + ".png", dpi=300, bbox_inches='tight')
			f1.savefig(plotfile + ".pdf", bbox_inches='tight')

			print(f"Saved {plotfile}.pdf ....")
		else:
			plt.show()
			pdb.set_trace()

		plt.close()
		gc.collect()


	def save_ret_stats(
		ret_stats,
		set_dict):

		"""
		Puts error statistics saved to the dictionary ret_stats into an xarray dataset. The dataset
		is then saved to a netcdf file.

		Parameters:
		-----------
		ret_stats : dict
			Dictionary containing float arrays or xarray datasets filled with error statistics of the
			neural network retrieval, evaluated with ERA5 data.
		set_dict : dict
			Dictionary containing additional information.
		"""

		def add_global_attributes(DS, set_dict):
			# global attributes:
			DS.attrs['title'] = f"Retrieval error statistics for {set_dict['pred']} based on ERA5 evaluation data"
			DS.attrs['institution'] = "Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
			DS.attrs['author'] = "Andreas Walbröl, a.walbroel@uni-koeln.de"
			DS.attrs['history'] = f"{dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}, created with retrieval_evaluation.py"
			DS.attrs['eval_data'] = "ERA5"
			DS.attrs['References'] = ("Hersbach et al. (2018), https://doi.org/10.24381/cds.bd0915c6. Hersbach et al. (2018), " +
										"https://doi.org/10.24381/cds.adbb2d47. Hersbach et al. (2020), https://doi.org/10.1002/qj.3803")

			return DS


		# differentiate between 1D and 2D retrieved variable (iwv vs temp and q):
		if set_dict['pred'] == 'iwv':
			# form xarray dataset:
			ret_stat_ds = xr.Dataset(coords={"bins": (['bins'], ret_stats['aux']['val_bins_centre'],
														{'long_name': "Centre of integrated water vapour bins",
														'units': "kg m-2"})})
			ret_stat_ds['bins_bnds'] = xr.DataArray(ret_stats['aux']['val_bins'], dims=['bins', 'nb'],
														attrs={'long_name': "Integrated water vapour bin boundaries",
																'comment': "First (last) value of dimension 'nb' is the lower (upper) boundary"})

			# save data variables:
			for err_key in ret_stats['syn'].keys():
				ret_stat_ds[err_key] = xr.DataArray(ret_stats['syn'][err_key], dims=['bins'])

				# attributes:
				ret_stat_ds[err_key].attrs = {'long_name': "", 'units': "kg m-2"}

			# set attributes:
			ret_stat_ds['rmse_bins_min'].attrs['long_name'] = "Minimum root mean squared error over 20 similar neural network runs"
			ret_stat_ds['rmse_bins_max'].attrs['long_name'] = "Maximum root mean squared error over 20 similar neural network runs"
			ret_stat_ds['rmse_bins_mean'].attrs['long_name'] = "Mean root mean squared error over 20 similar neural network runs"
			ret_stat_ds['rmse_bins_op'].attrs['long_name'] = "Root mean squared error of the final selected neural network run"
			ret_stat_ds['bias_bins_min'].attrs['long_name'] = "Minimum bias over 20 similar neural network runs"
			ret_stat_ds['bias_bins_max'].attrs['long_name'] = "Maximum bias over 20 similar neural network runs"
			ret_stat_ds['bias_bins_mean'].attrs['long_name'] = "Mean bias over 20 similar neural network runs"
			ret_stat_ds['bias_bins_op'].attrs['long_name'] = "Bias of the final selected neural network run"

			# global attributes:
			ret_stat_ds = add_global_attributes(ret_stat_ds, set_dict)


			# save as netcdf:
			outfile_name = set_dict['path_output'] + f"NN_syn_ret_ERA5_eval_data_error_stats_{set_dict['pred']}.nc"
			ret_stat_ds.to_netcdf(outfile_name, mode='w', format="NETCDF4")
			print(f"Saved {outfile_name}....")

			ret_stat_ds = ret_stat_ds.close()
			

		elif set_dict['pred'] in ['q', 'temp']:

			# labels and attributes for the dataset:
			units = {'temp': 'K', 'temp_bl': 'K', 'q': 'g kg-1'}
			long_names = {'rmse': "root mean squared error", 'stddev': "bias corrected rmse", 'bias': 'bias'}
			comments = {'rmse': 'rmse = sqrt(mean((prediction - reference)**2))', 'stddev': 'bias corrected rmse', 
						'bias': "computed as difference: bias = mean(prediction - reference, dim=data_samples)"}


			# split between bl and non bl if applicable (i.e., for 'temp'):
			for key in ['syn_20', 'syn_20_bl']:
				if key in ret_stats.keys():
					ret_stats[key] = ret_stats[key].rename({'stddev': 'stddev_tot'})		# synonym

					# rename dataset:
					ret_stat_ds = ret_stats[key]

					# set attributes:
					for dv in ret_stat_ds.data_vars:
						ret_stat_ds[dv].attrs['units'] = units[set_dict['pred']]

						if 'rmse_' in dv:
							ret_stat_ds[dv].attrs['long_name'] = long_names['rmse']
							ret_stat_ds[dv].attrs['comment'] = comments['rmse']
						elif 'bias_' in dv:
							ret_stat_ds[dv].attrs['long_name'] = long_names['bias']
							ret_stat_ds[dv].attrs['comment'] = comments['bias']
						elif 'stddev_' in dv:
							ret_stat_ds[dv].attrs['long_name'] = long_names['stddev']
							ret_stat_ds[dv].attrs['comment'] = comments['stddev']


						# add description depending on variable name suffix:
						if "_mean" in dv:
							ret_stat_ds[dv].attrs['long_name'] = f"mean {ret_stat_ds[dv].attrs['long_name']} over n_rand"
						elif "_std" in dv:
							ret_stat_ds[dv].attrs['long_name'] = f"standard deviation of {ret_stat_ds[dv].attrs['long_name']} over n_rand"
						elif "_max" in dv:
							ret_stat_ds[dv].attrs['long_name'] = f"max {ret_stat_ds[dv].attrs['long_name']} over n_rand"
						elif "_min" in dv:
							ret_stat_ds[dv].attrs['long_name'] = f"min {ret_stat_ds[dv].attrs['long_name']} over n_rand"
						elif "_op" in dv:
							ret_stat_ds[dv].attrs['long_name'] += " of the final selected seed"
					

					# global attributes:
					ret_stat_ds = add_global_attributes(ret_stat_ds, set_dict)

					# save as netcdf:
					outfile_suffix = set_dict['pred']
					if "_bl" in key: 	# adapt filename and attribute to be distinguishable from 'temp'
						outfile_suffix += '_bl'
						ret_stat_ds.attrs['title'] = ret_stat_ds.attrs['title'].replace(" temp ", " temp_bl ")
					outfile_name = set_dict['path_output'] + f"NN_syn_ret_ERA5_eval_data_error_stats_{outfile_suffix}.nc"
					ret_stat_ds.to_netcdf(outfile_name, mode='w', format="NETCDF4")
					print(f"Saved {outfile_name}....")

					ret_stat_ds = ret_stat_ds.close()


	# settings:
	set_dict = {'lw': 900,					# time window (in sec) from radiosonde launch_time to launch_time+lw for averaging
				'date_0': "2019-09-20",		# lower limit of dates for import of mosaic data (default: "2019-09-20")
				'date_1': "2020-10-12",		# upper limit of dates (default: "2020-10-12")
				'data_version': {'iwv': 'v00',
								'temp': 'v00', 'temp_bl': 'v00', 
								'q': 'v00'},					# indicates the version of the mosaic data to be used
				'save_figures': True,							# if True, plot is saved to file. If False, it won't.
				'op_rng': {'iwv': 806, 'q': 442, 'temp': 487, 'temp_bl': 472},	# rng seed used for final retrieval
				}

	# set paths:
	set_dict['path_plots'] = path_plots
	set_dict['path_output'] = path_output


	# check if output path exists:
	os.makedirs(os.path.dirname(set_dict['path_output']), exist_ok=True)


	# dictionary to convert units: [offset value, factor to multiply data with]: converted = (original+offset)*factor
	# keys have to be in set_dict['pred_list']!
	set_dict['unit_conv_dict'] = {'q': [0.0, 1000.]}		# from kg kg-1 to g kg-1

	set_dict['translate_dict'] = {'prw': 'iwv'}				# to translate variable and file names
	set_dict['bl_mode'] = False								# if boundary layer temperature profiles are selected


	# Identify evaluation data set files:
	files = {"nn_syn_eval_data": sorted(glob.glob(path_data[f"nn_syn_eval_data"] + "NN_syn_ret_eval_data_prediction*.nc")),
			}


	# set predictand::
	set_dict['pred'] = var

	print(f"Creating plots for {set_dict['pred']}....")

	# set alternative predictand name to address variables in the eval datasets:
	set_dict['p'] = set_dict['pred']		# to address variables in datasets


	# load MOSAiC observations (and predictions):
	if set_dict['pred'] in ['iwv', 'lwp']:
		NN_MOS_DS = import_hatpro_mirac_level2a_daterange_pangaea(path_data['nn_syn_mosaic'], set_dict['date_0'], set_dict['date_1'], 
															which_retrieval=set_dict['pred'], data_version=set_dict['data_version'][set_dict['pred']])
	elif set_dict['pred'] in ['q', 'temp']:
		NN_MOS_DS = import_hatpro_mirac_level2b_daterange_pangaea(path_data['nn_syn_mosaic'], set_dict['date_0'], set_dict['date_1'], 
															which_retrieval=set_dict['pred'], data_version=set_dict['data_version'][set_dict['pred']],
															around_radiosondes=True, path_radiosondes=path_data['radiosondes'], 
															s_version='level_2', mwr_avg=set_dict['lw'])

		# if temp: also load boundary layer temperature profiles:
		if set_dict['pred'] == 'temp':
			NN_MOS_BL_DS = import_hatpro_mirac_level2c_daterange_pangaea(path_data['nn_syn_mosaic'], set_dict['date_0'], set_dict['date_1'], 
															data_version=set_dict['data_version'][set_dict['pred']],
															around_radiosondes=True, path_radiosondes=path_data['radiosondes'], 
															s_version='level_2', mwr_avg=set_dict['lw']*2)		# OR 1800


	# Load evaluation data set: Find files for current predictand and import data:
	if set_dict['pred'] == 'temp':
		file_nn_syn = [file for file in files["nn_syn_eval_data"] if '_417.nc' in file]
		file_nn_syn_bl = [file for file in files["nn_syn_eval_data"] if '_424.nc' in file]
		NN_SYN_BL_DS = xr.open_dataset(file_nn_syn_bl[0])
	else:
		file_nn_syn = [file for file in files["nn_syn_eval_data"] if set_dict['pred'] in file]
	NN_SYN_DS = xr.open_dataset(file_nn_syn[0])


	# rename variables: cycle through data variables to find those related to the retrieved 
	# quantities: (e.g., prw, prw_offset)
	NN_MOS_DS = unify_names_mos(NN_MOS_DS, set_dict)
	if set_dict['pred'] == 'temp': NN_MOS_BL_DS = unify_names_mos(NN_MOS_BL_DS, set_dict)


	# import radiosonde data
	print("Importing radiosonde data....")
	sonde_dict = import_radiosonde_daterange(path_data['radiosondes'], set_dict['date_0'], set_dict['date_1'], 
											s_version='level_2', remove_failed=True)
	sonde_dict['launch_time_npdt'] = sonde_dict['launch_time'].astype('datetime64[s]')


	# Flag MWR predictions using HATPRO and MiRAC-P flags. Remove values before first successful calibration.
	# Remove values within the Exclusive Economic Zones (EEZs):
	print("Filtering data....")
	if set_dict['pred'] == 'temp':
		NN_MOS_DS, sonde_dict, NN_MOS_BL_DS = filter_data_mosaic(NN_MOS_DS, sonde_dict, NN_MOS_BL_DS)
	else:
		NN_MOS_DS, sonde_dict = filter_data_mosaic(NN_MOS_DS, sonde_dict)
	set_dict['n_sondes'] = len(sonde_dict['launch_time'])


	# convert units:
	if set_dict['pred'] in set_dict['unit_conv_dict'].keys():
		NN_MOS_DS = conv_units(NN_MOS_DS, set_dict, set_dict['p'])
		NN_SYN_DS = conv_units(NN_SYN_DS, set_dict, varname=set_dict['p'])
		NN_SYN_DS = conv_units(NN_SYN_DS, set_dict, varname=set_dict['p']+"_p")
	sonde_dict['q'] *= 1000.0		# to g kg**-1

	# interpolate sonde data to retrieval grid:
	if (set_dict['pred'] in ['q', 'temp']) and ('height_ret' not in sonde_dict.keys()):
		sonde_dict = sonde_to_ret_grid(sonde_dict, NN_MOS_DS.height.values)


	# find overlap of synergetic ret with radiosonde times:
	print("Find overlaps with MOSAiC radiosondes....")
	NN_MOS_DS, sonde_dict = overlap_with_radiosondes(NN_MOS_DS, sonde_dict, set_dict)
	if set_dict['pred'] == 'temp':
		NN_MOS_BL_DS, sonde_dict = overlap_with_radiosondes(NN_MOS_BL_DS, sonde_dict, set_dict, bl_mode=True)


	# compute error stats:
	ret_stats = dict()
	ret_stats['syn'] = compute_error_stats(NN_SYN_DS[set_dict['p']+"_p"].sel(n_rand=set_dict['op_rng'][set_dict['pred']]).values, 
										NN_SYN_DS[set_dict['p']].values, set_dict['pred'], NN_SYN_DS.height)
	ret_stats['syn']['mean_ref'] = NN_SYN_DS[set_dict['p']].values.mean(axis=0)	# average ERA5 profile as reference
	if set_dict['pred'] == 'iwv':
		ret_stats['mos'] = compute_error_stats(NN_MOS_DS[set_dict['p']+"_avg"].values, sonde_dict[set_dict['p'] + "_ret"],
												set_dict['pred'])

	elif set_dict['pred'] in ['temp', 'q']:
		ret_stats['mos'] = compute_error_stats(NN_MOS_DS[set_dict['p']+"_avg"].values, sonde_dict[set_dict['p'] + "_ret"],
												set_dict['pred'], NN_MOS_DS.height)
		ret_stats['mos']['mean_ret'] = np.nanmean(NN_MOS_DS[set_dict['p']+"_avg"].values, axis=0)	# average retrieved profile as reference
		ret_stats['mos']['mean_ref'] = sonde_dict[set_dict['p'] + "_ret"].mean(axis=0)	# average radiosonde profile as reference

		if set_dict['pred'] == 'temp':
			ret_stats['mos_bl'] = compute_error_stats(NN_MOS_BL_DS[set_dict['p']+"_avg"].values, sonde_dict[set_dict['p'] + "_ret"],
													set_dict['pred'], NN_MOS_BL_DS.height)
			ret_stats['syn_bl'] = compute_error_stats(NN_SYN_BL_DS[set_dict['p']+"_p"].sel(n_rand=set_dict['op_rng']['temp_bl']).values, 
											NN_SYN_BL_DS[set_dict['p']].values, set_dict['pred'], NN_SYN_BL_DS.height)


			# also compute error stats for each random seed for BL temperature profiles:
			ret_stats['syn_20_bl'] = dict()
			for rng_seed in NN_SYN_BL_DS.n_rand.values:
				ret_stats['syn_20_bl'][str(rng_seed)] = compute_error_stats(NN_SYN_BL_DS[set_dict['p']+"_p"].sel(n_rand=rng_seed).values, 
															NN_SYN_BL_DS[set_dict['p']].values, set_dict['pred'], NN_SYN_BL_DS.height.values)

			# convert the dictionary containing error estimates for each rng seed to xarray dataset:
			syn_20_ds = xr.Dataset(coords={'n_rand': NN_SYN_BL_DS.n_rand, 'height': NN_SYN_BL_DS.height})
			n_rand = len(syn_20_ds.n_rand)
			n_hgt = len(syn_20_ds.height)

			# init arrays and save dictionary value:
			for err_key in ['rmse_tot', 'bias_tot', 'stddev']:
				syn_20_ds[err_key] = xr.DataArray(np.full((n_rand,n_hgt), np.nan), dims=['n_rand', 'height'])
				for i_r, key in enumerate(ret_stats['syn_20_bl'].keys()):
					syn_20_ds[err_key][i_r,:] = ret_stats['syn_20_bl'][key][err_key]

				# compute mean, std, min, max over 20 rng seeds and also highlight the operational rng seed:
				syn_20_ds[err_key + '_mean'] = syn_20_ds[err_key].mean('n_rand')
				syn_20_ds[err_key + '_std'] = syn_20_ds[err_key].std('n_rand')
				syn_20_ds[err_key + '_max'] = syn_20_ds[err_key].max('n_rand')
				syn_20_ds[err_key + '_min'] = syn_20_ds[err_key].min('n_rand')
				syn_20_ds[err_key + '_op'] = syn_20_ds[err_key].sel(n_rand=set_dict['op_rng']['temp_bl'])

			ret_stats['syn_20_bl'] = syn_20_ds			# overwrite the dictionary
			del syn_20_ds


	# for profiles, ret_stat['syn'] is also needed for all rng seeds (not for IWV because that's computed 
	# differently for each IWV bin):
	if set_dict['pred'] in ['q', 'temp']:
		ret_stats['syn_20'] = dict()
		for rng_seed in NN_SYN_DS.n_rand.values:
			ret_stats['syn_20'][str(rng_seed)] = compute_error_stats(NN_SYN_DS[set_dict['p']+"_p"].sel(n_rand=rng_seed).values, 
														NN_SYN_DS[set_dict['p']].values, set_dict['pred'], NN_SYN_DS.height.values)

		# convert the dictionary containing error estimates for each rng seed to xarray dataset:
		syn_20_ds = xr.Dataset(coords={'n_rand': NN_SYN_DS.n_rand, 'height': NN_SYN_DS.height})
		n_rand = len(syn_20_ds.n_rand)
		n_hgt = len(syn_20_ds.height)

		# init arrays and save dictionary value:
		for err_key in ['rmse_tot', 'bias_tot', 'stddev']:
			syn_20_ds[err_key] = xr.DataArray(np.full((n_rand,n_hgt), np.nan), dims=['n_rand', 'height'])
			for i_r, key in enumerate(ret_stats['syn_20'].keys()):
				syn_20_ds[err_key][i_r,:] = ret_stats['syn_20'][key][err_key]

			# compute mean, std, min, max over 20 rng seeds and also highlight the operational rng seed:
			syn_20_ds[err_key + '_mean'] = syn_20_ds[err_key].mean('n_rand')
			syn_20_ds[err_key + '_std'] = syn_20_ds[err_key].std('n_rand')
			syn_20_ds[err_key + '_max'] = syn_20_ds[err_key].max('n_rand')
			syn_20_ds[err_key + '_min'] = syn_20_ds[err_key].min('n_rand')
			syn_20_ds[err_key + '_op'] = syn_20_ds[err_key].sel(n_rand=set_dict['op_rng'][set_dict['pred']])

		ret_stats['syn_20'] = syn_20_ds			# overwrite the dictionary
		del syn_20_ds


	# visualize errors:
	if set_dict['pred'] == 'iwv':
		ret_stats = iwv_err_diff_comp(NN_MOS_DS, sonde_dict, NN_SYN_DS, set_dict)

	elif set_dict['pred'] == 'q':
		bias_rmse_prof(ret_stats, set_dict, height=NN_SYN_DS.height)

	elif set_dict['pred'] == 'temp':
		bias_rmse_prof_temp(ret_stats, set_dict, height=NN_SYN_DS.height)


	# save error statistics to data set and to file: 
	save_ret_stats(ret_stats, set_dict)
