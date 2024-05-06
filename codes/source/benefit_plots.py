def run_benefit_plots(path_data, path_plots, var, appendix=False):

	"""
	In this script, plots for the analysis of the information benefit (error reduction) will be 
	created. Single instrument retrievals are compared with the synergy. Relative humidity 
	profiles will be computed from retrieved temperature and humidity profiles. Eventually, also 
	data to investigate errors at different atmospheric conditions (cloudiness, 2 m temperature) 
	will be loaded.

	- import MOSAiC obs and predictions for chosen predictand
	- process imported data
	- visualize

	Parameters:
	-----------
	path_data : dict
		Dictionary containing strings of the full paths of the synergetic retrieval output based on
		MOSAiC obs (key 'nn_syn_mosaic'), of the HATPRO-only retrievals ('mwr_pro'), MiRAC-P-only
		IWV retrieval ('nn_mir'), MOSAiC level 2 radiosondes ('radiosondes'), 
		Met City data ('metcity'), Cloudnet + low-level stratus mask + issue flag data set 
		('cloudnet', 'lls_mask' and 'issue_flag').
	path_plots : str
		String indicating where to save the plots.
	var : str
		String indicating which variable to evaluate. Options: 'iwv', 'q', 'temp'
	appendix : bool
		Indicates if this is one of the Appendix figures or main figures. If True, the plots are
		set up as Appendix figure.
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

	from data_tools import (compute_retrieval_statistics, compute_RMSE_profile)
	from import_data import (import_hatpro_mirac_level2a_daterange_pangaea, import_hatpro_mirac_level2b_daterange_pangaea,
							import_hatpro_level2a_daterange_pangaea, import_mirac_level2a_daterange_pangaea, 
							import_hatpro_level2b_daterange_pangaea, import_radiosonde_daterange)
	from met_tools import convert_abshum_to_spechum, convert_abshum_to_relhum


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
	c_H_fade = (0.067,0.29,0.769,0.15)	# MWR_PRO (HATPRO)
	c_M = (0,0.779,0.615)		# MiRAC-P retrieval
	c_G = (0.1,0.1,0.1)			# for grey scale (i.e., profiles)
	c_G_fade = (0.1,0.1,0.1,0.2)
	c_warm = (0.35,0.05,0.07)	# colour for warm conditions
	c_cold = (0.54,0.73,0.43)	# colour for cold conditions


	def preprocess_metcity(DS):

		"""
		Cut all but the desired variables of the MOSAiC Met City data.

		Parameters:
		-----------
		DS : xr dataset
			Dataset containing MetCity data from the MOSAiC expedition.
		"""

		# drop unused variables:
		wanted_vars = ['temp_2m', 'temp_2m_qc']
		for dv in DS.data_vars:
			if not dv in wanted_vars:
				DS = DS.drop(dv)

		# drop values where data quality is bad: DS.qc_flags yields that good quality is only
		# given for _qc == 0:
		DS = DS.sel(time=DS.temp_2m_qc==0)
		DS['temp_2m'] += 273.15		# convert to K

		return DS


	def clear_sky_only(
		DS,
		LLS_DS,
		truncate=True,
		ignore_x_lowest=1):

		"""
		Find clear sky scenes. Following classifications are also considered clear sky:
		aerosols, insects, aerosols & insects. The whole column must be 'clear sky'.
		A mask variable will only be set True if the entire column of a time stamp is
		'clear sky'. Also low level stratus clouds are respected that are missed by 
		Cloudnet but detected by another instrument. Time dependent dimensions will be 
		truncated to clear sky scenes.

		Due to noise in the cloudnet classification (at some time stamps more or less
		randomly one or two pixels of the column are non clear sky although all
		surrounding pixels are clear sky), time stamps where the number of pixels not in
		[0,8,9,10] doesn't exceed 7 are also considered clear sky. Some of the lowest height 
		levels in target_classification are not regarded because they indicated some non-clear
		sky conditions when the webcam clearly showed clear conditions.

		Parameters:
		-----------
		DS : xarray dataset
			Dataset containing Cloudnet target classification data.
		LLS_DS : xarray dataset
			Dataset containing low level stratus cloud masks that could not be identified with Cloudnet
			but were derived from a lidar (Polly).
		truncate : bool
			Defines if time dependent dimensions will be truncated to clear sky only cases.
			It will, if True.
		ignore_x_lowest : int
			Lowest x height levels that will be ignored when searching for noisy pixels.
		"""

		# Mask time stamps:
		n_time = len(DS.time)
		t_mask = np.full((n_time,), False)	# initialise mask with False
		for k in range(n_time):

			if ((np.all((DS.target_classification.values[k,:] == 0) | (DS.target_classification.values[k,:] == 8) |
			(DS.target_classification.values[k,:] == 9) | (DS.target_classification.values[k,:] == 10))) and (LLS_DS.lls_flag.values[k] == 0)):

				t_mask[k] = True

			elif ((np.count_nonzero((DS.target_classification.values[k,:] > 0) & (DS.target_classification.values[k,:] < 8)) <= 5) and 
					(LLS_DS.lls_flag.values[k] == 0)):

				# Make sure the non-clear sky pixels are not 'in a row':
				noisy_pxl = np.where((DS.target_classification.values[k,ignore_x_lowest:] > 0) & 
										(DS.target_classification.values[k,ignore_x_lowest:] < 8))[0]
				if np.all(np.diff(noisy_pxl) > 1): t_mask[k] = True

		if truncate:
			# Truncate time dependent dimensions:
			DS = DS.sel(time=t_mask)

		# set clear sky mask as attribute:
		DS['is_clear_sky'] = xr.DataArray(t_mask, dims=['time'], attrs={'long_name': "Clear sky flag"})

		return DS


	def unify_names(
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
		HAT_DS=xr.Dataset(),
		MIR_DS=xr.Dataset()):

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
		MIR_DS : xarray dataset
			Dataset containing predictions from MiRAC-P's NN retrieval.
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

		if MIR_DS:
			MIR_DS['time_npdt'] = MIR_DS.time.astype('datetime64[s]')

			# filter bad flags:
			ok_idx_m = np.where(MIR_DS['flag'] == 0.0)[0]
			MIR_DS = MIR_DS.isel(time=ok_idx_m)

			MIR_DS = MIR_DS.sel(time=slice(first_calib_HAT_MIR.astype(np.float64),None))
			outside_eez['mir'] = np.full((len(MIR_DS.time),), True)
			for EEZ_range in EEZ_periods_npdt.keys():
				outside_eez['mir'][(MIR_DS.time_npdt >= EEZ_periods_npdt[EEZ_range][0]) &
										(MIR_DS.time_npdt <= EEZ_periods_npdt[EEZ_range][1])] = False
			MIR_DS = MIR_DS.isel(time=outside_eez['mir'])

		return DS, sonde_dict, HAT_DS, MIR_DS


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

		elif set_dict['pred'] in ['temp', 'q', 'rh']:

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


		elif predictand_id in ['temp', 'temp_bl', 'q', 'rh']:

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


			# compute relative errors (might be useful for q profile):
			x_mean = np.nanmean(x_stuff, axis=0)
			error_dict['rmse_tot_rel'] = error_dict['rmse_tot'] / x_mean
			error_dict['bias_tot_rel'] = error_dict['bias_tot'] / x_mean
			error_dict['stddev_rel'] = error_dict['stddev'] / x_mean

		return error_dict


	def compute_error_stats_atm_cond(
		ret_stats,
		NN_MOS_DS, 
		HAT_DS, 
		sonde_dict, 
		set_dict):

		"""
		Compute retrieval error statistics with the function compute_error_stats for different 
		atmospheric conditions where the respective time indices are saved in set_dict['idx_warm']
		and set_dict['idx_cold'].

		Parameters:
		-----------
		ret_stats : dict
			Dictionary containing error stats for different data sets (synergy, HATPRO, ...), which 
			contain dictionaries with different error metrics (rmse, bias, ...).
		NN_MOS_DS : xarray dataset
			Dataset containing predictions made with NN_retrieval.py.save_obs_predictions, imported 
			with import_hatpro_mirac_level2a_daterange_pangaea or 
			import_hatpro_mirac_level2b_daterange_pangaea or ...level2c....
		HAT_DS : xarray dataset
			Dataset containing predictions from HATPRO's MWR_PRO retrieval.
		sonde_dict : dictionary
			Dictionary containing MOSAiC radiosonde level 2 data imported with 
			import_radiosonde_daterange.
		set_dict : dict
			Dictionary containing additional information. Must contain the indices for the atmospheric
			conditions: 'idx_warm' and 'idx_cold'
		"""

		# loop over conditions and compute error stats:
		for cond in ['cold', 'warm']:
			ret_stats[f'mos_{cond}'] = compute_error_stats(NN_MOS_DS[set_dict['pred']+"_avg"].isel(sondes=set_dict[f'idx_{cond}']).values, 
													sonde_dict[set_dict['pred'] + "_ret"][set_dict[f'idx_{cond}'],:],
													set_dict['pred'], NN_MOS_DS.height)
			ret_stats[f'hat_{cond}'] = compute_error_stats(HAT_DS[set_dict['pred']+"_avg"].isel(sondes=set_dict[f'idx_{cond}']).values, 
													sonde_dict[set_dict['pred'] + "_ret"][set_dict[f'idx_{cond}'],:],
													set_dict['pred'], HAT_DS.height)

		return ret_stats


	def iwv_err_diff_comp(
		NN_MOS_DS, 
		HAT_DS,
		MIR_DS,
		sonde_dict,
		set_dict):

		"""
		Visualize the performance of the synergetic Neural Network prediction against a reference 
		predictand (evaluation data set or MOSAiC radiosondes). Here, a composit of RMSE and bias 
		against the reference IWV will be plotted.

		Parameters:
		-----------
		NN_MOS_DS : xarray dataset
			Dataset created with NN_retrieval.py containing predicted MOSAiC observation data. 
		HAT_DS : xarray dataset
			Dataset containing predictions from HATPRO's MWR_PRO retrieval.
		MIR_DS : xarray dataset
			Dataset containing predictions from MiRAC-P's NN retrieval.
		sonde_dict : dict
			Dictionary containing the key 'iwv', which is a 1D array of floats containing the reference
			IWV in kg m**-2. 
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
				RMSE_bins[ibi] = np.sqrt(np.nanmean((prediction[idx_bin] - reference[idx_bin])**2))					# RMSE
				# RMSE_bins[ibi] = np.sqrt(np.nanmean((reference[idx_bin] - (prediction[idx_bin] - bias_ibi))**2))	# BIAS CORR. RMSE
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
		prediction_mos = NN_MOS_DS[set_dict['p']+"_avg"]
		predictand_mos = sonde_dict[set_dict['p']]
		prediction_hat = HAT_DS[set_dict['p']+"_avg"]
		prediction_mir = MIR_DS[set_dict['p']+"_avg"]


		# compute errors for each bin:
		RMSE_bins_mos, BIAS_bins_mos, N_bins_mos = compute_errors_bins(val_bins, prediction_mos.values, predictand_mos)
		RMSE_bins_hat, BIAS_bins_hat, N_bins_hat = compute_errors_bins(val_bins, prediction_hat.values, predictand_mos)
		RMSE_bins_mir, BIAS_bins_mir, N_bins_mir = compute_errors_bins(val_bins, prediction_mir.values, predictand_mos)


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
		rmse_mos_plot, = a1.plot(val_bins_plot, RMSE_bins_mos, color=c_S, linewidth=2, label='RMSE')
		bias_mos_plot, = a1.plot(val_bins_plot, BIAS_bins_mos, color=c_S, linewidth=2, linestyle='dashed', label='Bias')

		rmse_hat_plot, = a1.plot(val_bins_plot, RMSE_bins_hat, color=c_H, linewidth=2, label='RMSE')
		bias_hat_plot, = a1.plot(val_bins_plot, BIAS_bins_hat, color=c_H, linewidth=2, linestyle='dashed', label='Bias')

		rmse_mir_plot, = a1.plot(val_bins_plot, RMSE_bins_mir, color=c_M, linewidth=2, label='RMSE')
		bias_mir_plot, = a1.plot(val_bins_plot, BIAS_bins_mir, color=c_M, linewidth=2, linestyle='dashed', label='Bias')


		# Legends:
		leg_title = "Synergy"
		if set_dict['appendix']: leg_title="NN K-band"
		leg_syn = a1.legend(handles=[rmse_mos_plot, bias_mos_plot], loc='lower left', bbox_to_anchor=(0.01, 0.00), fontsize=fs_micro-4, 
							framealpha=0.5, title=leg_title, title_fontsize=fs_micro-4)
		a1.add_artist(leg_syn)
		leg_hat = a1.legend(handles=[rmse_hat_plot, bias_hat_plot], loc='lower center', bbox_to_anchor=(0.50, 0.00), fontsize=fs_micro-4, 
							framealpha=0.5, title='HATPRO', title_fontsize=fs_micro-4)
		a1.add_artist(leg_hat)
		leg_mir = a1.legend(handles=[rmse_mir_plot, bias_mir_plot], loc='lower right', bbox_to_anchor=(0.99, 0.00), fontsize=fs_micro-4,
							framealpha=0.5, title='MiRAC-P', title_fontsize=fs_micro-4)
		a1.add_artist(leg_mir)


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
		a1.set_xlabel("Radiosonde IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs_dwarf)

		if set_dict['save_figures']:
			plotname = f"MOSAiC_nn_syn_mwr_pro_nn_mir_{set_dict['p']}_err_diff_comp"
			if set_dict['appendix']: plotname = f"MOSAiC_KbandTB_only_NN_vs_mwr_pro_vs_nn_mir_{set_dict['p']}_err_diff_comp"
			plotfile = plotpath_dir + "/" + plotname
			f1.savefig(plotfile + ".png", dpi=300, bbox_inches='tight')
			f1.savefig(plotfile + ".pdf", bbox_inches='tight')

			print(f"Saved {plotfile}.pdf ....")
		else:
			plt.show()
			pdb.set_trace()

		plt.close()
		gc.collect()


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
		STD_mos_op = ret_stats['mos']['stddev']
		BIAS_mos_op = ret_stats['mos']['bias_tot']

		STD_hat = ret_stats['hat']['stddev']
		BIAS_hat = ret_stats['hat']['bias_tot']

		# convert units if needed:
		if set_dict['pred'] == 'rh':
			STD_mos_op *= 100.
			BIAS_mos_op *= 100.
			STD_hat *= 100.
			BIAS_hat *= 100.


		# dictionaries for adaptations:
		label_size_big = fs_dwarf
		label_size_small = fs_micro-4

		panel_id_dict = dict()		# dictionary containing further properties of the figure panel identifiers
		legend_pos = 'upper right'
		anchor_pos = (0.98, 0.99)
		bias_label = {'q': "$\mathrm{Bias}_{\mathrm{q}}$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)",
						'rh': "$\mathrm{Bias}_{\mathrm{RH}}$ (%)"}
		std_label = {'q': "$\mathrm{RMSE}_{\mathrm{corr, q}}$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)",
						'rh': "$\mathrm{RMSE}_{\mathrm{corr, RH}}$ (%)"}
		rel_bias_label = {'q': "Relative $\mathrm{bias}_{\mathrm{q}}$ (%)",
						'rh': "Relative $\mathrm{bias}_{\mathrm{RH}}$ (%)"}
		rel_std_label = {'q': "Relative $\mathrm{RMSE}_{\mathrm{corr, q}}$ (%)",
						'rh': "Relative $\mathrm{RMSE}_{\mathrm{corr, RH}}$ (%)"}


		f1 = plt.figure(figsize=(10,6.5))
		ax_bias = plt.subplot2grid((1,2), (0,0))			# bias profile
		ax_std = plt.subplot2grid((1,2), (0,1))				# std dev profile

		y_lim = np.array([0.0, height.max()])
		x_lim_std = {'q': np.array([0.0, 0.8]),		# in g kg-1,
					'rh': np.array([0, 80])}			# in %
		x_lim_bias = {'q': np.array([-0.5, 0.5]),	# in g kg-1
					'rh': np.array([-75, 75])}		# in %


		# bias profiles:
		ax_bias.plot(np.full_like(height, 0.0), height, color=(0,0,0), linewidth=1.0)	# helper line
		ax_bias.plot(BIAS_hat, height, color=c_H, linewidth=2)
		ax_bias.plot(BIAS_mos_op, height, color=c_S, linewidth=2)


		# std dev profiles:
		plot_label = "Synergy"
		if set_dict['appendix']: plot_label = "NN K-band"
		ax_std.plot(STD_hat, height, color=c_H, linewidth=2, label='HATPRO')
		ax_std.plot(STD_mos_op, height, color=c_S, linewidth=2, label=plot_label)


		# add relative errors (e.g., for q profiles):
		rel_plots = False			# boolean indicator if relative plots are added
		if (set_dict['pred'] == 'q') and (not set_dict['appendix']):
			rel_plots = True
			ax_bias2 = ax_bias.twiny()
			ax_bias2.plot(ret_stats['hat']['bias_tot_rel']*100., height, color=c_H, linestyle='dashed', linewidth=1.5)
			ax_bias2.plot(ret_stats['mos']['bias_tot_rel']*100., height, color=c_S, linestyle='dashed', linewidth=1.5)

			ax_std2 = ax_std.twiny()
			ax_std2.plot(ret_stats['hat']['stddev_rel']*100., height, color=c_H, linestyle='dashed', linewidth=1.5, label='HATPRO')
			ax_std2.plot(ret_stats['mos']['stddev_rel']*100., height, color=c_S, linestyle='dashed', linewidth=1.5, label='Synergy')


		# add errors for different atmospheric conditions if desired (rather only for
		# relative humidity because 'q' plot is already quite busy:
		if set_dict['atm_cond'] and (set_dict['pred'] == 'rh'):

			# write out the important error stats: also convert units
			COLD_STD_mos_op = ret_stats['mos_cold']['stddev']*100.
			COLD_BIAS_mos_op = ret_stats['mos_cold']['bias_tot']*100.
			COLD_STD_hat = ret_stats['hat_cold']['stddev']*100.
			COLD_BIAS_hat = ret_stats['hat_cold']['bias_tot']*100.

			WARM_STD_mos_op = ret_stats['mos_warm']['stddev']*100.
			WARM_BIAS_mos_op = ret_stats['mos_warm']['bias_tot']*100.
			WARM_STD_hat = ret_stats['hat_warm']['stddev']*100.
			WARM_BIAS_hat = ret_stats['hat_warm']['bias_tot']*100.

			# plot data:
			ax_bias.plot(COLD_BIAS_hat, height, color=c_H, linewidth=1.5, linestyle='dotted')
			ax_bias.plot(COLD_BIAS_mos_op, height, color=c_S, linewidth=1.5, linestyle='dotted')
			ax_bias.plot(WARM_BIAS_hat, height, color=c_H, linewidth=1.5, linestyle='dashed')
			ax_bias.plot(WARM_BIAS_mos_op, height, color=c_S, linewidth=1.5, linestyle='dashed')

			ax_std.plot(COLD_STD_hat, height, color=c_H, linewidth=1.5, linestyle='dotted', label='HATPRO, cold')
			ax_std.plot(COLD_STD_mos_op, height, color=c_S, linewidth=1.5, linestyle='dotted', label='Synergy, cold')
			ax_std.plot(WARM_STD_hat, height, color=c_H, linewidth=1.5, linestyle='dashed', label='HATPRO, warm')
			ax_std.plot(WARM_STD_mos_op, height, color=c_S, linewidth=1.5, linestyle='dashed', label='Synergy, warm')


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
			leg2 = ax_bias2.legend(lh, ll, loc=legend_pos, bbox_to_anchor=(anchor_pos[0], 0.78),
							fontsize=label_size_small, framealpha=0.5, title='Relative errors', title_fontsize=label_size_small)

			# adapt axis properties:
			ax_bias2.set_xlim(-50.,50.)
			ax_std2.set_xlim(0., 120.)
			for ax in [ax_bias2, ax_std2]: ax.tick_params(axis='both', labelsize=label_size_small)
			ax_bias2.set_xlabel(rel_bias_label[set_dict['pred']], fontsize=label_size_big, labelpad=8)
			ax_std2.set_xlabel(rel_std_label[set_dict['pred']], fontsize=label_size_big, labelpad=8)

		else:
			leg1 = ax_std.legend(lh, ll, loc=legend_pos, bbox_to_anchor=anchor_pos, 
							fontsize=label_size_small, framealpha=0.5)



		# axis lims:
		ax_bias.set_xlim(left=x_lim_bias[set_dict['pred']][0], right=x_lim_bias[set_dict['pred']][1])
		ax_std.set_xlim(left=x_lim_std[set_dict['pred']][0], right=x_lim_std[set_dict['pred']][1])


		for ax in [ax_bias, ax_std]:
			ax.set_ylim(bottom=y_lim[0], top=y_lim[1])

			ax.minorticks_on()
			ax.tick_params(axis='both', labelsize=label_size_small)
			ax.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# remove tick labels for y axis:
		ax_std.yaxis.set_ticklabels([])


		# labels:
		ax_bias.set_xlabel(bias_label[set_dict['pred']], fontsize=label_size_big)
		ax_std.set_xlabel(std_label[set_dict['pred']], fontsize=label_size_big)
		ax_bias.set_ylabel("Height (m)", fontsize=label_size_big)


		# figure labels:
		ax_bias.set_title("a)", loc='left', fontsize=label_size_big, **panel_id_dict)
		ax_std.set_title("b)", loc='left', fontsize=label_size_big, **panel_id_dict)

		plt.tight_layout()

		if set_dict['save_figures']:
			plotname = f"MOSAiC_nn_syn_mwr_pro_{set_dict['pred']}_err_profs"
			if set_dict['appendix']: plotname = "q_err_NN_KbandTBs_ONLY"
			if (set_dict['pred'] == 'rh') and set_dict['atm_cond']: plotname += "_atm_conditions"
			plotfile = plotpath_dir + "/" + plotname
			f1.savefig(plotfile + ".png", dpi=300, bbox_inches='tight')
			f1.savefig(plotfile + ".pdf", bbox_inches='tight')

			print(f"Saved {plotfile}.pdf ....")
		else:
			plt.show()
			pdb.set_trace()

		plt.close()
		gc.collect()


	def error_stat_table(
		ret_stats,
		set_dict,
		err_type):

		"""
		Visualize error stats (rmse, bias, std dev) for different regimes (height levels or predictand
		ranges) as a table for all available retrieval types.

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
		err_type : str
			String indicating the error type ("rmse", "bias", "stddev").
		"""

		# create output path if not existing:
		plotpath_dir = os.path.dirname(set_dict['path_plots'] + "err_tables/")
		if not os.path.exists(plotpath_dir):
			os.makedirs(plotpath_dir)


		# create table text (2D list of str):
		# First line: retrieval types and regime specifier
		retrieval_types = {'mos': "Synergy", 'hat': "HATPRO", 'mir': "MiRAC-P"}
		regime_specifier = {'iwv': "IWV range ($\mathrm{kg}\,\mathrm{m}^{-2}$)",
							'temp': "Height range (m)", 'temp_bl': "Height range (m)",
							'q': "Height range (m)", 'rh': "Height range (m)"}
		regimes = {'iwv': ["[0, 5)", "[5, 10)", "[10, 100)"], 
					'temp': ["[0, 1500)", "[1500, 5000)", "[5000, 15000)"],
					'temp_bl': ["[0, 1500)", "[1500, 5000)", "[5000, 15000)"],
					'rh': ["[0, 1500)", "[1500, 5000)", "[5000, 15000)"],
					'q': ["[0, 1500)", "[1500, 5000)", "[5000, 15000)"]}
		title_labels = {'iwv': "Integreated Water Vapour", 
						'temp': "Temperature profile", 
						'temp_bl': "Boundary layer temperature profile", 
						'q': "Specific humidity profile",
						'rh': "Relative humidity profile"}
		error_types = {'rmse': ['rmse_bot', 'rmse_mid', 'rmse_top', 'rmse_tot'],
						'bias': ['bias_bot', 'bias_mid', 'bias_top', 'bias_tot'],
						'stddev': ['stddev_bot', 'stddev_mid', 'stddev_top', 'stddev']}
		error_labels = {'rmse': "RMSE", 'bias': "Bias", 'stddev': "RMSE$_{\mathrm{corr}}$"}
		cell_text = list()

		# table header:
		table_header = list()
		table_header.append(regime_specifier[set_dict['pred']])
		for ret_type in retrieval_types.keys():
			table_header.append(retrieval_types[ret_type])

		# first line:
		cell_text.append(table_header)

		# second line: lowest/middle/highest/total regime + respective errors:
		# loop through retrieval types to get error estimates:
		err_bot = [regimes[set_dict['pred']][0]]
		err_mid = [regimes[set_dict['pred']][1]]
		err_top = [regimes[set_dict['pred']][2]]
		err_tot = ["Total"]
		for ret_type in retrieval_types.keys():
			if ret_type in ret_stats.keys():
				err_bot.append(f"{ret_stats[ret_type][error_types[err_type][0]]:.2f}")
				err_mid.append(f"{ret_stats[ret_type][error_types[err_type][1]]:.2f}")
				err_top.append(f"{ret_stats[ret_type][error_types[err_type][2]]:.2f}")
				if set_dict['pred'] in ['temp', 'temp_bl', 'q', 'rh']:
					# average over height levels:
					err_tot.append(f"{ret_stats[ret_type][error_types[err_type][3]].mean():.2f}")
				else:
					err_tot.append(f"{ret_stats[ret_type][error_types[err_type][3]]:.2f}")
			else:
				err_bot.append("-")
				err_mid.append("-")
				err_top.append("-")
				err_tot.append("-")

			
		cell_text.append(err_bot)
		cell_text.append(err_mid)
		cell_text.append(err_top)
		cell_text.append(err_tot)
		n_cols = len(cell_text[0])

		# make sure each row has got the same number of cells:
		for c_t in cell_text:
			assert n_cols == len(c_t)


		# create figure for table:
		f1 = plt.figure(figsize=(9,9))
		a1 = plt.axes()
		a1.set_axis_off()


		col_cols = list()
		for k in range(n_cols): col_cols.append((0.8,0.8,0.8))
		a1.table(cell_text[1:], cellLoc='right', rowLoc='center', colLabels=cell_text[0], colColours=col_cols, colLoc='center')

		a1.set_title(f"{title_labels[set_dict['pred']]} {error_labels[err_type]}", fontsize=fs_micro-4)

		if set_dict['save_figures']:
			plotname = f"MOSAiC_nn_syn_mwr_pro_nn_mir_{set_dict['pred']}_{err_type}_table"
			f1.savefig(plotpath_dir + "/" + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()


		plt.close()
		gc.collect()


	# settings:
	set_dict = {'lw': 900,					# time window (in sec) from radiosonde launch_time to launch_time+lw for averaging
				'date_0': "2019-09-20",		# lower limit of dates for import of mosaic data (default: "2019-09-20")
				'date_1': "2020-10-12",		# upper limit of dates (default: "2020-10-12")
				'data_version': {'iwv': 'v00',
								'temp': 'v00', 'temp_bl': 'v00', 'rh': 'v00',
								'q': 'v00'},					# indicates the version of the mosaic data to be used
				'atm_cond': True,			# if True, error will be visualized for different atm. conditions (evtl. not for each 
											# element of ['iwv', 'q', 'rh'])
				'save_figures': True,		# if True, plot is saved to file. If False, it won't.
				'print_error_stats': True,	# if True, error stats will be printed
				'appendix': appendix,		# if True, plots are adapted for supplemental/appendix figures
				}

	# set paths:
	set_dict['path_plots'] = path_plots

	# dictionary to convert units: [offset value, factor to multiply data with]: converted = (original+offset)*factor
	# keys have to be in ['iwv', 'q', 'rh']!
	set_dict['unit_conv_dict'] = {'q': [0.0, 1000.]}		# from kg kg-1 to g kg-1

	set_dict['translate_dict'] = {'prw': 'iwv'}				# to translate variable and file names
	set_dict['bl_mode'] = False								# if boundary layer temperature profiles are selected


	# set predictand:
	set_dict['pred'] = var
	print(f"Creating plots for {set_dict['pred']}....")

	# set alternative predictand name to address variables in the eval datasets:
	if set_dict['pred'] in ['iwv', 'q']:
		set_dict['p'] = set_dict['pred']		# to address variables in datasets
	else:
		set_dict['p'] = ['q', 'temp']


	# load MOSAiC observations (and predictions):
	if set_dict['pred'] in ['iwv', 'lwp']:
		NN_MOS_DS = import_hatpro_mirac_level2a_daterange_pangaea(path_data['nn_syn_mosaic'], set_dict['date_0'], set_dict['date_1'], 
															which_retrieval=set_dict['pred'], data_version=set_dict['data_version'][set_dict['pred']])

	elif set_dict['pred'] == 'q':
		NN_MOS_DS = import_hatpro_mirac_level2b_daterange_pangaea(path_data['nn_syn_mosaic'], set_dict['date_0'], set_dict['date_1'], 
															which_retrieval=set_dict['pred'], data_version=set_dict['data_version'][set_dict['pred']],
															around_radiosondes=True, path_radiosondes=path_data['radiosondes'], 
															s_version='level_2', mwr_avg=set_dict['lw'])

	elif set_dict['pred'] == 'rh':
		NN_MOS_DS = import_hatpro_mirac_level2b_daterange_pangaea(path_data['nn_syn_mosaic'], set_dict['date_0'], set_dict['date_1'], 
															which_retrieval='rh', data_version=set_dict['data_version']['rh'],
															around_radiosondes=True, path_radiosondes=path_data['radiosondes'], 
															s_version='level_2', mwr_avg=set_dict['lw'])


	# load old retrievals:
	print("Importing single-instrument retrievals....")
	HAT_DS = xr.Dataset()
	MIR_DS = xr.Dataset()
	if set_dict['pred'] == 'iwv':		# load MWR_PRO and MiRAC-P
		which_retrieval = {'iwv': 'prw', 'lwp': 'clwvi'}		# to translate between old and new names
		hatpro_dict = import_hatpro_level2a_daterange_pangaea(path_data['mwr_pro'], set_dict['date_0'], set_dict['date_1'], which_retrieval='iwv')
		mirac_dict = import_mirac_level2a_daterange_pangaea(path_data['nn_mir'], set_dict['date_0'], set_dict['date_1'], which_retrieval='iwv')

		# also needs to be reduced to times around radiosonde launches:
		hatpro_dict['time_npdt'] = hatpro_dict['time'].astype("datetime64[s]")
		mirac_dict['time_npdt'] = mirac_dict['time'].astype("datetime64[s]")

		# put into dataset:
		HAT_DS = xr.Dataset(coords={'time': (['time'], hatpro_dict['time'])})
		HAT_DS['flag'] = xr.DataArray(hatpro_dict['flag'], dims=['time'])
		HAT_DS['prw'] = xr.DataArray(hatpro_dict['prw'], dims=['time'])
		MIR_DS = xr.Dataset(coords={'time': (['time'], mirac_dict['time'])})
		MIR_DS['flag'] = xr.DataArray(mirac_dict['flag'], dims=['time'])
		MIR_DS['prw'] = xr.DataArray(mirac_dict['prw'], dims=['time'])

		# unify variable names: 
		HAT_DS = unify_names(HAT_DS, set_dict)
		MIR_DS = unify_names(MIR_DS, set_dict)

		del hatpro_dict, mirac_dict		# to save memory

	elif set_dict['pred'] in ['q', 'rh']:
		which_retrieval = {'temp': 'ta', 'q': 'hus'}		# to translate between old and new names
		which_retrieval_keys = {'temp': 'ta', 'q': 'hua'}	# to translate between old and new names
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


	# rename variables: cycle through data variables to find those related to the retrieved 
	# quantities: (e.g., prw, prw_offset)
	NN_MOS_DS = unify_names(NN_MOS_DS, set_dict)


	# import radiosonde data
	print("Importing radiosonde data....")
	sonde_dict = import_radiosonde_daterange(path_data['radiosondes'], set_dict['date_0'], set_dict['date_1'], 
											s_version='level_2', remove_failed=True)
	sonde_dict['launch_time_npdt'] = sonde_dict['launch_time'].astype('datetime64[s]')
		

	# Flag MWR predictions using HATPRO and MiRAC-P flags. Remove values before first successful calibration.
	# Remove values within the Exclusive Economic Zones (EEZs):
	print("Filtering data....")
	NN_MOS_DS, sonde_dict, HAT_DS, MIR_DS = filter_data_mosaic(NN_MOS_DS, sonde_dict, HAT_DS, MIR_DS)
	set_dict['n_sondes'] = len(sonde_dict['launch_time'])


	# convert units: for HATPRO, first create a dummy specific humidity variable, which is absolute
	# humidity for now:
	if type(set_dict['p']) == type(list()): # set_dict['p'] contains two predictands: q and temp
		for pred in set_dict['p']:
			if (pred in set_dict['unit_conv_dict'].keys()) and (pred in NN_MOS_DS.data_vars):
				NN_MOS_DS = conv_units(NN_MOS_DS, set_dict, pred)

		HAT_DS['q'] = xr.DataArray(HAT_DS.rho_v.values, dims=['time', 'height'])

	elif set_dict['p'] in set_dict['unit_conv_dict'].keys():
		NN_MOS_DS = conv_units(NN_MOS_DS, set_dict, set_dict['p'])
		HAT_DS['q'] = xr.DataArray(HAT_DS.rho_v.values, dims=['time', 'height'])	# actually abs. humidity in kg m-3

	sonde_dict['q'] *= 1000.0		# to g kg**-1


	# interpolate sonde data to retrieval grid:
	if (set_dict['pred'] in ['q', 'temp', 'rh']) and ('height_ret' not in sonde_dict.keys()):
		sonde_dict = sonde_to_ret_grid(sonde_dict, NN_MOS_DS.height.values)


	# find overlap of synergetic ret with radiosonde times:
	print("Find overlaps with MOSAiC radiosondes....")
	NN_MOS_DS = overlap_with_radiosondes(NN_MOS_DS, sonde_dict, set_dict)

	# because the synergy combines flags from both data sets, it has the min overlap with
	# radiosondes. Also reduce HATPRO and MiRAC-P data to that minimum
	if set_dict['pred'] == 'rh':
		intersct_overlap = ~np.isnan(NN_MOS_DS[f"{set_dict['pred']}_avg"])
	else:
		intersct_overlap = ~np.isnan(NN_MOS_DS[f"{set_dict['p']}_avg"])
	if intersct_overlap.ndim > 1: intersct_overlap = intersct_overlap.min('height')

	if HAT_DS: 
		HAT_DS = overlap_with_radiosondes(HAT_DS, sonde_dict, set_dict)

		# limit to radiosondes that are included in all two-three (NN_MOS_DS, HAT_DS, MIR_DS) data sets:
		if set_dict['pred'] == 'rh':
			for dvv in set_dict['p']:	# loop over [q, temp]
				HAT_DS[f"{dvv}_avg"] = HAT_DS[f"{dvv}_avg"].where(intersct_overlap, other=np.nan)
				HAT_DS[f"{dvv}_std"] = HAT_DS[f"{dvv}_std"].where(intersct_overlap, other=np.nan)
		else:
			HAT_DS[f"{set_dict['p']}_avg"] = HAT_DS[f"{set_dict['p']}_avg"].where(intersct_overlap, other=np.nan)
			HAT_DS[f"{set_dict['p']}_std"] = HAT_DS[f"{set_dict['p']}_std"].where(intersct_overlap, other=np.nan)

	if MIR_DS: 
		MIR_DS = overlap_with_radiosondes(MIR_DS, sonde_dict, set_dict)
		MIR_DS[f"{set_dict['p']}_avg"] = MIR_DS[f"{set_dict['p']}_avg"].where(intersct_overlap, other=np.nan)
		MIR_DS[f"{set_dict['p']}_std"] = MIR_DS[f"{set_dict['p']}_std"].where(intersct_overlap, other=np.nan)


	# load data to distinguish between different atmospheric conditions
	if set_dict['atm_cond'] and (set_dict['pred'] in ['rh']):

		print("Importing Met City and Cloudnet data....")

		# import Met City data:
		files = sorted(glob.glob(path_data['metcity'] + "mosmet.metcity.level3.4*.nc"))
		MET_DS = xr.open_mfdataset(files, concat_dim='time', combine='nested', preprocess=preprocess_metcity)		

		# import Cloudnet data:
		files = sorted(glob.glob(path_data['cloudnet'] + "*rv-polarstern_classification.nc"))
		CL_DS = xr.open_mfdataset(files, concat_dim='time', combine='nested')

		# add low level stratus mask: also have to repair the time axis because there are duplicates and jumps,
		# e.g. 2020-01-07T23:59:30 is followed by 2020-01-07T00:00:00 and then 2020-01-08T00:00:30.
		files = sorted(glob.glob(path_data['lls_mask'] + "*_polarstern_lls.nc"))
		LLS_DS = xr.open_mfdataset(files, concat_dim='time', combine='nested')
		LLS_DS = LLS_DS.sel(time=(LLS_DS.blowing_snow_flag==0))		# filter data that might be blowing snow events
		skip_jumps = np.where(np.diff(LLS_DS.time.values) > np.timedelta64(0,"s"))[0] + 1
		LLS_DS = LLS_DS.isel(time=skip_jumps)		# filters out these jumps in the time axis

		# add the cloudnet issue data set to flag data that is not reliable due to external influences:
		files = sorted(glob.glob(path_data['issue_flag'] + "*_polarstern_issues.nc"))
		ISS_DS = xr.open_mfdataset(files, concat_dim='time', combine='nested')


		# overlap with MWR data: preselect and then check if a tolerance of 900 sec is overshot:
		MET_DS = MET_DS.sel(time=sonde_dict['launch_time_npdt'], method='nearest')
		where_in_bounds = np.abs(MET_DS.time - sonde_dict['launch_time_npdt']) <= np.timedelta64(set_dict['lw'], "s")
		MET_DS = MET_DS.where(where_in_bounds)

		def time_overlap(time_t, reftimes):
			# Find overlaps of time with radiosonde times (as datetime64[s] or datetime64[ns] arrays):
			last_overlap_idx = 0		# index to reduce searching time
			time_idx = np.array([], dtype=np.int32)
			for lt in reftimes:
				overlap_idx = np.where((time_t >= lt) & (time_t < lt+np.timedelta64(set_dict['lw'], "s")))[0] + last_overlap_idx
				time_idx = np.concatenate((time_idx, overlap_idx))

				# remove times that were already checked (irrelevant for the upcoming radiosondes):
				if len(overlap_idx) > 0:
					time_t = time_t[(overlap_idx[-1]-last_overlap_idx):]
					last_overlap_idx = overlap_idx[-1]

			return time_idx

		time_idx = time_overlap(ISS_DS.time.values, sonde_dict['launch_time_npdt'])
		ISS_DS = ISS_DS.isel(time=time_idx)
		time_idx = time_overlap(CL_DS.time.values, sonde_dict['launch_time_npdt'])
		CL_DS = CL_DS.isel(time=time_idx)
		time_idx = time_overlap(LLS_DS.time.values, sonde_dict['launch_time_npdt'])
		LLS_DS = LLS_DS.isel(time=time_idx)

		# Bring issue data set on CL_DS time axis and filter CL_DS for time steps when
		# no issues were indicated:
		ISS_DS = ISS_DS.sel(time=CL_DS.time, method='nearest')
		ISS_DS = ISS_DS.load()
		where_in_bounds = np.abs(ISS_DS.time.values - CL_DS.time.values) <= np.timedelta64(set_dict['lw'], "s")
		ISS_DS['issue_bit'] = ISS_DS.issue_bit.where(where_in_bounds, other=np.nan)
		ISS_ok = (ISS_DS.issue_bit.values == 0) | (ISS_DS.issue_bit.values == 1)		# where no significant issues exist
		CL_DS = CL_DS.where(xr.DataArray(ISS_ok, dims=['time']))


		LLS_DS = LLS_DS.sel(time=CL_DS.time, method='nearest')
		LLS_DS = LLS_DS.load()
		where_in_bounds = np.abs(LLS_DS.time.values - CL_DS.time.values) <= np.timedelta64(set_dict['lw'], "s")
		LLS_DS['lls_flag'] = LLS_DS.lls_flag.where((where_in_bounds & xr.DataArray(ISS_ok, dims=['time'])), other=1.)

		MET_DS = MET_DS.load()
		CL_DS = CL_DS.drop('detection_status')
		CL_DS = CL_DS.load()

		# create clear sky flag:
		CL_DS = clear_sky_only(CL_DS, LLS_DS, truncate=False, ignore_x_lowest=1)

		# now, a radiosonde is considered clear sky if all CL_DS clear sky flag values within 15 min of
		# the radiosonde launch indicate clear sky conditions:
		sonde_idx = np.ones(CL_DS.time.shape)*(-1)
		last_overlap_idx = 0
		time_temp = CL_DS.time.values
		for k, lt in enumerate(sonde_dict['launch_time_npdt']):
			overlap_idx = np.where((time_temp >= lt) & (time_temp < lt+np.timedelta64(set_dict['lw'], "s")))[0] + last_overlap_idx
			sonde_idx[overlap_idx] = k

			# remove times that were already checked (irrelevant for the upcoming radiosondes):
			if len(overlap_idx) > 0:
				time_temp = time_temp[(overlap_idx[-1]-last_overlap_idx):]
				last_overlap_idx = overlap_idx[-1]


		# group by sonde index and select min of the clear sky flag:
		CL_DS['sonde_idx'] = xr.DataArray(sonde_idx, dims=['time'])
		sonde_idx = np.unique(sonde_idx)
		CL_DS['is_clear_sky'] = CL_DS.is_clear_sky.astype(np.int32)
		CL_DS_grouped = CL_DS.groupby('sonde_idx').min('time')

		# broadcast clear sky flag back to radiosonde array shape:
		clear_sky_flag = np.zeros(sonde_dict['launch_time_npdt'].shape)
		clear_sky_flag[sonde_idx.astype(np.int32)] = CL_DS_grouped.is_clear_sky.values
		clear_sky_flag = clear_sky_flag.astype('bool')		


		# dinstinguish atmospheric conditions: find radiosonde indices where conditions are cold and
		# where conditions are warm:
		set_dict['idx_cold'] = np.where((clear_sky_flag) & (MET_DS.temp_2m.values < 273.15) & (sonde_dict['iwv'] < 10))[0]
		set_dict['idx_warm'] = np.where((MET_DS.temp_2m.values >= 273.15) & (sonde_dict['iwv'] >= 10))[0]


	# Convert rho_v to specific humidity for HATPRO or compute relative humidity from temperature 
	# and humidity retrievals:
	if set_dict['pred'] == 'q':
		HAT_DS['q_avg'][:] = convert_abshum_to_spechum(HAT_DS.temp_avg.values, sonde_dict['pres_ret'], HAT_DS.q_avg.values)*1000.
	elif set_dict['pred'] == 'rh':
		HAT_DS['rh_avg'] = convert_abshum_to_relhum(HAT_DS.temp_avg, HAT_DS.q_avg)

		# limit HATPRO to 8000 m:
		HAT_DS = HAT_DS.sel(height=NN_MOS_DS.height)


	# compute error stats:
	ret_stats = dict()
	if set_dict['pred'] == 'iwv':
		sonde_dict[set_dict['pred'] + "_ret"] = sonde_dict[set_dict['pred']]	# renamed for comparison
		ret_stats['mos'] = compute_error_stats(NN_MOS_DS[set_dict['p']+"_avg"].values, sonde_dict[set_dict['p'] + "_ret"],
												set_dict['pred'])
		ret_stats['hat'] = compute_error_stats(HAT_DS[set_dict['p']+"_avg"].values, sonde_dict[set_dict['p'] + "_ret"],
												set_dict['pred'])
		ret_stats['mir'] = compute_error_stats(MIR_DS[set_dict['p']+"_avg"].values, sonde_dict[set_dict['p'] + "_ret"],
												set_dict['pred'])

	elif set_dict['pred'] == 'q':
		ret_stats['mos'] = compute_error_stats(NN_MOS_DS[set_dict['p']+"_avg"].values, sonde_dict[set_dict['p'] + "_ret"],
												set_dict['pred'], NN_MOS_DS.height)
		ret_stats['hat'] = compute_error_stats(HAT_DS[set_dict['p']+"_avg"].values, sonde_dict[set_dict['p'] + "_ret"],
												set_dict['pred'], HAT_DS.height)

	elif set_dict['pred'] == 'rh':
		ret_stats['mos'] = compute_error_stats(NN_MOS_DS[set_dict['pred']+"_avg"].values, sonde_dict[set_dict['pred'] + "_ret"],
												set_dict['pred'], NN_MOS_DS.height)
		ret_stats['hat'] = compute_error_stats(HAT_DS[set_dict['pred']+"_avg"].values, sonde_dict[set_dict['pred'] + "_ret"],
												set_dict['pred'], HAT_DS.height)


	# also compute error stats for different atmospheric conditions if desired:
	if set_dict['atm_cond'] and (set_dict['pred'] in ['rh']):
		ret_stats = compute_error_stats_atm_cond(ret_stats, NN_MOS_DS, HAT_DS, sonde_dict, set_dict)


	# visualize errors:
	if set_dict['pred'] == 'iwv':
		iwv_err_diff_comp(NN_MOS_DS, HAT_DS, MIR_DS, sonde_dict, set_dict)

	elif set_dict['pred'] in ['q', 'rh']:
		bias_rmse_prof(ret_stats, set_dict, height=NN_MOS_DS.height)


	# if desired, error stats can also be printed as table:
	if set_dict['print_error_stats']:
		error_stat_table(ret_stats, set_dict, err_type='rmse')
		error_stat_table(ret_stats, set_dict, err_type='bias')
		error_stat_table(ret_stats, set_dict, err_type='stddev')