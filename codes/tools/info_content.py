from copy import deepcopy
import datetime as dt
import multiprocessing
import sys
import os

wdir = os.getcwd() + "/"
remote = "/net/blanc/" in wdir

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(wdir[:-1]) + "/")
from data_tools import *
from met_tools import *

import pyPamtra
import pdb

os.environ['OPENBLAS_NUM_THREADS'] = "1"


# some general settings for plots:
fs = 14
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15


class info_content:
	"""
		Compute information content based on optimal estimation theory. 

		Option 1 for degrees of freedom:
		To obtain the degrees of freedom (DOF) from the Averaging Kernel (AK), we firstly need to 
		perturb each component of the state vector (x -> x') of the (eval) data sample step by 
		step to compute a new set of perturbed observations (y'). Then proceed to compute the 
		Jacobian matrix K (roughly dF(x)/dx ~ dy/dx). To obtain the AK, covariance matrix of the
		state vector (S_a) over the eval data set and the observation error covariance matrix
		(S_eps) must be provided or computed. Then, the AK is computed as 
		(K.T*S_eps^(-1)*K + S_a^(-1))^(-1) * K.T*S_eps^(-1)*K .

		Option 2 for degrees of freedom: 
		To obtain the degrees of freedom (DOF) from the Averaging Kernel (AK), we firstly need to 
		perturb each component of the state vector (x -> x') of the (eval) data sample step by 
		step to compute a new set of perturbed observations (y'). The new observations will be fed
		into the retrieval to generate a perturbed retrieved state vector (x_ret'). Differences of 
		x_ret' and x_ret divided by the difference of the (eval) data state vectors x' and x yields
		the AK for one eval data sample.

		Computing the gain matrix is simpler: Perturb the observation vector directly (y -> y')
		and have the retrieval generate a new x_ret'. The quotient of x_ret' - x_ret and y' - y
		yields a part of the gain matrix. This must be repeated for all obs. vector components
		(i.e., brightness temperature channels) to obtain the full gain matrix.

		For initialisation, we need:
		x : array of floats
			State vector of the (eval) data sample (not retrieved!). Has got either one or multiple
			components (i.e., height levels of a temperature profile). Currently, only the 
			following shape is supported: (data_samples, components).
		y : array of floats
			Observation vector, which must have the same number of data samples as x. Components of
			the observation vector could be, for example, brightness temperature channels. 
			Currently, only the following shape is supported: (data_samples, components).
		x_ret : array of floats
			Retrieved state vector. Must have the same number of components as x. Currently, only the
			following shape is supported: (data_samples, components).
		x_cov : array of floats
			State vector covariance matrix, computed as covariance over the data samples so that 
			the variance over the data samples is represented for each state vector component
			(e.g., height level). The shape is therefore (components, components).
		y_cov : array of floats
			Observation vector error covariance matrix indicating the measurement uncertainties 
			(e.g., brightness temperature noise) for each component. The shape is therefore
			(components, components).
		ax_samp : int
			Number indicating which axis of a numpy array represents the (eval) data samples.
		ax_comp : int
			Number indicating which axis of a numpy array corresponds to the state vector 
			components.
		perturbation : float
			Float that will be added to or multiplied with the state vector or observation vector, 
			depending on whether perturb_type is "add" or "multiply" selected and whether gain 
			matrix or AK will be computed.
		perturb_type : str
			String indicating whether the perturbation is to be added or multiplied to the state or
			obs vector. Valid options: 'add', 'multiply'
		aux_i : dict
			Dictionary that can contain various information. It must contain information describing
			the state vector in the key "predictand".
		suppl_data : dict
			Dictionary containing supplemental data needed to run the PAMTRA simulations for the 
			function new_obs: temperature, pressure, relative humidity (or specific humidity) 
			profiles, surface properties (temperature, type), geo-information (lat, lon, time),
			height grid, and hydrometeors.
		descriptor_file : str
			String indicating the path and filename of the hydrometeor descriptor file needed for
			PAMTRA simulations.

		**kwargs:
	"""

	def __init__(self, x, y, x_ret, x_cov, y_cov, ax_samp, ax_comp, perturbation, perturb_type, aux_i, 
					suppl_data, descriptor_file, **kwargs):
		
		# check for correct dimensions:
		assert x.shape == x_ret.shape
		assert y.shape[ax_samp] == x.shape[ax_samp]

		# set attributes:
		self.x = x
		self.y = y
		self.x_ret = x_ret
		self.perturb_type = perturb_type
		self.pert = perturbation
		self.ax_s = ax_samp
		self.ax_c = ax_comp
		self.aux_i = aux_i
		self.suppl_data = suppl_data
		self.descriptor_file = descriptor_file
		self.i_s = 0		# (eval) data sample index is set to 0 at the beginning (would also be done in function 'perturb')
		self.n_s = self.x.shape[self.ax_s]		# number of (eval) data samples
		self.n_cx = self.x.shape[self.ax_c]		# number of state vector components
		self.n_cy = self.y.shape[self.ax_c]		# number of obs vector components
		self.x_i = self.x[self.i_s,:]			# i-th sample of state vector

		# error cov matrices:
		self.x_cov = x_cov						# apriori state vector error covariance matrix (statistical errors of prior state)
		self.y_cov = y_cov						# observation error covariance matrix
		

		# initialize variables to be computed:
		self.K_i = np.zeros((self.n_cy, self.n_cx))				# jacobian matrix of the i-th data dample
		self.K = np.zeros((self.n_s, self.n_cy, self.n_cx))		# all Jacobian matrices (contains K_i for all i_s)
		self.AK_i = np.zeros((self.n_cx, self.n_cx))			# i-th AK
		self.AK = np.zeros((self.n_s, self.n_cx, self.n_cx))	# all AK matrices (contains AK_i for all i_s)
		self.AK_diag = np.zeros((self.n_s, self.n_cx))	# main diagonal of AK matrix for all eval samples and state vector components
		self.DOF = np.zeros((self.n_s,))				# saves Degrees of Freedom for each eval samples


	def perturb(self, wat, samp, comp):

		"""
		Perturbs a component of the state or observation vector with a given perturbation. Either
		additive or multiplicative perturbation will be performed.

		Parameters:
		-----------
		wat : str
			Specify if the observation or state vector is to be perturbed. Valid options: 'obs', 
			'state'
		samp : int
			Integer indicating which sample of the vector is processed. Must be within the range of
			the (eval) data set.
		comp : int or str
			Integer that indicates which component of the vector will be perturbed. Must be within
			the range of the respective vector. OR: comp can also be a string: "all" which perturbs 
			all components of the state or obs vector step by step and saved the perturbed component
			of the vector in a new vector, which will contain all perturbed components.
		"""

		if comp == "all":

			self.i_s = samp		# current sample index being worked on
			self.x_i = self.x[self.i_s,:]		# i-th sample of state vector
			self.y_i = self.y[self.i_s,:]		# i-th sample of obs vector

			if wat == 'state':
				orig = self.x_i
				n_comp = self.n_cx
			else:
				orig = self.y_i
				n_comp = self.n_cy

			# perturb: save perturbed components into new vector full of perturbations:
			# i.e., orig_p_vec[0] will have the first perturbed component, ...[1] has the second perturbed component....
			if self.perturb_type == 'add':
				orig_p_vec = orig + self.pert

			else:
				orig_p_vec = orig * self.pert


			# save perturbed state or obs vector to a square matrix where i.e., entry [0,:]
			# contains the entire vector with the first component being perturbed. The matrix's
			# diagonal entries equal orig_p_vec.
			pert_vec_mat = np.broadcast_to(orig, (n_comp,n_comp))
			pert_vec_mat = pert_vec_mat - np.diag(orig) + np.diag(orig_p_vec)
			if wat == 'state':

				self.x_ip_mat = pert_vec_mat	# perturbed state vector for each perturbed component (i-th sample)
				self.dx_i = orig_p_vec - orig	# disturb. of all comp. of i-th sample of state vector

			else:
				self.y_ip_mat = pert_vec_mat	# perturbed obs vector for each perturbed component (i-th sample)
				self.dy_i = orig_p_vec - orig	# disturb. of all comp. of i-th sample of obs vector


		elif type(comp) == type(0):

			self.i_s = samp		# current sample index being worked on
			self.i_c = comp		# current component index being worked on
			self.x_i = self.x[self.i_s,:]		# i-th sample of state vector
			self.y_i = self.y[self.i_s,:]		# i-th sample of obs vector

			if wat == 'state':
				orig = self.x_i
			else:
				orig = self.y_i

			# perturb:
			orig_p = deepcopy(orig)				# will contain the perturbed vector
			if self.perturb_type == 'add':
				orig_p[self.i_c] = orig[self.i_c] + self.pert
			else:
				orig_p[self.i_c] = orig[self.i_c] * self.pert

			# save to attributes:
			if wat == 'state':
				self.x_ip = orig_p				# perturbed state vector (i-th sample)
				self.dx_ij = self.x_ip[self.i_c] - self.x_i[self.i_c]	# disturb. of j-th comp. of i-th sample of state vector

			else:
				self.y_ip = orig_p				# perturbed obs vector (i-th sample)
				self.dy_ij = self.y_ip[self.i_c] - self.y_i[self.i_c]	# disturb. of j-th comp. of i-th sample of obs vector


	def new_obs(self, perturbed, what_data='single'):
		
		"""
		Simulations of brightness temperatures (TB) with PAMTRA based on the perturbed atmospheric state.

		Parameters:
		-----------
		perturbed : bool
			Bool to specify whether new obs are generated for a disturbed or undisturbed state vector.
			If True, new TBs are generated for the perturbed state vector.
		what_data : str
			String indicating what will be forward simulated. Options: 'single': A single atmospheric profile 
			(state vector) is simulated, 'comp': Use this option if you want to simulate obs vector for all perturbed 
			state vectors of the i-th (eval) data sample, 'samp': Simulate all state vectors in the (eval) data set.
		"""

		# identify what's the state vector:
		shape_pred_0 = 0
		shape_pred_1 = 0
		x_idx = dict()			# state vector x can consist of different meteorol. variables.
								# this identifies the indices where the single predictands are located
		for id_i, predictand in enumerate(self.aux_i['predictand']):
			# inquire shape of current predictand and its position in the output vector or prediction:
			shape_pred_0 = shape_pred_1
			shape_pred_1 = shape_pred_1 + self.aux_i['n_ax1'][predictand]
			x_idx[predictand] = [shape_pred_0, shape_pred_1]


		# create pam object:
		pam = pyPamtra.pyPamtra()


		# general settings:
		pam.nmlSet['passive'] = True						# passive simulation
		pam.nmlSet['active'] = False						# False: no radar
		if remote:
			pam.nmlSet['data_path'] = "/net/blanc/awalbroe/Codes/pamtra/"
		else:
			pam.nmlSet['data_path'] = "/home/tenweg/pamtra/"

		# define the pamtra profile: temp, relhum, pres, height, lat, lon, timestamp, lfrac, obs_height, ...
		pamData = dict()

		if what_data == 'samp':
			n_data = self.n_s
			shape2d = (n_data,1)

			# data needed for PAMTRA:
			lon = self.suppl_data['lon']
			lat = self.suppl_data['lat']
			timestamp = self.suppl_data['time']
			hgt_lev = self.suppl_data['height']
			pres_lev = self.suppl_data['pres']
			temp_sfc = self.suppl_data['temp_sfc']
			sfc_slf = self.suppl_data['sfc_slf']
			sfc_sif = self.suppl_data['sfc_sif']
			cwc = self.suppl_data['cwc']
			iwc = self.suppl_data['iwc']
			rwc = self.suppl_data['rwc']
			swc = self.suppl_data['swc']
			cwp = self.suppl_data['cwp']
			iwp = self.suppl_data['iwp']
			rwp = self.suppl_data['rwp']
			swp = self.suppl_data['swp']


			# use x_idx (who identified where which meteorological variable of the state vector is located)
			# to set the temp and rh data:
			if perturbed:
				pdb.set_trace()		# not yet coded, and I don't think that this case will be planned

			else:

				# start with suppl data (which has the full grid) and insert state vector where needed:
				temp_lev = np.zeros(self.suppl_data['temp'].shape)
				temp_lev[:,:] = self.suppl_data['temp']
				if 'temp' in self.aux_i['predictand']:
					pdb.set_trace() # check if grid extension works
					temp_lev[:, x_idx['temp'][0]:x_idx['temp'][1]] = self.x[:, x_idx['temp'][0]:x_idx['temp'][1]]

				# in case "unperturbed" is selected, we don't need to filter q out of the
				# state vector and convert it. We can just take the relative humidity instead 				# # or does it induce too many uncertainties to TBs
																											# # due to the conversion, and potentially overexpose
																											# # the effect by the perturbation lateron?
				# if 'q' in self.aux_i['predictand']:
					# # pdb.set_trace() # check for correct units
					# q_lev = self.x_i[x_idx['q'][0]:x_idx['q'][1]]

					# rho_v_lev = convert_spechum_to_abshum(temp_lev, 
														# self.suppl_data['pres'][self.i_s,:],
														# q_lev)
					# rh_lev = convert_abshum_to_relhum(temp_lev, rho_v_lev)

				# else:
				rh_lev = self.suppl_data['rh']


		elif what_data == 'single':
			shape2d = (1,1)

			# data needed for PAMTRA: select case
			lon = self.suppl_data['lon'][self.i_s]
			lat = self.suppl_data['lat'][self.i_s]
			timestamp = self.suppl_data['time'][self.i_s]
			hgt_lev = self.suppl_data['height'][self.i_s,:]
			pres_lev = self.suppl_data['pres'][self.i_s,:]
			temp_sfc = self.suppl_data['temp_sfc'][self.i_s]
			sfc_slf = self.suppl_data['sfc_slf'][self.i_s]
			sfc_sif = self.suppl_data['sfc_sif'][self.i_s]
			cwp = self.suppl_data['cwp'][self.i_s]
			iwp = self.suppl_data['iwp'][self.i_s]
			rwp = self.suppl_data['rwp'][self.i_s]
			swp = self.suppl_data['swp'][self.i_s]
			cwc = self.suppl_data['cwc'][self.i_s,:]
			iwc = self.suppl_data['cwc'][self.i_s,:]
			rwc = self.suppl_data['rwc'][self.i_s,:]
			swc = self.suppl_data['swc'][self.i_s,:]

			# use x_idx (who identified where which meteorological variable of the state vector is located)
			# to set the temp and rh data:
			n_hgt = len(hgt_lev)
			if perturbed:

				temp_lev = np.zeros((n_hgt,))
				temp_lev[:] = self.suppl_data['temp'][self.i_s,:]
				if 'temp' in self.aux_i['predictand']:
					temp_lev[x_idx['temp'][0]:x_idx['temp'][1]] = self.x_ip[x_idx['temp'][0]:x_idx['temp'][1]]

				# we need relative humidity: thus, if q is a predictand, it needs to be converted
				if 'q' in self.aux_i['predictand']:
					q_lev = np.zeros((n_hgt,))
					q_lev[:] = self.suppl_data['q'][self.i_s,:]
					q_lev[x_idx['q'][0]:x_idx['q'][1]] = self.x_ip[x_idx['q'][0]:x_idx['q'][1]]

					rh_lev = convert_spechum_to_relhum(temp_lev, pres_lev, q_lev*0.001)

				else:
					rh_lev = self.suppl_data['rh'][self.i_s,:]

			else:

				# also start with suppl data, but then insert x into the correct indices:
				temp_lev = np.zeros((n_hgt,))
				temp_lev[:] = self.suppl_data['temp'][self.i_s,:]
				if 'temp' in self.aux_i['predictand']:
					temp_lev[x_idx['temp'][0]:x_idx['temp'][1]] = self.x_i[x_idx['temp'][0]:x_idx['temp'][1]]

				# in case "unperturbed" is selected, we don't need to filter q out of the
				# state vector and convert it. We can just take the relative humidity instead 				# # or does it induce too many uncertainties to TBs
																											# # due to the conversion, and potentially overexpose
																											# # the effect by the perturbation lateron?
				# if 'q' in self.aux_i['predictand']:
					# # pdb.set_trace() # check for correct units
					# q_lev = self.x_i[x_idx['q'][0]:x_idx['q'][1]]

					# rho_v_lev = convert_spechum_to_abshum(temp_lev, 
														# self.suppl_data['pres'][self.i_s,:],
														# q_lev)
					# rh_lev = convert_abshum_to_relhum(temp_lev, rho_v_lev)

				# else:
				rh_lev = self.suppl_data['rh'][self.i_s,:]


		elif what_data == 'comp':
			n_data = self.n_cx

			# the retrieval grid can be different from the supplemental data grid to improve
			# PAMTRA simulations:
			n_data_ex = len(self.suppl_data['height'][self.i_s,:])
			if n_data != n_data_ex: n_data = n_data_ex
			shape2d = (n_data,1)

			# data needed for PAMTRA: suppl_data must still be from i-th data sample
			lon = np.broadcast_to(self.suppl_data['lon'][self.i_s], (n_data,))
			lat = np.broadcast_to(self.suppl_data['lat'][self.i_s], (n_data,))
			timestamp = np.broadcast_to(self.suppl_data['time'][self.i_s], (n_data,))
			hgt_lev = np.broadcast_to(self.suppl_data['height'][self.i_s,:], (n_data,n_data))
			pres_lev = np.broadcast_to(self.suppl_data['pres'][self.i_s,:], (n_data,n_data))
			temp_sfc = np.broadcast_to(self.suppl_data['temp_sfc'][self.i_s], (n_data,))
			sfc_slf = np.broadcast_to(self.suppl_data['sfc_slf'][self.i_s], (n_data,))
			sfc_sif = np.broadcast_to(self.suppl_data['sfc_sif'][self.i_s], (n_data,))
			cwp = np.broadcast_to(self.suppl_data['cwp'][self.i_s], (n_data,))
			iwp = np.broadcast_to(self.suppl_data['iwp'][self.i_s], (n_data,))
			rwp = np.broadcast_to(self.suppl_data['rwp'][self.i_s], (n_data,))
			swp = np.broadcast_to(self.suppl_data['swp'][self.i_s], (n_data,))
			cwc = np.broadcast_to(self.suppl_data['cwc'][self.i_s,:], (n_data,n_data))
			iwc = np.broadcast_to(self.suppl_data['iwc'][self.i_s,:], (n_data,n_data))
			rwc = np.broadcast_to(self.suppl_data['rwc'][self.i_s,:], (n_data,n_data))
			swc = np.broadcast_to(self.suppl_data['swc'][self.i_s,:], (n_data,n_data))


			# use x_idx (who identified where which meteorological variable of the state vector is located)
			# to set the temp and rh data:
			if perturbed:

				# start with the suppl data and insert the (perturbed) state vector where needed:
				temp_lev = np.zeros((n_data,n_data))
				temp_lev[:,:] = np.broadcast_to(self.suppl_data['temp'][self.i_s,:], (n_data,n_data))
				if 'temp' in self.aux_i['predictand']:
					temp_lev_ip = self.x_ip_mat[:, x_idx['temp'][0]:x_idx['temp'][1]]
					temp_lev[x_idx['temp'][0]:x_idx['temp'][1],x_idx['temp'][0]:x_idx['temp'][1]] = temp_lev_ip

				# we need relative humidity: thus, if q is a predictand, it needs to be converted
				if 'q' in self.aux_i['predictand']:
					q_lev = np.zeros((n_data,n_data))
					q_lev[:,:] = np.broadcast_to(self.suppl_data['q'][self.i_s,:], (n_data,n_data))
					q_lev_ip = self.x_ip_mat[:, x_idx['q'][0]:x_idx['q'][1]]		# in g kg-1
					q_lev[x_idx['q'][0]:x_idx['q'][1],x_idx['q'][0]:x_idx['q'][1]] = q_lev_ip

					rh_lev = convert_spechum_to_relhum(temp_lev, pres_lev, q_lev*0.001)

				else:
					rh_lev = np.broadcast_to(self.suppl_data['rh'][self.i_s,:], (n_data,n_data))

			else:
				pdb.set_trace()		# I also don't expect this case to happen


		# make sure relative humidity doesn't exceed sensible values:
		rh_lev = rh_lev*100.0		# convert to %
		rh_lev[rh_lev > 105.0] = 105.0
		rh_lev[rh_lev < 0.0] = 0.0


		# write data into pamData dict:
		pamData['lon'] = np.reshape(lon, shape2d)
		pamData['lat'] = np.reshape(lat, shape2d)
		pamData['timestamp'] = timestamp
		pamData['hgt_lev'] = hgt_lev


		# put meteo data into pamData:
		shape3d = shape2d + (pamData['hgt_lev'].shape[-1],)
		pamData['press_lev'] = np.reshape(pres_lev, shape3d)
		pamData['relhum_lev'] = np.reshape(rh_lev, shape3d)
		pamData['temp_lev'] = np.reshape(temp_lev, shape3d)

		# Surface data: exclude wind because influence is negligible
		pamData['groundtemp'] = temp_sfc
		pamData['sfc_slf'] = sfc_slf
		pamData['sfc_sif'] = sfc_sif


		# surface properties:
		pamData['sfc_type'] = np.around(pamData['sfc_slf']).astype('int32')
		pamData['sfc_model'] = np.zeros(shape2d, dtype='int32')
		pamData['sfc_refl'] = np.chararray(shape2d, unicode=True)
		pamData['sfc_refl'][:] = 'F'
		pamData['sfc_refl'][pamData['sfc_type'] > 0] = 'S'		# specular over land

		# change properties over sea ice:
		ice_idx = (pamData['sfc_sif'] > 0)
		pamData['sfc_type'][ice_idx] = 1
		pamData['sfc_model'][ice_idx] = 0
		pamData['sfc_refl'][ice_idx] = 'L'

		pamData['obs_height'] = np.broadcast_to(np.array([0.0]), shape2d + (1,))
		

		# 4d variables: hydrometeors: distribute the given CWP, IWP, RWP, SWP linearly:
		shape3d_lay = shape2d + (pamData['hgt_lev'].shape[-1]-1,)
		shape4d = shape2d + (pamData['hgt_lev'].shape[-1]-1, 4)


		# get hydrometeors on layers instead of levels. shape of each hyd met should be shape3d_lay
		cwc_lay = 0.5*(cwc[:,1:] + cwc[:,:-1])
		iwc_lay = 0.5*(iwc[:,1:] + iwc[:,:-1])
		rwc_lay = 0.5*(rwc[:,1:] + rwc[:,:-1])
		swc_lay = 0.5*(swc[:,1:] + swc[:,:-1])


		pamData['hydro_q'] = np.zeros(shape4d)
		pamData["hydro_q"][:,:,:,0] = np.reshape(cwc_lay, shape4d[:3])
		pamData["hydro_q"][:,:,:,1] = np.reshape(iwc_lay, shape4d[:3])
		pamData["hydro_q"][:,:,:,2] = np.reshape(rwc_lay, shape4d[:3])
		pamData["hydro_q"][:,:,:,3] = np.reshape(swc_lay, shape4d[:3])

		# load descriptorfile:
		pam.df.readFile(self.descriptor_file)


		# create pamtra profile from pamData and run pamtra at all specified frequencies:
		# if G band frequencies are included, make sure to simulate both sides of the absorption band:
		freqs = self.suppl_data['freqs']
		if np.any((freqs >= 183.31) & (freqs <= 200.0)):		
			# then G band is selected (but only upper freq labels are used). Also include lower 
			# G band frequencies:
			freqs = np.concatenate((freqs, np.array([175.810, 178.310, 179.810, 180.810, 181.810, 182.710])))
		freqs = np.sort(freqs)


		pam.createProfile(**pamData)

		if what_data == 'single':
			pam.runPamtra(freqs)
		else:
			n_cpus = int(multiprocessing.cpu_count()*0.75)
			pam.runParallelPamtra(freqs, pp_deltaX=1, pp_deltaY=0, pp_deltaF=0, pp_local_workers=n_cpus)
			# pam.runPamtra(freqs)


		# post process PAMTRA TBs:
		# TBs: angles: 0 = nadir; -1 = zenith <<->> angle[0] = 180.0; angle[-1] = 0.0
		TB = pam.r['tb'][:,0,0,-1,:,:].mean(axis=-1)
		if np.any((freqs >= 170.0) & (freqs <= 200.0)):
			TB, freqs = Gband_double_side_band_average(TB, freqs)

		# select the TBs used as obs vector:
		TB_obs, freq_obs = select_MWR_channels(TB, freqs, band=self.aux_i['predictor_TBs'], return_idx=0)

		# update obs vector
		if perturbed:
			if what_data == 'comp':
				self.y_ip_mat = TB_obs[:self.n_cx,:]	# (self.n_cx, self.n_cy) matrix of obs vector based on
														# perturbed state vector. self.y_ip_max[0,:] yields the
														# obs vector for the 0-th-component-perturbed state vector
			elif what_data == 'single':
				self.y_ip = TB_obs[0,:]				# reduce to 1D array, removing the unnecessary dimension

		else:
			if what_data == 'samp':
				self.y = TB_obs
			elif what_data == 'single':
				self.y[self.i_s,:] = TB_obs[0,:]	# remove unnecessary dimension
				self.y_i = self.y[self.i_s,:]		# i-th sample of obs vector


	def compute_dx_ret_i(self, comp):

		"""
		Computes the difference of the perturbed retrieved state vector (generated from perturbed obs 
		vector) and the original retrieved state vector x_ret for the current i-th (eval) data sample.

		Parameters:
		-----------
		comp : int or str
			Integer that indicates which component of the vector will be perturbed. Must be within
			the range of the respective vector. OR: comp can also be a string: "all" which perturbs 
			all components of the state or obs vector step by step and saved the perturbed component
			of the vector in a new vector, which will contain all perturbed components.
		"""

		if comp == 'all':
			self.dx_ret_i_mat = self.x_ret_ip_mat - self.x_ret[self.i_s,:]	# (n_c, n_c) array where the first row dx_ret_i_mat[0,:]
																	# indictes the ret state vector difference when the first 
																	# component was perturbed

		elif type(comp) == type(0):	############
			self.dx_ret_i = self.x_ret_ip - self.x_ret[self.i_s,:]


	def compute_jacobian_step(self):

		"""
		Computes the j-th (self.i_c) column of the Jacobian K with entries K_aj = dy_ia / dx_ij where dy_ia is the a-th
		component of the difference between the perturbed and reference obs vector of (eval) data sample i. dx_ij is the 
		j-th component of the diff between the perturbed and reference state vector of (eval) data sample i.
		"""

		jacobian_j = (self.y_ip - self.y_i) / self.dx_ij		# j-th column of the jacobian
		self.K_i[:,self.i_c] = jacobian_j
		self.K[self.i_s,:,:] = self.K_i


	def compute_jacobian(self):

		"""
		Computes the Jacobian K with entries K_aj = dy_ia / dx_ij where dy_ia is the a-th component of the 
		difference between the perturbed and reference obs vector of (eval) data sample i. dx_ij is the 
		j-th component of the diff between the perturbed and reference state vector of (eval) data sample i.
		"""

		jacobian = np.zeros((self.n_cy, self.n_cx))

		# loop through obs vector components:
		for a in range(self.n_cy):
			jacobian[a,:] = (self.y_ip_mat[:,a] - self.y_i[a]) / self.dx_i

		self.K_i = jacobian
		self.K[self.i_s,:,:] = self.K_i


	def compute_col_of_AK_i(self):

		"""
		Computes the j-th (i_j-th) column of the Averaging Kernel matrix of data sample i.
		This function is needed when considering each component step by step, meaning that 'all' 
		has been used for comp in the other functions. Also, the main diagonal is set when all
		columns of the AK have been computed.
		"""

		self.AK_i[:,self.i_c] = self.dx_ret_i / self.dx_ij

		if self.i_c == self.n_cx - 1:
			self.AK_diag[self.i_s,:] = np.diag(self.AK_i)


	def compute_AK_i(self, how):

		"""
		Computes the the entire  Averaging Kernel matrix of data sample i. Use this function when
		all components have been worked on in one batch (i.e., comp == 'all' in the functions above).
		Also, the main diagonal is set. The Averaging Kernel will either be computed via the 
		dx_ret / dx or via the matrix multiplication scheme.

		Parameters:
		-----------
		how : str
			String to choose between two ways of computing the AK matrix. Valid options: 'matrix': it
			requires to run perturb('state', i_s, 'all'), new_obs(True, what_data='comp') and 
			compute_jacobian(); 'ret': requires perturb('state', i_s, 'all'), new_obs(True, what_data='comp'),
			generation of new (perturbed) x_ret from perturbed obs vector, and compute_dx_ret_i('all').
		"""

		if how == 'ret':
			for jj in range(self.n_cx):
				self.AK_i[:,jj] = self.dx_ret_i_mat[jj,:] / self.dx_i[jj]

		elif how == 'matrix':
			x_cov_inv = np.linalg.inv(self.x_cov)
			y_cov_inv = np.linalg.inv(self.y_cov)
			KTSeK = self.K_i.T @ y_cov_inv @ self.K_i
			self.AK_i = np.linalg.inv(KTSeK + x_cov_inv) @ KTSeK


		else:
			raise ValueError("Argument 'how' of the function compute_AK_i must be either 'ret' or 'matrix'.")

		self.AK_diag[self.i_s,:] = np.diag(self.AK_i)
		self.AK[self.i_s,:,:] = self.AK_i


	def compute_DOF(self):

		"""
		Computes the degrees of freedom (DOF) from the trace of the AK of the i-th data sample.
		"""

		self.DOF[self.i_s] = np.trace(self.AK_i)


	def visualise_AK_i(self):

		"""
		Visualises the main diagonal of the AK of the i-th (eval) data sample.
		"""

		units_AK = {'temp': "(K/K)",
					'q': "(g$\,$kg$^{-1}$/g$\,$kg$^{-1}$)"}

		f1 = plt.figure(figsize=(7,11))
		a1 = plt.axes()

		# axis limits: eventually limit height if grid had been extended for PAMTRA:
		height_i = self.suppl_data['height'][self.i_s,:self.n_cx]
		ax_lims = {'y': [0.0, height_i.max()]}

		# plot data:
		a1.plot(self.AK_diag[self.i_s,:], height_i, color=(0,0,0), linewidth=1.25)

		# aux info:
		a1.text(0.98, 0.98, f"DOF = {self.DOF[self.i_s]}", 
				ha='right', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				transform=a1.transAxes)

		# legend, colorbar

		# set axis limits:
		a1.set_ylim(ax_lims['y'][0], ax_lims['y'][1])

		# - set ticks and tick labels and parameters:
		a1.tick_params(axis='both', labelsize=fs_dwarf)

		# grid:
		a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# labels:
		a1.set_ylabel("Height (m)", fontsize=fs)
		a1.set_xlabel(f"Averaging Kernel diagonal {units_AK[self.aux_i['predictand'][0]]}", fontsize=fs)

		plt.show()


	def visualise_mean_AK(self, path_output):

		"""
		Visualises the main diagonal of the mean of all (eval) data sample AKs.

		Parameters:
		-----------
		path_output : str
			Path to save the plot to.
		"""

		units_AK = {'temp': "(K/K)",
					'q': "(g$\,$kg$^{-1}$/g$\,$kg$^{-1}$)"}

		mean_AK = np.mean(self.AK_diag, axis=0)
		std_AK = np.std(self.AK_diag, axis=0)
		mean_height = np.mean(self.suppl_data['height'][:,:self.n_cx], axis=0)
		
		f1 = plt.figure(figsize=(7,11))
		a1 = plt.axes()

		# axis limits:
		ax_lims = {'y': [0.0, self.suppl_data['height'][:,:self.n_cx].max()]}

		# plot data:
		a1.plot(mean_AK, mean_height, color=(0,0,0), linewidth=1.25, label='Mean')
		a1.fill_betweenx(mean_height, mean_AK - std_AK, mean_AK + std_AK, color=(0,0,0,0.2), label='Std. deviation')

		# aux info:
		a1.text(0.98, 0.98, f"DOF = {np.mean(self.DOF):.2f}$\pm${np.std(self.DOF):.2f}", 
				ha='right', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				transform=a1.transAxes)

		# legend, colorbar
		lh, ll = a1.get_legend_handles_labels()
		a1.legend(lh, ll, loc='upper right', bbox_to_anchor=(0.98, 0.90), fontsize=fs_small)

		# set axis limits:
		a1.set_ylim(ax_lims['y'][0], ax_lims['y'][1])

		# - set ticks and tick labels and parameters:
		a1.tick_params(axis='both', labelsize=fs_dwarf)

		# grid:
		a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# labels:
		a1.set_ylabel("Height (m)", fontsize=fs)
		a1.set_xlabel(f"Averaging Kernel diagonal {units_AK[self.aux_i['predictand'][0]]}", fontsize=fs)


		# check if output path exists:
		plotpath_dir = os.path.dirname(path_output)
		if not os.path.exists(plotpath_dir):
			os.makedirs(plotpath_dir)

		plot_file = path_output + f"NN_syn_ret_info_content_AK_diag_mean_DOF_{self.aux_i['file_descr']}.png"
		f1.savefig(plot_file, dpi=300, bbox_inches='tight')
		print(f" Saved {plot_file}")


	def save_info_content_data(self, path_output):

		"""
		Saves the AK_diag, DOF, and supplementary data of all (eval) data samples to a netCDF file in the specified path.

		Parameters:
		-----------
		path_output : str
			Path to save the netCDF file to.
		"""

		# check if output path exists: if it doesn't, create it:
		path_output_dir = os.path.dirname(path_output)
		if not os.path.exists(path_output_dir):
			os.makedirs(path_output_dir)


		# unit labels:
		x_l = {'temp': "K", 'q': "g kg-1"}		# state vector units
		x_l = x_l[self.aux_i['predictand'][0]]
		y_l = "K"								# obs vector units


		# create xarray Dataset:
		DS = xr.Dataset(coords={'n_s': 	(['n_s'], np.arange(self.n_s), {'long_name': "number of data samples"}),
								'n_cx': (['n_cx'], np.arange(self.n_cx), {'long_name': "number of state vector components"}),
								'n_cy': (['n_cy'], np.arange(self.n_cy), {'long_name': "number of observation vector components"})})

		# save data into it:
		if 'i_eval' in self.aux_i.keys():
			DS['idx_eval'] = xr.DataArray(self.aux_i['i_eval'], dims=['n_s'],
											attrs={	'long_name': "dataset indices used for information content",
													'comment': ("only a subset of the entire dataset has been used to compute information content. " +
																"The indices represent a subset of the entire dataset to track which samples " +
																"have been selected.")})
		DS['height'] = xr.DataArray(self.suppl_data['height'][:,:self.n_cx], dims=['n_s', 'n_cx'], 
									attrs={	'long_name': "height grid for state vector (profile)",
											'units': "m"})
		DS['freqs'] = xr.DataArray(self.suppl_data['freqs'], dims=['n_cy'],
									attrs={	'long_name': "Frequency", 
											'units': "GHz"})
		DS['pert_type'] = xr.DataArray(self.perturb_type, attrs={'long_name': "perturbation type (either additive or multiplicative)"})
		DS['perturbation'] = xr.DataArray(self.pert, attrs={'long_name': "perturbation factor or summand"})
		DS['y_cov'] = xr.DataArray(self.y_cov, dims=['n_cy', 'n_cy'], 
									attrs={	'long_name': "observation vetor error covariance matrix",
											'units': f"({y_l}**2)"})
		DS['x_cov'] = xr.DataArray(self.x_cov, dims=['n_cx', 'n_cx'], 
									attrs={	'long_name': "state vector apriori error covariance matrix",
											'units': f"({x_l}**2)"})
		DS['K'] = xr.DataArray(self.K, dims=['n_s', 'n_cy', 'n_cx'],
									attrs={	'long_name': "Jacobian matrix for each data sample",
											'units': f"{y_l} / {x_l}"})
		DS['AK'] = xr.DataArray(self.AK, dims=['n_s', 'n_cx', 'n_cx'],
									attrs={	'long_name': "Averaging Kernel matrix for each data sample",
											'units': f"({x_l} / {x_l})"})
		DS['AK_diag'] = xr.DataArray(self.AK_diag, dims=['n_s', 'n_cx'], 
									attrs={	'long_name': "main diagonal of the Averaging Kernel matrix for each data sample",
											'units': f"({x_l} / {x_l})"})
		DS['DOF'] = xr.DataArray(self.DOF, dims=['n_s'], attrs={'long_name': "Degrees Of Freedom, computed as trace of the Averaging Kernel matrix",
																'units': f"({x_l} / {x_l})"})


		# GLOBAL ATTRIBUTES:
		DS.attrs['title'] = f"Information content output for state vector: {self.aux_i['predictand'][0]}"
		DS.attrs['author'] = "Andreas Walbroel (a.walbroel@uni-koeln.de), Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
		DS.attrs['predictor_TBs'] = self.aux_i['predictor_TBs']
		DS.attrs['setup_id'] = self.aux_i['file_descr']
		DS.attrs['python_version'] = f"python version: {sys.version}"
		DS.attrs['python_packages'] = f"numpy: {np.__version__}, xarray: {xr.__version__}, matplotlib: {mpl.__version__}, "

		datetime_utc = dt.datetime.utcnow()
		DS.attrs['processing_date'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")

		# export to netCDF:
		DS.to_netcdf(path_output + f"NN_synergetic_ret_info_content_{self.aux_i['file_descr']}.nc", mode='w', format='NETCDF4')
		DS.close()




