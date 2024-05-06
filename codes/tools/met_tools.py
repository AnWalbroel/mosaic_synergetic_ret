import numpy as np
import xarray as xr
import pdb


# constants:
R_d = 287.0597  # gas constant of dry air, in J kg-1 K-1
R_v = 461.5  	# gas constant of water vapour, in J kg-1 K-1
M_dv = R_d / R_v # molar mass ratio , in ()
e_0 = 611		# saturation water vapour pressure at freezing point (273.15 K), in Pa
T0 = 273.15		# freezing temperature, in K
g = 9.80665 	# gravitation acceleration, in m s^-2 (from https://doi.org/10.6028/NIST.SP.330-2019 )
c_pd = 1005.7	# specific heat capacity of dry air at constant pressure, in J kg-1 K-1
c_vd = 719.0	# specific heat capacity of dry air at constant volume, in J kg-1 K-1
c_h2o = 4187.0	# specific heat capacity of water at 15 deg C; in J kg-1 K-1
L_v = 2.501e+06	# latent heat of vaporization, in J kg-1
omega_earth = 2*np.pi / 86164.09	# earth's angular velocity: World Book Encyclopedia Vol 6. Illinois: World Book Inc.: 1984: 12.


def compute_IWV(
	rho_v,
	z,
	nan_threshold=0.0,
	scheme='balanced'):

	"""
	Compute Integrated Water Vapour (also known as precipitable water content)
	out of absolute humidity (in kg m^-3) and height (in m).
	The moisture data may contain certain number gaps (up to nan_threshold*n_levels) but
	the height variable must be free of gaps.

	Parameters:
	-----------
	rho_v : array of floats
		One dimensional array of absolute humidity in kg m^-3.
	z : array of floats
		One dimensional array of sorted height axis (ascending order) in m.
	nan_threshold : float, optional
		Threshold describing the fraction of nan values of the total height level
		number that is still permitted for computation.
	scheme : str, optional
		Chose the scheme 'balanced' or 'top_weighted'. They differ in the way the altitude
		levels are used to compute IWV. Recommendation and default: 'balanced'
	"""

	# Check if the height axis is sorted in ascending order:
	if np.any(np.diff(z) < 0):
		print("Warning! Height axis must be in ascending order to compute the integrated" +
			" water vapour.")

		# if the pressure data is okay until 300 hPa, compute IWV nonetheless and truncate the
		# profile beyond:
		where_broken = np.where(np.diff(z) < 0)[0]		# when where_broken == 152, then z[153] - z[152] is broken
		if z[where_broken[0]] < 9000.0:	# then, sufficient altitude doesn't have valid data valid data, return IWV=nan
			return IWV

	# truncate data to non nan height or pressure levels:
	non_nan_idx = np.where(~np.isnan(z))[0]
	q = q[non_nan_idx[0]:non_nan_idx[-1]+1]
	z = z[non_nan_idx[0]:non_nan_idx[-1]+1]

	# check if height axis is free of gaps:
	if np.any(np.isnan(np.diff(z))): 
		print("Height axis contains gaps. Aborted IWV computation.")
		return IWV


	n_height = len(z)
	# Check if rho_v has got any gaps:
	n_nans_rho_v = np.count_nonzero(np.isnan(rho_v))


	# If no nans exist, the computation is simpler. If some nans exist IWV will still be
	# computed but needs to look for the next non-nan value. If too many nans exist IWV
	# won't be computed.
	if scheme == 'balanced':
		if (n_nans_rho_v == 0):

			IWV = 0.0
			for k in range(n_height):
				if k == 0:		# bottom of grid
					dz = 0.5*(z[k+1] - z[k])		# just a half of a level difference
					IWV = IWV + rho_v[k]*dz

				elif k == n_height-1:	# top of grid
					dz = 0.5*(z[k] - z[k-1])		# the other half level difference
					IWV = IWV + rho_v[k]*dz

				else:			# mid of grid
					dz = 0.5*(z[k+1] - z[k-1])
					IWV = IWV + rho_v[k]*dz

		elif n_nans_rho_v / n_height < nan_threshold:

			# Loop through height grid:
			IWV = 0.0
			k = 0
			prev_nonnan_idx = -1
			while k < n_height:
				
				# check if hum on current level is nan:
				# if so search for the next non-nan level:
				if np.isnan(rho_v[k]):
					next_nonnan_idx = np.where(~np.isnan(rho_v[k:]))[0]

					if (len(next_nonnan_idx) > 0) and (prev_nonnan_idx >= 0):	# mid or near top of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of rho_v
						IWV += 0.25*(rho_v[next_nonnan_idx] + rho_v[prev_nonnan_idx])*(z[k+1] - z[k-1])
					
					elif (len(next_nonnan_idx) > 0) and (prev_nonnan_idx < 0):	# bottom of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of rho_v

						# fixing height grid variable in case only the lowest measurement doesn't exist:
						if np.isnan(z[0]) and not (np.isnan(z[1]+z[2])):
							IWV += 0.5*rho_v[next_nonnan_idx]*(z[2] - z[1])
						else:
							IWV += 0.5*rho_v[next_nonnan_idx]*(z[k+1] - z[k])
						

					else: # reached top of grid
						IWV += 0.0

				else:
					prev_nonnan_idx = k

					if k == 0:			# bottom of grid
						IWV += 0.5*rho_v[k]*(z[k+1] - z[k])
					elif k == 1 and np.isnan(z[k-1]):	# next to bottom of grid
						IWV += 0.5*rho_v[k]*(z[k+1] - z[k])
					elif (k > 0) and (k < n_height-1):	# mid of grid
						IWV += 0.5*rho_v[k]*(z[k+1] - z[k-1])
					else:				# top of grid
						IWV += 0.5*rho_v[k]*(z[-1] - z[-2])

				k += 1		

		else:
			IWV = np.nan


	elif scheme == 'top_weighted':
		if (n_nans_rho_v == 0):

			IWV = 0.0
			for k in range(n_height):
				if k < n_height-2:		# bottom or mid of grid
					dz = z[k+1] - z[k]
					IWV = IWV + rho_v[k]*dz

				else:	# top and next to top of grid
					dz = 0.5*(z[-1] - z[-2])		# half the height for top two levels
					IWV = IWV + rho_v[k]*dz

		elif n_nans_rho_v / n_height < nan_threshold:

			# Loop through height grid:
			IWV = 0.0
			k = 0
			prev_nonnan_idx = -1
			while k < n_height:
				
				# check if hum on current level is nan:
				# if so search for the next non-nan level:
				if np.isnan(rho_v[k]):
					next_nonnan_idx = np.where(~np.isnan(rho_v[k:]))[0]

					if (len(next_nonnan_idx) > 0) and (prev_nonnan_idx >= 0):	# mid of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of rho_v

						if k+1 != n_height-1:
							IWV += 0.5*(rho_v[next_nonnan_idx] + rho_v[prev_nonnan_idx])*(z[k+1] - z[k])
						else:	# near top of grid
							IWV += 0.25*(rho_v[next_nonnan_idx] + rho_v[prev_nonnan_idx])*(z[k+1] - z[k])
					
					elif (len(next_nonnan_idx) > 0) and (prev_nonnan_idx < 0):	# bottom of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of rho_v

						if np.isnan(z[0]) and not (np.isnan(z[1]+z[2])):
							IWV += rho_v[next_nonnan_idx]*(z[2] - z[1])
						else:
							IWV += rho_v[next_nonnan_idx]*(z[k+1] - z[k])
						

					else: # reached top of grid
						IWV += 0.0

				else:
					prev_nonnan_idx = k

					if k < n_height-2:	# bottom or mid of grid
						IWV += rho_v[k]*(z[k+1] - z[k])
					else:				# top of grid
						IWV += 0.5*rho_v[k]*(z[-1] - z[-2])

				k += 1		

		else:
			IWV = np.nan
		
	return IWV


def compute_IWV_q(
	q,
	press,
	nan_threshold=0.0,
	scheme='balanced'):

	"""
	Compute Integrated Water Vapour (also known as precipitable water content)
	out of specific humidity (in kg kg^-1), gravitational constant and air pressure (in Pa).
	The moisture data may contain certain number gaps (up to nan_threshold*n_levels) but
	the height variable must be free of gaps.

	Parameters:
	-----------
	q : array of floats
		One dimensional array of specific humidity in kg kg^-1.
	press : array of floats
		One dimensional array of pressure in Pa.
	nan_threshold : float, optional
		Threshold describing the fraction of nan values of the total height level
		number that is still permitted for computation.
	scheme : str, optional
		Chose the scheme 'balanced' or 'top_weighted'. They differ in the way the altitude
		levels are used to compute IWV. Recommendation and default: 'balanced'
	"""

	IWV = np.nan

	# Check if the Pressure axis is sorted in descending order:
	if np.any(np.diff(press) > 0):
		print("Warning! Height axis must be in ascending order (pressure in descending) to compute the integrated" +
			" water vapour.")

		# if the pressure data is okay until 300 hPa, compute IWV nonetheless and truncate the
		# profile beyond:
		where_broken = np.where(np.diff(press) > 0)[0]		# when where_broken == 152, then press[153] - press[152] is broken
		if press[where_broken[0]] > 30000.0:	# then, sufficient altitude doesn't have valid data valid data, return IWV=nan
			return IWV

	# truncate data to non nan height or pressure levels:
	non_nan_idx = np.where(~np.isnan(press))[0]
	q = q[non_nan_idx[0]:non_nan_idx[-1]+1]
	press = press[non_nan_idx[0]:non_nan_idx[-1]+1]

	# check if height axis is free of gaps:
	if np.any(np.isnan(np.diff(press))): 
		print("Height axis contains gaps. Aborted IWV computation.")
		return IWV


	n_height = len(press)
	# Check if q has got any gaps:
	n_nans = np.count_nonzero(np.isnan(q))


	# If no nans exist, the computation is simpler. If some nans exist IWV will still be
	# computed but needs to look for the next non-nan value. If too many nans exist IWV
	# won't be computed.
	if scheme == 'balanced':
		if (n_nans == 0):

			IWV = 0.0
			for k in range(n_height):
				if k == 0:		# bottom of grid
					dp = 0.5*(press[k+1] - press[k])		# just a half of a level difference
					IWV = IWV - q[k]*dp

				elif k == n_height-1:	# top of grid
					dp = 0.5*(press[k] - press[k-1])		# the other half level difference
					IWV = IWV - q[k]*dp

				else:			# mid of grid
					dp = 0.5*(press[k+1] - press[k-1])
					IWV = IWV - q[k]*dp


		elif n_nans / n_height < nan_threshold:

			# Loop through height grid:
			IWV = 0.0
			k = 0
			prev_nonnan_idx = -1
			while k < n_height:

				# check if hum on current level is nan:
				# if so search for the next non-nan level:
				if np.isnan(q[k]):
					next_nonnan_idx = np.where(~np.isnan(q[k:]))[0]

					if (len(next_nonnan_idx) > 0) and (prev_nonnan_idx >= 0):	# mid or near top of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of rho_v
						IWV -= 0.25*(q[next_nonnan_idx] + q[prev_nonnan_idx])*(press[k+1] - press[k-1])
					
					elif (len(next_nonnan_idx) > 0) and (prev_nonnan_idx < 0):	# bottom of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of q

						# fixing height grid variable in case only the lowest measurement doesn't exist:
						if np.isnan(press[0]) and not (np.isnan(press[1]+press[2])):
							IWV -= 0.5*q[next_nonnan_idx]*(press[2] - press[1])
						else:
							IWV -= 0.5*q[next_nonnan_idx]*(press[k+1] - press[k])

					else: # reached top of grid
						IWV += 0.0

				else:
					prev_nonnan_idx = k

					if k == 0:			# bottom of grid
						IWV -= 0.5*q[k]*(press[k+1] - press[k])
					elif k == 1 and np.isnan(press[k-1]):		# next to bottom of grid
						IWV -= 0.5*q[k]*(press[k+1] - press[k])
					elif (k > 0) and (k < n_height-1):	# mid of grid
						IWV -= 0.5*q[k]*(press[k+1] - press[k-1])
					else:				# top of grid
						IWV -= 0.5*q[k]*(press[-1] - press[-2])

				k += 1

		else:
			IWV = np.nan


	elif scheme == 'top_weighted':
		if (n_nans == 0):

			IWV = 0.0
			for k in range(n_height):
				if k < n_height-2:		# bottom or mid of grid
					dp = press[k+1] - press[k]
					IWV = IWV - q[k]*dp

				else:	# top and next to top of grid
					dp = 0.5*(press[-1] - press[-2])		# half the height for top two levels
					IWV = IWV - q[k]*dp

		elif n_nans / n_height < nan_threshold:

			# Loop through height grid:
			IWV = 0.0
			k = 0
			prev_nonnan_idx = -1
			while k < n_height:
				
				# check if hum on current level is nan:
				# if so search for the next non-nan level:
				if np.isnan(q[k]):
					next_nonnan_idx = np.where(~np.isnan(q[k:]))[0]

					if (len(next_nonnan_idx) > 0) and (prev_nonnan_idx >= 0):	# mid of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of q

						if k+1 != n_height-1:
							IWV -= 0.5*(q[next_nonnan_idx] + q[prev_nonnan_idx])*(press[k+1] - press[k])
						else:	# near top of grid
							IWV -= 0.25*(q[next_nonnan_idx] + q[prev_nonnan_idx])*(press[k+1] - press[k])
					
					elif (len(next_nonnan_idx) > 0) and (prev_nonnan_idx < 0):	# bottom of height grid
						next_nonnan_idx = next_nonnan_idx[0] + k	# plus k because searched over part of q

						# fixing height grid variable in case only the lowest measurement doesn't exist:
						if np.isnan(press[0]) and not (np.isnan(press[1]+press[2])):
							IWV -= q[next_nonnan_idx]*(press[2] - press[1])
						else:
							IWV -= q[next_nonnan_idx]*(press[k+1] - press[k])
						

					else: # reached top of grid
						IWV += 0.0

				else:
					prev_nonnan_idx = k

					if k < n_height-2:	# bottom or mid of grid
						IWV -= q[k]*(press[k+1] - press[k])
					else:				# top of grid
						IWV -= 0.5*q[k]*(press[-1] - press[-2])

				k += 1

		else:
			IWV = np.nan


	IWV = IWV / g		# yet had to be divided by gravitational acceleration

	return IWV


def wspeed_wdir_to_u_v(
	wspeed,
	wdir,
	convention='towards'):

	"""
	This will compute u and v wind components from wind speed and wind direction
	(in deg from northward facing wind). u and v will have the same units as
	wspeed. The default convention is that wdir indicates where the wind will flow
	to. Note, that meteorological wind direction is defined as from where the wind is coming
	from.

	Parameters:
	-----------
	wspeed : array of float or int
		Wind speed array.
	wdir : array of float or int
		Wind direction in deg from northward facing (or northerly) wind (for convention
		= towards, the wind flows northwards, wdir is 0; for convention = from, the wind
		comes from the north for wdir = 0).
	convention : str
		Convention of how wdir is to be interpreted. Options: 'towards' means that
		wdir indicates where the wind points to (where parcels will move to); 'from'
		means that wdir indicates where the wind comes from.
	"""

	if convention == 'towards':
		wdir_rad = np.radians(wdir)
		u = np.sin(wdir_rad)*wspeed
		v = np.cos(wdir_rad)*wspeed

	elif convention == 'from':
		wdir_rad = np.radians(wdir+180)
		wdir_rad[wdir_rad > 2*np.pi] -= 2*np.pi

		u = np.sin(wdir_rad)*wspeed
		v = np.cos(wdir_rad)*wspeed

	return u, v


def u_v_to_wspeed_wdir(
	u,
	v,
	convention='towards'):

	"""
	This will compute wind speed (in units of u and v) and wind direction (in deg from 
	northward facing or from north coming wind (depends on convention)) from u and v wind 
	components.The default convention is that wdir indicates where the wind will flow
	to. Note, that meteorological wind direction is defined as from where the wind is coming
	from.

	Parameters:
	-----------
	u : array of float or int
		Zonal component of wind (eastwards > 0).
	v : array of float or int
		Meridional component of wind (northwards > 0).
	convention : str
		Convention of how wdir is to be interpreted. Options: 'towards' means that
		wdir indicates where the wind points to (where parcels will move to); 'from'
		means that wdir indicates where the wind comes from.
	"""

	assert u.shape == v.shape	# check if both have the same dimension

	# flatten array and put it back into shape later.
	u_shape = u.shape
	u = u.flatten()
	v = v.flatten()
	wspeed = (u**2.0 + v**2.0)**0.5

	if convention == 'from':
		u *= (-1.0)
		v *= (-1.0)

	# distinguish the two semi circles to compute the correct wind direction:
	u_greater_0 = np.where(u >= 0)[0]
	u_smaller_0 = np.where(u < 0)[0]

	# compute wind direction based on the semi circle:
	wdir = np.zeros(u.shape)
	wdir[u_greater_0] = np.arccos(v[u_greater_0] / wspeed[u_greater_0])
	wdir[u_smaller_0] = 2.0*np.pi - np.arccos(v[u_smaller_0] / wspeed[u_smaller_0])

	# convert wdir to deg:
	wdir = np.degrees(wdir)

	# back to old shape:
	wspeed = np.reshape(wspeed, u_shape)
	wdir = np.reshape(wdir, u_shape)

	return wspeed, wdir


def potential_temperature(
	press,
	temp,
	press_sfc=100000.0,
	height_axis=None):

	"""
	Computes potential temperature theta from pressure and temperature of a certain level, and
	surface pressure according to theta = T*(p_s/p)**(R/c_p).

	Parameters:
	temp : array of floats
		Temperature at a certain height level in K.
	press : array of floats
		Pressure (best in Pa, else: same units as press_sfc) at a certain height level. Shape can
		be equal to temp.shape but can also be a 1D array.
	press_sfc : float
		Surface or reference pressure (in same units as press, preferably in Pa or hPa). Usually
		100000 Pa. 
	height_axis : int or None
		Identifier to locate the height axis of temp (i.e., 0, 1 or 2).
	"""

	if press.ndim == 1:	# expand press to shape of temp
		n_press = len(press)
		
		if height_axis == None:
			raise ValueError("Please specify which is the height axis of the temperature data as integer.")

		else:
			# build new shape list
			press_shape_new = list()
			for k in range(temp.ndim): press_shape_new.append(1)
			press_shape_new[height_axis] = temp.shape[height_axis]
			press = np.reshape(press, press_shape_new)

			# repeat pressure values:
			for k, tt in enumerate(temp.shape):
				if k != height_axis:
					press = np.repeat(press, tt, axis=k)

			# compute pot. temperature:
			theta = temp*(press_sfc/press)**(R_d/c_pd)

	elif press.shape == temp.shape:
		theta = temp*(press_sfc/press)**(R_d/c_pd)

	return theta


def e_sat(
	temp,
	which_algo='hyland_and_wexler'):

	"""
	Calculates the saturation pressure over water after Goff and Gratch (1946)
	or Hyland and Wexler (1983).
	Source: Smithsonian Tables 1984, after Goff and Gratch 1946
	http://cires.colorado.edu/~voemel/vp.html
	http://hurri.kean.edu/~yoh/calculations/satvap/satvap.html

	e_sat_gg_water in Pa.

	Parameters:
	-----------
	temp : array of floats
		Array of temperature (in K).
	which_algo : str
		Specify which algorithm is chosen to compute e_sat (in Pa). Options:
		'hyland_and_wexler' (default), 'goff_and_gratch'
	"""

	if which_algo == 'hyland_and_wexler':
		e_sat_gg_water = temp**(0.65459673e+01) * np.exp(-0.58002206e+04 / temp + 0.13914993e+01 - 0.48640239e-01*temp + 
								0.41764768e-04*(temp**2) - 0.14452093e-07*(temp**3))

	elif which_algo == 'goff_and_gratch':
		e_sat_gg_water = 100 * 1013.246 * 10**(-7.90298*(373.16/temp-1) + 5.02808*np.log10(
				373.16/temp) - 1.3816e-7*(10**(11.344*(1-temp/373.16))-1) + 8.1328e-3 * (10**(-3.49149*(373.16/temp-1))-1))

	return e_sat_gg_water


def convert_rh_to_abshum(
	temp,
	relhum):

	"""
	Convert array of relative humidity (between 0 and 1) to absolute humidity
	in kg m^-3. 

	Saturation water vapour pressure computation is based on: see e_sat(temp).

	Parameters:
	-----------
	temp : array of floats
		Array of temperature (in K).
	relhum : array of floats
		Array of relative humidity (between 0 and 1).
	"""

	e_sat_water = e_sat(temp)

	rho_v = relhum * e_sat_water / (R_v * temp)

	return rho_v


def convert_rh_to_spechum(
	temp,
	pres,
	relhum):

	"""
	Convert array of relative humidity (between 0 and 1) to specific humidity
	in kg kg^-1.

	Saturation water vapour pressure computation is based on: see e_sat(temp).

	Parameters:
	-----------
	temp : array of floats
		Array of temperature (in K).
	pres : array of floats
		Array of pressure (in Pa).
	relhum : array of floats
		Array of relative humidity (between 0 and 1).
	"""

	e_sat_water = e_sat(temp)

	e = e_sat_water * relhum
	q = M_dv * e / (e*(M_dv - 1) + pres)

	return q
	
	
def convert_abshum_to_spechum(
	temp,
	pres,
	abshum):

	"""
	Convert array of absolute humidity (kg m^-3) to specific humidity
	in kg kg^-1.

	Parameters:
	-----------
	temp : array of floats
		Array of temperature (in K).
	pres : array of floats
		Array of pressure (in Pa).
	abshum : array of floats
		Array of absolute humidity (in kg m^-3).
	"""

	q = abshum / (abshum*(1 - 1/M_dv) + (pres/(R_d*temp)))

	return q


def convert_spechum_to_mix_rat(
	q,
	q_add=np.nan):

	"""
	Convert array (of float) of specific humidity (kg kg-1) to water vapour 
	mixing ratio (in kg kg-1). Also other hydrometeors (cloud liquid, 
	cloud rain water, snow, ice) can be respected.

	Parameters:
	-----------
	q : float or array of floats
		Specific humidity in kg kg-1.
	q_add : float or array of floats
		Sum of other hydrometeors (i.e., cloud liquid, cloud ice, snow, rain) as
		'specific' contents (in kg kg-1). 
	"""

	if ((type(q_add) == type(np.array([]))) and q_add.size == 0) or ((type(q_add) == float) and (np.isnan(q_add))):
		r_v = q / (1 - q)
	else:
		r_v = q / (1 - q - q_add)

	return r_v


def convert_relhum_to_mix_rat(
	relhum,
	temp,
	pres):

	"""
	Convert relative humidity (in [0,1]) to water vapour mixing ratio (in kg kg-1).

	Parameters:
	-----------
	relhum : array of floats or float
		Array of relative humidity (between 0 and 1).
	temp : array of floats or float
		Array of temperature (in K).
	pres : array of floats or float
		Array of air pressure (in Pa).
	"""

	# convert relhum to abshum:
	abshum = convert_rh_to_abshum(temp, relhum)
	r_v = abshum / ((pres - e_sat(temp)*relhum) / (R_d * temp))

	return r_v


def rho_air(
	pres,
	temp,
	abshum):

	"""
	Compute the density of air (in kg m-3) with a certain moisture load.

	Parameters:
	-----------
	pres : array of floats
		Array of pressure (in Pa).
	temp : array of floats
		Array of temperature (in K).
	abshum : array of floats
		Array of absolute humidity (in kg m^-3).
	"""

	rho = (pres - abshum*R_v*temp) / (R_d*temp) + abshum

	return rho


def convert_spechum_to_abshum(
	temp,
	pres,
	q):

	"""
	Convert array of specific humidity (kg kg^-1) to absolute humidity
	in kg m^-3.

	Parameters:
	-----------
	temp : array of floats
		Array of temperature (in K).
	pres : array of floats
		Array of pressure (in Pa).
	q : array of floats
		Array of specific humidity (in kg kg^-1).
	"""

	abshum = pres / (R_d*temp*(1/q + 1/M_dv - 1))

	return abshum


def convert_abshum_to_relhum(
	temp,
	abshum):

	"""
	Convert array of absolute humidity (in kg m^-3) to relative humidity (in [0...1]).

	Parameters:
	-----------
	temp : array of floats
		Array of temperature (in K).
	abshum : array of floats
		Array of absolute humidity (in kg m^-3).
	"""

	e = abshum*R_v*temp
	e_sat_water = e_sat(temp)
	relhum = e/e_sat_water

	return relhum


def convert_spechum_to_relhum(
	temp,
	pres,
	q):

	"""
	Convert array of specific humidity (kg kg^-1) to relative humidity
	in [0...1].

	Parameters:
	-----------
	temp : array of floats
		Array of temperature (in K).
	pres : array of floats
		Array of pressure (in Pa).
	q : array of floats
		Array of specific humidity (in kg kg^-1).
	"""

	e = pres / (M_dv*(1/q + 1/M_dv - 1))
	e_sat_water = e_sat(temp)
	relhum = e/e_sat_water

	return relhum


def equiv_pot_temperature(
	temp,
	pres,
	relhum=np.array([]),
	q=np.array([]),
	q_hyd=np.array([]),
	neglect_rtc=True):

	"""
	Computes the equivalent potential temperature following 
	https://glossary.ametsoc.org/wiki/Equivalent_potential_temperature .
	The given air pressure must be reduced to partial pressure of dry air.
	temp, pres, relhum, q and q_hyd must have the same shape. Either relhum
	or q must be provided.

	Parameters:
	-----------
	temp : array of floats
		Temperature in K.
	pres : rray of floats
		Air pressure in Pa.
	relhum : array of floats
		Relative humidity in [0,1].
	q : array of floats
		Specific humidity in kg kg-1.
	q_hyd : array of floats
		Specific content of several hydrometeors (i.e., cloud liquid, ice, snow, rain)
		in kg kg-1. Can be neglected
	neglect_rtc : bool
		Option whether to neglect the terms r_t*c_h2o (setting r_t = 0) or not.
		According to https://glossary.ametsoc.org/wiki/Equivalent_potential_temperature
		both can be used with good accuracy.
	"""

	if (relhum.size == 0) and (q.size == 0):
		raise ValueError("Specific or relative humidity must be provided.")
	elif q.size == 0:
		r_v = convert_relhum_to_mix_rat(relhum, temp, pres)
		e = e_sat(temp) * relhum
	else:
		r_v = convert_spechum_to_mix_rat(q, q_hyd)
		e = pres / (1 + M_dv*(1/q - 1))		# partial pressure of water vapour in Pa

	if q_hyd.size == 0:
		neglect_rtc = True

	pres_dry = pres - e					# partial pressure of dry air in Pa

	# compute total water mixing ratio (vapour, liquid, ice, snow, rain) in kg kg-1
	if neglect_rtc:
		r_t = np.zeros(temp.shape)		# total water mixing ratio (vapour, liquid, ice, snow, rain)
	else:
		# convert q_hyd + q to r_t
		r_t = convert_spechum_to_mix_rat(q_hyd + q)

	cpd_rtc = c_pd + r_t*c_h2o
	theta_e = temp * (100000.0 / pres_dry)**(R_d / cpd_rtc) * relhum**(-r_v*R_v / cpd_rtc) * np.exp(L_v*r_v / (cpd_rtc*temp))

	return theta_e


def Z_from_GP(
	gp):

	"""
	Computes geopotential height (in m) from geopotential.

	Parameters:
	gp : float or array of float
		Geopotential in m^2 s^-2.
	"""

	return gp / g
