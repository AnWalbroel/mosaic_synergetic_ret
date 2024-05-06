import numpy as np
import copy
import datetime as dt
import xarray as xr
import pandas as pd
import os
import glob
import pdb
import warnings


def running_mean(x, N):

    """
    Moving average of a 1D array x with a window width of N

    Parameters:
    -----------
    x : array of floats
        1D data vector of which the running mean is to be taken.
    N : int
        Running mean window width.
    """
    x = x.astype(np.float64)
    x_m = copy.deepcopy(x)
    
    # run through the array:
    for k in range(len(x)):
        if k%400000 == 0: print(k/len(x))   # output required to avoid the ssh connection to
                                            # be automatically dropped

        # Identify which indices are addressed for the running
        # mean of the current index k:
        if N%2 == 0:    # even:
            rm_range = np.arange(k - int(N/2), k + int(N/2), dtype = np.int32)
        else:           # odd:
            rm_range = np.arange(k - int(N/2), k + int(N/2) + 1, dtype=np.int32)

        # remove indices that exceed array bounds:
        rm_range = rm_range[(rm_range >= 0) & (rm_range < len(x))]

        # moving average:
        x_m[k] = np.mean(x[rm_range])
    
    return x_m


def running_mean_datetime(x, N, t):

    """
    Moving average of a 1D array x with a window width of N in seconds.
    Here it is required to find out the actual window range. E.g. if
    the desired window width is 300 seconds but the measurement rate
    is one/minute, the actual window width is 5.

    Parameters:
    -----------
    x : array of floats
        1D data vector of which the running mean is to be taken.
    N : int
        Running mean window width in seconds.
    t : array of floats
        1D time vector (in seconds since a reference time) required to
        compute the actual running mean window width.
    """

    x = x.astype(np.float64)
    x_m = copy.deepcopy(x)

    n_x = len(x)
    is_even = (N%2 == 0)        # if N is even: is_even is True

    # ii = np.arange(len(x))    # array of indices, required for the 'slow version'

    # inquire mean delta time to get an idea of how broad a window
    # must be (roughly) <-> used to speed up computation time:
    mdt = np.nanmean(t[1:] - t[:-1])
    look_range = int(np.ceil(N/mdt))
    
    # run through the array:
    look_save = 0
    for k in range(n_x):    # k, t_c in enumerate(t)?
        if k%400000 == 0: print(k/n_x)  # output required to avoid ssh connection to
                                        # be automatically dropped

        # Identify the correct running mean window width from the current
        # time t_c:
        t_c = t[k]
        if is_even: # even:
            t_c_plus = t_c + int(N/2)
            t_c_minus = t_c - int(N/2)
            # t_range = t[(t >= t_c_minus) & (t <= t_c_plus)]   # not required
        else:           # odd:
            t_c_plus = t_c + int(N/2) + 1
            t_c_minus = t_c - int(N/2)
            # t_range = t[(t >= t_c_minus) & (t <= t_c_plus)]   # not required

        # rm_range_SAVE = ii[(t >= t_c_minus) & (t <= t_c_plus)]        # very slow for large time axis array but also works
        # faster:

        if (k > look_range) and (k < n_x - look_range): # in between
            look_save = k-look_range
            rm_range = np.argwhere((t[k-look_range:k+look_range] >= t_c_minus) & (t[k-look_range:k+look_range] <= t_c_plus)).flatten() + look_save
        elif k <= look_range:   # lower end of array
            look_save = 0
            rm_range = np.argwhere((t[:k+look_range] >= t_c_minus) & (t[:k+look_range] <= t_c_plus)).flatten()
        else:   # upper end of array
            look_save = k-look_range
            rm_range = np.argwhere((t[k-look_range:] >= t_c_minus) & (t[k-look_range:] <= t_c_plus)).flatten() + look_save
            

        # moving average:
        x_m[k] = np.mean(x[rm_range])
    
    return x_m


def running_mean_pdtime(x, N, t):
    """
    Running mean of a 1D array x with a window width of N seconds.

    Parameters:
    -----------
    x : array of floats
        1D data vector of which the running mean is to be taken.
    N : int
        Running mean window width in seconds.
    t : array of floats
        1D time vector (in numpy datetim64[ns]) required to
        compute the actual running mean window width.
    """

    # first, create xarray DataArray and convert it to pandas DataFrame:
    x_DA = xr.DataArray(x, dims=['time'], coords={'time': (['time'], t)})
    x_DF = x_DA.to_dataframe(name='x')

    # compute running mean (rolling mean): center=True is recommended to have a 5-min running
    # at 2020-01-01T14:00:00 from 2020-01-01T13:57:30 until 2020-01-01T14:02:30.
    x_rm = x_DF.rolling(f"{int(N)}S", center=True).mean().to_xarray().x

    return x_rm.values


def running_mean_time_2D(x, N, t, axis=0):

    """
    Moving average of a 2D+ array x with a window width of N in seconds.
    The moving average will be taken over the specifiec axis.
    Here it is required to find out the actual window range. E.g. if
    the desired window width is 300 seconds but the measurement rate
    is one/minute, the actual window width is 5.

    Parameters:
    -----------
    x : array of floats
        Data array (multi-dim) of which the running mean is to be taken for a
        certain axis.
    N : int
        Running mean window width in seconds.
    t : array of floats
        1D time vector (in seconds since a reference time) required to
        compute the actual running mean window width.
    axis : int
        Indicates, which axis represents the time axis, over which the moving
        average will be taken. Default: 0
    """

    # check if shape of x is correct:
    n_x = x.shape[axis]
    assert n_x == len(t)

    x = x.astype(np.float64)
    x_m = copy.deepcopy(x)

    is_even = (N%2 == 0)        # if N is even: is_even is True

    # inquire mean delta time to get an idea of how broad a window
    # must be (roughly) <-> used to speed up computation time:
    mdt = np.nanmean(np.diff(t))
    look_range = int(np.ceil(N/mdt))
    
    # run through the array:
    look_save = 0
    for k in range(n_x):    # k, t_c in enumerate(t)?
        if k%400000 == 0: print(k/n_x)  # output required to avoid ssh connection to
                                        # be automatically dropped

        # Identify the correct running mean window width from the current
        # time t_c:
        t_c = t[k]
        if is_even: # even:
            t_c_plus = t_c + int(N/2)
            t_c_minus = t_c - int(N/2)
        else:           # odd:
            t_c_plus = t_c + int(N/2) + 1
            t_c_minus = t_c - int(N/2)


        if (k > look_range) and (k < n_x - look_range): # in between
            look_save = k-look_range
            rm_range = np.argwhere((t[k-look_range:k+look_range] >= t_c_minus) & (t[k-look_range:k+look_range] <= t_c_plus)).flatten() + look_save
        elif k <= look_range:   # lower end of array
            look_save = 0
            rm_range = np.argwhere((t[:k+look_range] >= t_c_minus) & (t[:k+look_range] <= t_c_plus)).flatten()
        else:   # upper end of array
            look_save = k-look_range
            rm_range = np.argwhere((t[k-look_range:] >= t_c_minus) & (t[k-look_range:] <= t_c_plus)).flatten() + look_save
            

        # moving average:
        x_m[k] = np.mean(x[rm_range], axis=axis)
    
    return x_m


def datetime_to_epochtime(dt_array):
    
    """
    This tool creates a 1D array (or of seconds since 1970-01-01 00:00:00 UTC
    (type: float) out of a datetime object or an array of datetime objects.

    Parameters:
    -----------
    dt_array : array of datetime objects or datetime object
        Array (1D) that includes datetime objects. Alternatively, dt_array is directly a
        datetime object.
    """

    reftime = dt.datetime(1970,1,1)

    try:
        sec_epochtime = np.asarray([(dtt - reftime).total_seconds() for dtt in dt_array])
    except TypeError:   # then, dt_array is no array
        sec_epochtime = (dt_array - reftime).total_seconds()

    return sec_epochtime


def numpydatetime64_to_epochtime(npdt_array):

    """
    Converts numpy datetime64 array to array in seconds since 1970-01-01 00:00:00 UTC (type:
    float).
    Alternatively, just use "some_array.astype(np.float64)" or it might be needed to first
    convert to some_array.astype("datetime64[s]").astype(np.float64).

    Parameters:
    -----------
    npdt_array : numpy array of type np.datetime64 or np.datetime64 type
        Array (1D) or directly a np.datetime64 type variable.
    """

    sec_epochtime = npdt_array.astype(np.timedelta64) / np.timedelta64(1, 's')

    return sec_epochtime


def numpydatetime64_to_reftime(
    npdt_array, 
    reftime):

    """
    Converts numpy datetime64 array to array in seconds since a reftime as type:
    float. Reftime could be for example: "2017-01-01 00:00:00" (in UTC)

    Parameters:
    -----------
    npdt_array : numpy array of type np.datetime64 or np.datetime64 type
        Array (1D) or directly a np.datetime64 type variable.
    reftime : str
        Specification of the reference time in "yyyy-mm-dd HH:MM:SS" (in UTC).
    """

    time_dt = numpydatetime64_to_datetime(npdt_array)

    reftime = dt.datetime.strptime(reftime, "%Y-%m-%d %H:%M:%S")

    try:
        sec_epochtime = np.asarray([(dtt - reftime).total_seconds() for dtt in time_dt])
    except TypeError:   # then, time_dt is no array
        sec_epochtime = (time_dt - reftime).total_seconds()

    return sec_epochtime


def numpydatetime64_to_datetime(npdt_array):

    """
    Converts numpy datetime64 array to a datetime object array.

    Parameters:
    -----------
    npdt_array : numpy array of type np.datetime64 or np.datetime64 type
        Array (1D) or directly a np.datetime64 type variable.
    """

    sec_epochtime = npdt_array.astype(np.timedelta64) / np.timedelta64(1, 's')

    # sec_epochtime can be an array or just a float
    if sec_epochtime.ndim > 0:
        time_dt = np.asarray([dt.datetime.utcfromtimestamp(tt) for tt in sec_epochtime])

    else:
        time_dt = dt.datetime.utcfromtimestamp(sec_epochtime)

    return time_dt


def compute_DOY(
    time,
    return_dt=True,
    reshape=False):

    """
    Compute the cos and sin of the day of the year for a given time.

    Parameters:
    -----------
    time : numpy array of floats or float
        Time data (must be in seconds since 1970-01-01 00:00:00 UTC) used to compute 
        the cos and sin of the day of the year.
    return_dt : bool
        If True the datetime object/array used for the computation is returned as well.
    reshape : bool
        If True an additional dimension of length 1 will be added to DOY_1 and DOY_2 via
        reshaping.
    """

    time_dt = np.asarray([dt.datetime.utcfromtimestamp(ttt) for ttt in time])

    DOY = np.asarray([(ttt - dt.datetime(ttt.year,1,1)).days*2*np.pi/365 for ttt in time_dt])
    DOY_1 = np.cos(DOY)
    DOY_2 = np.sin(DOY)

    if reshape:
        n_data = len(time)
        DOY_1 = np.reshape(DOY_1, (n_data,1))
        DOY_2 = np.reshape(DOY_2, (n_data,1))

    if return_dt:
        return DOY_1, DOY_2, time_dt
    else:
        return DOY_1, DOY_2


def break_str_into_lines(
    le_string,
    n_max,
    split_at=' ',
    keep_split_char=False):

    """
    Break a long strings into multiple lines if a certain number of chars may
    not be exceeded per line. String will be split into two lines if its length
    is > n_max but <= 2*n_max.

    Parameters:
    -----------
    le_string : str
        String that will be broken into several lines depending on n_max.
    n_max : int
        Max number of chars allowed in one line.
    split_at : str
        Character to look for where the string will be broken. Default: space ' '
    keep_split_char : bool
        If True, the split char indicated by split_at will not be removed (useful for "-" as split char).
        Default: False
    """

    n_str = len(le_string)
    if n_str > n_max:
        # if string is > 2*n_max, then it has to be split into three lines, ...:
        n_lines = (n_str-1) // n_max        # // is flooring division

        # look though the string in backwards direction to find the first space before index n_max:
        le_string_bw = le_string[::-1]
        new_line_str = "\n"

        for k in range(n_lines):
            space_place = le_string_bw.find(split_at, n_str - (k+1)*n_max)
            if keep_split_char:
                le_string_bw = le_string_bw[:space_place].replace("\n","") + new_line_str + le_string_bw[space_place:]
            else:
                le_string_bw = le_string_bw[:space_place] + new_line_str + le_string_bw[space_place+1:]

        # reverse the string again
        le_string = le_string_bw[::-1]

    return le_string


def bin_to_dec(b_in):

    """
    Converts a binary number given as string to normal decimal number (as integer).

    Parameters:
    -----------
    b_in : str
        String of a binary number that may either directly start with the
        binary number or start with "0b".
    """

    d_out = 0       # output as decimal number (int or float)
    if "b" in b_in:
        b_in = b_in[b_in.find("b")+1:]  # actual bin number starts after "b"
    b_len = len(b_in)

    for ii, a in enumerate(b_in): d_out += int(a)*2**(b_len-ii-1)

    return d_out


def compute_retrieval_statistics(
    x_stuff,
    y_stuff,
    compute_stddev=False):

    """
    Compute bias, RMSE and Pearson correlation coefficient (and optionally the standard deviation)
    from x and y data.

    Parameters:
    x_stuff : float or array of floats
        Data that is to be plotted on the x axis.
    y_stuff : float or array of floats
        Data that is to be plotted on the y axis.
    compute_stddev : bool
        If True, the standard deviation is computed (bias corrected RMSE).
    """

    where_nonnan = np.argwhere(~np.isnan(y_stuff) & ~np.isnan(x_stuff)).flatten()
                    # -> must be used to ignore nans in corrcoef
    stat_dict = {   'N': np.count_nonzero(~np.isnan(x_stuff) & ~np.isnan(y_stuff)),
                    'bias': np.nanmean(y_stuff - x_stuff),
                    'rmse': np.sqrt(np.nanmean((x_stuff - y_stuff)**2)),
                    'R': np.corrcoef(x_stuff[where_nonnan], y_stuff[where_nonnan])[0,1]}

    if compute_stddev:
        stat_dict['stddev'] = np.sqrt(np.nanmean((x_stuff - (y_stuff - stat_dict['bias']))**2))

    return stat_dict


def compute_RMSE_profile(
    x,
    x_o,
    which_axis=0):

    """
    Compute RMSE 'profile' of a i.e., (height x time)-matrix (e.g. temperature profile):
    RMSE(z_i) = sqrt(mean((x - x_o)^2, dims='time'))
    
    Parameters:
    -----------
    x : 2D array of numerical
        Data matrix whose deviation from a reference is desired.
    x_o : 2d array of numerical
        Data matrix of the reference.
    which_axis : int
        Indicator which axis is to be averaged over. For the RMSE profile, you would
        want to average over time!
    """

    if which_axis not in [0, 1]:
        raise ValueError("'which_axis' must be either 0 or 1!")

    return np.sqrt(np.nanmean((x - x_o)**2, axis=which_axis))


def build_K_reg(
    y,
    order=1):

    """
    Constructs the observation matrix typically used for regression retrievals where the
    rows indicate the samples (i.e., time series). The first column usually contains "1"
    only and the remaining columns contain observations in first and higher order.

    Parameters:
    -----------
    y : array of floats
        Observation vector. Must be a numpy array with M observations and N samples. The 
        shape must be N x M. (Also if M == 1, y must be a 2D array.)
    order : int
        Defines the order of the regression equation. Options: i.e., 1, 2, 3. Default:
        1
    """

    n_obs = y.shape[1]      # == M
    n_samples = y.shape[0]  # == N

    assert y.shape == (n_samples,n_obs)

    # generate regression matrix K out of obs vector:
    K_reg = np.ones((n_samples, order*n_obs+1))
    K_reg[:,1:n_obs+1] = y

    if order > 1:
        for kk in range(order-1):
            jj = kk + 1
            K_reg[:,jj*n_obs+1:(jj+1)*n_obs+1] = y**(jj+1)

    return K_reg


def regression(
    x,
    y,
    order=1):
    
    """
    Computes regression coefficients m_est to map observations y (i.e., brightness temperatures)
    to state variable x (i.e., temperature profile at one height level, or IWV). The regression
    order can also be specified.
    
    Parameters:
    -----------
    x : array of floats
        State variable vector. Must be a numpy array with N samples (N = training data size).
    y : array of floats
        Observation vector. Must be a numpy array with M observations (i.e., M frequencies) 
        and N samples. The shape must be N x M. (Also if M == 1, y must be a 2D array.)
    order : int
        Defines the order of the regression equation. Options: i.e., 1, 2, 3. Default:
        1
    """

    # Generate matrix from observations:
    K_reg = build_K_reg(y, order)

    # compute m_est
    K_reg_T = K_reg.T
    m_est = np.linalg.inv(K_reg_T@K_reg)@K_reg_T@x

    return m_est


def Gband_double_side_band_average(
    TB,
    freqs,
    xarray_compatibility=False,
    freq_dim_name=""):

    """
    Computes the double side band average of TBs that contain both
    sides of the G band absorption line. Returns either only the TBs
    or both the TBs and frequency labels with double side band avg.
    If xarray_compatibility is True, also more dimensional TB arrays
    can be included. Then, also the frequency dimension name must be
    supplied.

    Parameters:
    -----------
    TB : array of floats
        Brightness temperature array. Must have the following shape
        (time x frequency). More dimensions and other shapes are only
        allowed if xarray_compatibility=True.
    freqs : array of floats
        1D Array containing the frequencies of the TBs. The array must be
        sorted in ascending order.
    xarray_compatibility : bool
        If True, xarray utilities can be used, also allowing TBs of other
        shapes than (time x frequency). Then, also freq_dim_name must be
        provided.
    freq_dim_name : str
        Name of the xarray frequency dimension. Must be specified if 
        xarray_compatibility=True.
    """

    if xarray_compatibility and not freq_dim_name:
        raise ValueError("Please specify 'freq_dim_name' when using the xarray compatible mode.")

    # Double side band average for G band if G band frequencies are available, which must first be clarified:
    # Determine, which frequencies are around the G band w.v. absorption line:
    g_upper_end = 183.31 + 15
    g_lower_end = 183.31 - 15
    g_freq = np.where((freqs > g_lower_end) & (freqs < g_upper_end))[0]
    non_g_freq = np.where(~((freqs > g_lower_end) & (freqs < g_upper_end)))[0]

    TB_dsba = copy.deepcopy(TB)

    if g_freq.size > 0: # G band within frequencies
        g_low = np.where((freqs <= 183.31) & (freqs >= g_lower_end))[0]
        g_high = np.where((freqs >= 183.31) & (freqs <= g_upper_end))[0]

        assert len(g_low) == len(g_high)
        if not xarray_compatibility:
            for jj in range(len(g_high)):
                TB_dsba[:,jj] = (TB[:,g_low[-1-jj]] + TB[:,g_high[jj]])/2.0

        else:
            for jj in range(len(g_high)):
                # TB_dsba[{freq_dim_name: jj}] = (TB[{freq_dim_name: g_low[-1-jj]}] + TB[{freq_dim_name: g_high[jj]}])/2.0
                TB_dsba[{freq_dim_name: g_high[jj]}] = (TB[{freq_dim_name: g_low[-1-jj]}] + TB[{freq_dim_name: g_high[jj]}])/2.0


    # Indices for sorting:
    idx_have = np.concatenate((g_high, non_g_freq), axis=0)
    idx_sorted = np.argsort(idx_have)

    # truncate and append the unedited frequencies (e.g. 243 and 340 GHz):
    if not xarray_compatibility:
        TB_dsba = TB_dsba[:,:len(g_low)]
        TB_dsba = np.concatenate((TB_dsba, TB[:,non_g_freq]), axis=1)

        # Now, the array just needs to be sorted correctly:
        TB_dsba = TB_dsba[:,idx_sorted]

        # define freq_dsba (usually, the upper side of the G band is then used as
        # frequency label:
        freq_dsba = np.concatenate((freqs[g_high], freqs[non_g_freq]))[idx_sorted]

    else:
        # TB_dsba = TB_dsba[{freq_dim_name: slice(0,len(g_low))}]
        TB_dsba = TB_dsba[{freq_dim_name: g_high}]
        TB_dsba = xr.concat([TB_dsba, TB[{freq_dim_name: non_g_freq}]], dim=freq_dim_name)

        # Now, the array just needs to be sorted correctly:
        TB_dsba = TB_dsba[{freq_dim_name: idx_sorted}]

        # define freq_dsba (usually, the upper side of the G band is then used as
        # frequency label:
        freq_dsba = xr.concat([freqs[g_high], freqs[non_g_freq]], dim=freq_dim_name)[idx_sorted]


    return TB_dsba, freq_dsba


def Fband_double_side_band_average(
    TB,
    freqs,
    xarray_compatibility=False,
    freq_dim_name=""):

    """
    Computes the double side band average of TBs that contain both
    sides of the F band absorption line. Returns either only the TBs
    or both the TBs and frequency labels with double side band avg.

    Parameters:
    -----------
    TB : array of floats
        Brightness temperature array. Must have the following shape
        (time x frequency).
    freqs : array of floats
        1D Array containing the frequencies of the TBs. The array must be
        sorted in ascending order.
    xarray_compatibility : bool
        If True, xarray utilities can be used, also allowing TBs of other
        shapes than (time x frequency). Then, also freq_dim_name must be
        provided.
    freq_dim_name : str
        Name of the xarray frequency dimension. Must be specified if 
        xarray_compatibility=True.
    """

    if xarray_compatibility and not freq_dim_name:
        raise ValueError("Please specify 'freq_dim_name' when using the xarray compatible mode.")

    # Double side band average for F band if F band frequencies are available, which must first be clarified:
    # Determine, which frequencies are around the F band w.v. absorption line:
    upper_end = 118.75 + 10
    lower_end = 118.75 - 10
    f_freq = np.where((freqs > lower_end) & (freqs < upper_end))[0]
    non_f_freq = np.where(~((freqs > lower_end) & (freqs < upper_end)))[0]

    TB_dsba = copy.deepcopy(TB)
    
    if f_freq.size > 0: # F band within frequencies
        low = np.where((freqs <= 118.75) & (freqs >= lower_end))[0]
        high = np.where((freqs >= 118.75) & (freqs <= upper_end))[0]

        assert len(low) == len(high)
        if not xarray_compatibility:
            for jj in range(len(high)):
                TB_dsba[:,jj] = (TB[:,low[-1-jj]] + TB[:,high[jj]])/2.0

        else:
            for jj in range(len(high)):
                TB_dsba[{freq_dim_name: jj}] = (TB[{freq_dim_name: low[-1-jj]}] + TB[{freq_dim_name: high[jj]}])/2.0


    # Indices for sorting:
    idx_have = np.concatenate((high, non_f_freq), axis=0)
    idx_sorted = np.argsort(idx_have)

    # truncate and append the unedited frequencies (e.g. 243 and 340 GHz):
    if not xarray_compatibility:
        TB_dsba = TB_dsba[:,:len(low)]
        TB_dsba = np.concatenate((TB_dsba, TB[:,non_f_freq]), axis=1)

        # Now, the array just needs to be sorted correctly:
        TB_dsba = TB_dsba[:,idx_sorted]

        # define freq_dsba (usually, the upper side of the G band is then used as
        # frequency label:
        freq_dsba = np.concatenate((freqs[high], freqs[non_f_freq]))[idx_sorted]

    else:
        TB_dsba = TB_dsba[{freq_dim_name: slice(0,len(low))}]
        TB_dsba = xr.concat([TB_dsba, TB[{freq_dim_name: non_f_freq}]], dim=freq_dim_name)

        # Now, the array just needs to be sorted correctly:
        TB_dsba = TB_dsba[{freq_dim_name: idx_sorted}]

        # define freq_dsba (usually, the upper side of the G band is then used as
        # frequency label:
        freq_dsba = xr.concat([freqs[high], freqs[non_f_freq]], dim=freq_dim_name)[idx_sorted]

    return TB_dsba, freq_dsba


def select_MWR_channels(
    TB,
    freq,
    band,
    return_idx=0):

    """
    This function selects certain frequencies (channels) of brightness temperatures (TBs)
    from a given set of TBs. The output will therefore be a subset of the input TBs. Single
    frequencies cannot be selected but only 'bands' (e.g. K band, V band, ...). Combinations
    are also possible.

    Parameters:
    -----------
    TB : array of floats
        2D array (i.e., time x freq; freq must be the second dimension) or higher dimensional
        array (where freq must be on axis -1) of TBs (in K).
    freq : array of floats
        1D array of frequencies (in GHz).
    band : str
        Specify the frequencies to be selected. Valid options:
        'K': 20-40 GHz, 'V': 50-60 GHz, 'W': 85-95 GHz, 'F': 110-130 GHz, 'G': 170-200 GHz,
        '243/340': 240-350 GHz
        Combinations are also possible: e.g. 'K+V+W' = 20-95 GHz
    return_idx : int
        If 0 the frq_idx list is not returned and merely TB and freq are returned.
        If 1 TB, freq, and frq_idx are returned. If 2 only frq_idx is returned.
    """

    # define dict of band limits:
    band_lims = {   'K': [20, 40],
                    'V': [50, 60],
                    'W': [85, 95],
                    'F': [110, 130],
                    'G': [170, 200],
                    '243/340': [240, 350]}

    # split band input:
    band_split = band.split('+')

    # cycle through all bands:
    frq_idx = list()
    for k, baba in enumerate(band_split):
        # find the right indices for the appropriate frequencies:
        frq_idx_temp = np.where((freq >= band_lims[baba][0]) & (freq <= band_lims[baba][1]))[0]
        for fit in frq_idx_temp: frq_idx.append(fit)

    # sort the list and select TBs:
    frq_idx = sorted(frq_idx)
    TB = TB[..., frq_idx]
    freq = freq[frq_idx]

    if return_idx == 0:
        return TB, freq

    elif return_idx == 1:
        return TB, freq, frq_idx

    elif return_idx == 2:
        return frq_idx

    else:
        raise ValueError("'return_idx' in function 'select_MWR_channels' must be an integer. Valid options: 0, 1, 2")


def filter_time(
    time_have,
    time_wanted,
    window=0,
    around=False):

    """
    This function returns a mask (True, False) when the first argument (time_have) is in
    the range time_wanted:time_wanted+window (in seconds) (for around=False) or in the
    range time_wanted-window:time_wanted+window.

    It is important that time_have and time_wanted have the same units (e.g., seconds 
    since 1970-01-01 00:00:00 UTC). t_mask will be True when time_have and time_wanted
    overlap according to 'window' and 'around'. The overlap always includes the boundaries
    (e.g., time_have >= time_wanted & time_have <= time_wanted + window).

    Parameters:
    -----------
    time_have : 1D array of float or int
        Time array that should be masked so that you will know, when time_have overlaps
        with time_wanted.
    time_wanted : 1D array of float or int
        Time array around which 
    window : int or float
        Window in seconds around time_wanted (or merely from time_wanted until time_wanted
        + window) that will be set True in the returned mask. If window = 0, the closest
        match will be used.
    around : bool
        If True, time_wanted - window : time_wanted + window is considered. If False,
        time_wanted : time_wanted + window is considered.
    """

    if not isinstance(around, bool):
        return TypeError("Argument 'around' must be boolean.")

    # Initialise mask with False. Overlap with time_wanted will then be set True.
    have_shape = time_have.shape
    t_mask = np.full(have_shape, False)

    if window > 0:
        if around:  # search window is in both directions around time_wanted
            for tw in time_wanted:
                idx = np.where((time_have >= tw - window) & (time_have <= tw + window))[0]
                t_mask[idx] = True

        else:       # search window only in one direction
            for tw in time_wanted:
                idx = np.where((time_have >= tw) & (time_have <= tw + window))[0]
                t_mask[idx] = True

    else:   # window <= 0: use closest match; around = True or False doesn't matter here
        for tw in time_wanted:
            idx = np.argmin(np.abs(time_have - tw)).flatten()
            t_mask[idx] = True

    return t_mask


def find_files_daterange(
    all_files, 
    date_start_dt, 
    date_end_dt,
    idx,
    file_dt_fmt="%Y%m%d"):

    """
    Filter from a given set of files the correct ones within the date range
    date_start_dt - date_end_dt (including start and end date).

    Parameters:
    -----------
    all_files : list of str
        List of str that includes all the files.
    date_start_dt : datetime object
        Start date as a datetime object.
    date_end_dt : datetime object
        End date as a datetime object.
    idx : list of int
        List of int where the first entry specifies the start and the second one
        the end of the date string in any all_files item (i.e., [-17,-9]).
    file_dt_fmt : str
        String indicating the date format in the file names. E.g., "%Y%m%d" for
        "20190725".
    """

    files = list()
    for pot_file in all_files:
        # check if file is within our date range:
        file_dt = dt.datetime.strptime(pot_file[idx[0]:idx[1]], file_dt_fmt)
        if (file_dt >= date_start_dt) & (file_dt <= date_end_dt):
            files.append(pot_file)

    return files


def change_colormap_len(
    cmap, 
    n_new):

    """
    Changes the number of colours of a given colourmap (cmap) to the number given by n_new.

    Parameters:
    -----------
    cmap : matplotlib.colors element
        Colourmap used by matplotlib.
    n_new : int
        Number of colours the new colourmap should have ('length of colourmap').
    """

    import matplotlib as mpl

    len_cmap = cmap.shape[0]    # length of colourmap
    n_rgba = cmap.shape[1]      # rgb + alpha

    cmap_new = np.zeros((n_new, n_rgba))
    for m in range(n_rgba):
        cmap_new[:,m] = np.interp(np.linspace(0, 1, n_new), np.linspace(0, 1, len_cmap), cmap[:,m])

    cmap_new = mpl.colors.ListedColormap(cmap_new)

    return cmap_new
