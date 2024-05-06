def run_training_data_map(path_data_dict, path_plots):

    """
    This script creates a map plot showing the ERA5 training data grid points for the retrieval
    and the MOSAiC track. The mean sea ice concentration is also included.
    - import training data for a year
    - import sea ice concentration data and MOSAiC track
    - visualize

    Parameters:
    -----------
    path_data_dict : dict
        Dictionary containing strings of the full paths of the ERA5 training data, ERA5 sea ice 
        data, MOASiC Polarstern track data and Cartopy background. The keys of the dictionary must
        be 'era5', 'era5_sic', 'ps_track' and 'cartopy_background', respectively.
    path_plots : str
        Full path where the plot should be saved to.
    """

    import gc
    import datetime as dt
    import pdb
    import glob
    import os
    import sys


    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as mpl_pe
    import numpy as np
    import xarray as xr
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt
    from cmcrameri import cm

    from data_tools import change_colormap_len
    from import_data import import_PS_mastertrack


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
                pstrack_dict['time_sec'][m] = np.datetime64(current_line[0]).astype('datetime64[s]').astype('float64')

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


        # create xarray dataset:
        pstrack_ds = xr.Dataset(coords={'time': (['time'], pstrack_dict['time_sec'].astype('datetime64[s]').astype('datetime64[ns]'))})
        pstrack_ds['Longitude'] = xr.DataArray(pstrack_dict['Longitude'], dims=['time'])
        pstrack_ds['Latitude'] = xr.DataArray(pstrack_dict['Latitude'], dims=['time'])

        return pstrack_ds


    # background image for cartopy:
    os.environ['CARTOPY_USER_BACKGROUNDS'] = path_data_dict['cartopy_background']


    # paths:
    path_data = path_data_dict
    path_plots = path_plots


    # additional settings:
    set_dict = {'save_figures': True,
                '1D_aligned': True}

    os.makedirs(os.path.dirname(path_plots), exist_ok=True)

    if set_dict['1D_aligned']:
        dims_2d_list = ['time', 'x']
        dims_3d_list = ['time', 'x', 'z']
    else:
        dims_2d_list = ['time', 'x', 'y']
        dims_3d_list = ['time', 'x', 'y', 'z']

    # import data:
    files = sorted(glob.glob(path_data['era5'] + "ERA5_syn_ret_*.nc"))
    era5_atmos = xr.open_mfdataset(files[0], concat_dim='time', combine='nested')
    SIC_DS = xr.open_dataset(path_data['era5_sic'] + "ERA5_single_level_SIC_arctic_2001-2018.nc")


    # load MOSAiC track data:
    files_ps_track = sorted(glob.glob(path_data['ps_track'] + "PS122*.tab"))
    ps_track_list = list()
    for k, pstrack_file in enumerate(files_ps_track):
        pstrack_ds = import_PS_mastertrack_tab(pstrack_file)
        ps_track_list.append(pstrack_ds)
    PS_TRACK_DS = xr.concat(ps_track_list, dim='time', join='outer')
    del ps_track_list


    # visualize:
    fs = 16
    fs_small = fs - 2
    fs_dwarf = fs_small - 2

    # colours and colourmaps:
    levels_0 = np.arange(0.0, 100.1, 2.5)
    n_levels = len(levels_0)
    cmap_o = mpl.cm.get_cmap('Blues_r', n_levels)
    cmap_o = cmap_o(range(n_levels))
    cmap_o[0,:] = np.array([153./255., 179./255., 204./255., 1.0])
    cmap_target = cmap_o[4,:]
    for k in range(cmap_o.shape[1]): cmap_o[:5,k] = np.linspace(cmap_o[0,k], cmap_target[k], 5)
    cmap_o = mpl.colors.ListedColormap(cmap_o)

    cmap_track = cm.vik(range(len(cm.vik.colors)))      # colourmap for MOSAiC drift track
    n_levels = len(PS_TRACK_DS.Latitude)
    cmap_track = change_colormap_len(cmap_track, n_levels)


    data_plot = SIC_DS.siconc*100.
    marker_size = 9.0

    # map_settings:
    # lon_centre = 0.0
    # lat_centre = 75.0
    # lon_lat_extent = [-60.0, 60.0, 60.0, 90.0]        # (zoomed in)
    # sel_projection = ccrs.Orthographic(central_longitude=lon_centre, central_latitude=lat_centre)
    lon_centre = 8.5
    lat_centre = 79.5
    lon_lat_extent = [-20.0, 35.0, 75, 90.0]
    sel_projection = ccrs.NearsidePerspective(central_longitude=lon_centre, central_latitude=lat_centre, satellite_height=1800000)


    # some extra info for the plot:
    station_coords = {'Ny-\u00C5lesund': [11.93, 78.92]}


    f1 = plt.figure(figsize=(6,7.5))
    a1 = plt.axes(projection=sel_projection)
    a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
    a1.background_img(name='NaturalEarthRelief', resolution='high')


    # add some land marks:
    a1.coastlines(resolution="50m", zorder=9999.0, linewidth=0.5)
    a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
    a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
    a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
    gridlines = a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

    PlateCarree_mpl_transformer = ccrs.PlateCarree()._as_mpl_transform(a1)
    text_transform = mpl.transforms.offset_copy(PlateCarree_mpl_transformer, units='dots', 
                                                x=marker_size*2.4, y=marker_size*2.4)


    # plot MOSAiC track:
    time_range = PS_TRACK_DS.time[-1] - PS_TRACK_DS.time[0]
    normed_time = (PS_TRACK_DS.time.values - PS_TRACK_DS.time[0].values) / time_range.values
    a1.plot(PS_TRACK_DS.Longitude, PS_TRACK_DS.Latitude, color=(0,0,0), linewidth=2,
            path_effects=[mpl_pe.Stroke(linewidth=4.0, foreground=(0,0,0)), mpl_pe.Normal()],
            transform=ccrs.PlateCarree(), zorder=9997.0)
    sc_track = a1.scatter(PS_TRACK_DS.Longitude, PS_TRACK_DS.Latitude, s=4, c=normed_time, vmin=0, vmax=1, 
                        cmap=cmap_track, marker='o',
                        edgecolors='none', transform=ccrs.PlateCarree(), zorder=9998.0)


    # plot SIC datadata:
    contour_0 = a1.contourf(data_plot.longitude.values, data_plot.latitude.values, data_plot.mean('time').values, 
                            cmap=cmap_o, levels=levels_0, transform=ccrs.PlateCarree())


    # plot the grid points:
    sel_lats = era5_atmos.lat[:].values
    sel_lons = era5_atmos.lon[:].values
    ii = 0
    for slat, slon in zip(sel_lats, sel_lons):
        if ii == 0:
            a1.plot(slon, slat, linestyle='none', color=(0,1,1), marker='o', markersize=marker_size, 
                        markeredgecolor=(0,0,0), transform=ccrs.PlateCarree(), zorder=10000.0, label='Selected grid points')
        else:
            a1.plot(slon, slat, linestyle='none', color=(0,1,1), marker='o', markersize=marker_size, 
                        markeredgecolor=(0,0,0), transform=ccrs.PlateCarree(), zorder=10000.0)
        ii +=1



    # place markers and labels:
    a1.plot(station_coords['Ny-\u00C5lesund'][0], station_coords['Ny-\u00C5lesund'][1], color=(1,0,0),
            marker='^', markersize=marker_size, markeredgecolor=(0,0,0),
            transform=ccrs.PlateCarree(), zorder=10000.0)

    a1.text(station_coords['Ny-\u00C5lesund'][0], station_coords['Ny-\u00C5lesund'][1], 'Ny-\u00C5lesund',
            ha='left', va='bottom',
            color=(1,0,0), fontsize=fs_dwarf, transform=text_transform, 
            bbox={'facecolor': (1,1,1), 'edgecolor': (0,0,0), 'boxstyle': 'square'},
            zorder=10000.0)


    # colorbar(s) and legends:
    lh, ll = a1.get_legend_handles_labels()
    leg = a1.legend(lh, ll, loc='upper left', fontsize=fs_dwarf-2)
    leg = leg.set(zorder=10001.0)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cb_axis = inset_axes(a1, width="80%", height="2%", loc='lower center')
    cb_track = f1.colorbar(mappable=sc_track, cax=cb_axis, orientation='horizontal')
    cb_track.set_ticks([0.05,0.95], labels=["Sep 2019", "Oct 2020"], fontsize=fs_dwarf-2, 
                        path_effects=[mpl_pe.Stroke(linewidth=2.5, foreground=(1,1,1)), mpl_pe.Normal()])
    cb_track.ax.xaxis.set_ticks_position('top')
    cb_track.set_label(label="MOSAiC drift track", labelpad=-25, fontsize=fs_dwarf-2,
                        path_effects=[mpl_pe.Stroke(linewidth=2.5, foreground=(1,1,1)), mpl_pe.Normal()])


    cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='max', orientation='horizontal', 
                            fraction=0.06, pad=0.04, shrink=1.00)
    cb_var.set_label(label="Sea ice concentration (%)", fontsize=fs_small)
    cb_var.ax.tick_params(labelsize=fs_dwarf)


    # customize grid lines:
    gridlines.xlocator = mpl.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
    gridlines.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gridlines.top_labels = False
    gridlines.left_labels = False

    if set_dict['save_figures']:
        plotname = f"ERA5_syn_ret_training_map_plot_overview"
        f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
        f1.savefig(path_plots + plotname + ".pdf", bbox_inches='tight')
    else:
        plt.show()
        pdb.set_trace()

    f1.clf()
    plt.close()
    gc.collect()