def run_DOF_plots(path_data, path_plots):

    """
    In this script, the Degrees Of Freedom (DOF) of the Neural Network based temperature and
    humidity profile retrievals will be visualized. DOFs have been computed for various predictor
    combinations with the evaluation dataset (using NN_retrieval.py exec_type='i_cont'). Also 
    different weather conditions will be highlighted to assess the sensitivity of DOF to
    certain conditions. The following plots will be generated: Cumulative diagonal entries of
    the Averaging Kernel matrix, boxplot showing DOF over the info content data samples (subset
    of evaluation dataset).
    - import information content data
    - import evaluation dataset
    - cut evaluation dataset to subset of information content estimation
    - differentiate between certain weather conditions
    - visualize cumulative information content
    - visualize DOF in boxplot

    Parameters:
    -----------
    path_data : dict
        Dictionary containing strings of the full paths of the ERA5 evaluation data (key 'eval')
        and information content estimation output (key 'i_cont').
    path_plots : str
        String indicating where to save the plots.
    """

    import os
    import sys
    import glob
    import pdb
    import gc

    import numpy as np
    import xarray as xr
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from data_tools import break_str_into_lines


    # font and marker sizes
    fs = 18
    fs_small = fs - 2
    fs_dwarf = fs_small - 2
    fs_micro = fs_dwarf - 2
    msize = 9.0

    # colours:
    col_dict = {'OP': (1,0.72,0.12),        # synergetic retrieval
                'K': (0.067,0.29,0.769),    # MWR_PRO (HATPRO)
                'K+V': 'slategrey', 
                'K+G': 'darkcyan', 
                'K+V+G': 'darkkhaki'}
    col_fade_dict = {'OP': (1,0.72,0.12,0.15),  # synergetic ret, but less alpha
                    'K': (0.067,0.29,0.769,0.15)}


    def cumulative_DOF(
        I_DS_dict,
        set_dict):

        """
        Visualizes the information content by plotting the diagonal entries of the Averaging Kernel
        matrix in a cumulative way against height.

        Parameters:
        -----------
        I_DS_dict : dictionary of xarray datasets
            Dictionary containing xarray datasets for each set of predictors used for information
            content estimation for the predictand. Each dataset has been created with 
            NN_retrieval.py exec_type='i_cont'. 
        set_dict : dict
            Dictionary containing additional information.
        """

        # create output path if not existing:
        plotpath_dir = os.path.dirname(set_dict['path_plots'] + f"{set_dict['p']}/")
        if not os.path.exists(plotpath_dir):
            os.makedirs(plotpath_dir)

        # reduce unnecessary dimensions of height:
        height = I_DS_dict['OP'].height.values
        if height.ndim == 2:
            height = height[0,:]


        # compute vertical gradient of the Averaging Kernel diagonal:
        # grad_AK_diag = dict()
        # for i_k, key in enumerate(I_DS_dict.keys()):
            # AK_diag_key = I_DS_dict[key].AK_diag.median('n_s').values
            # grad_AK_diag[key] = (AK_diag_key[1:] - AK_diag_key[:-1]) / (height[1:] - height[:-1])


        # dictionaries for adaptations:
        DOF_label_dict = {'temp': "Temperature profile cumulative DOF",
                        'temp_bl': "Temperature profile cumulative DOF",
                        'q': "Specific humidity profile cumulative DOF"}


        f1 = plt.figure(figsize=(7,9))
        a1 = plt.axes()
        a2 = a1.twiny()

        y_lim = np.array([0.0, height.max()])
        if set_dict['p'] in ['temp', 'temp_bl']:
            x_lim = np.array([0, 5])
        elif set_dict['p'] == 'q':
            x_lim = np.array([0, 3])


        # plot data:
        color_array = np.array([(4,89,147), (219,96,0), (17,128,17), (180,12,13), (109,57,46), 
                                (192,89,161), (96,96,96), (155,156,7), (0,157,173)]) / 255.
        for i_k, key in enumerate(I_DS_dict.keys()):
            if key == 'OP':
                a1.plot(np.median(np.cumsum(I_DS_dict[key].AK_diag, axis=1), axis=0), height, color=(0,0,0), linewidth=1.5, label=key)
                a1.plot([I_DS_dict[key].DOF.median('n_s'), I_DS_dict[key].DOF.median('n_s')], y_lim, color=(0,0,0), 
                            linestyle='dashed', linewidth=1)

            else:
                a1.plot(np.median(np.cumsum(I_DS_dict[key].AK_diag, axis=1), axis=0), height, color=tuple(color_array[i_k]), linewidth=1.2, label=key)
                a1.plot([I_DS_dict[key].DOF.median('n_s'), I_DS_dict[key].DOF.median('n_s')], y_lim, color=tuple(color_array[i_k]), 
                            linestyle='dashed', linewidth=1)


        # legends:
        lh, ll = a1.get_legend_handles_labels()
        a1.legend(lh, ll, loc='upper left', bbox_to_anchor=(0.02, 0.99), fontsize=fs_micro, framealpha=0.5)

        # axis lims:
        a1.set_xlim(x_lim[0], x_lim[1])
        a1.set_ylim(y_lim[0], y_lim[1])

        # set ticks and tick labels and parameters:
        a1.tick_params(axis='both', labelsize=fs_micro)

        # grid:
        a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

        # labels:
        a1.set_xlabel(DOF_label_dict[set_dict['p']], fontsize=fs_dwarf)
        a1.set_ylabel("Height (m)", fontsize=fs_dwarf)


        if set_dict['save_figures']:
            plotname = f"NN_syn_ret_info_content_median_DOF_{set_dict['p']}"
            plotfile = plotpath_dir + "/" + plotname + ".png"
            f1.savefig(plotfile, dpi=300, bbox_inches='tight')
            print(f"Saved {plotfile}")
        else:
            plt.show()
            pdb.set_trace()

        plt.close()
        gc.collect()


    def DOF_boxplot(
        I_DS_dict,
        idx_dict,
        set_dict):

        """
        Visualizes the Degrees Of Freedom (DOF) of different predictor combinations with boxplots where
        the box range, whiskers, ... indicate the spread over the information content subset of the 
        evaluation dataset.

        Parameters:
        -----------
        I_DS_dict : dictionary of xarray datasets
            Dictionary containing xarray datasets for each set of predictors used for information
            content estimation for the predictand. Each dataset has been created with 
            NN_retrieval.py exec_type='i_cont'. 
        idx_dict : dict
            Dictionary containing indices for different weather conditions for each key of the I_DS_dict
            (each set of predictors).
        set_dict : dict
            Dictionary containing additional information.
        """

        # create output path if not existing:
        plotpath_dir = os.path.dirname(set_dict['path_plots'] + f"{set_dict['p']}/")
        if not os.path.exists(plotpath_dir):
            os.makedirs(plotpath_dir)


        # variables for adaptations of plot properties:
        labels_boxplot = [*I_DS_dict.keys()]        # labels for boxplot
        labels_boxplot[0] = "All"
        y_label_dict = {'temp': "Temperature profile DOF",
                        'temp_bl': "Temperature profile DOF",
                        'q': "Specific humidity profile DOF"}
        legend_loc = 'upper left'
        legend_anchor = [1.01, 1.01]

        # number of boxes to show and identify where the boxes are supposed to be on the x axis:
        n_boxes = len(labels_boxplot)
        pos_boxes = [k+1 for k in range(n_boxes)]


        def make_boxplot_great_again(bp):   # change linewidth to 1.5
            plt.setp(bp['boxes'], color=(0,0,0), linewidth=1.5)
            plt.setp(bp['whiskers'], color=(0,0,0), linewidth=1.5)
            plt.setp(bp['caps'], color=(0,0,0), linewidth=1.5)
            plt.setp(bp['medians'], color=(0,0,0), linewidth=1.5)


        weather_colours = {'clear_sky': np.array([255,255,255])/255,
                            'warm_moist': np.array([255,196,68])/255,
                            'cold_dry': np.array([213,99,255])/255,
                            'warm': np.array([206,0,8])/255,
                            'cold': np.array([0,246,255])/255,
                            'moist': np.array([68,255,68])/255,
                            'dry': np.array([135,99,0])/255}

        weather_labels = {'clear_sky': "Clear sky",
                            'warm_moist': "Mild & moist",
                            'cold_dry': "Cool & dry",
                            'warm': "Warm",
                            'cold': "Cold",
                            'moist': "Very moist",
                            'dry': "Very dry"}
        alt_weather_labels = {'clear_sky': "IWP, SWP, LWP < 1$\,\mathrm{g}\,\mathrm{m}^{-2}$",
                                'warm_moist': "T$_{\mathrm{2 m}}$ >= 273$\,$K, IWV >= 10$\,\mathrm{kg}\,\mathrm{m}^{-2}$",
                                'cold_dry': "T$_{\mathrm{2 m}}$ < 273$\,$K, IWV < 10$\,\mathrm{kg}\,\mathrm{m}^{-2}$",
                                'warm': "T$_{\mathrm{2 m}}$ > 275$\,$K",
                                'cold': "T$_{\mathrm{2 m}}$ < 258.15$\,$K",
                                'moist': "IWV >= 15$\,\mathrm{kg}\,\mathrm{m}^{-2}$",
                                'dry': "IWV < 5$\,\mathrm{kg}\,\mathrm{m}^{-2}$"}

        for wt in weather_labels.keys():
            weather_labels[wt] = weather_labels[wt] + ":\n" + alt_weather_labels[wt]
        n_w_types = len(weather_labels.keys())      # number of weather types


        f1 = plt.figure(figsize=(12,6))
        a1 = plt.axes()

        if set_dict['p'] in ['temp', 'temp_bl']:
            y_lims = np.array([0, 5])           # DOF
        elif set_dict['p'] == 'q':
            y_lims = np.array([1, 4])           # DOF

        x_lims = np.array([pos_boxes[0]-1, pos_boxes[-1]+1])

        # BOXPLOT of data:
        bp_plots = dict()
        bp_plots_markers = dict()
        bwidth = 0.5
        for k, key in enumerate(I_DS_dict.keys()):
            bp_plots[str(k)] = a1.boxplot(I_DS_dict[key].DOF.values, positions=[pos_boxes[k]], sym='.', widths=bwidth)
            make_boxplot_great_again(bp_plots[str(k)])


            # add weather condition markers (median DOF):
            for i_w, wt in enumerate(idx_dict.keys()):
                wt_idx = idx_dict[wt][key]      # indices for current weather type and predictor set
                DOF_wt = I_DS_dict[key].DOF[wt_idx].median('n_s')

                x_pos_marker = bwidth*i_w / (n_w_types-1) + pos_boxes[k] - 0.5*bwidth
                if k == 0:
                    bp_plots_markers[wt], = a1.plot([x_pos_marker], [DOF_wt], color=tuple(weather_colours[wt]), 
                                marker='h', markersize=msize, linestyle='none', markeredgecolor=(0,0,0),
                                label=weather_labels[wt])
                else:
                    a1.plot([x_pos_marker], [DOF_wt], color=tuple(weather_colours[wt]), marker='h', 
                                markersize=msize, linestyle='none', markeredgecolor=(0,0,0))


            # add aux info (texts, annotations):
            a1.text(pos_boxes[k], 1.00*np.diff(y_lims)+y_lims[0], f"{np.median(I_DS_dict[key].DOF.values):.2f}",
                    color=(0,0,0), fontsize=fs_micro, ha='center', va='bottom', 
                    transform=a1.transData)


        a1.text(x_lims[0], 1.0*np.diff(y_lims)+y_lims[0], "Median:",
                    color=(0,0,0), fontsize=fs_micro, ha='left', va='bottom', 
                    transform=a1.transData)

        # legend/colorbar:
        # pdb.set_trace()
        # one legend for group 1: cold and dry conditions and one for group 2: warm and moist conditions
        cold_keys = ['clear_sky', 'cold', 'dry', 'cold_dry']
        warm_keys = ['warm', 'warm_moist', 'moist']
        handles1 = [value for key, value in bp_plots_markers.items() if key in cold_keys]
        leg1 = a1.legend(handles=handles1, loc=legend_loc, fontsize=fs_micro, bbox_to_anchor=legend_anchor, 
                            title='Cold conditions', title_fontsize=fs_dwarf, frameon=False)
        a1.add_artist(leg1)

        handles2 = [value for key, value in bp_plots_markers.items() if key in warm_keys]
        leg2 = a1.legend(handles=handles2, loc=legend_loc, fontsize=fs_micro, bbox_to_anchor=[legend_anchor[0], 0.45], 
                            title='Warm conditions', title_fontsize=fs_dwarf, frameon=False)

        # lh, ll = a1.get_legend_handles_labels()
        # a1.legend(lh, ll, loc=legend_loc, fontsize=fs_micro, bbox_to_anchor=legend_anchor, frameon=False)#, framealpha=0.5)


        # set axis limits:
        a1.set_ylim(y_lims[0], y_lims[1])
        a1.set_xlim(x_lims[0], x_lims[1])

        # set ticks and tick labels and parameters: for plotting, break string into several lines if too long
        a1.set_xticks(pos_boxes)
        labels_boxplot = [break_str_into_lines(lab, 9, split_at='+', keep_split_char=True) for lab in labels_boxplot]
        a1.set_xticklabels(labels_boxplot)
        a1.tick_params(axis='both', labelsize=fs_micro)

        # grid:
        a1.grid(which='both', axis='y', color=(0.5,0.5,0.5), alpha=0.5)
        a1.set_axisbelow(True)


        # set labels:
        a1.set_xlabel("Frequency bands", fontsize=fs_dwarf)
        a1.set_ylabel(y_label_dict[set_dict['p']], fontsize=fs_dwarf)

        # adjust axis width:
        a1_pos = a1.get_position().bounds
        a1.set_position([a1_pos[0], a1_pos[1], 0.7*a1_pos[2], a1_pos[3]])

        if set_dict['save_figures']:

            plotname = f"NN_syn_ret_info_content_DOF_boxplot_{set_dict['p']}"
            plotfile = plotpath_dir + "/" + plotname
            f1.savefig(plotfile + ".png", dpi=300, bbox_inches='tight')
            f1.savefig(plotfile + ".pdf", bbox_inches='tight')
            print(f"Saved {plotfile}.png.")             

        else:
            plt.show()
            pdb.set_trace()

        plt.close()


    def AK_resolution_plot(
        I_DS_dict,
        set_dict):

        """
        Visualizes the vertical resolution estimated from the diagonal values of the Averaging
        Kernel matrix and the vertical grid spacing. 

        Parameters:
        -----------
        I_DS_dict : dictionary of xarray datasets
            Dictionary containing xarray datasets for each set of predictors used for information
            content estimation for the predictand. Each dataset has been created with 
            NN_retrieval.py exec_type='i_cont'. 
        set_dict : dict
            Dictionary containing additional information.
        """

        # create output path if not existing:
        plotpath_dir = os.path.dirname(set_dict['path_plots'] + f"{set_dict['p']}/")
        if not os.path.exists(plotpath_dir):
            os.makedirs(plotpath_dir)

        # compute height grid spacing as 'layers': mean layer height diff: dz_i = (z_i+1 - z_i-1)*0.5
        # diff at top of grid can be larger because original height grid for info content was extended
        # ([..., 9000, 9500, 10000, 11000, 12000, ...]). Thus, the top must have 0.5*(11000-9500)
        # at sfc, grid spacing must be (z0 + z1)/2
        height = I_DS_dict['OP'].height.mean('n_s').values
        dz_0 = np.concatenate((np.array([0]), np.diff(height)))
        dz_1 = np.concatenate((np.diff(height), np.array([np.diff(height)[-1]])))
        dz = 0.5*(dz_0+dz_1)


        # variables for adaptations of plot properties:
        labels_boxplot = {'OP': "All frequencies", 'K': "K--band"}
        y_label_dict = {'temp': "Temperature profile DOF",
                        'temp_bl': "Temperature profile DOF",
                        'q': "Specific humidity profile DOF"}
        legend_loc = 'upper left'
        zorders = {'OP': 35., 'K': 31, 'K+V': 25, 'K+G': 27, 'K+V+G': 29}
        linewidths = {'OP': 2.5, 'K': 2.5, 'K+V': 1.5, 'K+G': 1.5, 'K+V+G': 1.5}


        f1 = plt.figure(figsize=(5,6))
        a1 = plt.axes()

        y_lims = np.array([0, 10000.])          # height grid lims in m
        x_lims = np.array([0., 10500.])         # estimated height resolution limits in m

        # plot height resolutions:
        for k, key in enumerate(I_DS_dict.keys()):
            AK_mean = I_DS_dict[key].AK_diag.mean('n_s')

            if key in ['OP', 'K']:  # compute AK mean +/- std for resolution estimation
                # AK_p = AK_mean + I_DS_dict[key].AK_diag.std('n_s')
                # AK_m = AK_mean - I_DS_dict[key].AK_diag.std('n_s')
                # a1.fill_betweenx(height, dz/AK_m, dz/AK_p, color=col_fade_dict[key],
                                # label=f"{labels_boxplot[k]} std. range", zorder=zorders[key])

                a1.plot(dz / AK_mean, height, color=col_dict[key], 
                        linewidth=linewidths[key], label=labels_boxplot[key], zorder=zorders[key]+1)


        # legend/colorbar:
        lh, ll = a1.get_legend_handles_labels()
        leg1 = a1.legend(lh, ll, loc=legend_loc, fontsize=fs_micro, framealpha=0.5)
        a1.add_artist(leg1)

        # set axis limits:
        a1.set_ylim(y_lims)
        a1.set_xlim(x_lims)

        # set ticks and tick labels and parameters:
        a1.minorticks_on()
        a1.tick_params(axis='both', labelsize=fs_micro)

        # grid:
        a1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)


        # set labels:
        a1.set_xlabel("Effective resolution (m)", fontsize=fs_dwarf)
        a1.set_ylabel("Height (m)", fontsize=fs_dwarf)

        # adjust axis width:
        f1.tight_layout()

        if set_dict['save_figures']:

            plotname = f"NN_syn_ret_info_content_AK_resolution_{set_dict['p']}"
            plotfile = plotpath_dir + "/" + plotname
            f1.savefig(plotfile + ".png", dpi=300, bbox_inches='tight')
            f1.savefig(plotfile + ".pdf", bbox_inches='tight')
            print(f"Saved {plotfile}.png.")             

        else:
            plt.show()
            pdb.set_trace()

        plt.close()


    def Jacobian_matrix_plot(
        I_DS_dict, 
        set_dict):

        """
        Visualizes the Jacobian matrix (dy/dx with y: observations (i.e., TBs) and x: state vector
        (i.e., q profile)) averaged over the samples (i.e., evaluation dataset).

        Parameters:
        -----------
        I_DS_dict : dictionary of xarray datasets
            Dictionary containing xarray datasets for each set of predictors used for information
            content estimation for the predictand. Each dataset has been created with 
            NN_retrieval.py exec_type='i_cont'. 
        set_dict : dict
            Dictionary containing additional information.
        """

        # create output path if not existing:
        plotpath_dir = os.path.dirname(set_dict['path_plots'] + f"{set_dict['p']}/")
        if not os.path.exists(plotpath_dir):
            os.makedirs(plotpath_dir)

        # frequency labels for x axis:
        freq_labels = I_DS_dict['OP'].freqs.values.astype("str")

        # colormap levels and colour selection:
        pdb.set_trace()
        K_levels = np.arange(0, 5., 0.01)
        K_cmap = mpl.cm.get_cmap('nipy_spectral', len(K_levels))


        f1 = plt.figure(figsize=(10,8))
        a1 = plt.axes()

        y_lims = np.array([I_DS_dict['OP'].height.min().values, I_DS_dict['OP'].height.max().values])   # height in m
        x_lims = np.array([I_DS_dict['OP'].n_cy[0].values, I_DS_dict['OP'].n_cy[-1].values])

        # plotting data:
        yy, xx = np.meshgrid(I_DS_dict['OP'].height.mean('n_s').values, I_DS_dict['OP'].n_cy.values)
        K_plot = a1.contourf(xx, yy, I_DS_dict['OP'].K.mean('n_s'), levels=K_levels, cmap=K_cmap, extend='both')


        # legend/colorbar:
        shrinkval = 0.9
        cb = f1.colorbar(mappable=K_plot, ax=a1, orientation='vertical', extend='both', fraction=0.09, pad=0.01)#, shrink=shrinkval)
        cb.set_label(label=f"Jacobian ({I_DS_dict['OP'].K.units})", fontsize=fs_micro)
        cb.ax.tick_params(labelsize=fs_micro)


        # set axis limits:
        a1.set_ylim(y_lims)
        a1.set_xlim(x_lims)

        # set ticks and tick labels and parameters: for plotting, break string into several lines if too long
        a1.set_xticks(I_DS_dict['OP'].n_cy.values[::2])
        a1.set_xticklabels(freq_labels[::2])
        a1.tick_params(axis='y', labelsize=fs_micro)
        a1.tick_params(axis='x', labelsize=fs_micro-2)


        # set labels:
        a1.set_xlabel("Frequency (GHz)", fontsize=fs_dwarf)
        a1.set_ylabel("Height (m)", fontsize=fs_dwarf)

        if set_dict['save_figures']:

            plotname = f"NN_syn_ret_info_content_Jacobian_matrix_{set_dict['p']}"
            plotfile = plotpath_dir + "/" + plotname + ".png"
            f1.savefig(plotfile, dpi=300, bbox_inches='tight')
            print(f"Saved {plotfile}")              

        else:
            plt.show()
            pdb.set_trace()

        plt.close()


    # settings:
    set_dict = {'save_figures': True,
                'path_plots': path_plots,
                'tc_dict': {'q': ["472", "481", "482", "483", "484"],       # dictionary showing the 'test_purpose.yaml' test cases
                            'temp': ["417", "478", "479", "480"],           # for each predictand
                            'temp_bl': ["424"]},
                'id_dict': {'472': 'OP', '417': 'OP', '424': 'OP'},         # dictionary to identify the type of the test case:
                                                                            # OP: operational, K: K band only, K+V: K and V band, ...
                'plot_jacobian': False,     # if True, Jacobian matrix of OP is visualized
                'plot_cumul_DOF': False,    # if True, the cumulative sum of the diagonal entries of the AK are visualized
                'plot_AK_resolution': True, # if True, the vertical resolution is estimated from the AK and the height 
                                            # grid and visualized
                }


    # import info content data for each predictand; set predictand:
    set_dict['p'] = 'q'
    print(f"Information content visualization for {set_dict['p']}....")

    # In the evaluation data file names, temp and temp_bl is only distinguished via test case 
    # number and not named 'temp_bl'. Therefore, set_dict['p_file'] is useful to find the correct
    # file.
    if set_dict['p'] != 'temp_bl':
        set_dict['p_file'] = set_dict['p']
    else:
        set_dict['p_file'] = 'temp'


    # identify files:
    files = list()
    I_DS_dict = dict()
    for tc in set_dict['tc_dict'][set_dict['p']]:
        file = sorted(glob.glob(path_data['i_cont'] + f"*_{tc}.nc"))
        if len(file) > 0: 
            file = file[0]
        else:
            continue
        files.append(file)
        I_DS_dict[tc] = xr.open_dataset(file)

        # improve identifier of I_DS_dict by using the predictors
        if tc not in ['417', '424', '472']:
            set_dict['id_dict'][tc] = I_DS_dict[tc].predictor_TBs

        I_DS_dict[set_dict['id_dict'][tc]] = I_DS_dict[tc]      # rename dictionary items
        del I_DS_dict[tc]

    n_ic = len(files)       # number of information content estimates for current predictand


    # Load evaluation data:
    files_eval = sorted(glob.glob(path_data['eval'] + f"NN_syn_ret_eval_data_prediction_{set_dict['p_file']}_*.nc"))
    if len(files_eval) == 1:
        files_eval = files_eval[0]
    elif len(files_eval) > 1:   # for temp and temp_bl: identify correct file
        tc_op = {'temp': "417", 'temp_bl': "424"}
        pdb.set_trace() # test if it works as intended
        files_eval = [f_eval for f_eval in files_eval if tc_op[set_dict['p'] in f_eval]]
    else:
        pdb.set_trace()     # no files found

    # load dataset and select information content subset:
    E_DS = xr.open_dataset(files_eval)
    E_DS_sel = E_DS.isel(n_s=I_DS_dict['OP'].idx_eval.values)


    # reduce dataset to information content subset and find indices for certain weather conditions:
    idx_dict = dict()
    idx_dict['clear_sky'] = dict()
    idx_dict['cold'] = dict()
    idx_dict['dry'] = dict()
    idx_dict['cold_dry'] = dict()
    idx_dict['warm'] = dict()
    idx_dict['warm_moist'] = dict()
    idx_dict['moist'] = dict()
    for key in I_DS_dict.keys():

        # weather conditions:
        idx_dict['clear_sky'][key] = np.where((E_DS_sel.lwp < 0.001) & (E_DS_sel.iwp < 0.001) & 
                                        (E_DS_sel.swp < 0.001))[0]
        idx_dict['warm_moist'][key] = np.where((E_DS_sel.temp[:,0] >= 273.0) & (E_DS_sel.iwv >= 10.0))[0]
        idx_dict['cold_dry'][key] = np.where((E_DS_sel.temp[:,0] < 273.0) & (E_DS_sel.iwv < 10.0))[0]
        idx_dict['warm'][key] = np.where(E_DS_sel.temp[:,0] > 275.0)[0]
        idx_dict['cold'][key] = np.where(E_DS_sel.temp[:,0] < 258.15)[0]
        idx_dict['moist'][key] = np.where(E_DS_sel.iwv >= 15.0)[0]
        idx_dict['dry'][key] = np.where(E_DS_sel.iwv < 5.0)[0]

    del E_DS


    # visualize information content (cumulative AK diagonal):
    if set_dict['plot_cumul_DOF']:
        cumulative_DOF(I_DS_dict, set_dict)

    # boxplot of DOF including different weather conditions:
    DOF_boxplot(I_DS_dict, idx_dict, set_dict)

    # if desired: plot the estimated vertical resolution of the retrieved profiles:
    if set_dict['plot_AK_resolution']:
        AK_resolution_plot(I_DS_dict, set_dict)

    # if desired: Jacobian Matrix plot:
    if set_dict['plot_jacobian']:
        Jacobian_matrix_plot(I_DS_dict, set_dict)