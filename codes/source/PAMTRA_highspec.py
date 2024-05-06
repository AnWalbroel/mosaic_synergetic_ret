def run_PAMTRA_highspec(path_sim_radiosondes, path_plots):

    """
    This program is used to visualize forward simulated MOSAiC radiosondes for two
    test cases.
    Winter case: 2020-03-05 04:54:32 UTC
    Summer case: 2020-08-05 22:54:39 UTC

    Parameters:
    -----------
    path_sim_radiosondes : str
        Full path of the simulated radiosonde data. Must contain the files
        pam_out_RS_level_2_summer_20200805_225439Z.nc and 
        pam_out_RS_level_2_winter_20200305_045432Z.nc.
    path_plots : str
        Full path where the plot should be saved to.
    """


    import numpy as np
    import datetime as dt
    import glob
    import xarray as xr

    import pdb
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


    # Paths:
    path_output = path_sim_radiosondes      # path of PAMTRA output directory
    path_plot = path_plots


    # Import data:
    file_winter = glob.glob(path_output + "*_winter_*.nc")
    file_summer = glob.glob(path_output + "*_summer_*.nc")
    PAM_DS_winter = xr.open_dataset(file_winter[0])
    PAM_DS_summer = xr.open_dataset(file_summer[0])


    mwr_freqs = np.array([  22.240, 23.040, 23.840, 25.440, 26.240, 27.840, 31.400,
                                    51.260, 52.280, 53.860, 54.940, 56.660, 57.300, 58.000,
                                    175.810, 178.310, 179.810, 180.810, 181.810, 182.710,
                                    183.910, 184.810, 185.810, 186.810, 188.310, 190.810,
                                    243.000, 340.000])
    
    fs = 18

    fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(16,6))
    ylims = np.array([0, 295])


    x_plot = PAM_DS_winter.frequency.values
    y_plot = PAM_DS_winter.tb[0,0,0,-1,:,:].mean(axis=-1).values
    plot_winter = ax0.plot(x_plot, y_plot, color=(0,0,0), linewidth=1.25, linestyle='dashed',
                                label="Winter, IWV = 0.9$\,\mathrm{kg}\,\mathrm{m}^{-2}$")

    x_plot = PAM_DS_summer.frequency.values
    y_plot = PAM_DS_summer.tb[0,0,0,-1,:,:].mean(axis=-1).values
    plot_summer = ax0.plot(x_plot, y_plot, color=(0,0,0), linewidth=1.25,
                                label="Summer, IWV = 16.1$\,\mathrm{kg}\,\mathrm{m}^{-2}$")

    # fill between:
    ax0.fill_between(x_plot, y1=y_plot, y2=0.0, facecolor=(1.0,0.855,0.5,0.2))

    # add markers for HATPRO and MiRAC-P frequencies:
    for frq in mwr_freqs:
        if frq < 170.0: # then HATPRO
            ax0.plot([frq, frq], ylims, color=(17.0/255.0,74.0/255.0,196.0/255.0), linewidth=0.75, zorder=-2)

        else:
            ax0.plot([frq, frq], ylims, color=(0.0,199.0/255.0,157.0/255.0), linewidth=0.75, zorder=-2)

    # dummy for legend:
    ax0.plot([np.nan, np.nan], [np.nan, np.nan], color=(17.0/255.0,74.0/255.0,196.0/255.0), linewidth=1.5, label="HATPRO frequencies")
    ax0.plot([np.nan, np.nan], [np.nan, np.nan], color=(0.0,199.0/255.0,157.0/255.0), linewidth=1.5, label="MiRAC-P frequencies")

    # add band identifiers (text labels):
    band_labels = ["K", "V", "G", "243", "340"]
    band_bounds = {'K': [20, 35], 'V': [50, 60], 'G': [170, 200], "243": [230, 250], '340': [335, 345]}
    for band in band_labels:
        frq_band = mwr_freqs[(mwr_freqs >= band_bounds[band][0]) & (mwr_freqs <= band_bounds[band][1])]
        avg_freq_band = np.mean(frq_band)
        ax0.text(avg_freq_band, 1.00*np.diff(ylims)+ylims[0], 
                    f"{band}", 
                    ha='center', va='bottom', color=(0,0,0), fontsize=fs+2, fontweight='bold', transform=ax0.transData)

    # and another one indicating the meaning of K, V, ...:
    ax0.text(0.00, 1.00, "Freq. bands", 
            ha='right', va='bottom', color=(0,0,0), fontsize=fs+2, transform=ax0.transAxes)

    ax0.set_xlim(left=0.0, right=x_plot[-1])
    ax0.set_ylim(bottom=ylims[0], top=ylims[1])

    han, solo = ax0.get_legend_handles_labels()
    le_leg = ax0.legend(handles=han, labels=solo, loc='upper left', bbox_to_anchor=(0.47,0.59), fontsize=fs-2, framealpha=1.0)

    # ax0.set_title("Simulated TBs - Microwave spectrum", fontsize=fs, pad=24)
    ax0.set_xlabel("Frequency (GHz)", fontsize=fs, labelpad=0.75)
    ax0.set_ylabel("TB (K)", fontsize=fs, labelpad=0.75)

    ax0.minorticks_on()
    # ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
    ax0.tick_params(axis='both', labelsize=fs-2)


    # fig0.savefig(path_plot + "MOSAiC_radiosonde_PAMTRA_1-500GHz_highres.png", dpi=400, bbox_inches='tight')
    fig0.savefig(path_plot + "MOSAiC_radiosonde_PAMTRA_1-500GHz_highres.pdf", bbox_inches='tight')
    print(f"Saved {path_plot + 'MOSAiC_radiosonde_PAMTRA_1-500GHz_highres.pdf'}")
