# Synergetic retrieval combining low- and high-frequency microwave radiometer observations for enhanced water vapour products

Create a directory (e.g., `/home/user/synergetic_ret/`) where you clone or copy the codes to `/home/user/synergetic_ret/codes/`. Data (see below) must then be downloaded and moved to respective subdirectories in `/home/user/synergetic_ret/data/`. Plots will be saved to `/home/user/synergetic_ret/plots/`.

To recreate figures shown in (INSERT PUBLICATION), execute [create_plots.py](./codes/create_plots.py) as follows:

- `python3 create_plots.py "1"` to recreate Figure 1

- `python3 create_plots.py "B1"` to recreate Figure B1 (Appendix)

The string following the script name indicates the figure. Note that for the appendix figures, the [NN_retrieval.py](./codes/source/NN_retrieval.py)
script is executed which was used with different python packages (see below).

To recreate the retrievals, execute [synergetic_ret.py](./codes/synergetic_ret.py) as follows:

- `python3 synergetic_retrieval.py "prw"` for integrated water vapour retrieval

- `python3 synergetic_retrieval.py "q"` for specific humidity profile retrieval

The string following the script name indicates which retrieval to perform.


## REQUIREMENTS
### Data:
-----
1) Retrieved IWV, and temperature, specific humidity and relative humidity profiles from MOSAiC observations:
	Download data from PANGAEA (...) and (...). Move files to `.../data/retrieval_output/mosaic/`.				# # # # # # # # # # # # DOI PENDING # # # # # # # # # # # # # 

2) Retrieved IWV, and temperature and specific humidity profiles from synthetic ERA5 observations:
	Download data from https://doi.org/10.5281/zenodo.10998146 . Move files to 
	`.../data/retrieval_output/era5/`.

3) Information content estimation:
	Download data from https://doi.org/10.5281/zenodo.10997692 and move files to 
	`.../data/info_content/`.

4) MOSAiC level 2 radiosondes:
	Download data from https://doi.org/10.1594/PANGAEA.928656 . Move files to 
	`.../data/MOSAiC_radiosondes/`. Execute script `.../codes/source/PANGAEA_tab_to_nc.py` 
	with `python3 PANGAEA_tab_to_nc.py "level_2"` to convert .tab files to .nc.

5) MOSAiC level 3 radiosondes:
	Only needed if the relative humidity product of the synergetic retrieval should be recreated.
	Download data from https://doi.org/10.1594/PANGAEA.961881 and move files to
	`.../data/MOSAiC_radiosondes_level_3/raw/`. Execute `.../codes/source/PANGAEA_tab_to_nc.py` with
	`python3 PANGAEA_tab_to_nc.py "level_3"` .

6) Simulated MOSAiC radiosondes: 
	Download from https://doi.org/10.5281/zenodo.11092210, move files to 
	`.../data/MOSAiC_PAMTRA_highspec/`.

7) HATPRO level 2 data:
	Download data from https://doi.org/10.1594/PANGAEA.941389 and move to `.../data/hatpro_l2/`.

8) MiRAC-P IWV data:
	Download data from https://doi.org/10.1594/PANGAEA.941470 and move to `.../data/mirac-p_l2/`.

9) MOSAiC Met City 2 m temperature data:
	Download data from https://doi.org/10.18739/A2PV6B83F and move to `.../data/MOSAiC_metcity/`.

10) MOSAiC cloudnet target classification:
	Download target classification data from https://doi.org/10.60656/60ea0dd0a99746ba and move
	to `.../data/MOSAiC_cloudnet/`.

11) MOSAiC cloudnet low-level stratus mask:
	Download low-level stratus mask data from https://doi.org/10.1594/PANGAEA.961789 and move
	to `.../data/MOSAiC_cloudnet/LLS_mask/`.

12) MOSAiC cloudnte issue data set:
	Download issue data set from https://doi.org/10.5281/zenodo.7310858 and move to 
	`.../data/MOSAiC_cloudnet/issue_flag/`.

13) ERA5 sea ice data: 
	Run `.../codes/source/get_era5_sic.py`, move files to `.../data/ERA5_data/`.

14) ERA5 data on model levels:
	Run `.../codes/source/get_era5_model_level.py`, move files to `.../data/ERA5_data/`.
	Run `.../codes/source/ERA5_process_model_levels.py`.

15) MOASiC Polarstern track data: 
	Download from https://doi.org/10.1594/PANGAEA.924668 , https://doi.org/10.1594/PANGAEA.924674 ,
	https://doi.org/10.1594/PANGAEA.924681 , https://doi.org/10.1594/PANGAEA.926829 , 
	https://doi.org/10.1594/PANGAEA.926910 , move files to `.../data/polarstern_track/`.

16) ERA5 training, validation and evaluation data:
	Download from https://doi.org/10.5281/zenodo.10997365 , move files to `.../data/training_data/`.

17) cartopy background data: Following https://docs.dkrz.de/doc/visualization/sw/python/source_code/python-matplotlib-example-high-resolution-background-image-plot.html
	Download high resolution version from 
	https://www.naturalearthdata.com/downloads/10m-natural-earth-1/10m-natural-earth-1-with-shaded-relief-water-and-drainages/ ,
	unpack NE1_HR_LC_SR_W_DR.tif and convert to png. Save
```
{"__comment__": "JSON file specifying background images. env CARTOPY_USER_BACKGROUNDS, ax.background_img()",
  "NaturalEarthRelief": {
    "__comment__": "Natural Earth I with shaded Relief, water, and drainage",
    "__source__": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/raster/NE1_HR_LC_SR_W_DR.zip",
    "__projection__": "PlateCarree",
    "high": "NE1_HR_LC_SR_W_DR.png"
  }
}
```
  to `images.json`. Move NE1_HR_LC_SR_W_DR.png and `images.json` to `.../data/cartopy_background/`. 

18) HATPRO TB data:
	If an Appendix figure should be recreated, also download data from 
	https://doi.org/10.1594/PANGAEA.941356 and move to `.../data/hatpro_l1/`.

19) MiRAC-P TB data:
	If an Appendix figure should be recreated, also download data from 
	https://doi.org/10.1594/PANGAEA.941407 and move to `.../data/mirac-p_l1/`.

--------------


### Python packages:
---
For all non-appendix figures created with `create_plots.py`:
All non-appendix figures do not require tensorflow because the retrieval does not have to be 
executed. For these figures, a separate conda environment with the following "main" packages 
were used:
- python version: 3.9.16 (main, Jan 11 2023, 16:05:54) [GCC 11.2.0]
- numpy: 1.22.3
- netCDF4: 1.6.2
- matplotlib: 3.5.2
- xarray: 2023.1.0
- pandas: 1.4.4
- yaml: 6.0
- cartopy: 0.18.0
- cmcrameri: 1.8



For the execution of the retrievals (`synergetic_retrieval.py`) and to create the appendix figures
with `create_plots.py`, the following packages were used:
- python version: 3.10.6 (main, Nov 14 2022, 16:10:14) [GCC 11.3.0], 
- tensorflow: 2.10.0, keras: 2.10.0,
- numpy: 1.21.5,
- sklearn: 1.2.1,
- netCDF4: 1.5.8,
- matplotlib: 3.6.3,
- xarray: 2023.1.0,
- pandas: 1.5.3
- pyPamtra: 1.0.0 or higher
- yaml: 5.4.1

**Further requirements** if `synergetic_retrieval.py` is executed. Install the Passive and Active 
Microwave Radiative Transfer Model ([PAMTRA](https://doi.org/10.5194/gmd-13-4229-2020)), following
the [installation instructions](https://pamtra.readthedocs.io/en/latest/installation.html).
