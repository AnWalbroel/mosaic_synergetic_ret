import os
import sys
import pdb

wdir = os.getcwd() + "/"


"""
	Script to manage and create plots for the synergistic retrieval study. Execute this script
	in the following way:
	To create Figure 1: python3 create_plots.py "1"
	- Figure 2: python3 create_plots.py "2"
	- Figure B1 in Appendix B: python3 create_plots.py "B1"
"""

# Paths:
path_scripts = os.path.dirname(wdir) + "/source/"
path_tools = os.path.dirname(wdir) + "/tools/"
path_data_base = os.path.dirname(wdir[:-1]) + "/data/"
path_plots_base = os.path.dirname(wdir[:-1]) + "/plots/"

sys.path.insert(0, path_scripts)
sys.path.insert(0, path_tools)


# settings:
which_figure = sys.argv[1] 	# set the number of the figure of the manuscript as string,
							# e.g., "1" for Figure 1, "B1" for Figure B1 in Appendix B



# Fig. 1: Simulated TBs:
if which_figure == "1":
	from PAMTRA_highspec import run_PAMTRA_highspec
	path_radiosondes = path_data_base + "MOSAiC_PAMTRA_highspec/"	# forward simulated radiosondes
	path_plots = path_plots_base + "MOSAiC_PAMTRA_highspec/"
	os.makedirs(os.path.dirname(path_plots), exist_ok=True)	# create path if not existing

	run_PAMTRA_highspec(path_radiosondes, path_plots)


# Fig. 2: Training data map:
if which_figure == "2":
	from training_data_map import run_training_data_map
	path_data =  {'era5': path_data_base + "training_data/",	# retrieval training data
				'era5_sic': path_data_base + "ERA5_data/",		# ERA5 sea ice concentr. data
				'ps_track': path_data_base + "polarstern_track/",	# Polarstern track data
				'cartopy_background': path_data_base + "cartopy_background/"}	# background data for plot
	path_plots =  path_plots_base + "training_data_map/"
	os.makedirs(os.path.dirname(path_plots), exist_ok=True)

	run_training_data_map(path_data, path_plots)


# Fig. 3-5: Evaluation IWV, q, temp:
if which_figure in ["3", "4", "5"]:
	from retrieval_evaluation import run_retrieval_evaluation
	# # # # # # # # # # # # # path_data = {'nn_syn_mosaic': path_data_base + "retrieval_output/mosaic/",	# synergy of hatpro and mirac-p
				# # # # # # # # # # # # # 'nn_syn_eval_data': path_data_base + "retrieval_output/era5/",	# ERA5 evaluation data
				# # # # # # # # # # # # # 'radiosondes': path_data_base + "MOSAiC_radiosondes/"}			# MOSAiC level 2 radiosondes
	path_data = {'nn_syn_mosaic': "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/tests_01/output/l2/",	# synergy of hatpro and mirac-p
				'nn_syn_eval_data': "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/tests_01/output/eval_data/",	# ERA5 evaluation data
				'radiosondes': "/mnt/f/heavy_data/MOSAiC_radiosondes/"}			# MOSAiC level 2 radiosondes
	path_plots = path_plots_base + "retrieval_evaluation/"
	path_output = path_data_base + "retrieval_evaluation/"
	os.makedirs(os.path.dirname(path_plots), exist_ok=True)
	os.makedirs(os.path.dirname(path_output), exist_ok=True)

	var_dict = {'3': 'iwv', '4': 'q', '5': 'temp'}
	run_retrieval_evaluation(path_data, path_plots, path_output, var_dict[which_figure])


# Fig. 6, 7, 9: Retrieval benefit IWV, q, relative humidity:
if which_figure in ["6", "7", "9"]:
	from benefit_plots import run_benefit_plots
	# # # # # # # # # # # # # # # # # # # # # # # # path_data = {'nn_syn_mosaic': path_data_base + "retrieval_output/mosaic/",	# synergy of hatpro and mirac-p
				# # # # # # # # # # # # # # # # # # # # # # # # 'mwr_pro': path_data_base + "hatpro_l2/",		# hatpro retrievals
				# # # # # # # # # # # # # # # # # # # # # # # # 'nn_mir': path_data_base + "mirac-p_l2/",		# mirac-p IWV retrieval
				# # # # # # # # # # # # # # # # # # # # # # # # 'radiosondes': path_data_base + "MOSAiC_radiosondes/",	# MOSAiC level 2 radiosondes
				# # # # # # # # # # # # # # # # # # # # # # # # 'metcity': path_data_base + "MOSAiC_metcity/",	# MOSAiC met city data
				# # # # # # # # # # # # # # # # # # # # # # # # 'cloudnet': path_data_base + "MOSAiC_cloudnet/",# MOSAiC cloudnet target classification
				# # # # # # # # # # # # # # # # # # # # # # # # 'lls_mask': path_data_base + "MOSAiC_cloudnet/LLS_mask/",	# MOSAiC cloudnet low-level stratus mask
				# # # # # # # # # # # # # # # # # # # # # # # # 'issue_flag': path_data_base + "MOSAiC_cloudnet/issue_flag/"}	# MOSAiC cloudnet issue data set
	path_data = {'nn_syn_mosaic': "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/tests_01/output/l2/",
				'mwr_pro': "/mnt/f/heavy_data/MOSAiC_radiometers/HATPRO_l2_v01/",
				'nn_mir': "/mnt/f/heavy_data/MOSAiC_radiometers/MiRAC-P_l2_v01/",
				'radiosondes': "/mnt/f/heavy_data/MOSAiC_radiosondes/",
				'metcity': "/mnt/f/heavy_data/MOSAiC_metcity/",
				'cloudnet': "/mnt/f/heavy_data/MOSAiC_cloudnet/",
				'lls_mask': "/mnt/f/heavy_data/MOSAiC_cloudnet/LLS_mask/",
				'issue_flag': "/mnt/f/heavy_data/MOSAiC_cloudnet/issue_flag/"}
	path_plots = path_plots_base + "retrieval_benefit/"
	os.makedirs(os.path.dirname(path_plots), exist_ok=True)

	var_dict = {'6': 'iwv', '7': 'q', '9': 'rh'}
	run_benefit_plots(path_data, path_plots, var_dict[which_figure])


# Fig. 8: DOF boxplot:
if which_figure == "8":
	from DOF_plots import run_DOF_plots
	path_data = {'eval': path_data_base + "retrieval_output/era5/",	# ERA5 evaluation data
				'i_cont': path_data_base + "info_content/"}			# information content output
	path_plots = path_plots_base + "info_content/"
	os.makedirs(os.path.dirname(path_plots), exist_ok=True)

	run_DOF_plots(path_data, path_plots)


# Fig. 10: Specific humidity inversion case study:
if which_figure == "10":
	from q_inv_case_study import run_q_inv_case_study
	# # # # # # # # # # # # # # # # # # # # path_data = {'mwr_syn': path_data_base + "retrieval_output/mosaic/",	# synergy of hatpro and mirac-p
				# # # # # # # # # # # # # # # # # # # # 'mwr_hat': path_data_base + "hatpro_l2/",		# hatpro humidity profiles only
				# # # # # # # # # # # # # # # # # # # # 'rs': path_data_base + "MOSAiC_radiosondes/",	# MOSAiC level 2 radiosondes
				# # # # # # # # # # # # # # # # # # # # 'era5_m': path_data_base + "ERA5_data/"}		# ERA5 on model levels
	path_data = {'mwr_syn': "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/tests_01/output/l2/",	# synergy of hatpro and mirac-p
				'mwr_hat': "/mnt/f/heavy_data/MOSAiC_radiometers/HATPRO_l2_v01/",		# hatpro humidity profiles only
				'rs': "/mnt/f/heavy_data/MOSAiC_radiosondes/",			# MOSAiC level 2 radiosondes
				'era5_m': path_data_base + "ERA5_data/"}	# ERA5 on model levels
	path_plots = path_plots_base + "q_inv_case_study/"
	os.makedirs(os.path.dirname(path_plots), exist_ok=True)

	run_q_inv_case_study(path_data, path_plots)


# Fig. B1: IWV comparison:
if which_figure == "B1":

	# first, run the retrieval to get the needed data: NOTE: EXECUTING THE RETRIEVAL REPLACES
	# THE EXISTING _prw_ files IN path_data_base + "retrieval_output/mosaic/"!
	from NN_retrieval import run_NN_retrieval
	from benefit_plots import run_benefit_plots

	# paths:
	# # # # # # # # # # # # # # # # path_data = {'path_output': path_data_base + "retrieval_output/mosaic/",	# path where output is saved to
				# # # # # # # # # # # # # # # # 'path_output_info': path_data_base + "info_content/",
				# # # # # # # # # # # # # # # # 'path_output_pred_ref': path_data_base + "retrieval_output/validation_prediction_and_reference/",
				# # # # # # # # # # # # # # # # 'path_data': path_data_base + "training_data/",			# path of training/test data
				# # # # # # # # # # # # # # # # 'path_data_eval': path_data_base + "training_data/",
				# # # # # # # # # # # # # # # # 'path_ps_track': path_data_base + "polarstern_track/",		# path of Polarstern track data
				# # # # # # # # # # # # # # # # 'path_tb_obs': {'hatpro': path_data_base + "hatpro_l1/",
								# # # # # # # # # # # # # # # # 'mirac-p': path_data_base + "mirac-p_l1/"},	# path of published level 1 tb data
				# # # # # # # # # # # # # # # # 'path_tb_offsets': "",								# path of the MOSAiC MWR TB offset correction
				# # # # # # # # # # # # # # # # 'path_old_ret': path_data_base + "hatpro_l2/",					# path of old HATPRO retrievals
				# # # # # # # # # # # # # # # # 'path_rs_obs': path_data_base + "MOSAiC_radiosondes/",
				# # # # # # # # # # # # # # # # 'descriptor_file': path_scripts + "descriptor_file_ecmwf.txt",	# for PAMTRA simulations
				# # # # # # # # # # # # # # # # 'test_purpose': path_scripts}
	# # # # # # # # # # # # # # # # path_plots = {'path_plots': path_plots_base + "synergetic_ret/",
				# # # # # # # # # # # # # # # # 'path_plots_info': path_plots_base + "synergetic_ret/info_content/"}


	path_data = {'path_output': path_data_base + "retrieval_output/mosaic/",	# path where output is saved to
				'path_output_info': path_data_base + "info_content/",
				'path_output_pred_ref': path_data_base + "retrieval_output/validation_prediction_and_reference/",
				'path_data': "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/training_data_01/merged/new_z_grid_plus/",			# path of training/test data
				'path_data_eval': "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/training_data_01/merged/new_z_grid_plus/",
				'path_ps_track': "/mnt/f/heavy_data/polarstern_track/",		# path of Polarstern track data
				'path_tb_obs': {'hatpro': path_data_base + "hatpro_l1/",
								'mirac-p': path_data_base + "mirac-p_l1/"},	# path of published level 1 tb data
				'path_tb_offsets': "",								# path of the MOSAiC MWR TB offset correction
				'path_old_ret': "/mnt/f/heavy_data/MOSAiC_radiometers/HATPRO_l2_v01/",					# path of old HATPRO retrievals
				'path_rs_obs': "/mnt/f/heavy_data/MOSAiC_radiosondes/",
				'descriptor_file': path_scripts + "descriptor_file_ecmwf.txt",	# for PAMTRA simulations
				'test_purpose': path_scripts}
	path_plots = {'path_plots': path_plots_base + "synergetic_ret/",
				'path_plots_info': path_plots_base + "synergetic_ret/info_content/"}
	for key, value in path_plots.items():
		os.makedirs(os.path.dirname(value), exist_ok=True)

	# execute retrieval; then create plot.
	run_NN_retrieval(path_data, path_plots, test_id="486", exec_type='op_ret', eval_mode=False)

	# # # # # # # # # # # # # # # # # # # # # # # # path_data = {'nn_syn_mosaic': path_data_base + "retrieval_output/mosaic/",	# synergy of hatpro and mirac-p
				# # # # # # # # # # # # # # # # # # # # # # # # 'mwr_pro': path_data_base + "hatpro_l2/",		# hatpro retrievals
				# # # # # # # # # # # # # # # # # # # # # # # # 'nn_mir': path_data_base + "mirac-p_l2/",		# mirac-p IWV retrieval
				# # # # # # # # # # # # # # # # # # # # # # # # 'radiosondes': path_data_base + "MOSAiC_radiosondes/",	# MOSAiC level 2 radiosondes
				# # # # # # # # # # # # # # # # # # # # # # # # 'metcity': path_data_base + "MOSAiC_metcity/",	# MOSAiC met city data
				# # # # # # # # # # # # # # # # # # # # # # # # 'cloudnet': path_data_base + "MOSAiC_cloudnet/",# MOSAiC cloudnet target classification
				# # # # # # # # # # # # # # # # # # # # # # # # 'lls_mask': path_data_base + "MOSAiC_cloudnet/LLS_mask/",	# MOSAiC cloudnet low-level stratus mask
				# # # # # # # # # # # # # # # # # # # # # # # # 'issue_flag': path_data_base + "MOSAiC_cloudnet/issue_flag/"}	# MOSAiC cloudnet issue data set
	path_data = {'nn_syn_mosaic': path_data_base + "retrieval_output/mosaic/",
				'mwr_pro': "/mnt/f/heavy_data/MOSAiC_radiometers/HATPRO_l2_v01/",
				'nn_mir': "/mnt/f/heavy_data/MOSAiC_radiometers/MiRAC-P_l2_v01/",
				'radiosondes': "/mnt/f/heavy_data/MOSAiC_radiosondes/",
				'metcity': "/mnt/f/heavy_data/MOSAiC_metcity/",
				'cloudnet': "/mnt/f/heavy_data/MOSAiC_cloudnet/",
				'lls_mask': "/mnt/f/heavy_data/MOSAiC_cloudnet/LLS_mask/",
				'issue_flag': "/mnt/f/heavy_data/MOSAiC_cloudnet/issue_flag/"}
	path_plots = path_plots_base + "retrieval_benefit/"
	os.makedirs(os.path.dirname(path_plots), exist_ok=True)
	run_benefit_plots(path_data, path_plots, 'iwv', appendix=True)


# Fig. B1: q comparison:
if which_figure == "B2":

	# first, run the retrieval to get the needed data: NOTE: EXECUTING THE RETRIEVAL REPLACES
	# THE EXISTING _prw_ files IN path_data_base + "retrieval_output/mosaic/"!
	from NN_retrieval import run_NN_retrieval
	from benefit_plots import run_benefit_plots

	# paths:
	# # # # # # # # # # # # # # # # path_data = {'path_output': path_data_base + "retrieval_output/mosaic/",	# path where output is saved to
				# # # # # # # # # # # # # # # # 'path_output_info': path_data_base + "info_content/",
				# # # # # # # # # # # # # # # # 'path_output_pred_ref': path_data_base + "retrieval_output/validation_prediction_and_reference/",
				# # # # # # # # # # # # # # # # 'path_data': path_data_base + "training_data/",			# path of training/test data
				# # # # # # # # # # # # # # # # 'path_data_eval': path_data_base + "training_data/",
				# # # # # # # # # # # # # # # # 'path_ps_track': path_data_base + "polarstern_track/",		# path of Polarstern track data
				# # # # # # # # # # # # # # # # 'path_tb_obs': {'hatpro': path_data_base + "hatpro_l1/",
								# # # # # # # # # # # # # # # # 'mirac-p': path_data_base + "mirac-p_l1/"},	# path of published level 1 tb data
				# # # # # # # # # # # # # # # # 'path_tb_offsets': "",								# path of the MOSAiC MWR TB offset correction
				# # # # # # # # # # # # # # # # 'path_old_ret': path_data_base + "hatpro_l2/",					# path of old HATPRO retrievals
				# # # # # # # # # # # # # # # # 'path_rs_obs': path_data_base + "MOSAiC_radiosondes/",
				# # # # # # # # # # # # # # # # 'descriptor_file': path_scripts + "descriptor_file_ecmwf.txt",	# for PAMTRA simulations
				# # # # # # # # # # # # # # # # 'test_purpose': path_scripts}
	# # # # # # # # # # # # # # # # path_plots = {'path_plots': path_plots_base + "synergetic_ret/",
				# # # # # # # # # # # # # # # # 'path_plots_info': path_plots_base + "synergetic_ret/info_content/"}


	path_data = {'path_output': path_data_base + "retrieval_output/mosaic/",	# path where output is saved to
				'path_output_info': path_data_base + "info_content/",
				'path_output_pred_ref': path_data_base + "retrieval_output/validation_prediction_and_reference/",
				'path_data': "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/training_data_01/merged/new_z_grid_plus/",			# path of training/test data
				'path_data_eval': "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/training_data_01/merged/new_z_grid_plus/",
				'path_ps_track': "/mnt/f/heavy_data/polarstern_track/",		# path of Polarstern track data
				'path_tb_obs': {'hatpro': path_data_base + "hatpro_l1/",
								'mirac-p': path_data_base + "mirac-p_l1/"},	# path of published level 1 tb data
				'path_tb_offsets': "",								# path of the MOSAiC MWR TB offset correction
				'path_old_ret': "/mnt/f/heavy_data/MOSAiC_radiometers/HATPRO_l2_v01/",					# path of old HATPRO retrievals
				'path_rs_obs': "/mnt/f/heavy_data/MOSAiC_radiosondes/",
				'descriptor_file': path_scripts + "descriptor_file_ecmwf.txt",	# for PAMTRA simulations
				'test_purpose': path_scripts}
	path_plots = {'path_plots': path_plots_base + "synergetic_ret/",
				'path_plots_info': path_plots_base + "synergetic_ret/info_content/"}
	for key, value in path_plots.items():
		os.makedirs(os.path.dirname(value), exist_ok=True)

	# execute retrieval; then create plot.
	run_NN_retrieval(path_data, path_plots, test_id="485", exec_type='op_ret', eval_mode=False)

	# # # # # # # # # # # # # # # # # # # # # # # # path_data = {'nn_syn_mosaic': path_data_base + "retrieval_output/mosaic/",	# synergy of hatpro and mirac-p
				# # # # # # # # # # # # # # # # # # # # # # # # 'mwr_pro': path_data_base + "hatpro_l2/",		# hatpro retrievals
				# # # # # # # # # # # # # # # # # # # # # # # # 'nn_mir': path_data_base + "mirac-p_l2/",		# mirac-p IWV retrieval
				# # # # # # # # # # # # # # # # # # # # # # # # 'radiosondes': path_data_base + "MOSAiC_radiosondes/",	# MOSAiC level 2 radiosondes
				# # # # # # # # # # # # # # # # # # # # # # # # 'metcity': path_data_base + "MOSAiC_metcity/",	# MOSAiC met city data
				# # # # # # # # # # # # # # # # # # # # # # # # 'cloudnet': path_data_base + "MOSAiC_cloudnet/",# MOSAiC cloudnet target classification
				# # # # # # # # # # # # # # # # # # # # # # # # 'lls_mask': path_data_base + "MOSAiC_cloudnet/LLS_mask/",	# MOSAiC cloudnet low-level stratus mask
				# # # # # # # # # # # # # # # # # # # # # # # # 'issue_flag': path_data_base + "MOSAiC_cloudnet/issue_flag/"}	# MOSAiC cloudnet issue data set
	path_data = {'nn_syn_mosaic': path_data_base + "retrieval_output/mosaic/",
				'mwr_pro': "/mnt/f/heavy_data/MOSAiC_radiometers/HATPRO_l2_v01/",
				'nn_mir': "/mnt/f/heavy_data/MOSAiC_radiometers/MiRAC-P_l2_v01/",
				'radiosondes': "/mnt/f/heavy_data/MOSAiC_radiosondes/",
				'metcity': "/mnt/f/heavy_data/MOSAiC_metcity/",
				'cloudnet': "/mnt/f/heavy_data/MOSAiC_cloudnet/",
				'lls_mask': "/mnt/f/heavy_data/MOSAiC_cloudnet/LLS_mask/",
				'issue_flag': "/mnt/f/heavy_data/MOSAiC_cloudnet/issue_flag/"}
	path_plots = path_plots_base + "retrieval_benefit/"
	os.makedirs(os.path.dirname(path_plots), exist_ok=True)
	run_benefit_plots(path_data, path_plots, 'q', appendix=True)