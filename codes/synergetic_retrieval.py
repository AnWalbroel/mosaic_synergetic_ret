import os
import sys
import pdb

wdir = os.getcwd() + "/"


"""
    Script to manage and create retrieval output for the synergistic retrieval study. Execute this script
    in the following way:
    To create IWV retrieval: python3 synergetic_retrieval.py "prw"
    To create temperature profile retrieval: python3 synergetic_retrieval.py "temp"
    To create boundary layer temperature profile retrieval: python3 synergetic_retrieval.py "temp_bl"
    To create q profile retrieval: python3 synergetic_retrieval.py "q"
"""

# Paths:
path_scripts = os.path.dirname(wdir) + "/source/"
path_tools = os.path.dirname(wdir) + "/tools/"
path_data_base = os.path.dirname(wdir[:-1]) + "/data/"
path_plots_base = os.path.dirname(wdir[:-1]) + "/plots/"

sys.path.insert(0, path_scripts)
sys.path.insert(0, path_tools)


# settings:
which_ret = sys.argv[1]     # set the number of the figure of the manuscript as string,
                            # e.g., "prw" for IWV, "temp_bl" for boundarly layer temperature profile


# Run the retrieval to get the output data: NOTE: EXECUTING THE RETRIEVAL REPLACES
# THE EXISTING _prw_ files IN path_data_base + "retrieval_output/mosaic/"!
from NN_retrieval import run_NN_retrieval

# paths:
path_data = {'path_output': path_data_base + "retrieval_output/mosaic/",    # path where output is saved to
            'path_output_info': path_data_base + "info_content/",
            'path_output_pred_ref': path_data_base + "retrieval_output/validation_prediction_and_reference/",
            'path_data': path_data_base + "training_data/",         # path of training/test data
            'path_data_eval': path_data_base + "training_data/",
            'path_ps_track': path_data_base + "polarstern_track/",      # path of Polarstern track data
            'path_tb_obs': {'hatpro': path_data_base + "hatpro_l1/",
                            'mirac-p': path_data_base + "mirac-p_l1/"}, # path of published level 1 tb data
            'path_tb_offsets': "",                              # path of the MOSAiC MWR TB offset correction
            'path_old_ret': path_data_base + "hatpro_l2/",                  # path of old HATPRO retrievals
            'path_rs_obs': path_data_base + "MOSAiC_radiosondes/",
            'descriptor_file': path_scripts + "descriptor_file_ecmwf.txt",  # for PAMTRA simulations
            'test_purpose': path_scripts}
path_plots = {'path_plots': path_plots_base + "synergetic_ret/",
            'path_plots_info': path_plots_base + "synergetic_ret/info_content/"}

for key, value in path_plots.items():
    os.makedirs(os.path.dirname(value), exist_ok=True)


# execute retrieval
if which_ret == 'prw':
    run_NN_retrieval(path_data, path_plots, test_id="126", exec_type='op_ret', eval_mode=False)
if which_ret == 'temp':
    run_NN_retrieval(path_data, path_plots, test_id="417", exec_type='op_ret', eval_mode=False)
if which_ret == 'temp_bl':
    run_NN_retrieval(path_data, path_plots, test_id="424", exec_type='op_ret', eval_mode=False)
if which_ret == 'q':
    run_NN_retrieval(path_data, path_plots, test_id="472", exec_type='op_ret', eval_mode=False)