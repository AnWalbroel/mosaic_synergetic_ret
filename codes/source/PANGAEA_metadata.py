import datetime as dt
import os
import pdb
import glob

wdir = os.getcwd() + "/"
remote = ((("/net/blanc/" in wdir) | ("/work/awalbroe/" in wdir)) and ("/mnt/f/" not in wdir))		# identify if the code is executed on the blanc computer or at home

import numpy as np
import xarray as xr


# Event list according to PANGAEA: https://www.pangaea.de/expeditions/byproject/MOSAiC
# also includes start and end times of events:
event_list = {'leg1': ['PS122/1_1-38', np.datetime64("2019-09-30T06:00:00"), np.datetime64("2019-12-13T09:04:45")],
				'leg2': ['PS122/2_14-18', np.datetime64("2019-12-13T09:11:00"), np.datetime64("2020-02-27T10:56:00")],
				'leg3': ['PS122/3_28-6', np.datetime64("2020-02-27T10:57:00"), np.datetime64("2020-06-08T23:59:00")],
				'leg4_1': ['PS122/4_43-11', np.datetime64("2020-06-09T00:00:00"), np.datetime64("2020-06-27T07:00:00")],
				'leg4_2': ['PS122/4_43-145', np.datetime64("2020-06-27T07:01:00"), np.datetime64("2020-08-12T09:59:00")],
				'leg5': ['PS122/5_58-3', np.datetime64("2020-08-12T10:00:00"), np.datetime64("2020-10-12T23:59:59")]}


def extract_metadata(
	files,
	path_output,
	product):

	"""
	Imports from all files (one after another), identifies the correct 'event',
	extracts metadata, and writes them to a text file.

	Parameters:
	-----------
	files : list of str
		Path and filename of all files from which metadata is to be extracted.
	path_output : str
		Path where the metadata will be saved to (as .txt).
	product : str
		Specifies the data product. Valid options are given in set_dict['products'].
	"""

	metadata = {'event': list(), 'filename': list(), 'start_time': list(), 'start_lat': list(),
				'start_lon': list(), 'end_time': list(), 'end_lat': list(), 'end_lon': list()}
	for file in files:
		DS = xr.open_dataset(file)

		ds_time = DS.time.values
		n_time = len(ds_time)
		current_date = ds_time[round(n_time/2)].astype('datetime64[D]')

		# catch overlap of more than one event:
		if current_date == np.datetime64("2019-12-13"):
			metadata['event'].append('PS122/1_1-38,PS122/2_14-18')

		elif current_date == np.datetime64("2020-02-27"):
			metadata['event'].append('PS122/2_14-18,PS122/3_28-6')

		elif current_date == np.datetime64("2020-06-08"):
			metadata['event'].append('PS122/3_28-6')

		elif current_date == np.datetime64("2020-06-09"):
			metadata['event'].append('PS122/4_43-11')

		elif current_date == np.datetime64("2020-06-27"):
			metadata['event'].append('PS122/4_43-11,PS122/4_43-145')

		elif current_date == np.datetime64("2020-08-12"):
			metadata['event'].append('PS122/4_43-145,PS122/5_58-3')

		else:
			
			# identify unambiguous event:
			for event in event_list.keys():
				# matches one event:
				if np.all((ds_time >= event_list[event][1]) & (ds_time <= event_list[event][2])):
					metadata['event'].append(event_list[event][0])


		# append metadata:
		metadata['filename'].append(os.path.basename(file))
		metadata['start_time'].append(str(ds_time[0].astype('datetime64[m]')))
		metadata['start_lat'].append(f"{DS.lat.values[0]:.5f}")
		metadata['start_lon'].append(f"{DS.lon.values[0]:.5f}")
		metadata['end_time'].append(str(ds_time[-1].astype('datetime64[m]')))
		metadata['end_lat'].append(f"{DS.lat.values[-1]:.5f}")
		metadata['end_lon'].append(f"{DS.lon.values[-1]:.5f}")

		DS = DS.close()


	n_files = len(metadata['filename'])
	for mkey in metadata.keys():
		assert len(metadata[mkey]) == n_files

	# reorganize list for file writing:
	metadata_write = list()
	for idx, file in enumerate(metadata['filename']):
		# list of all metadata for the current file:
		list_temp = [metadata[mkey][idx] for mkey in metadata.keys()]
		metadata_write.append(list_temp)

	# write to file:
	output_file = path_output + f"metadata_PANGAEA_uoc_hatpro_lhumpro-243-340_l2_{product}.txt"
	headerline = [mkey for mkey in metadata.keys()]
	with open(output_file, 'w') as f:
		f.write('\t'.join(headerline) + '\n')
		f.writelines('\t'.join(list_row) + '\n' for list_row in metadata_write)
	

###################################################################################################


"""
	Quick script to read out lat, lon and time at beginning and end of each dataset
	uploaded to PANGAEA (HATPRO L1 and L2, MiRAC-P L1 and L2).

	- Import data
	- read out information
	- save to tab limited text file
"""

if remote:
	path_data = {'synergy': "/net/blanc/awalbroe/Data/synergetic_ret/tests_01/output/l2/"}
	path_output = "/net/blanc/awalbroe/Data/synergetic_ret/tests_01/output/l2/"
else:
	path_data = {'synergy': "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/tests_01/output/l2/"}
	path_output = "/mnt/f/heavy_data/synergetic_ret/synergetic_ret/tests_01/output/l2/"


# settings:
set_dict = {'products': ['prw', 'q', 'temp', 'temp_bl', 'rh']
			}

# create path if non-existent:
os.makedirs(os.path.dirname(path_output), exist_ok=True)


# Import data: Loop through data products:
for key in set_dict['products']:

	product = key.upper()
	print(f"Extracting metadata from {product}.")
	files = sorted(glob.glob(path_data['synergy'] + f"MOSAiC_uoc_hatpro_lhumpro-243-340_l2_{key}_v*.nc"))
	extract_metadata(files, path_output, product=product)


print("Done....")