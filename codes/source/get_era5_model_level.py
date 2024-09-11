"""
    Get ERA5 model level data:
    latitude: 90, 75
    longitude: -180, 180
    time: hourly, 2019-11-20
"""

#!/usr/bin/env python
import cdsapi

c = cdsapi.Client()
date_str = "2019-11-20"
date_str_file = date_str.replace("-","").replace("/","")

c.retrieve("reanalysis-era5-complete", {
    "date": date_str,    # adapt date
    "levelist": "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50/51/52/53/54/55/56/57/58/59/60/61/62/63/64/65/66/67/68/69/70/71/72/73/74/75/76/77/78/79/80/81/82/83/84/85/86/87/88/89/90/91/92/93/94/95/96/97/98/99/100/101/102/103/104/105/106/107/108/109/110/111/112/113/114/115/116/117/118/119/120/121/122/123/124/125/126/127/128/129/130/131/132/133/134/135/136/137",    # 1: top of atmosphere, 137: lowest model level
    "levtype": "ml",
    "param": "129/130/133/152", # Full information at https://apps.ecmwf.int/codes/grib/param-db/ ; 129, 130, 133, 152 must always be included
    "stream": "oper",
    "time": '00/to/23/by/1', # adapt time; for 00, 03, 06, ..., 21 UTC, use: '00/to/23/by/3'
    "type": "an",
    "area": '87./118.0/84.0/125.0', # north/west/south/east
    "grid": '0.25/0.25',      # latitude/longitude grid
    "format": 'netcdf'
}, f"MOSAiC_ERA5_model_levels_temp_q_{date_str_file}.nc") # output name
