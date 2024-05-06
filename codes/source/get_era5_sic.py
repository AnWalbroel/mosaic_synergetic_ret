"""
	Get ERA5 sea ice concentration data:
	latitude: 90, 65
	longitude: -180, 180
	time: daily at 12 UTC from 2001-01-01 to 2018-12-31
"""


import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'sea_ice_cover',
        'year': ['2001', '2002', '2003', 
                 '2004', '2005', '2006',
                 '2007', '2008', '2009',
                 '2010', '2011', '2012',
                 '2013', '2014', '2015',
                 '2016', '2017', '2018',],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': '12:00',
        'area': [
            90, -180, 65,
            180,
        ],
    },
    'ERA5_single_level_SIC_arctic_2001-2018.nc')