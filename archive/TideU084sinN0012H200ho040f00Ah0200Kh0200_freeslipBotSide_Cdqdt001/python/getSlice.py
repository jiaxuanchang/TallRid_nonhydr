import xarray as xr

#for f in ['038', '073', '100', '126', '141']:
if True:
    td = 'LWT1kmlowU0T025Amp305f100'
    todo = '../results/{}/input/spinup.nc'.format(td)

    with xr.open_dataset(todo) as ds:
        print(ds)
        ds = ds.isel(j=10, j_g=10)
        ds.to_netcdf('../reduceddata/AllSlicesSU{}.nc'.format(td))
