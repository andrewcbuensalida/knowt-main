embed = np.memmap('data/corpus_hpr/sentences.embeddings.memmap', shape=embed.shape, mode='r+')
import numpy as np
import xarray as xr
!free
!free -h
v = np.random.randn(3, 384)

x = xr.open_dataset('data/corpus_hpr/sentences.embeddings.netcdf')
!free
%timeit v.dot(embed.T)
embed = np.memmap('data/corpus_hpr/sentences.embeddings.memmap', shape=embed.shape, mode='r+')
who
shape = (41531,384)
x.shape
x.dims
x.dims.shape
list(x.dims)
shape == (x.dims['sentence'], x.dims['dim'])
x.sizes
list(x.sizes)
shape == (x.sizes['sentence'], x.sizes['dim'])
shape = (x.sizes['sentence'], x.sizes['dim'])
embed = np.memmap('data/corpus_hpr/sentences.embeddings.memmap', shape=shape, mode='r+')
!free -h
!free
%timeit v.dot(embed.T)
%timeit v.dot(embed.T)
%timeit v.dot(x.T)
%timeit v.dot(x.transpose())
x = xr.open_dataset('data/corpus_hpr/sentences.embeddings.netcdf', shape=(shape[1], shape[0]))
x = xr.open_dataset?
x = x.transpose()
%timeit v.dot(x)
v = xarrayDataArray(v)
v = xarray.DataArray(v)
v = xr.DataArray(v)
%timeit v.dot(x)
xr.dot(v, x)
v = np.random.randn(3, 384)
xr.dot(v, x)
x.dims
da_a = xr.DataArray(np.arange(3 * 2).reshape(3, 2), dims=["a", "b"])
da_b = xr.DataArray(np.arange(3 * 2 * 2).reshape(3, 2, 2), dims=["a", "b", "c"])
da_c = xr.DataArray(np.arange(2 * 3).reshape(2, 3), dims=["c", "d"])
da_a.dot(da_c)
da_a
type(da_a)
type(x)
x0 = x.to_dataarray()
!free
8217772 - 8507388
!free -h
ls -hal data/corpus_hpr/sentences.embeddings.netcdf
ls -hal data/corpus_hpr/sentences.embeddings.joblib
ls -hal data/corpus_hpr/sentences.embeddings.memmap
8302724 - 8507388
%timeit v.dot(x0)
%timeit v.dot(x0)
hist -o -p -f scripts/speed_test_xarray_vs_memmap.hist.ipy
memstart = 8268476
memopendataset = 8302724
memnpmemmap = 8217772
memxtodataarray = 8507388
!free
memend = 8432168
hist -o -p -f scripts/speed_test_xarray_vs_memmap.hist.ipy
hist -f scripts/speed_test_xarray_vs_memmap.hist.py
