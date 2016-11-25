<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/utils/io_utils.py#L8)</span>
### HDF5Matrix

```python
keras.utils.io_utils.HDF5Matrix(datapath, dataset, start=0, end=None, normalizer=None)
```

Representation of HDF5 dataset which can be used instead of a
Numpy array.

__Example__


```python
X_data = HDF5Matrix('input/file.hdf5', 'data')
model.predict(X_data)
```

Providing start and end allows use of a slice of the dataset.

Optionally, a normalizer function (or lambda) can be given. This will
be called on every slice of data retrieved.

__Arguments__

- __datapath__: string, path to a HDF5 file
- __dataset__: string, name of the HDF5 dataset in the file specified
	in datapath
- __start__: int, start of desired slice of the specified dataset
- __end__: int, end of desired slice of the specified dataset
- __normalizer__: function to be called on data when retrieved

