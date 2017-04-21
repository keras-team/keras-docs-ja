<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/utils/io_utils.py#L8)</span>
### HDF5Matrix

```python
keras.utils.io_utils.HDF5Matrix(datapath, dataset, start=0, end=None, normalizer=None)
```

Numpyの配列の代わりに使えるHDF5データセットの表現です．

__例__


```python
X_data = HDF5Matrix('input/file.hdf5', 'data')
model.predict(X_data)
```

startとendを指定することでデータセットをスライスすることができます．

normalizer関数（やラムダ式）を渡すことができます．
normalizer関数は取得されたすべてのスライスに適用されます．

__引数__

- __datapath__: 文字列．HDF5ファイルへのパス
- __dataset__: 文字列．datapathで指定されたファイル中におけるHDF5データセットの名前
- __start__: 整数．指定されたデータセットのスライス開始位置
- __end__: 整数．指定されたデータセットのスライス終了位置
- __normalizer__: 読み込まれた時にデータに対して適用する関数
