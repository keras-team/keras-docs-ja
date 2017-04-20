<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L54)</span>
### MaxPooling1D

```python
keras.layers.pooling.MaxPooling1D(pool_size=2, strides=None, padding='valid')
```

時系列データのマックスプーリング演算．

__Arguments__

- __pool_size__: マックスプーリングを適用する領域のサイズを指定します．
- __strides__: ストライド値．整数もしくはNoneで指定します．Noneの場合は，`pool_size`の値が適用されます．
- __padding__: `'valid'`か`'same'`のいずれかです．

__Input shape__

`(batch_size, steps, features)`の3次元テンソル．

__Output shape__

`(batch_size, downsampled_steps, features)`の3次元テンソル．



----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L174)</span>
### MaxPooling2D

```python
keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

空間データのマックスプーリング演算．

__Arguments__

- __pool_size__: ダウンスケールする係数を決める
  2つの整数のタプル（垂直，水平）．
  (2, 2) は画像をそれぞれの次元で半分にします．
- __strides__: ストライド値．2つの整数からなるタプル，もしくはNoneで指定します．
Noneの場合は，`pool_size`の値が適用されます．
- __padding__: `'valid'`か`'same'`のいずれかです．
- __data_format__: `"channels_last"`(デフォルト)か`"channels_first"`を指定します. `"channels_last"`の場合，入力の型は`(batch, height, width, channels)`となり，`"channels_first"`の場合は`(batch, channels, height, width)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__Input shape__

- `data_format='channels_last'`の場合， `(batch_size, rows, cols, channels)`の4次元テンソル．
- `data_format='channels_first'`の場合， `(batch_size, channels, rows, cols)`の4次元テンソル．

__Output shape__

- `data_format='channels_last'`の場合，   
`(batch_size, pooled_rows, pooled_cols, channels)`の4次元テンソル．
- `data_format='channels_first'`の場合，   
`(batch_size, channels, pooled_rows, pooled_cols)`の4次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L323)</span>
### MaxPooling3D

```python
keras.layers.pooling.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

3次元データ（空間もしくは時空間）に対するマックスプーリング演算．

__Arguments__

- __pool_size__: 3つの整数のタプル(dim1, dim2, dim3)，
  ダウンスケールするための係数．
  (2, 2, 2)は3次元入力のサイズをそれぞれの次元で半分にします．
- __strides__: ストライド値．3つの整数のタプルもしくはNoneで指定します．
- __padding__: `'valid'`か`'same'`のいずれかです．
- __data_format__: `"channels_last"`(デフォルト)か`"channels_first"`を指定します. `"channels_last"`の場合，入力の型は`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`となり，`"channels_first"`の場合は`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__Input shape__

- `data_format='channels_last'`の場合，   
`(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`の5次元テンソル．
- `data_format='channels_first'`の場合，   
`(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`の5次元テンソル．

__Output shape__

- `data_format='channels_last'`の場合，  
`(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`の5次元テンソル．
- `data_format='channels_first'`の場合，   
`(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`の5次元テンソル．


----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L84)</span>
### AveragePooling1D

```python
keras.layers.pooling.AveragePooling1D(pool_size=2, strides=None, padding='valid')
```

時系列データのための平均プーリング演算．

__Arguments__

- __pool_size__: 整数．ダウンスケールする係数．
- __strides__: ストライド値．整数もしくはNone．
    Noneの場合は，`pool_size`の値が適用されます．
- __padding__: `'valid'`か`'same'`のいずれかです．

__Input shape__

`(batch_size, steps, features)`の3次元テンソル．

__Output shape__

`(batch_size, downsampled_steps, features)`の3次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L216)</span>
### AveragePooling2D

```python
keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

空間データのための平均プーリング演算．

__Arguments__

- __pool_size__: ダウンスケールする係数を決める
  2つの整数のタプル（垂直，水平）．
  (2, 2) は画像をそれぞれの次元で半分にします．
- __strides__: ストライド値．2つの整数のタプルもしくはNone．
    Noneの場合は，`pool_size`の値が適用されます．
- __padding__: `'valid'`か`'same'`のいずれかです．
- __data_format__: `channels_last`(デフォルト)か`"channels_first"`を指定します. `"channels_last"`の場合，入力の型は`(batch, height, width, channels)`となり，`"channels_first"`の場合は`(batch, channels, height, width)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__Input shape__

- `data_format='channels_last'`の場合，`(batch_size, rows, cols, channels)`の4次元テンソル．
- `data_format='channels_first'`の場合，`(batch_size, channels, rows, cols)`の4次元テンソル．

__Output shape__

- `data_format='channels_last'`の場合，   
`(batch_size, pooled_rows, pooled_cols, channels)`の4次元テンソル．
- `data_format='channels_first'`の場合，   
`(batch_size, channels, pooled_rows, pooled_cols)`の4次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L363)</span>
### AveragePooling3D

```python
keras.layers.pooling.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

3次元データ（空間もしくは時空間）に対する平均プーリング演算．

__Arguments__

- __pool_size__: 3つの整数のタプル(dim1, dim2, dim3)，
  ダウンスケールするための係数．
  (2, 2, 2)は3次元入力のサイズをそれぞれの次元で半分にします．
- __strides__: ストライド値．3つの整数のタプルもしくはNone．
- __border_mode__: `'valid'`か`'same'`のいずれかです．
- __data_format__: `"channels_last"`(デフォルト)か`"channels_first"`を指定します. `"channels_last"`の場合，入力の型は`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`となり，`"channels_first"`の場合は`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__Input shape__

- `data_format='channels_last'`の場合，   
`(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`の5次元テンソル．
- `data_format='channels_first'`の場合，   
`(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`の5次元テンソル．


__Output shape__

- `data_format='channels_last'`の場合，  
`(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`の5次元テンソル．
- `data_format='channels_first'`の場合，   
`(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`の5次元テンソル．


----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L430)</span>
### GlobalMaxPooling1D

```python
keras.layers.pooling.GlobalMaxPooling1D()
```

時系列データのためのグルーバルなマックスプーリング演算．  
グローバルとは"特徴マップ全てに対して"という意味です．

__Input shape__

`(batch_size, steps, features)`の3次元テンソル．

__Output shape__

`(batch_size, channels)`の2次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L416)</span>
### GlobalAveragePooling1D

```python
keras.layers.pooling.GlobalAveragePooling1D()
```

時系列データのためのグルーバルな平均プーリング演算．  
グローバルとは"特徴マップ全てに対して"という意味です．

__Input shape__

`(batch_size, steps, features).`の3次元テンソル．

__Output shape__

`(batch_size, channels)`の2次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L497)</span>
### GlobalMaxPooling2D

```python
keras.layers.pooling.GlobalMaxPooling2D(data_format=None)
```

空間データのグルーバルなマックスプーリング演算．  
グローバルとは"特徴マップ全てに対して"という意味です．

__Auguments__

- __data_format__: `"channels_last"`(デフォルト)か`"channels_first"`を指定します. `"channels_last"`の場合，入力の型は`(batch, height, width, channels)`となり，`"channels_first"`の場合は`(batch, channels, height, width)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__Input shape__

- `data_format='channels_last'`の場合， `(batch_size, rows, cols, channels)`の4次元テンソル．
- `data_format='channels_first'`の場合， `(batch_size, channels, rows, cols)`の4次元テンソル．

__Output shape__

`(batch_size, channels)`の2次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L469)</span>
### GlobalAveragePooling2D

```python
keras.layers.pooling.GlobalAveragePooling2D(data_format=None)
```

空間データのグルーバルな平均プーリング演算．  
グローバルとは"特徴マップ全てに対して"という意味です．

__Auguments__

- __data_format__: `"channels_last"`(デフォルト)か`"channels_first"`を指定します. `"channels_last"`の場合，入力の型は`(batch, height, width, channels)`となり，`"channels_first"`の場合は`(batch, channels, height, width)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__Input shape__

- `data_format='channels_last'`の場合， `(batch_size, rows, cols, channels)`の4次元テンソル．
- `data_format='channels_first'`の場合， `(batch_size, channels, rows, cols)`の4次元テンソル．

__Output shape__

`(batch_size, channels)`の2次元テンソル．
