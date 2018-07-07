<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L57)</span>
### MaxPooling1D

```python
keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')
```

時系列データのマックスプーリング演算．

__引数__

- __pool_size__: マックスプーリングを適用する領域のサイズを指定します．
- __strides__: ストライド値．整数もしくはNoneで指定します．Noneの場合は，`pool_size`の値が適用されます．
- __padding__: `'valid'`か`'same'`のいずれかです．

__入力のshape__

`(batch_size, steps, features)`の3階テンソル．

__出力のshape__

`(batch_size, downsampled_steps, features)`の3階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L170)</span>
### MaxPooling2D

```python
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

空間データのマックスプーリング演算．

__引数__

- __pool_size__: ダウンスケールする係数を決める
  2つの整数のタプル（垂直，水平）．
  (2, 2) は画像をそれぞれの次元で半分にします．
  1つの整数しか指定ないと，それぞれの次元に対して同じ値が用いられます．
- __strides__: ストライド値．2つの整数からなるタプル，もしくはNoneで指定します．
Noneの場合は，`pool_size`の値が適用されます．
- __padding__: `'valid'`か`'same'`のいずれかです．
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力のshapeは`(batch, height, width, channels)`となり，`"channels_first"`の場合は`(batch, channels, height, width)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__入力のshape__

- `data_format='channels_last'`の場合， `(batch_size, rows, cols, channels)`の4階テンソル．
- `data_format='channels_first'`の場合， `(batch_size, channels, rows, cols)`の4階テンソル．

__出力のshape__

- `data_format='channels_last'`の場合，   
`(batch_size, pooled_rows, pooled_cols, channels)`の4階テンソル．
- `data_format='channels_first'`の場合，   
`(batch_size, channels, pooled_rows, pooled_cols)`の4階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L339)</span>
### MaxPooling3D

```python
keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

3次元データ（空間もしくは時空間）に対するマックスプーリング演算．

__引数__

- __pool_size__: 3つの整数のタプル(dim1, dim2, dim3)，
  ダウンスケールするための係数．
  (2, 2, 2)は3次元入力のサイズをそれぞれの次元で半分にします．
- __strides__: ストライド値．3つの整数のタプルもしくはNoneで指定します．
- __padding__: `'valid'`か`'same'`のいずれかです．
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力のshapeは`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`となり，`"channels_first"`の場合は`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__入力のshape__

- `data_format='channels_last'`の場合，   
`(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`の5階テンソル．
- `data_format='channels_first'`の場合，   
`(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`の5階テンソル．

__出力のshape__

- `data_format='channels_last'`の場合，  
`(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`の5階テンソル．
- `data_format='channels_first'`の場合，   
`(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`の5階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L87)</span>
### AveragePooling1D

```python
keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid')
```

時系列データのための平均プーリング演算．

__引数__

- __pool_size__: 整数．ダウンスケールする係数．
- __strides__: ストライド値．整数もしくはNone．
    Noneの場合は，`pool_size`の値が適用されます．
- __padding__: `'valid'`か`'same'`のいずれかです．

__入力のshape__

`(batch_size, steps, features)`の3階テンソル．

__出力のshape__

`(batch_size, downsampled_steps, features)`の3階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L225)</span>
### AveragePooling2D

```python
keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

空間データのための平均プーリング演算．

__引数__

- __pool_size__: ダウンスケールする係数を決める
  2つの整数のタプル（垂直，水平）．
  (2, 2) は画像をそれぞれの次元で半分にします．
  1つの整数しか指定ないと，それぞれの次元に対して同じ値が用いられます．
- __strides__: ストライド値．2つの整数のタプルもしくはNone．
    Noneの場合は，`pool_size`の値が適用されます．
- __padding__: `'valid'`か`'same'`のいずれかです．
- __data_format__: `channels_last`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力のshapeは`(batch, height, width, channels)`となり，`"channels_first"`の場合は`(batch, channels, height, width)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__入力のshape__

- `data_format='channels_last'`の場合，`(batch_size, rows, cols, channels)`の4階テンソル．
- `data_format='channels_first'`の場合，`(batch_size, channels, rows, cols)`の4階テンソル．

__出力のshape__

- `data_format='channels_last'`の場合，   
`(batch_size, pooled_rows, pooled_cols, channels)`の4階テンソル．
- `data_format='channels_first'`の場合，   
`(batch_size, channels, pooled_rows, pooled_cols)`の4階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L389)</span>
### AveragePooling3D

```python
keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

3次元データ（空間もしくは時空間）に対する平均プーリング演算．

__引数__

- __pool_size__: 3つの整数のタプル(dim1, dim2, dim3)，
  ダウンスケールするための係数．
  (2, 2, 2)は3次元入力のサイズをそれぞれの次元で半分にします．
- __strides__: ストライド値．3つの整数のタプルもしくはNone．
- __border_mode__: `'valid'`か`'same'`のいずれかです．
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力のshapeは`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`となり，`"channels_first"`の場合は`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__入力のshape__

- `data_format='channels_last'`の場合，   
`(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`の5階テンソル．
- `data_format='channels_first'`の場合，   
`(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`の5階テンソル．

__出力のshape__

- `data_format='channels_last'`の場合，  
`(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`の5階テンソル．
- `data_format='channels_first'`の場合，   
`(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`の5階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L470)</span>
### GlobalMaxPooling1D

```python
keras.layers.GlobalMaxPooling1D()
```

時系列データのためのグローバルなマックスプーリング演算．

__入力のshape__

`(batch_size, steps, features)`の3階テンソル．

__出力のshape__

`(batch_size, channels)`の2階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L455)</span>
### GlobalAveragePooling1D

```python
keras.layers.GlobalAveragePooling1D()
```

時系列データのためのグローバルな平均プーリング演算．

__入力のshape__

`(batch_size, steps, features).`の3階テンソル．

__出力のshape__

`(batch_size, channels)`の2階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L545)</span>
### GlobalMaxPooling2D

```python
keras.layers.GlobalMaxPooling2D(data_format=None)
```

空間データのグローバルなマックスプーリング演算．

__引数__

- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力のshapeは`(batch, height, width, channels)`となり，`"channels_first"`の場合は`(batch, channels, height, width)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__入力のshape__

- `data_format='channels_last'`の場合， `(batch_size, rows, cols, channels)`の4階テンソル．
- `data_format='channels_first'`の場合， `(batch_size, channels, rows, cols)`の4階テンソル．

__出力のshape__

`(batch_size, channels)`の2階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L510)</span>
### GlobalAveragePooling2D

```python
keras.layers.GlobalAveragePooling2D(data_format=None)
```

空間データのグローバルな平均プーリング演算．  

__引数__

- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力のshapeは`(batch, height, width, channels)`となり，`"channels_first"`の場合は`(batch, channels, height, width)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__入力のshape__

- `data_format='channels_last'`の場合， `(batch_size, rows, cols, channels)`の4階テンソル．
- `data_format='channels_first'`の場合， `(batch_size, channels, rows, cols)`の4階テンソル．

__出力のshape__

`(batch_size, channels)`の2階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L640)</span>
### GlobalMaxPooling3D

```python
keras.layers.GlobalMaxPooling3D(data_format=None)
```

3次元データに対するグローバルなマックスプーリング演算．  

__引数__

- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力のshapeは`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`となり，`"channels_first"`の場合は`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__入力のshape__

- `data_format='channels_last'`の場合， `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`の5階テンソル．
- `data_format='channels_first'`の場合， `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`の5階テンソル．

__出力のshape__

`(batch_size, channels)`の2階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L605)</span>
### GlobalAveragePooling3D

```python
keras.layers.GlobalAveragePooling3D(data_format=None)
```

3次元データに対するグローバルな平均プーリング演算．  

__引数__

- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力のshapeは`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`となり，`"channels_first"`の場合は`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__入力のshape__

- `data_format='channels_last'`の場合， `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`の5階テンソル．
- `data_format='channels_first'`の場合， `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`の5階テンソル．

__出力のshape__

`(batch_size, channels)`の2階テンソル．
