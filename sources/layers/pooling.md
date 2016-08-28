<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L54)</span>
### MaxPooling1D

```python
keras.layers.pooling.MaxPooling1D(pool_length=2, stride=None, border_mode='valid')
```

時系列データのマックスプーリング演算．

__Input shape__

配列サイズ `(samples, steps, features)`の3次元テンソル．

__Output shape__

配列サイズ `(samples, downsampled_steps, features)`の3次元テンソル．

__Arguments__

- __pool_length__: マックスプーリングを適用する領域のサイズ
- __stride__: 整数もしくはNone．Stride値．ダウンスケールする係数．2は入力を半分にする．
    Noneの場合は，デフォルトで`pool_length`になる．
- __border_mode__: 'valid'か'same'．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L174)</span>
### MaxPooling2D

```python
keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default')
```

空間データのマックスプーリング演算．

__Arguments__

- __pool_size__: ダウンスケールする係数を決める
  2つの整数のタプル（垂直，水平）．
  (2, 2) は画像をそれぞれの次元で半分にします．
- __strides__: 2つの整数のタプルもしくはNone．Strides値．
    Noneの場合は，デフォルトで`pool_length`になる．
- __border_mode__: 'valid'か'same'．
    - __Note__: 現時点では'same'はTensorFlowでのみ動きます．
- __dim_ordering__:'th'か'tf'．
  'th'モードのときはチャネルの次元（深さ）はindex 1に，
  'tf'モードではindex 3に．
  デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_dim_ordering`の値です．値を設定していなければ，"th"になります．

__Input shape__

dim_ordering='th'の場合，`(samples, channels, rows, cols)`の4次元テンソル，
もしくは
dim_ordering='tf'の場合，`(samples, rows, cols, channels)`の4次元テンソル．

__Output shape__

dim_ordering='th'の場合，4D tensor with shape:
`(nb_samples, channels, pooled_rows, pooled_cols)`の4次元テンソル，
もしくは
dim_ordering='tf'の場合，`(samples, pooled_rows, pooled_cols, channels)`の4次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L323)</span>
### MaxPooling3D

```python
keras.layers.pooling.MaxPooling3D(pool_size=(2, 2, 2), strides=None, border_mode='valid', dim_ordering='default')
```

3次元データ（空間もしくは時空間）に対するマックスプーリング演算．

__Arguments__

- __pool_size__: 3つの整数のタプル(dim1, dim2, dim3)，
  ダウンスケールするための係数．
  (2, 2, 2)は3次元入力のサイズをそれぞれの次元で半分にします．
- __strides__: 3つの整数のタプルもしくはNone．Strides値．
- __border_mode__: 'valid'か'same'．
- __dim_ordering__: 'th'か'tf'．
  'th'モードのときはチャネルの次元（深さ）はindex 1に，
  'tf'モードではindex 4に．
  デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_dim_ordering`の値です．値を設定していなければ，"th"になります．

__Input shape__

dim_ordering='th'の場合，配列サイズ
`(samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3)` の5次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, len_pool_dim1, len_pool_dim2, len_pool_dim3, channels)`の5次元テンソル．

__Output shape__

dim_ordering='th'の場合，配列サイズ
`(nb_samples, channels, pooled_dim1, pooled_dim2, pooled_dim3)`の5次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, pooled_dim1, pooled_dim2, pooled_dim3, channels)`の5次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L84)</span>
### AveragePooling1D

```python
keras.layers.pooling.AveragePooling1D(pool_length=2, stride=None, border_mode='valid')
```

時系列データのための平均プーリング演算．

__Arguments__

- __pool_length__: ダウンスケールする係数．2は入力を半分にします．
- __stride__: 整数もしくはNone．Stride値．
    Noneの場合は，デフォルトで`pool_length`になる．
- __border_mode__: 'valid'か'same'．
    - __Note__: 現時点では'same'はTensorFlowでのみ動きます．

__Input shape__

配列サイズ`(samples, steps, features)`の3次元テンソル．

__Output shape__

配列サイズ`(samples, downsampled_steps, features)`の3次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L216)</span>
### AveragePooling2D

```python
keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default')
```

空間データのための平均プーリング演算．

__Arguments__

- __pool_size__: ダウンスケールする係数を決める
  2つの整数のタプル（垂直，水平）．
  (2, 2) は画像をそれぞれの次元で半分にします．
- __strides__: 2つの整数のタプルもしくはNone．Strides値．
    Noneの場合は，デフォルトで`pool_length`になる．
- __border_mode__: 'valid'か'same'．
  - __Note__: 現時点では'same'はTensorFlowでのみ動きます．
- __dim_ordering__: 'th'か'tf'．
  'th'モードのときはチャネルの次元（深さ）はindex 1に，
  'tf'モードではindex 3に．
  デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_dim_ordering`の値です．値を設定していなければ，"th"になります．

__Input shape__

dim_ordering='th'の場合，配列サイズ
`(samples, channels, rows, cols)`の4次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, rows, cols, channels)`の4次元テンソル．

__Output shape__

dim_ordering='th'の場合，配列サイズ
`(nb_samples, channels, pooled_rows, pooled_cols)` の4次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, pooled_rows, pooled_cols, channels)`の4次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L363)</span>
### AveragePooling3D

```python
keras.layers.pooling.AveragePooling3D(pool_size=(2, 2, 2), strides=None, border_mode='valid', dim_ordering='default')
```

3次元データ（空間もしくは時空間）に対する平均プーリング演算．

__Arguments__

- __pool_size__: 3つの整数のタプル(dim1, dim2, dim3)，
  ダウンスケールするための係数．
  (2, 2, 2)は3次元入力のサイズをそれぞれの次元で半分にします．
- __strides__: 3つの整数のタプルもしくはNone．Strides値．
- __border_mode__: 'valid'か'same'．
- __dim_ordering__: 'th'か'tf'．
  'th'モードのときはチャネルの次元（深さ）はindex 1に，
  'tf'モードではindex 4に．
  デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_dim_ordering`の値です．値を設定していなければ，"th"になります．

__Input shape__

dim_ordering='th'の場合，配列サイズ
`(samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3)`の5次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, len_pool_dim1, len_pool_dim2, len_pool_dim3, channels)`の5次元テンソル．

__Output shape__

dim_ordering='th'の場合，配列サイズ
`(nb_samples, channels, pooled_dim1, pooled_dim2, pooled_dim3)`の5次元テンソル
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, pooled_dim1, pooled_dim2, pooled_dim3, channels)`の5次元テンソル
