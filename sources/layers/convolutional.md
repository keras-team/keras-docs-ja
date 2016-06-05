<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L20)</span>
### Convolution1D

```python
keras.layers.convolutional.Convolution1D(nb_filter, filter_length, init='uniform', activation='linear', weights=None, border_mode='valid', subsample_length=1, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
```

１次元入力の近傍をフィルターする畳み込み演算．このレイヤーを第一層に使う場合，キーワード引数として`input_dim`（整数値、例えば128次元ベクトル系列には1128）を指定するか`input_shape`（整数のタプル，例えば10個の128次元ベクトル系列のでは(10, 128)）を指定してください．

__Example__


```python
# apply a convolution 1d of length 3 to a sequence with 10 timesteps,
# with 64 output filters
model = Sequential()
model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
# now model.output_shape == (None, 10, 64)

# add a new conv1d on top
model.add(Convolution1D(32, 3, border_mode='same'))
# now model.output_shape == (None, 10, 32)
```

__Arguments__

- __nb_filter__: 使用する畳み込みカーネルの数（出力の次元）．
- __filter_length__: それぞれのフィルターの（空間もしくは時間的な）長さ．
- __init__: レイヤーの重みの初期化関数の名前（[initializations](../initializations.md)参照），
	もしくは重み初期化に用いるTheano関数。このパラメータは`weights`引数を与えない場合にのみ有効です．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)参照），
	もしくは要素ごとのTheano関数。もしなにも指定しなければ活性化は一切適用されません（つまり"線形"活性a(x) = x）．
- __weights__: 初期重みとして設定されるnumpy配列のリスト．
- __border_mode__: 'valid' あるいは 'same'．
- __subsample_length__: 出力を部分サンプルするときの長さ．　<>check
- __W_regularizer__: メインの重み行列に適用される[WeightRegularizer](../regularizers.md)（例えばL1やL2正則化）のインスタンス．
- __b_regularizer__: バイアス項に適用される[WeightRegularizer](../regularizers.md)のインスタンス．
- __activity_regularizer__: ネットワーク出力に適用される[ActivityRegularizer](../regularizers.md)のインスタンス．
- __W_constraint__: メインの重み行列に適用される[constraints](../constraints.md)モジュール（例えばmaxnorm, nonneg）のインスタンス．
- __b_constraint__: バイアス項に適用される[constraints](../constraints.md)モジュールのインスタンス．
- __bias__: バイアス項を含むかどうか（レイヤをアフィンにするか線形にするか）．
- __input_dim__: 入力のチャネル/次元数．
	このレイヤーがモデルの初めのレイヤーの場合，
	この引数もしくはキーワード引数`input_shape`を指定する必要があります．
- __input_length__: 入力系列が一定のときのその長さ．
	この引数は上流の`Flatten`そして`Dense`レイヤーを繋ぐときに必要となります．　<>check
	これがないとdense出力の配列サイズを計算することができません．

__Input shape__

配列サイズ`(samples, steps, input_dim)`の3次元テンソル．

__Output shape__

配列サイズ`(samples, new_steps, nb_filter)`の3次元テンソル．
`steps`値はパディングにより変っている可能性あり．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L191)</span>
### Convolution2D

```python
keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='th', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```

２次元入力のフィルター窓の畳み込み演算子
このレイヤーをモデルの第一層に使うときはキーワード引数`input_shape`
整数のタプル，サンプル軸を含まない）を指定してください．
例えば128x128 RGBのピクチャーでは`input_shape=(3, 128, 128)`．

__Examples__


```python
# apply a 3x3 convolution with 64 output filters on a 256x256 image:
model = Sequential()
model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
# now model.output_shape == (None, 64, 256, 256)

# add a 3x3 convolution on top, with 32 output filters:
model.add(Convolution2D(32, 3, 3, border_mode='same'))
# now model.output_shape == (None, 32, 256, 256)
```

__Arguments__

- __nb_filter__: Number of convolution filters to use.
- __nb_row__: Number of rows in the convolution kernel.
- __nb_col__: Number of columns in the convolution kernel.
- __init__: name of initialization function for the weights of the layer
	(see [initializations](../initializations.md)), or alternatively,
	Theano function to use for weights initialization.
	This parameter is only relevant if you don't pass
	a `weights` argument.
- __activation__: name of activation function to use
	(see [activations](../activations.md)),
	or alternatively, elementwise Theano function.
	If you don't specify anything, no activation is applied
	(ie. "linear" activation: a(x) = x).
- __weights__: list of numpy arrays to set as initial weights.
- __border_mode__: 'valid' or 'same'.
- __subsample__: tuple of length 2. Factor by which to subsample output.
	Also called strides elsewhere.
- __W_regularizer__: instance of [WeightRegularizer](../regularizers.md)
	(eg. L1 or L2 regularization), applied to the main weights matrix.
- __b_regularizer__: instance of [WeightRegularizer](../regularizers.md),
	applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
	applied to the network output.
- __W_constraint__: instance of the [constraints](../constraints.md) module
	(eg. maxnorm, nonneg), applied to the main weights matrix.
- __b_constraint__: instance of the [constraints](../constraints.md) module,
	applied to the bias.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 3.
- __bias__: whether to include a bias (i.e. make the layer affine rather than linear).

__Input shape__

4D tensor with shape:
`(samples, channels, rows, cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, rows, cols, channels)` if dim_ordering='tf'.

__Output shape__

4D tensor with shape:
`(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
`rows` and `cols` values might have changed due to padding.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L383)</span>
### Convolution3D

```python
keras.layers.convolutional.Convolution3D(nb_filter, kernel_dim1, kernel_dim2, kernel_dim3, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1, 1), dim_ordering='th', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```

Convolution operator for filtering windows of three-dimensional inputs.
When using this layer as the first layer in a model,
provide the keyword argument `input_shape`
(tuple of integers, does not include the sample axis),
e.g. `input_shape=(3, 10, 128, 128)` for 10 frames of 128x128 RGB pictures.

- __Note__: this layer will only work with Theano for the time being.

__Arguments__

- __nb_filter__: Number of convolution filters to use.
- __kernel_dim1__: Length of the first dimension in the covolution kernel.
- __kernel_dim2__: Length of the second dimension in the convolution kernel.
- __kernel_dim3__: Length of the third dimension in the convolution kernel.
- __init__: name of initialization function for the weights of the layer
	(see [initializations](../initializations.md)), or alternatively,
	Theano function to use for weights initialization.
	This parameter is only relevant if you don't pass
	a `weights` argument.
- __activation__: name of activation function to use
	(see [activations](../activations.md)),
	or alternatively, elementwise Theano function.
	If you don't specify anything, no activation is applied
	(ie. "linear" activation: a(x) = x).
- __weights__: list of numpy arrays to set as initial weights.
- __border_mode__: 'valid' or 'same'.
- __subsample__: tuple of length 3. Factor by which to subsample output.
	Also called strides elsewhere.
	- __Note__: 'subsample' is implemented by slicing the output of conv3d with strides=(1,1,1).
- __W_regularizer__: instance of [WeightRegularizer](../regularizers.md)
	(eg. L1 or L2 regularization), applied to the main weights matrix.
- __b_regularizer__: instance of [WeightRegularizer](../regularizers.md),
	applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
	applied to the network output.
- __W_constraint__: instance of the [constraints](../constraints.md) module
	(eg. maxnorm, nonneg), applied to the main weights matrix.
- __b_constraint__: instance of the [constraints](../constraints.md) module,
	applied to the bias.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 4.
- __bias__: whether to include a bias (i.e. make the layer affine rather than linear).

__Input shape__

5D tensor with shape:
`(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if dim_ordering='tf'.

__Output shape__

5D tensor with shape:
`(samples, nb_filter, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, nb_filter)` if dim_ordering='tf'.
`new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have changed due to padding.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L628)</span>
### MaxPooling1D

```python
keras.layers.convolutional.MaxPooling1D(pool_length=2, stride=None, border_mode='valid')
```

Max pooling operation for temporal data.

__Input shape__

3D tensor with shape: `(samples, steps, features)`.

__Output shape__

3D tensor with shape: `(samples, downsampled_steps, features)`.

__Arguments__

- __pool_length__: factor by which to downscale. 2 will halve the input.
- __stride__: integer or None. Stride value.
- __border_mode__: 'valid' or 'same'.
	- __Note__: 'same' will only work with TensorFlow for the time being.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L743)</span>
### MaxPooling2D

```python
keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th')
```

Max pooling operation for spatial data.

__Arguments__

- __pool_size__: tuple of 2 integers,
	factors by which to downscale (vertical, horizontal).
	(2, 2) will halve the image in each dimension.
- __strides__: tuple of 2 integers, or None. Strides values.
- __border_mode__: 'valid' or 'same'.
	- __Note__: 'same' will only work with TensorFlow for the time being.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 3.

__Input shape__

4D tensor with shape:
`(samples, channels, rows, cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, rows, cols, channels)` if dim_ordering='tf'.

__Output shape__

4D tensor with shape:
`(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L882)</span>
### MaxPooling3D

```python
keras.layers.convolutional.MaxPooling3D(pool_size=(2, 2, 2), strides=None, border_mode='valid', dim_ordering='th')
```

Max pooling operation for 3D data (spatial or spatio-temporal).

- __Note__: this layer will only work with Theano for the time being.

__Arguments__

- __pool_size__: tuple of 3 integers,
	factors by which to downscale (dim1, dim2, dim3).
	(2, 2, 2) will halve the size of the 3D input in each dimension.
- __strides__: tuple of 3 integers, or None. Strides values.
- __border_mode__: 'valid' or 'same'.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 4.

__Input shape__

5D tensor with shape:
`(samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, len_pool_dim1, len_pool_dim2, len_pool_dim3, channels)` if dim_ordering='tf'.

__Output shape__

5D tensor with shape:
`(nb_samples, channels, pooled_dim1, pooled_dim2, pooled_dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, pooled_dim1, pooled_dim2, pooled_dim3, channels)` if dim_ordering='tf'.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L656)</span>
### AveragePooling1D

```python
keras.layers.convolutional.AveragePooling1D(pool_length=2, stride=None, border_mode='valid')
```

Average pooling for temporal data.

__Arguments__

- __pool_length__: factor by which to downscale. 2 will halve the input.
- __stride__: integer or None. Stride value.
- __border_mode__: 'valid' or 'same'.
	- __Note__: 'same' will only work with TensorFlow for the time being.

__Input shape__

3D tensor with shape: `(samples, steps, features)`.

__Output shape__

3D tensor with shape: `(samples, downsampled_steps, features)`.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L781)</span>
### AveragePooling2D

```python
keras.layers.convolutional.AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th')
```

Average pooling operation for spatial data.

__Arguments__

- __pool_size__: tuple of 2 integers,
	factors by which to downscale (vertical, horizontal).
	(2, 2) will halve the image in each dimension.
- __strides__: tuple of 2 integers, or None. Strides values.
- __border_mode__: 'valid' or 'same'.
	- __Note__: 'same' will only work with TensorFlow for the time being.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 3.

__Input shape__

4D tensor with shape:
`(samples, channels, rows, cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, rows, cols, channels)` if dim_ordering='tf'.

__Output shape__

4D tensor with shape:
`(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L924)</span>
### AveragePooling3D

```python
keras.layers.convolutional.AveragePooling3D(pool_size=(2, 2, 2), strides=None, border_mode='valid', dim_ordering='th')
```

Average pooling operation for 3D data (spatial or spatio-temporal).

- __Note__: this layer will only work with Theano for the time being.

__Arguments__

- __pool_size__: tuple of 3 integers,
	factors by which to downscale (dim1, dim2, dim3).
	(2, 2, 2) will halve the size of the 3D input in each dimension.
- __strides__: tuple of 3 integers, or None. Strides values.
- __border_mode__: 'valid' or 'same'.
- __dim_ordering__: 'th' or 'tf'. In 'th' mode, the channels dimension
	(the depth) is at index 1, in 'tf' mode is it at index 4.

__Input shape__

5D tensor with shape:
`(samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, len_pool_dim1, len_pool_dim2, len_pool_dim3, channels)` if dim_ordering='tf'.

__Output shape__

5D tensor with shape:
`(nb_samples, channels, pooled_dim1, pooled_dim2, pooled_dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, pooled_dim1, pooled_dim2, pooled_dim3, channels)` if dim_ordering='tf'.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L966)</span>
### UpSampling1D

```python
keras.layers.convolutional.UpSampling1D(length=2)
```

Repeat each temporal step `length` times along the time axis.

__Arguments__

- __length__: integer. Upsampling factor.

__Input shape__

3D tensor with shape: `(samples, steps, features)`.

__Output shape__

3D tensor with shape: `(samples, upsampled_steps, features)`.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L997)</span>
### UpSampling2D

```python
keras.layers.convolutional.UpSampling2D(size=(2, 2), dim_ordering='th')
```

Repeat the rows and columns of the data
by size[0] and size[1] respectively.

__Arguments__

- __size__: tuple of 2 integers. The upsampling factors for rows and columns.
- __dim_ordering__: 'th' or 'tf'.
	In 'th' mode, the channels dimension (the depth)
	is at index 1, in 'tf' mode is it at index 3.

__Input shape__

4D tensor with shape:
`(samples, channels, rows, cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, rows, cols, channels)` if dim_ordering='tf'.

__Output shape__

4D tensor with shape:
`(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
or 4D tensor with shape:
`(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1051)</span>
### UpSampling3D

```python
keras.layers.convolutional.UpSampling3D(size=(2, 2, 2), dim_ordering='th')
```

Repeat the first, second and third dimension of the data
by size[0], size[1] and size[2] respectively.

- __Note__: this layer will only work with Theano for the time being.

__Arguments__

- __size__: tuple of 3 integers. The upsampling factors for dim1, dim2 and dim3.
- __dim_ordering__: 'th' or 'tf'.
	In 'th' mode, the channels dimension (the depth)
	is at index 1, in 'tf' mode is it at index 4.

__Input shape__

5D tensor with shape:
`(samples, channels, dim1, dim2, dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, dim1, dim2, dim3, channels)` if dim_ordering='tf'.

__Output shape__

5D tensor with shape:
`(samples, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)` if dim_ordering='th'
or 5D tensor with shape:
`(samples, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)` if dim_ordering='tf'.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1112)</span>
### ZeroPadding1D

```python
keras.layers.convolutional.ZeroPadding1D(padding=1)
```

Zero-padding layer for 1D input (e.g. temporal sequence).

__Arguments__

- __padding__: int
	How many zeros to add at the beginning and end of
	the padding dimension (axis 1).

__Input shape__

3D tensor with shape (samples, axis_to_pad, features)

__Output shape__

3D tensor with shape (samples, padded_axis, features)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1147)</span>
### ZeroPadding2D

```python
keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th')
```

Zero-padding layer for 2D input (e.g. picture).

__Arguments__

- __padding__: tuple of int (length 2)
	How many zeros to add at the beginning and end of
	the 2 padding dimensions (axis 3 and 4).

__Input shape__

4D tensor with shape:
(samples, depth, first_axis_to_pad, second_axis_to_pad)

__Output shape__

4D tensor with shape:
(samples, depth, first_padded_axis, second_padded_axis)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1199)</span>
### ZeroPadding3D

```python
keras.layers.convolutional.ZeroPadding3D(padding=(1, 1, 1), dim_ordering='th')
```

Zero-padding layer for 3D data (spatial or spatio-temporal).

- __Note__: this layer will only work with Theano for the time being.

__Arguments__

- __padding__: tuple of int (length 3)
	How many zeros to add at the beginning and end of
	the 3 padding dimensions (axis 3, 4 and 5).

__Input shape__

5D tensor with shape:
(samples, depth, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)

__Output shape__

5D tensor with shape:
(samples, depth, first_padded_axis, second_padded_axis, third_axis_to_pad)
