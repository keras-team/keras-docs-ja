<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L20)</span>
### Convolution1D

```python
keras.layers.convolutional.Convolution1D(nb_filter, filter_length, init='uniform', activation='linear', weights=None, border_mode='valid', subsample_length=1, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
```

１次元入力の近傍をフィルターする畳み込み演算．このレイヤーを第一層に使う場合，キーワード引数として`input_dim`（整数値，例えば128次元ベクトル系列には1128）を指定するか`input_shape`（整数のタプル，例えば10個の128次元ベクトル系列のでは(10, 128)）を指定してください．

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
	もしくは重み初期化に用いるTheano関数．このパラメータは`weights`引数を与えない場合にのみ有効です．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)参照），
	もしくは要素ごとのTheano関数．
	もしなにも指定しなければ活性化は一切適用されません（つまり"線形"活性a(x) = x）．
- __weights__: 初期重みとして設定されるnumpy配列のリスト．
- __border_mode__: 'valid' あるいは 'same'．
- __subsample_length__: 出力を部分サンプルするときの長さ．　<!--check-->
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
	この引数は上流の`Flatten`そして`Dense`レイヤーを繋ぐときに必要となります．　<!--check-->
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

２次元入力をフィルターする畳み込み演算子．
このレイヤーをモデルの第一層に使うときはキーワード引数`input_shape`
（整数のタプル，サンプル軸を含まない）を指定してください．
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

- __nb_filter__: 使用する畳み込みカーネルの数.
- __nb_row__: 畳み込みカーネルの行数．
- __nb_col__: 畳み込みカーネルの列数．
- __init__: レイヤーの重みの初期化関数の名前
	（[initializations](../initializations.md)参照），
	もしくは重み初期化に用いるTheano関数．
	このパラメータは`weights`引数を与えない場合にのみ有効です．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)参照），
	もしくは要素ごとのTheano関数．
	もしなにも指定しなければ活性化は一切適用されません（つまり"線形"活性a(x) = x）．
- __weights__: 初期重みとして設定されるnumpy配列のリスト．
- __border_mode__:  'valid' あるいは 'same'．
- __subsample__: 長さ２のタプル．出力を部分サンプルするときの長さ．
	別の場所ではstrideとも呼ぶ.
- __W_regularizer__: メインの重み行列に適用される[WeightRegularizer](../regularizers.md)
	（例えばL1やL2正則化）のインスタンス．
- __b_regularizer__: バイアス項に適用される[WeightRegularizer](../regularizers.md)のインスタンス．
- __activity_regularizer__: ネットワーク出力に適用される[ActivityRegularizer](../regularizers.md)のインスタンス．
- __W_constraint__: メインの重み行列に適用される[constraints](../constraints.md)モジュール（例えばmaxnorm, nonneg）のインスタンス．
- __b_constraint__: バイアス項に適用される[constraints](../constraints.md)モジュールのインスタンス．
- __dim_ordering__: 'th'か'tf'．'th'モードのときはチャネルの次元（深さ）はindex 1に，
	'tf'モードではindex 3に．
- __bias__: バイアス項を含むかどうか（レイヤをアフィンにするか線形にするか）．

__Input shape__

dim_ordering='th'の場合，配列サイズ
`(samples, channels, rows, cols)`の４次元テンソル，あるいは
dim_ordering='tf'の場合，配列サイズ
`(samples, rows, cols, channels)`の４次元テンソル

__Output shape__

dim_ordering='th'の場合，配列サイズ`(samples, nb_filter, new_rows, new_cols)`の４次元テンソル，
あるいは
dim_ordering='tf'の場合，配列サイズ`(samples, new_rows, new_cols, nb_filter)`の４次元テンソル
`rows`と`cols`値はパディングにより変っている可能性あり．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L383)</span>
### Convolution3D

```python
keras.layers.convolutional.Convolution3D(nb_filter, kernel_dim1, kernel_dim2, kernel_dim3, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1, 1), dim_ordering='th', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```

３次元入力をフィルターする畳み込み演算子．
このレイヤーをモデルの第一層に使うときはキーワード引数`input_shape`
（整数のタプル，サンプル軸を含まない）を指定してください．
例えば10フレームの128x128 RGBのピクチャーでは`input_shape=(3, 10, 128, 128)`．

- __Note__: このレイヤーは現時点ではTheanoにしか働きません．

__Arguments__

- __nb_filter__: 使用する畳み込みカーネルの数.
- __kernel_dim1__: 畳み込みカーネルの１次元目の長さ．
- __kernel_dim2__: 畳み込みカーネルの２次元目の長さ．
- __kernel_dim3__: 畳み込みカーネルの３次元目の長さ．
- __init__: レイヤーの重みの初期化関数の名前
	（[initializations](../initializations.md)参照），
	もしくは重み初期化に用いるTheano関数．
	このパラメータは`weights`引数を与えない場合にのみ有効です．
- __activation__: 使用する活性化関数の名前
	（[activations](../activations.md)参照），もしくは要素ごとのTheano関数．
	もしなにも指定しなければ活性化は一切適用されません（つまり"線形"活性a(x) = x）．
- __weights__: 初期重みとして設定されるnumpy配列のリスト．
- __border_mode__: 'valid'か'same'．
- __subsample__: 長さ３のタプル．出力を部分サンプルするときの長さ．
	別の場所ではstrideとも呼ぶ.
	- __Note__: 'subsample'はconv3dの出力をstrides=(1,1,1)でスライスすることで実装されている．
- __W_regularizer__: メインの重み行列に適用される[WeightRegularizer](../regularizers.md)
	（例えばL1やL2正則化）のインスタンス．
- __b_regularizer__: バイアス項に適用される[WeightRegularizer](../regularizers.md)のインスタンス．
- __activity_regularizer__: ネットワーク出力に適用される[ActivityRegularizer](../regularizers.md)のインスタンス．
- __W_constraint__: メインの重み行列に適用される[constraints](../constraints.md)モジュール（例えばmaxnorm, nonneg）のインスタンス．
- __b_constraint__: バイアス項に適用される[constraints](../constraints.md)モジュールのインスタンス．
- __dim_ordering__: 'th'か'tf'．'th'モードのときはチャネルの次元（深さ）はindex 1に，
	'tf'モードではindex 4に．
- __bias__: バイアス項を含むかどうか（レイヤをアフィンにするか線形にするか）．

__Input shape__

dim_ordering='th'の場合，配列サイズ
`(samples, channels, conv_dim1, conv_dim2, conv_dim3)`の５次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, conv_dim1, conv_dim2, conv_dim3, channels)`の５次元テンソル．

__Output shape__

dim_ordering='th'の場合配列サイズ
`(samples, nb_filter, new_conv_dim1, new_conv_dim2, new_conv_dim3)`の５次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, nb_filter)`の５次元テンソル．
`new_conv_dim1`， `new_conv_dim2`，及び `new_conv_dim3`値はパディングにより変わっている可能性あり．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L628)</span>
### MaxPooling1D

```python
keras.layers.convolutional.MaxPooling1D(pool_length=2, stride=None, border_mode='valid')
```

時系列データのマックスプーリング演算子．

__Input shape__

配列サイズ `(samples, steps, features)`の３次元テンソル．

__Output shape__

配列サイズ `(samples, downsampled_steps, features)`の３次元テンソル．

__Arguments__

- __pool_length__: ダウンスケールする係数．２は入力を半分にする．
- __stride__: 整数もしくはNone．Stride値．
- __border_mode__: 'valid'か'same'.
	- __Note__: 現時点では'same'はTensorFlowでのみ動きます．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L743)</span>
### MaxPooling2D

```python
keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th')
```

空間データのマックスプーリング演算子．

__Arguments__

- __pool_size__: ダウンスケールする係数を決める
	２つの整数のタプル（垂直，水平）．
	(2, 2) は画像をそれぞれの次元で半分にします．
- __strides__: ２つの整数のタプルもしくはNone．Strides値．
- __border_mode__: 'valid'か'same'.
	- __Note__: 現時点では'same'はTensorFlowでのみ動きます．
- __dim_ordering__:'th'か'tf'.
	'th'モードのときはチャネルの次元（深さ）はindex 1に，
	'tf'モードではindex 3に．

__Input shape__

dim_ordering='th'の場合，`(samples, channels, rows, cols)`の４次元テンソル，
もしくは
dim_ordering='tf'の場合，`(samples, rows, cols, channels)`の４次元テンソル，

__Output shape__

dim_ordering='th'の場合，4D tensor with shape:
`(nb_samples, channels, pooled_rows, pooled_cols)`の４次元テンソル，
もしくは
dim_ordering='tf'の場合，`(samples, pooled_rows, pooled_cols, channels)`の４次元テンソル，

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L882)</span>
### MaxPooling3D

```python
keras.layers.convolutional.MaxPooling3D(pool_size=(2, 2, 2), strides=None, border_mode='valid', dim_ordering='th')
```

３次元データ（空間もしくは時空間）に対するマクスプーリング演算子

- __Note__: 現時点ではこのレイヤーは'Theano'でのみ動きます．

__Arguments__

- __pool_size__: ３つの整数のタプル(dim1, dim2, dim3)，
	ダウンスケールするための係数．
	(2, 2, 2)は３次元入力のサイズをそれぞれの次元で半分にします．
- __strides__: ３つの整数のタプルもしくはNone．Strides値．
- __border_mode__: 'valid'か'same'.
- __dim_ordering__: 'th'か'tf'.
	'th'モードのときはチャネルの次元（深さ）はindex 1に，
	'tf'モードではindex 4に．

__Input shape__

dim_ordering='th'の場合，配列サイズ
`(samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3)` の５次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, len_pool_dim1, len_pool_dim2, len_pool_dim3, channels)`の５次元テンソル．

__Output shape__

dim_ordering='th'の場合，配列サイズ
`(nb_samples, channels, pooled_dim1, pooled_dim2, pooled_dim3)`の５次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, pooled_dim1, pooled_dim2, pooled_dim3, channels)`の５次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L656)</span>
### AveragePooling1D

```python
keras.layers.convolutional.AveragePooling1D(pool_length=2, stride=None, border_mode='valid')
```

時系列データのための平均プーリング演算子．

__Arguments__

- __pool_length__: ダウンスケールする係数．２は入力を半分にします．
- __stride__: 整数もしくはNone．Stride値．
- __border_mode__: 'valid'か'same'.
	- __Note__: 現時点では'same'はTensorFlowでのみ動きます．

__Input shape__

配列サイズ`(samples, steps, features)`の３次元テンソル．

__Output shape__

配列サイズ`(samples, downsampled_steps, features)`の３次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L781)</span>
### AveragePooling2D

```python
keras.layers.convolutional.AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th')
```

空間データのための平均プーリング演算子．

__Arguments__

- __pool_size__: ダウンスケールする係数を決める
	２つの整数のタプル（垂直，水平）．
	(2, 2) は画像をそれぞれの次元で半分にします．
- __strides__: ２つの整数のタプルもしくはNone．Strides値．
- __border_mode__: 'valid'か'same'.
	- __Note__: 現時点では'same'はTensorFlowでのみ動きます．
- __dim_ordering__: 'th'か'tf'.
	'th'モードのときはチャネルの次元（深さ）はindex 1に，
	'tf'モードではindex 3に．

__Input shape__

dim_ordering='th'の場合，配列サイズ
`(samples, channels, rows, cols)`の４次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, rows, cols, channels)`の４次元テンソル．

__Output shape__

dim_ordering='th'の場合，配列サイズ
`(nb_samples, channels, pooled_rows, pooled_cols)` の４次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, pooled_rows, pooled_cols, channels)`の４次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L924)</span>
### AveragePooling3D

```python
keras.layers.convolutional.AveragePooling3D(pool_size=(2, 2, 2), strides=None, border_mode='valid', dim_ordering='th')
```

３次元データ（空間もしくは時空間）に対する平均プーリング演算子．

- __Note__: 現時点ではこのレイヤーは'Theano'でのみ動きます．

__Arguments__

- __pool_size__: ３つの整数のタプル(dim1, dim2, dim3)，
	ダウンスケールするための係数．
	(2, 2, 2)は３次元入力のサイズをそれぞれの次元で半分にします．
- __strides__: ３つの整数のタプルもしくはNone．Strides値．
- __border_mode__: 'valid'か'same'.
- __dim_ordering__: 'th'か'tf'.
	'th'モードのときはチャネルの次元（深さ）はindex 1に，
	'tf'モードではindex 4に．

__Input shape__

dim_ordering='th'の場合，配列サイズ
`(samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3)`の５次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, len_pool_dim1, len_pool_dim2, len_pool_dim3, channels)`の５次元テンソル．

__Output shape__

dim_ordering='th'の場合，配列サイズ
`(nb_samples, channels, pooled_dim1, pooled_dim2, pooled_dim3)`の５次元テンソル
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, pooled_dim1, pooled_dim2, pooled_dim3, channels)`の５次元テンソル

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
