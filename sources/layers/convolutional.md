<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L14)</span>
### Convolution1D

```python
keras.layers.convolutional.Convolution1D(nb_filter, filter_length, init='uniform', activation='linear', weights=None, border_mode='valid', subsample_length=1, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
```

1次元入力の近傍をフィルターする畳み込み演算．このレイヤーを第一層に使う場合，キーワード引数として`input_dim`（整数値，例えば128次元ベクトル系列には128）を指定するか`input_shape`（整数のタプル，例えば10個の128次元ベクトル系列のでは(10, 128)）を指定してください．

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
- __weights__: 初期重みとして設定されるNumpy配列のリスト．
- __border_mode__: 'valid' あるいは 'same'．
- __subsample_length__: 出力を部分サンプルするときの長さ．
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
	この引数は上流の`Flatten`そして`Dense`レイヤーを繋ぐときに必要となります．
	これがないとdense出力の配列サイズを計算することができません．

__Input shape__

配列サイズ`(samples, steps, input_dim)`の3次元テンソル．

__Output shape__

配列サイズ`(samples, new_steps, nb_filter)`の3次元テンソル．
`steps`値はパディングにより変わっている可能性あり．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L186)</span>
### Convolution2D

```python
keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```

2次元入力をフィルターする畳み込み演算．
このレイヤーをモデルの第一層に使うときはキーワード引数`input_shape`
（整数のタプル，サンプル軸を含まない）を指定してください．
例えば128x128 RGB画像では`input_shape=(3, 128, 128)`．

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

- __nb_filter__: 使用する畳み込みカーネルの数．
- __nb_row__: 畳み込みカーネルの行数．
- __nb_col__: 畳み込みカーネルの列数．
- __init__: レイヤーの重みの初期化関数の名前
	（[initializations](../initializations.md)参照），
	もしくは重み初期化に用いるTheano関数．
	このパラメータは`weights`引数を与えない場合にのみ有効です．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)参照），
	もしくは要素ごとのTheano関数．
	もしなにも指定しなければ活性化は一切適用されません（つまり"線形"活性a(x) = x）．
- __weights__: 初期重みとして設定されるNumpy配列のリスト．
- __border_mode__:  'valid' あるいは 'same'．
- __subsample__: 長さ2のタプル．出力を部分サンプルするときの長さ．
	別の場所ではstrideとも呼ぶ．
- __W_regularizer__: メインの重み行列に適用される[WeightRegularizer](../regularizers.md)
	（例えばL1やL2正則化）のインスタンス．
- __b_regularizer__: バイアス項に適用される[WeightRegularizer](../regularizers.md)のインスタンス．
- __activity_regularizer__: ネットワーク出力に適用される[ActivityRegularizer](../regularizers.md)のインスタンス．
- __W_constraint__: メインの重み行列に適用される[constraints](../constraints.md)モジュール（例えばmaxnorm, nonneg）のインスタンス．
- __b_constraint__: バイアス項に適用される[constraints](../constraints.md)モジュールのインスタンス．
- __dim_ordering__: 'th'か'tf'．'th'モードのときはチャネルの次元（深さ）はindex 1に，
  'tf'モードではindex 3に．
  デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_dim_ordering`の値です．値を設定していなければ，"th"になります．
- __bias__: バイアス項を含むかどうか（レイヤをアフィンにするか線形にするか）．

__Input shape__

dim_ordering='th'の場合，配列サイズ
`(samples, channels, rows, cols)`の4次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, rows, cols, channels)`の4次元テンソル

__Output shape__

dim_ordering='th'の場合，配列サイズ`(samples, nb_filter, new_rows, new_cols)`の4次元テンソル，
あるいは
dim_ordering='tf'の場合，配列サイズ`(samples, new_rows, new_cols, nb_filter)`の4次元テンソル
`rows`と`cols`値はパディングにより変わっている可能性あり．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L828)</span>
### Convolution3D

```python
keras.layers.convolutional.Convolution3D(nb_filter, kernel_dim1, kernel_dim2, kernel_dim3, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```

3次元入力をフィルターする畳み込み演算．
このレイヤーをモデルの第一層に使うときはキーワード引数`input_shape`
（整数のタプル，サンプル軸を含まない）を指定してください．
例えば10フレームの128x128 RGB画像では`input_shape=(3, 10, 128, 128)`．

- __Note__: このレイヤーは現時点ではTheanoでのみ動きます．

__Arguments__

- __nb_filter__: 使用する畳み込みカーネルの数．
- __kernel_dim1__: 畳み込みカーネルの1次元目の長さ．
- __kernel_dim2__: 畳み込みカーネルの2次元目の長さ．
- __kernel_dim3__: 畳み込みカーネルの3次元目の長さ．
- __init__: レイヤーの重みの初期化関数の名前
	（[initializations](../initializations.md)参照），
	もしくは重み初期化に用いるTheano関数．
	このパラメータは`weights`引数を与えない場合にのみ有効です．
- __activation__: 使用する活性化関数の名前
	（[activations](../activations.md)参照），もしくは要素ごとのTheano関数．
	もしなにも指定しなければ活性化は一切適用されません（つまり"線形"活性a(x) = x）．
- __weights__: 初期重みとして設定されるNumpy配列のリスト．
- __border_mode__: 'valid'か'same'．
- __subsample__: 長さ3のタプル．出力を部分サンプルするときの長さ．
	別の場所ではstrideとも呼ぶ．
	- __Note__: 'subsample'はconv3dの出力をstrides=(1,1,1)でスライスすることで実装されている．
- __W_regularizer__: メインの重み行列に適用される[WeightRegularizer](../regularizers.md)
	（例えばL1やL2正則化）のインスタンス．
- __b_regularizer__: バイアス項に適用される[WeightRegularizer](../regularizers.md)のインスタンス．
- __activity_regularizer__: ネットワーク出力に適用される[ActivityRegularizer](../regularizers.md)のインスタンス．
- __W_constraint__: メインの重み行列に適用される[constraints](../constraints.md)モジュール（例えばmaxnorm, nonneg）のインスタンス．
- __b_constraint__: バイアス項に適用される[constraints](../constraints.md)モジュールのインスタンス．
- __dim_ordering__: 'th'か'tf'．'th'モードのときはチャネルの次元（深さ）はindex 1に，
  'tf'モードではindex 4に．
  デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_dim_ordering`の値です．値を設定していなければ，"th"になります．
- __bias__: バイアス項を含むかどうか（レイヤをアフィンにするか線形にするか）．

__Input shape__

dim_ordering='th'の場合，配列サイズ
`(samples, channels, conv_dim1, conv_dim2, conv_dim3)`の5次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, conv_dim1, conv_dim2, conv_dim3, channels)`の5次元テンソル．

__Output shape__

dim_ordering='th'の場合配列サイズ
`(samples, nb_filter, new_conv_dim1, new_conv_dim2, new_conv_dim3)`の5次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, nb_filter)`の5次元テンソル．
`new_conv_dim1`， `new_conv_dim2`，及び `new_conv_dim3`値はパディングにより変わっている可能性あり．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1029)</span>
### UpSampling1D

```python
keras.layers.convolutional.UpSampling1D(length=2)
```

時間軸方向にそれぞれの時間ステップを`length`回繰り返す．

__Arguments__

- __length__: 整数．アップサンプリング係数．

__Input shape__

配列サイズ `(samples, steps, features)`の3次元テンソル．

__Output shape__

配列サイズ `(samples, upsampled_steps, features)`の3次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1061)</span>
### UpSampling2D

```python
keras.layers.convolutional.UpSampling2D(size=(2, 2), dim_ordering='default')
```

データの行と列をそれぞれsize[0]及びsize[1]回繰り返す．

__Arguments__

- __size__: 2つの整数のタプル．行と列のアップサンプリング係数．
- __dim_ordering__: 'th'か'tf'．
  'th'モードのときはチャネルの次元（深さ）はindex 1に，
  'tf'モードではindex 3に．
  デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_dim_ordering`の値です．値を設定していなければ，"th"になります．

__Input shape__

4D tensor with shape:
dim_ordering='th'の場合，配列サイズ`(samples, channels, rows, cols)`の4次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, rows, cols, channels)`の4次元テンソル．

__Output shape__

dim_ordering='th'の場合，配列サイズ
`(samples, channels, upsampled_rows, upsampled_cols)`の4次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, upsampled_rows, upsampled_cols, channels)`の4次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1124)</span>
### UpSampling3D

```python
keras.layers.convolutional.UpSampling3D(size=(2, 2, 2), dim_ordering='default')
```

データの1番目，2番目，3番目の次元をそれぞれsize[0]，size[1]，size[2]だけ繰り返す．

- __Note__: 現時点ではこのレイヤーは'Theano'でのみ動きます．

__Arguments__

- __size__: 3つの整数のタプル．dim1，dim2 and dim3のアップサンプリング係数．
- __dim_ordering__: 'th'か'tf'．
  'th'モードのときはチャネルの次元（深さ）はindex 1に，
  'tf'モードではindex 4に．
  デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_dim_ordering`の値です．値を設定していなければ，"th"になります．

__Input shape__

dim_ordering='th'の場合，配列サイズ
`(samples, channels, dim1, dim2, dim3)`の5次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, dim1, dim2, dim3, channels)`の5次元テンソル．

__Output shape__

dim_ordering='th'の場合，配列サイズ
`(samples, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`の5次元テンソル．
もしくはdim_ordering='tf'の場合，配列サイズ
`(samples, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`の5次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1191)</span>
### ZeroPadding1D

```python
keras.layers.convolutional.ZeroPadding1D(padding=1)
```

一次元入力（例，時系列）に対するゼロパディングレイヤー．

__Arguments__

- __padding__: 整数．
	パディング次元（axis 1）の始めと終わりにいくつのゼロを加えるか．

__Input shape__

配列サイズ`(samples, axis_to_pad, features)`の3次元テンソル．

__Output shape__

配列サイズ`(samples, padded_axis, features)`の3次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1226)</span>
### ZeroPadding2D

```python
keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='default')
```

2次元入力（例，画像）のためのゼロパディングレイヤー

__Arguments__

- __padding__: 整数のタプル（長さ2）．
	2つのパディング次元(axis 3 と 4)の始めと終わりにいくつのゼロを加えるか．
デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_dim_ordering`の値です．値を設定していなければ，"th"になります．
- __dim_ordering__: 'th'か'tf'．
  'th'モードのときはチャネルの次元（深さ）はindex 1に，
  'tf'モードではindex 3に．
  デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_dim_ordering`の値です．値を設定していなければ，"th"になります．

__Input shape__

配列サイズ`(samples, depth, first_axis_to_pad, second_axis_to_pad)`の4次元テンソル．

__Output shape__

配列サイズ`(samples, depth, first_padded_axis, second_padded_axis)`の4次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1286)</span>
### ZeroPadding3D

```python
keras.layers.convolutional.ZeroPadding3D(padding=(1, 1, 1), dim_ordering='default')
```

3次元データ（空間及び時空間）のためのゼロパディングレイヤー．

- __Note__: 現時点ではこのレイヤーは'Theano'でのみ動きます．

__Arguments__

- __padding__: 整数のタプル（長さ3）
	3つのパディング次元(axis 3, 4 and 5)の始めと終わりにいくつのゼロを加えるか．
- __dim_ordering__: 'th'か'tf'．
  'th'モードのときはチャネルの次元（深さ）はindex 1に，
  'tf'モードではindex 4に．
  デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_dim_ordering`の値です．値を設定していなければ，"th"になります．

__Input shape__

配列サイズ`(samples, depth, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)`の5次元テンソル．

__Output shape__

配列サイズ`(samples, depth, first_padded_axis, second_padded_axis, third_axis_to_pad)`の5次元テンソル．
