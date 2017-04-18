<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L233)</span>
### Conv1D

```python
keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

1次元入力の近傍をフィルターする畳み込み層．  

`use_bias`をTrueにすると，バイアスベクトルが出力に加えられます．`activation`が`None`でない場合，指定した活性化関数が出力に適用されます．

このレイヤーを第一層に使う場合，キーワード引数として`input_shape`（整数のタプルか`None`．例えば10個の128次元ベクトルの場合ならば`(10, 128)`，任意個数の128次元ベクトルの場合は`(None, 128)`）を指定してください．

__Arguments__

- __filters__: 使用するカーネルの数（出力の次元）．
- __kernel_size__: それぞれのフィルターの（空間もしくは時間的な）長さ．
- __strides__: カーネルのストライドを指定します.
- __padding__: `"valid"`，`"same"`，`casual`のいずれか．
- __init__: レイヤーの重みの初期化関数の名前（[initializations](../initializations.md)参照），
	もしくは重み初期化に用いるTheano関数．このパラメータは`weights`引数を与えない場合にのみ有効です．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)参照），
	もしくは要素ごとのTheano関数．
	もしなにも指定しなければ活性化は一切適用されません（つまり"線形"活性a(x) = x）．
- __dilation_rate__: 膨張率．整数か単一の整数からなるタプル/リストを指定します．現在，`dilation_rate` value != 1 とすると，strides value != 1を指定することはできません．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)参照），
  何も指定しなければ，活性化は一切適用されません（つまり"線形"活性a(x) = x）．
- __use_bias__: バイアスベクトルを加えるかどうかを指定します．
- __kernel_initializer__: カーネルの重み行列の初期値を指定します．（[initializers](../initializations.md)参照）
- __bias_initializer__: バイアスベクトルの初期値を指定します．（[initializers](../initializations.md)参照）
- __kernel_regularizer__: カーネルの重みに適用させるRegularizerを指定します．（[regularizer](../regularizers.md)参照）
- __bias_regularizer__: バイアスに適用させるRegularizerを指定します．（[regularizer](../regularizers.md)参照）
- __activity_regularizer__: 出力テンソルに適用させるRegularizerを指定します．（[regularizer](../regularizers.md)参照）
- __kernel_constraint__: カーネルの行列に適用させるConstraintを指定します．（[constraint](../constraints.md)参照）
- __bias_constraint__: バイアスベクトルに適用させるConstraintを指定します．（[constraint](../constraints.md)参照）

__Input shape__

配列サイズ`(batch_size, steps, input_dim)`の3次元テンソル．

__Output shape__

配列サイズ`(batch_size, new_steps, nb_filter)`の3次元テンソル．
`steps`はパディングにより変わっている可能性があります．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L340)</span>
### Conv2D

```python
keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2次元入力をフィルターする畳み込み層．  

`use_bias`をTrueにすると，バイアスベクトルが出力に加えられます．`activation`が`None`でない場合，指定した活性化関数が出力に適用されます．

このレイヤーをモデルの第一層に使うときはキーワード引数`input_shape`
（整数のタプル，サンプル軸を含まない）を指定してください．
例えば，`data_format="channels_last"`のとき，128x128 RGB画像では`input_shape=(128, 128, 3)`となります．

__Arguments__

- __filters__: 使用するカーネルの数．
- __kernel_size__: 畳み込みカーネルの幅と高さを指定します. タプル/リストでカーネルの幅と高さをそれぞれ指定でき，整数の場合は正方形のカーネルになります．
- __strides__: カーネルのストライドを指定します. 二つの整数からなるタプル/リストで縦と横のストライドをそれぞれ指定でき，整数の場合は幅と高さが同様のストライドになります．
- __padding__: `"valid"`か`"same"`のどちらかを指定します．
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力の型は`"(batch, height, width, channels)"`となり，`"channels_first"`の場合は`"(batch, channels, height, width)"`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．
- __dilation_rate__: 膨張率．整数か二つの整数からなるタプル/リストを指定します．単一の整数の場合，それぞれの次元に同一の値が適用されます．現在，`dilation_rate` value != 1 とすると，strides value != 1を指定することはできません．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)参照），
  何も指定しなければ，活性化は一切適用されません（つまり"線形"活性a(x) = x）．
- __use_bias__: 真偽値で，バイアスベクトルを加えるかどうかを指定します．
- __kernel_initializer__: カーネルの重み行列の初期値を指定します．（[initializers](../initializations.md)参照）
- __bias_initializer__: バイアスベクトルの初期値を指定します．（[initializers](../initializations.md)参照）
- __kernel_regularizer__: カーネルの重み行列に適用させるRegularizerを指定します．（[regularizer](../regularizers.md)参照）
- __bias_regularizer__: バイアスベクトルに適用させるRegularizerを指定します．（[regularizer](../regularizers.md)参照）
- __activity_regularizer__: 出力テンソルに適用させるRegularizerを指定します．（[regularizer](../regularizers.md)参照）
- __kernel_constraint__: カーネルの行列に適用させるConstraintを指定します．（[constraint](../constraints.md)参照）
- __bias_constraint__: バイアスベクトルに適用させるConstraintを指定します．（[constraint](../constraints.md)参照）

__Input shape__

data_format='channels_first'の場合，
`(batch_size, channels, rows, cols)`の4次元テンソル．
data_format='channels_last'の場合，
`(batch_size, rows, cols, channels)`の4次元テンソルになります．

__Output shape__

data_format='channels_first'の場合，
`(samples, channels, rows, cols)`の4次元テンソル．
data_format='channels_last'の場合，
`(samples, rows, cols, channels)`の4次元テンソルになります．`rows`と`cols`値はパディングにより変わっている可能性があります．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L806)</span>
### SeparableConv2D

```python
keras.layers.convolutional.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```
Depthwiseな二次元separable畳み込み層．

separable畳み込み演算は，depthwiseの空間的な畳み込み（各入力チャネルに別々に作用する）を実行し，続いてpointwiseに畳み込みを行い，両者の出力チャネルを混合します．`depth_multiplier`は，出力チャネルを生成するための入力チャネルの数を指定します．

separable畳み込み演算はひとつのカーネルをふたつの小さなカーネルに分解する方法として，直感的に理解することができます．もしくはInception blockの極端な例として考えることもできます．

__Arguments__

- __filters__: 使用するカーネルの数．
- __kernel_size__: 畳み込みカーネルの幅と高さを指定します. タプル/リストで長方形のカーネルを指定でき，一つの整数の場合は正方形のカーネルになります．
- __strides__: カーネルのストライドを指定します. 二つの整数からなるタプル/リストで縦と横のストライドをそれぞれ指定でき，一つの整数の場合は幅と高さで同様のストライドになります．
- __padding__: `"valid"`か`"same"`のどちらかを指定します．
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力の型は`"(batch, height, width, channels)"`となり，`"channels_first"`の場合は`"(batch, channels, height, width)"`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．
- __depth_multiplier__: 各入力チャネルに対するdepthwiseな畳み込みチャネルの数．深さ方向畳み込みチャネルの出力総数は，`filters_in * depth_multiplier`に等しくなります．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)参照），
  何も指定しなければ，活性化は一切適用されません（つまり"線形"活性a(x) = x）．
- __use_bias__: 真偽値で，バイアスベクトルを加えるかどうかを指定します．
- __depthwise_initializer__: カーネルの重み行列の初期値をdepthwiseに指定します．（[initializers](../initializations.md)参照）
- __pointwise_initializer__: カーネルの重み行列の初期値をpointwiseに指定します．（[initializers](../initializations.md)参照）
- __bias_initializer__: バイアスベクトルの初期値を指定します．（[initializers](../initializations.md)参照）
- __depthwise_regularizer__: 重み行列に対し，"depthwise"に適用させるRegularizerを指定します．（[ regularizer](../regularizers.md)参照）
- __pointwise_regularizer__: 重み行列に対し，"pointwise"に適用させるRegularizerを指定します．（[ regularizer](../regularizers.md)参照）
- __bias_regularizer__: バイアスベクトルに適用させるRegularizerを指定します．（[regularizer](../regularizers.md)参照）
- __activity_regularizer__: 出力テンソルに適用させるRegularizerを指定します．（[regularizer](../regularizers.md)参照）
- __depthwise_constraint__: カーネル行列に対し，"depthwise"に適用させるConstraintを指定します．（[constraint](../constraints.md)参照）
- __depthwise_constraint__: カーネル行列に対し，"pointwise"に適用させるConstraintを指定します．（[constraint](../constraints.md)参照）
- __bias_constraint__: バイアスベクトルに適用させるConstraintを指定します．（[constraint](../constraints.md)参照）

__Input shape__

data_format='channels_first'の場合，
`(batch_size, channels, rows, cols)`の4次元テンソル．
data_format='channels_last'の場合，
`(batch_size, rows, cols, channels)`の4次元テンソルになります．

__Output shape__

data_format='channels_first'の場合，
`(batch, channels, rows, cols)`の4次元テンソル．
data_format='channels_last'の場合，
`(batch, rows, cols, channels)`の4次元テンソルになります．`rows`と`cols`値はパディングにより変わっている可能性があります．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L589)</span>
### Conv2DTranspose

```python
keras.layers.convolutional.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2次元入力のためのtransposed畳み込みレイヤー．

一般的に，transposed畳み込み演算は通常の畳み込みに対して逆の演算を行いたい時に使われます．つまり，なんらかの畳み込み演算の出力を，接続パターンを保ちながら入力の形に変換する層です．

このレイヤーをモデルの第一層に使うときはキーワード引数`input_shape`
（整数のタプル，サンプル軸を含まない）を指定してください．
例えば`data_format="channels_last"`のとき，128x128 RGB画像では`input_shape=(3, 128, 128)`となります．

__Arguments__

- __filters__: 使用するカーネルの数．
- __kernel_size__: 畳み込みカーネルの幅と高さを指定します．タプル/リストでカーネルの幅と高さをそれぞれ指定でき，一つの整数の場合は正方形のカーネルになります．
- __strides__: カーネルのストライドを指定します. 二つの整数からなるタプル/リストで縦と横のストライドをそれぞれ指定でき，一つの整数の場合は幅と高さで同様のストライドになります．
- __padding__: `"valid"`か`"same"`のどちらかを指定します．
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力の型は`"(batch, height, width, channels)"`となり，`"channels_first"`の場合は`"(batch, channels, height, width)"`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．
- __dilation_rate__: 膨張率．整数か二つの整数からなるタプル/リストを指定します．単一の整数の場合，それぞれの次元に同一の値が適用されます．現在，`dilation_rate` value != 1 とすると，strides value != 1を指定することはできません．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)参照），
  何も指定しなければ，活性化は一切適用されません（つまり"線形"活性a(x) = x）．
- __use_bias__: 真偽値で，バイアスベクトルを加えるかどうかを指定します．
- __kernel_initializer__: カーネルの重み行列の初期値を指定します．（[initializers](../initializations.md)参照）
- __bias_initializer__: バイアスベクトルの初期値を指定します．（[initializers](../initializations.md)参照）
- __kernel_regularizer__: カーネルの重み行列に適用させるRegularizer関数を指定します．（[ regularizer](../regularizers.md)参照）
- __bias_regularizer__: バイアスベクトルに適用させるRegularizer関数を指定します．（[ regularizer](../regularizers.md)参照）
- __activity_regularizer__: 出力テンソルに適用させるRegularizer関数を指定します．（[ regularizer](../regularizers.md)参照）
- __kernel_constraint__: カーネルの行列に適用させるConstraint関数を指定します．（[Constraint](../constraints.md)参照）
- __bias_constraint__: バイアスベクトルに適用させるConstraint関数を指定します．（[Constraint](../constraints.md)参照）

__Input shape__

data_format='channels_first'の場合，
`(batch, channels, rows, cols)`の4次元テンソル．
data_format='channels_last'の場合，
`(batch, rows, cols, channels)`の4次元テンソルになります．

__Output shape__

data_format='channels_first'の場合，
`(batch, filters, new_rows, new_cols)`の4次元テンソル．
data_format='channels_last'の場合，
`(batch, new_rows, new_cols, channels)`の4次元テンソルになります．`rows`と`cols`値はパディングにより変わっている可能性があります．

__References__

[1] [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285 "arXiv:1603.07285v1 [stat.ML]")  
[2] [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L464)</span>
### Conv3D

```python
keras.layers.convolutional.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

3次元入力をフィルターする畳み込み演算．  


このレイヤーをモデルの第一層に使うときはキーワード引数`input_shape`
（整数のタプル，サンプル軸を含まない）を指定してください．
例えば10フレームの128x128 RGB画像では`input_shape=(3, 10, 128, 128)`．

__Arguments__

- __filters__: 使用する畳み込みカーネルの数です．
- __kernel_size__: 畳み込みカーネルの幅と高さを指定します. タプル/リストで長方形のカーネルを指定でき，一つの整数の場合は正方形のカーネルになります．
- __strides__: カーネルのストライドを指定します. 二つの整数からなるタプル/リストで縦と横のストライドを独立に指定でき，一つの整数の場合は縦横で同様のストライドになります．
- __padding__: `"valid"`か`"same"`のどちらかを指定します．
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します．   
`"channels_last"`の場合，入力は`"(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)"`．  
`"channels_first"`の場合は`"(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)"`となります．  
デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．
- __dilation_rate__: 膨張率．整数か三つの整数からなるタプル/リストを指定します．単一の整数の場合，それぞれの次元に同一の値が適用されます．現在，`dilation_rate` value != 1 とすると，strides value != 1を指定することはできません．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)参照），
  何も指定しなければ，活性化は一切適用されません（つまり"線形"活性a(x) = x）．
- __use_bias__: 真偽値で，バイアスベクトルを加えるかどうかを指定します．
- __kernel_initializer__: カーネルの重み行列の初期値を指定します．（[initializers](../initializations.md)参照）
- __bias_initializer__: バイアスベクトルの初期値を指定します．（[initializers](../initializations.md)参照）
- __kernel_regularizer__: カーネルの重み行列に適用させるRegularizer関数を指定します．（[ regularizer](../regularizers.md)参照）
- __bias_regularizer__: バイアスベクトルに適用させるRegularizer関数を指定します．（[ regularizer](../regularizers.md)参照）
- __activity_regularizer__: 出力テンソルに適用させるRegularizer関数を指定します．（[ regularizer](../regularizers.md)参照）
- __kernel_constraint__: カーネルの行列に適用させるConstraint関数を指定します．（[Constraint](../constraints.md)参照）
- __bias_constraint__: バイアスベクトルに適用させるConstraint関数を指定します．（[Constraint](../constraints.md)参照）

__Input shape__

data_format='channels_first'の場合，
`(samples, channels, conv_dim1, conv_dim2, conv_dim3)`の5次元テンソル．
data_format='channels_last'の場合，
`(samples, conv_dim1, conv_dim2, conv_dim3, channels)`の5次元テンソルになります．

__Output shape__

data_format='channels_first'の場合，
`(samples, channels, conv_dim1, conv_dim2, conv_dim3)`の5次元テンソル．
data_format='channels_last'の場合，
`(samples, conv_dim1, conv_dim2, conv_dim3, channels)`の5次元テンソルになります．`conv_dim1`，`conv_dim2`，`conv_dim3`の値はパディングにより変わっている可能性があります．

----
<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1480)</span>
### Cropping1D

```python
keras.layers.convolutional.Cropping1D(cropping=(1, 1))
```

一次元の入力をクロップする（切り落とす）層．  
クロップは時間軸に沿って実行されます（axis 1）．

__Arguments__

- __cropping__: クロップしたいユニットの数を指定します．二つの整数からなるタプルで指定した場合は，それぞれ両側からクロップします．一つの整数の場合は，両側から同数のユニットをクロップします．

__Input Shape__

`(batch, axis_to_crop, features)`の三次元テンソル．

__Output Shape__

`(batch, cropped_axis, features)`の三次元テンソル．

----
<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1525)</span>
### Cropping2D

```python
keras.layers.convolutional.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
```

二次元の入力をクロップする（切り落とす）層．  
クロップは幅と高さに対して実行されます．

__Arguments__

- __cropping__: 整数, タプル（二つの整数）, タプル（二つの整数）のタプルのいずれか．
	- 整数: 幅と高さに対称なクロップが実行されます．
	- タプル（二つの整数）: 幅と高さでそれぞれ対称なクロップが実行されます．  
	`(symmetric_height_crop, symmetric_width_crop)`
	- タプルのタプル: 四辺それぞれにクロップが実行されます．  
	`(top_crop, bottom_crop), (left_crop, right_crop))`
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します.   
`"channels_last"`の場合，入力は`"(batch, height, width, channels)"`．  
`"channels_first"`の場合は`"(batch, channels, height, width)"`となります．  
デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__Input Shape__

`data_format`が`"channels_last"`の場合，`(batch, rows, cols, channels)`．  
`"channels_first"`の場合，`(batch, channels, rows, cols)`の四次元テンソル．

__Output Shape__

`data_format`が`"channels_last"`の場合，`(batch, cropped_rows, cropped_cols, channels)`．
`"channels_first"`の場合，`(batch, channels, cropped_rows, cropped_cols)`の四次元テンソル．

----
<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1668)</span>
### Cropping3D

```python
keras.layers.convolutional.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), data_format=None)
```

三次元の入力をクロップする（切り落とす）層．  

__Arguments__

- __cropping__: 整数, タプル（二つの整数），タプル（二つの整数）のタプル，のいずれか．
	- 整数: 三つの軸に対して対称なクロップが実行されます．
	- タプル（三つの整数）: 三つの軸に対して，それぞれ対称なクロップが実行されます．  
	`(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`
	- タプルのタプル: 六面それぞれにクロップが実行されます．  
	`((left_dim1_crop, right_dim1_crop), (left_dim2_crop, right_dim2_crop), (left_dim3_crop, right_dim3_crop))`
	- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します．   
	`"channels_last"`の場合，入力は`"(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)"`．`"channels_first"`の場合は`"(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)"`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__Input Shape__


`data_format`が`"channels_last"`の場合，`(batch, first_cropped_axis, second_cropped_axis, third_cropped_axis, depth)`．`"channels_first"`の場合，`(batch, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)`の五次元テンソル．

__Output Shape__

`data_format`が`"channels_last"`の場合，`(batch, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop, depth)`． `"channels_first"`の場合，`(batch, depth, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)`の五次元テンソル．

----
<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1032)</span>
### UpSampling1D

```python
keras.layers.convolutional.UpSampling1D(size=2)
```

時間軸方向にそれぞれの時間ステップを`size`回繰り返します．

__Arguments__

- __size__: 整数．upsampling係数．

__Input shape__

`(batch, steps, features)`の3次元テンソル．

__Output shape__

`(batch, upsampled_steps, features)`の3次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1067)</span>
### UpSampling2D

```python
keras.layers.convolutional.UpSampling2D(size=(2, 2), data_format=None)
```

データの行と列をそれぞれsize[0]及びsize[1]回繰り返します．

__Arguments__

- __size__: 整数か2つの整数のタプル．行と列のupsampling係数．
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力の型は`"(batch, height, width, channels)"`となり，`"channels_first"`の場合は`"(batch, channels, height, width)"`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__Input shape__

data_format='channels_last'の場合，
`(batch, rows, cols, channels)`の4次元テンソル．
data_format='channels_first'の場合，
`(batch, channels, rows, cols)`の4次元テンソルになります．

__Output shape__

data_format='channels_first'の場合，
`(batch, channels, upsampled_rows, upsampled_cols)`の4次元テンソル．
data_format='channels_last'の場合，
`(batch, upsampled_rows, upsampled_cols, channels)`の4次元テンソルになります．
----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1136)</span>
### UpSampling3D

```python
keras.layers.convolutional.UpSampling3D(size=(2, 2, 2), data_format=None)
```

データの1番目，2番目，3番目の次元をそれぞれsize[0]，size[1]，size[2]だけ繰り返す．


__Arguments__

- __size__: 3つの整数のタプル．dim1，dim2，dim3のアップサンプリング係数．
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力の型は`"(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)"`となり，`"channels_first"`の場合は`"(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)"`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__Input shape__

data_format='channels_last'の場合，
`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`の5次元テンソル．
data_format='channels_first'の場合，
`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`の5次元テンソルになります．


__Output shape__

data_format='channels_last'の場合，
`(batch, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`の5次元テンソル．
data_format='channels_first'の場合，
`(batch, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`の5次元テンソルになります．
----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1210)</span>
### ZeroPadding1D

```python
keras.layers.convolutional.ZeroPadding1D(padding=1)
```

一次元入力（例，時系列）に対するゼロパディングレイヤー．

__Arguments__

- __padding__: 整数，タプル（二つの整数）のいずれか．
	- 整数: パディング次元（axis 1）の始めと終わりにいくつのゼロを加えるか．
	- タプル: 始めと終わりにそれぞれいくつのゼロを加えるか．  
	`(left_pad, right_pad)`


__Input shape__

`(batch, axis_to_pad, features)`の3次元テンソル．

__Output shape__

`(batch, padded_axis, features)`の3次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1252)</span>
### ZeroPadding2D

```python
keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format=None)
```

2次元入力（例，画像）のためのゼロパディングレイヤー

__Arguments__

- __padding__: 整数，タプル（二つの整数），タプル（二つの整数）のタプル．
	- 整数: 幅と高さにたいして対称なパディングを実行する．
	- タプル: 幅と高さ，それぞれに対して対称なパディングを実行する．  
	`(symmetric_height_pad, symmetric_width_pad)`
	- タプルのタプル:　四辺それぞれにパディング．  
	`((top_pad, bottom_pad), (left_pad, right_pad))`
	- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します.   
	`"channels_last"`の場合，入力は`"(batch, height, width, channels)"`．  
	`"channels_first"`の場合は`"(batch, channels, height, width)"`となります．  
	デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

	__Input Shape__

	`data_format`が`"channels_last"`の場合，`(batch, rows, cols, channels)`．  
	`"channels_first"`の場合，`(batch, channels, rows, cols)`の四次元テンソル．

	__Output Shape__

	`data_format`が`"channels_last"`の場合，`(batch, padded_rows, padded_cols, channels)`．
	`"channels_first"`の場合，`(batch, channels, padded_rows, padded_cols)`の四次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L1362)</span>
### ZeroPadding3D

```python
keras.layers.convolutional.ZeroPadding3D(padding=(1, 1, 1), data_format=None)
```

3次元データ（空間及び時空間）のためのゼロパディングレイヤー．

__Arguments__

- __cropping__: 整数, タプル（二つの整数），タプル（二つの整数）のタプル，のいずれか．
	- 整数: 三つの軸に対して対称なパディングが実行されます．
	- タプル（三つの整数）: 三つの軸に対して，それぞれ対称なパディングが実行されます．  
	`(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`
	- タプルのタプル: 六面それぞれにパディングが実行されます．
	`((left_dim1_pad, right_dim1_pad), (left_dim2_pad, right_dim2_pad), (left_dim3_pad, right_dim3_pad))`
	- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します．   
	`"channels_last"`の場合，入力は`"(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)"`．`"channels_first"`の場合は`"(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)"`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__Input Shape__


`data_format`が`"channels_last"`の場合，`(batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad, depth)`．`"channels_first"`の場合，`(batch, depth, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)`の五次元テンソル．

__Output Shape__

`data_format`が`"channels_last"`の場合，`(batch, first_padded_axis, second_padded_axis, third_axis_to_pad, depth)`． `"channels_first"`の場合，`(batch, depth, first_padded_axis, second_padded_axis, third_axis_to_pad)`の五次元テンソル．
