<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L237)</span>
### Conv1D

```python
keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
1次元の畳み込みレイヤー（例えば時間的な畳込み）．

このレイヤーは畳み込みカーネルを生成します．これはレイヤーの入力を単一の空間的（または時間的）次元で畳み込んで，出力のテンソルを作ります．
`use_bias`をTrueにすると，バイアスベクトルが出力に加えられます．`activation`が`None`でない場合，指定した活性化関数が出力に適用されます．

このレイヤーを第一層に使う場合，キーワード引数として`input_shape`（整数のタプルか`None`．例えば10個の128次元ベクトルの場合ならば`(10, 128)`，任意個数の128次元ベクトルの場合は`(None, 128)`）を指定してください．

__引数__

- __filters__: 整数，出力空間の次元（つまり畳み込みにおける出力フィルタの数）．
- __kernel_size__: 整数か単一の整数からなるタプル/リストで，1次元の畳み込みウィンドウの長さを指定します．
- __strides__: 整数か単一の整数からなるタプル/リストで，畳み込みのストライドの長さを指定します.
  strides value != 1とすると`dilation_rate` value != 1と指定できません．
- __padding__: `"valid"`，`"same"`，`"causal"`のいずれか（大文字小文字の区別はしない）．`"valid"`はパディングを行いません．`"same"`は元の入力と同じ長さを出力がもつように入力にパディングします．`"causal"`はcausal（dilated）畳み込み．例えば，output[t]はinput[t+1]に依存しません．時間的順序を無視すべきでない時系列データをモデリングする際に有効です．[WaveNet: A Generative Model for Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499)を参照して下さい.
- __dilation_rate__: 整数か単一の整数からなるタプル/リストで，dilated convolutionで使われる膨張率を指定します．
  現在，`dilation_rate` value != 1 とすると，`strides` value != 1を指定することはできません．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)を参照），
  何も指定しなければ，活性化は一切適用されません（つまり"線形"活性`a(x) = x`）．
- __use_bias__: 真理値で，バイアスベクトルを加えるかどうかを指定します．
- __kernel_initializer__: `kernel`の重み行列の初期値を指定します．（[initializers](../initializers.md)を参照）
- __bias_initializer__: バイアスベクトルの初期値を指定します．（[initializers](../initializers.md)を参照）
- __kernel_regularizer__: `kernel`の重みに適用させるRegularizerを指定します．（[regularizer](../regularizers.md)を参照）
- __bias_regularizer__: バイアスに適用させるRegularizerを指定します．（[regularizer](../regularizers.md)を参照）
- __activity_regularizer__: 出力テンソルに適用させるRegularizerを指定します．（[regularizer](../regularizers.md)を参照）
- __kernel_constraint__: カーネルの行列に適用させるConstraintを指定します．（[constraint](../constraints.md)を参照）
- __bias_constraint__: バイアスベクトルに適用させるConstraintを指定します．（[constraint](../constraints.md)を参照）

__入力のshape__

shapeが`(batch_size, steps, input_dim)`の3階テンソル．

__出力のshape__

shapeが`(batch_size, new_steps, nb_filter)`の3階テンソル．
`steps`はパディングにより変わっている可能性があります．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L347)</span>
### Conv2D

```python
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2次元の畳み込みレイヤー（例えばイメージに対する空間的畳み込み）．

このレイヤーは畳み込みカーネルを生成します．これはレイヤーの入力を畳み込んで，出力のテンソルを作ります．
`use_bias`をTrueにすると，バイアスベクトルが出力に加えられます．`activation`が`None`でない場合，指定した活性化関数が出力に適用されます．

このレイヤーをモデルの第1層に使うときはキーワード引数`input_shape`
（整数のタプル，サンプル軸を含まない）を指定してください．
例えば，`data_format="channels_last"`のとき，128x128 RGB画像では`input_shape=(128, 128, 3)`となります．

__引数__

- __filters__: 整数で，出力空間の次元（つまり畳み込みにおける出力フィルタの数）．
- __kernel_size__: 整数か2つの整数からなるタプル/リストで，2次元の畳み込みウィンドウの幅と高さを指定します.  単一の整数の場合は全ての次元に対して同じ値を指定します．単一の整数の場合は正方形のカーネルになります．
- __strides__: 整数か2つの整数からなるタプル/リストで畳み込みの縦と横のストライドをそれぞれ指定できます．単一の整数の場合は幅と高さが同様のストライドになります．strides value != 1とすると`dilation_rate` value != 1と指定できません．
- __padding__: `"valid"`か`"same"`のどちらかを指定します．
- __data_format__: 文字列で，`"channels_last"`（デフォルト）か`"channels_first"`のどちらかを指定します．これは入力における次元の順序です． `"channels_last"`の場合，入力のshapeは`"(batch, height, width, channels)"`となり，`"channels_first"`の場合は`"(batch, channels, height, width)"`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．
- __dilation_rate__: 整数か2つの整数からなるタプル/リストで，dilated convolutionで使われる膨張率を指定します．
  現在，`dilation_rate` value != 1 とすると，strides value != 1を指定することはできません．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)を参照），
  何も指定しなければ，活性化は一切適用されません（つまり"線形"活性`a(x) = x`）．
- __use_bias__: 真理値で，バイアスベクトルを加えるかどうかを指定します．
- __kernel_initializer__: `kernel`の重み行列の初期値を指定します．（[initializers](../initializers.md)を参照）
- __bias_initializer__: バイアスベクトルの初期値を指定します．（[initializers](../initializers.md)を参照）
- __kernel_regularizer__: `kernel`の重み行列に適用させるRegularizerを指定します．（[regularizer](../regularizers.md)を参照）
- __bias_regularizer__: バイアスベクトルに適用させるRegularizerを指定します．（[regularizer](../regularizers.md)を参照）
- __activity_regularizer__: 出力テンソルに適用させるRegularizerを指定します．（[regularizer](../regularizers.md)を参照）
- __kernel_constraint__: カーネルの行列に適用させるConstraintを指定します．（[constraint](../constraints.md)を参照）
- __bias_constraint__: バイアスベクトルに適用させるConstraintを指定します．（[constraint](../constraints.md)を参照）

__入力のshape__

data_format='channels_first'の場合，
`(batch_size, channels, rows, cols)`の4階テンソル．
data_format='channels_last'の場合，
`(batch_size, rows, cols, channels)`の4階テンソルになります．

__出力のshape__

data_format='channels_first'の場合，
`(samples, channels, rows, cols)`の4階テンソル．
data_format='channels_last'の場合，
`(samples, rows, cols, channels)`の4階テンソルになります．`rows`と`cols`値はパディングにより変わっている可能性があります．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1384)</span>
### SeparableConv2D

```python
keras.layers.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```

Depthwiseな2次元separable畳み込み層．

separable畳み込み演算は，depthwiseの空間的な畳み込み（各入力チャネルに別々に作用する）を実行し，続いてpointwiseに畳み込みを行い，両者の出力チャネルを混合します．`depth_multiplier`は，出力チャネルを生成するための入力チャネルの数を指定します．

separable畳み込み演算はひとつのカーネルをふたつの小さなカーネルに分解する方法として，直感的に理解することができます．もしくはInception blockの極端な例として考えることもできます．

__引数__

- __filters__: 整数で，出力空間の次元（つまり畳み込みにおける出力フィルタの数）．
- __kernel_size__: 整数か2つの整数からなるタプル/リストで，2次元の畳み込みウィンドウの幅と高さを指定します. 単一の整数の場合は全ての次元に対して同じ値を指定します．
- __strides__: 整数か2つの整数からなるタプル/リストで畳み込みの縦と横のストライドをそれぞれ指定できます．単一の整数の場合は幅と高さが同様のストライドになります．strides value != 1とすると`dilation_rate` value != 1と指定できません．
- __padding__: `"valid"`か`"same"`のどちらかを指定します．
- __data_format__: 文字列で，`"channels_last"`（デフォルト）か`"channels_first"`のどちらかを指定します．これは入力における次元の順序です． `"channels_last"`の場合，入力のshapeは`"(batch, height, width, channels)"`となり，`"channels_first"`の場合は`"(batch, channels, height, width)"`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．
- __depth_multiplier__: 各入力チャネルに対するdepthwiseな畳み込みチャネルの数．深さ方向畳み込みチャネルの出力総数は，`filters_in * depth_multiplier`に等しくなります．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)を参照），
  何も指定しなければ，活性化は一切適用されません（つまり"線形"活性`a(x) = x`）．
- __use_bias__: 真理値で，バイアスベクトルを加えるかどうかを指定します．
- __depthwise_initializer__: カーネルの重み行列の初期値をdepthwiseに指定します．（[initializers](../initializers.md)を参照）
- __pointwise_initializer__: カーネルの重み行列の初期値をpointwiseに指定します．（[initializers](../initializers.md)を参照）
- __bias_initializer__: バイアスベクトルの初期値を指定します．（[initializers](../initializers.md)を参照）
- __depthwise_regularizer__: 重み行列に対し，"depthwise"に適用させるRegularizerを指定します．（[ regularizer](../regularizers.md)を参照）
- __pointwise_regularizer__: 重み行列に対し，"pointwise"に適用させるRegularizerを指定します．（[ regularizer](../regularizers.md)を参照）
- __bias_regularizer__: バイアスベクトルに適用させるRegularizerを指定します．（[regularizer](../regularizers.md)を参照）
- __activity_regularizer__: 出力テンソルに適用させるRegularizerを指定します．（[regularizer](../regularizers.md)を参照）
- __depthwise_constraint__: カーネル行列に対し，"depthwise"に適用させるConstraintを指定します．（[constraint](../constraints.md)を参照）
- __depthwise_constraint__: カーネル行列に対し，"pointwise"に適用させるConstraintを指定します．（[constraint](../constraints.md)を参照）
- __bias_constraint__: バイアスベクトルに適用させるConstraintを指定します．（[constraint](../constraints.md)を参照）

__入力のshape__

data_format='channels_first'の場合，
`(batch_size, channels, rows, cols)`の4階テンソル．
data_format='channels_last'の場合，
`(batch_size, rows, cols, channels)`の4階テンソルになります．

__出力のshape__

data_format='channels_first'の場合，
`(batch, channels, rows, cols)`の4階テンソル．
data_format='channels_last'の場合，
`(batch, rows, cols, channels)`の4階テンソルになります．`rows`と`cols`値はパディングにより変わっている可能性があります．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L596)</span>
### Conv2DTranspose

```python
keras.layers.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

transposed畳み込みレイヤー（Deconvolutionとも呼ばれます）．

一般的に，transposed畳み込み演算は通常の畳み込みに対して逆の演算を行いたい時に使われます．つまり，なんらかの畳み込み演算の出力を，接続パターンを保ちながら入力の形に変換する層です．

このレイヤーをモデルの第一層に使うときはキーワード引数`input_shape`
（整数のタプル，サンプル軸を含まない）を指定してください．
例えば`data_format="channels_last"`のとき，128x128 RGB画像では`input_shape=(128, 128, 3)`となります．

__引数__

- __filters__: 整数で，出力空間の次元（つまり畳み込みにおける出力フィルタの数）．
- __kernel_size__: 整数か2つの整数からなるタプル/リストで，2次元の畳み込みウィンドウの幅と高さを指定します.  単一の整数の場合は全ての次元に対して同じ値を指定します．
- __strides__: 整数か2つの整数からなるタプル/リストで畳み込みの縦と横のストライドをそれぞれ指定できます．単一の整数の場合は幅と高さが同様のストライドになります．strides value != 1とすると`dilation_rate` value != 1と指定できません．
- __padding__: `"valid"`か`"same"`のどちらかを指定します．
- __data_format__: 文字列で，`"channels_last"`（デフォルト）か`"channels_first"`のどちらかを指定します．これは入力における次元の順序です． `"channels_last"`の場合，入力のshapeは`"(batch, height, width, channels)"`となり，`"channels_first"`の場合は`"(batch, channels, height, width)"`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．
- __dilation_rate__: 整数か2つの整数からなるタプル/リストで，dilated convolutionで使われる膨張率を指定します．
  現在，`dilation_rate` value != 1 とすると，strides value != 1を指定することはできません．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)を参照），
  何も指定しなければ，活性化は一切適用されません（つまり"線形"活性`a(x) = x`）．
- __use_bias__: 真理値で，バイアスベクトルを加えるかどうかを指定します．
- __kernel_initializer__: `kernel`の重み行列の初期値を指定します．（[initializers](../initializers.md)を参照）
- __bias_initializer__: バイアスベクトルの初期値を指定します．（[initializers](../initializers.md)を参照）
- __kernel_regularizer__: `kernel`の重み行列に適用させるRegularizerを指定します．（[regularizer](../regularizers.md)を参照）
- __bias_regularizer__: バイアスベクトルに適用させるRegularizer関数を指定します．（[regularizer](../regularizers.md)を参照）
- __activity_regularizer__: 出力テンソルに適用させるRegularizer関数を指定します．（[regularizer](../regularizers.md)を参照）
- __kernel_constraint__: カーネルの行列に適用させるConstraint関数を指定します．（[Constraint](../constraints.md)を参照）
- __bias_constraint__: バイアスベクトルに適用させるConstraint関数を指定します．（[Constraint](../constraints.md)を参照）

__入力のshape__

data_format='channels_first'の場合，
`(batch, channels, rows, cols)`の4階テンソル．
data_format='channels_last'の場合，
`(batch, rows, cols, channels)`の4階テンソルになります．

__出力のshape__

data_format='channels_first'の場合，
`(batch, filters, new_rows, new_cols)`の4階テンソル．
data_format='channels_last'の場合，
`(batch, new_rows, new_cols, channels)`の4階テンソルになります．`rows`と`cols`値はパディングにより変わっている可能性があります．

__参考文献__

- [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285v1)
- [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L471)</span>
### Conv3D

```python
keras.layers.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

3次元入力をフィルターする畳み込み演算（例えば高さを含めた空間の畳込み）．

このレイヤーは畳み込みカーネルを生成します．これはレイヤーの入力を畳み込んで，出力のテンソルを作ります．
`use_bias`をTrueにすると，バイアスベクトルが出力に加えられます．`activation`が`None`でない場合，指定した活性化関数が出力に適用されます．

このレイヤーをモデルの第一層に使うときはキーワード引数`input_shape`
（整数のタプル，サンプル軸を含まない）を指定してください．
例えば`data_format="channels_last"`の場合，シングルチャネルの128x128x128の立体は`input_shape=(128, 128, 128, 1)`です．

__引数__

- __filters__: 整数で，出力空間の次元（つまり畳み込みにおける出力フィルタの数）．
- __kernel_size__: 整数か3つの整数からなるタプル/リストで，3次元の畳み込みウィンドウの幅と高さを指定します. 単一の整数の場合は全ての次元に対して同じ値を指定します．
- __strides__: 整数か3つの整数からなるタプル/リストで畳み込みの縦と横のストライドをそれぞれ指定できます．単一の整数の場合は幅と高さが同様のストライドになります．strides value != 1とすると`dilation_rate` value != 1と指定できません．
- __padding__: `"valid"`か`"same"`のどちらかを指定します．
- __data_format__: 文字列で，`"channels_last"`（デフォルト）か`"channels_first"`のどちらかを指定します．これは入力における次元の順序です． `"channels_last"`の場合，入力のshapeは`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`となり，`"channels_first"`の場合は`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．
- __dilation_rate__: 整数か3つの整数からなるタプル/リストで，dilated convolutionで使われる膨張率を指定します．
  現在，`dilation_rate` value != 1 とすると，strides value != 1を指定することはできません．
- __activation__: 使用する活性化関数の名前（[activations](../activations.md)を参照），
  何も指定しなければ，活性化は一切適用されません（つまり"線形"活性`a(x) = x`）．
- __use_bias__: 真理値で，バイアスベクトルを加えるかどうかを指定します．
- __kernel_initializer__: `kernel`の重み行列の初期値を指定します．（[initializers](../initializers.md)を参照）
- __bias_initializer__: バイアスベクトルの初期値を指定します．（[initializers](../initializers.md)を参照）
- __kernel_regularizer__: `kernel`の重み行列に適用させるRegularizerを指定します．（[regularizer](../regularizers.md)を参照）
- __bias_regularizer__: バイアスベクトルに適用させるRegularizer関数を指定します．（[regularizer](../regularizers.md)を参照）
- __activity_regularizer__: 出力テンソルに適用させるRegularizer関数を指定します．（[regularizer](../regularizers.md)を参照）
- __kernel_constraint__: カーネルの行列に適用させるConstraint関数を指定します．（[Constraint](../constraints.md)を参照）
- __bias_constraint__: バイアスベクトルに適用させるConstraint関数を指定します．（[Constraint](../constraints.md)を参照）

__入力のshape__

data_format='channels_first'の場合，
`(samples, channels, conv_dim1, conv_dim2, conv_dim3)`の5階テンソル．
data_format='channels_last'の場合，
`(samples, conv_dim1, conv_dim2, conv_dim3, channels)`の5階テンソルになります．

__出力のshape__

data_format='channels_first'の場合，
`(samples, channels, conv_dim1, conv_dim2, conv_dim3)`の5階テンソル．
data_format='channels_last'の場合，
`(samples, conv_dim1, conv_dim2, conv_dim3, channels)`の5階テンソルになります．`conv_dim1`，`conv_dim2`，`conv_dim3`の値はパディングにより変わっている可能性があります．

----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2159)</span>
### Cropping1D

```python
keras.layers.Cropping1D(cropping=(1, 1))
```

一次元の入力をクロップする（切り落とす）層（例えば時間の配列）．

クロップは時間軸に沿って実行されます(axis 1)．

__引数__

- __cropping__: 整数か長さ2の整数のタプルで，クロップしたいユニットの数を指定します．2つの整数からなるタプルで指定した場合は，それぞれ両側からクロップします．1つの整数の場合は，両側から同数のユニットをクロップします．

__入力のShape__

`(batch, axis_to_crop, features)`の3階テンソル．

__出力のShape__

`(batch, cropped_axis, features)`の3階テンソル．

----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2204)</span>
### Cropping2D

```python
keras.layers.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
```

二次元の入力をクロップする（切り落とす）層（例えば画像）．

クロップは幅と高さに対して実行されます．

__引数__

- __cropping__: 整数, タプル（2つの整数）, タプル（2つの整数）のタプルのいずれか．
    - 整数: 幅と高さに対称なクロップが実行されます．
    - タプル（2つの整数）: 幅と高さでそれぞれ対称なクロップが実行されます．
    `(symmetric_height_crop, symmetric_width_crop)`
    - タプルのタプル: 四辺それぞれにクロップが実行されます．
    `(top_crop, bottom_crop), (left_crop, right_crop))`
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します.
`"channels_last"`の場合，入力は`"(batch, height, width, channels)"`．
`"channels_first"`の場合は`"(batch, channels, height, width)"`となります．
デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__入力のShape__

`data_format`が`"channels_last"`の場合，`(batch, rows, cols, channels)`．
`"channels_first"`の場合，`(batch, channels, rows, cols)`の4階テンソル．

__出力のShape__

`data_format`が`"channels_last"`の場合，`(batch, cropped_rows, cropped_cols, channels)`．
`"channels_first"`の場合，`(batch, channels, cropped_rows, cropped_cols)`の4階テンソル．

__例__

```python
# Crop the input 2D images or feature maps
model = Sequential()
model.add(Cropping2D(cropping=((2, 2), (4, 4)),
                     input_shape=(28, 28, 3)))
# now model.output_shape == (None, 24, 20, 3)
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Cropping2D(cropping=((2, 2), (2, 2))))
# now model.output_shape == (None, 20, 16. 64)
```

----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2347)</span>
### Cropping3D

```python
keras.layers.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), data_format=None)
```

三次元の入力をクロップする（切り落とす）層（例えば空間や時空間）．

__引数__

- __cropping__: 整数, タプル（2つの整数），タプル（2つの整数）のタプル，のいずれか．
    - 整数: 3つの軸に対して対称なクロップが実行されます．
    - タプル（3つの整数）: 3つの軸に対して，それぞれ対称なクロップが実行されます．
    `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`
    - タプルのタプル: 六面それぞれにクロップが実行されます．
    `((left_dim1_crop, right_dim1_crop), (left_dim2_crop, right_dim2_crop), (left_dim3_crop, right_dim3_crop))`
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します．
`"channels_last"`の場合，入力は`"(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)"`．`"channels_first"`の場合は`"(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)"`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__入力のShape__


`data_format`が`"channels_last"`の場合，`(batch, first_cropped_axis, second_cropped_axis, third_cropped_axis, depth)`．`"channels_first"`の場合，`(batch, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)`の5階テンソル．

__出力のShape__

`data_format`が`"channels_last"`の場合，`(batch, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop, depth)`． `"channels_first"`の場合，`(batch, depth, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)`の5階テンソル．

----
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1711)</span>
### UpSampling1D

```python
keras.layers.UpSampling1D(size=2)
```

1次元の入力に対するアップサンプリングレイヤー．

時間軸方向にそれぞれの時間ステップを`size`回繰り返します．

__引数__

- __size__: 整数．upsampling係数．

__入力のshape__

`(batch, steps, features)`の3階テンソル．

__出力のshape__

`(batch, upsampled_steps, features)`の3階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1746)</span>
### UpSampling2D

```python
keras.layers.UpSampling2D(size=(2, 2), data_format=None)
```

2次元の入力に対するアップサンプリングレイヤー．

データの行と列をそれぞれsize[0]及びsize[1]回繰り返します．

__引数__

- __size__: 整数か2つの整数のタプル．行と列のupsampling係数．
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力のshapeは`"(batch, height, width, channels)"`となり，`"channels_first"`の場合は`"(batch, channels, height, width)"`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__入力のshape__

data_format='channels_last'の場合，
`(batch, rows, cols, channels)`の4階テンソル．
data_format='channels_first'の場合，
`(batch, channels, rows, cols)`の4階テンソルになります．

__出力のshape__

data_format='channels_first'の場合，
`(batch, channels, upsampled_rows, upsampled_cols)`の4階テンソル．
data_format='channels_last'の場合，
`(batch, upsampled_rows, upsampled_cols, channels)`の4階テンソルになります．
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1815)</span>
### UpSampling3D

```python
keras.layers.UpSampling3D(size=(2, 2, 2), data_format=None)
```

3次元の入力に対するアップサンプリングレイヤー．

データの1番目，2番目，3番目の次元をそれぞれsize[0]，size[1]，size[2]だけ繰り返す．


__引数__

- __size__: 3つの整数のタプル．dim1，dim2，dim3のアップサンプリング係数．
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します. `"channels_last"`の場合，入力のshapeは`"(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)"`となり，`"channels_first"`の場合は`"(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)"`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__入力のshape__

data_format='channels_last'の場合，
`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`の5階テンソル．
data_format='channels_first'の場合，
`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`の5階テンソルになります．

__出力のshape__

data_format='channels_last'の場合，
`(batch, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`の5階テンソル．
data_format='channels_first'の場合，
`(batch, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`の5階テンソルになります．
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1889)</span>
### ZeroPadding1D

```python
keras.layers.ZeroPadding1D(padding=1)
```

一次元入力（例，時系列）に対するゼロパディングレイヤー．

__引数__

- __padding__: 整数，タプル（2つの整数）のいずれか．
    - 整数: パディング次元(axis 1)の始めと終わりにいくつのゼロを加えるか．
    - （長さ2の）整数のタプル: 始めと終わりにそれぞれいくつのゼロを加えるか．
    `(left_pad, right_pad)`

__入力のshape__

`(batch, axis_to_pad, features)`の3階テンソル．

__出力のshape__

`(batch, padded_axis, features)`の3階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1931)</span>
### ZeroPadding2D

```python
keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None)
```

2次元入力(例，画像)のためのゼロパディングレイヤー

このレイヤーは画像のテンソルの上下左右にゼロの行と列を追加します．

__引数__

- __padding__: 整数，タプル（2つの整数），タプル（2つの整数）のタプル．
    - 整数: 幅と高さにたいして対称なパディングを実行する．
    - タプル: 幅と高さ，それぞれに対して対称なパディングを実行する．
    `(symmetric_height_pad, symmetric_width_pad)`
    - タプルのタプル: 四辺それぞれにパディング．
    `((top_pad, bottom_pad), (left_pad, right_pad))`
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します.
`"channels_last"`の場合，入力は`"(batch, height, width, channels)"`．
`"channels_first"`の場合は`"(batch, channels, height, width)"`となります．
デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__入力のShape__

`data_format`が`"channels_last"`の場合，`(batch, rows, cols, channels)`．
`"channels_first"`の場合，`(batch, channels, rows, cols)`の4階テンソル．

__出力のShape__

`data_format`が`"channels_last"`の場合，`(batch, padded_rows, padded_cols, channels)`．
`"channels_first"`の場合，`(batch, channels, padded_rows, padded_cols)`の4階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2041)</span>
### ZeroPadding3D

```python
keras.layers.ZeroPadding3D(padding=(1, 1, 1), data_format=None)
```

3次元データ(空間及び時空間)のためのゼロパディングレイヤー．

__引数__

- __cropping__: 整数, タプル（2つの整数），タプル（2つの整数）のタプル，のいずれか．
    - 整数: 3つの軸に対して対称なパディングが実行されます．
    - タプル（3つの整数）: 3つの軸に対して，それぞれ対称なパディングが実行されます．
    `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`
    - タプルのタプル: 六面それぞれにパディングが実行されます．
    `((left_dim1_pad, right_dim1_pad), (left_dim2_pad, right_dim2_pad), (left_dim3_pad, right_dim3_pad))`
- __data_format__: `"channels_last"`（デフォルト）か`"channels_first"`を指定します．
`"channels_last"`の場合，入力は`"(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)"`．`"channels_first"`の場合は`"(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)"`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

__入力のShape__


`data_format`が`"channels_last"`の場合，`(batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad, depth)`．`"channels_first"`の場合，`(batch, depth, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)`の5階テンソル．

__出力のShape__

`data_format`が`"channels_last"`の場合，`(batch, first_padded_axis, second_padded_axis, third_axis_to_pad, depth)`． `"channels_first"`の場合，`(batch, depth, first_padded_axis, second_padded_axis, third_axis_to_pad)`の5階テンソル．
