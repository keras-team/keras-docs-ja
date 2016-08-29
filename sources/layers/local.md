
<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/local.py#L10)</span>
### LocallyConnected1D

```python
keras.layers.local.LocallyConnected1D(nb_filter, filter_length, init='uniform', activation='linear', weights=None, border_mode='valid', subsample_length=1, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
```

重みが共有されないこと，すなわち，異なるフィルタの集合が異なる入力のパッチに適用されること，以外は`LocallyConnected1D`は`Convolution1D`と似たように動作します．
このレイヤーを第一層に使う場合，キーワード引数として`input_dim` (整数値，例えば128次元ベクトル系列には128) を指定するか`input_shape` (整数のタプル，例えば10個の128次元ベクトル系列では`input_shape=(10, 128)`) を指定してください．
また，この層は入力のshapeをすべて指定することでのみ利用可能です (`None` をもつ次元は許容できません)．

__Example__


```python
# apply a unshared weight convolution 1d of length 3 to a sequence with
# 10 timesteps, with 64 output filters
model = Sequential()
model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
# now model.output_shape == (None, 8, 64)
# add a new conv1d on top
model.add(LocallyConnected1D(32, 3))
# now model.output_shape == (None, 6, 32)
```

__Arguments__

- __nb_filter__: 出力の次元．
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
<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/local.py#L188)</span>
### LocallyConnected2D

```python
keras.layers.local.LocallyConnected2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```

重みが共有されないこと，すなわち，異なるフィルタの集合が異なる入力のパッチに適用されること，以外は`LocallyConnected2D`は`Convolution2D`と似たように動作します．
このレイヤーをモデルの第一層に使うときはキーワード引数`input_shape`（整数のタプル，サンプル軸を含まない）を指定してください．
例えば128x128のRGB画像では`input_shape=(3, 128, 128)`とします．
また，この層は入力のshapeをすべて指定することでのみ利用可能です (`None` をもつ次元は許容できません).

__Example__


```python
# apply a 3x3 unshared weights convolution with 64 output filters on a 32x32 image:
model = Sequential()
model.add(LocallyConnected2D(64, 3, 3, input_shape=(3, 32, 32)))
# now model.output_shape == (None, 64, 30, 30)
# notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64 parameters

# add a 3x3 unshared weights convolution on top, with 32 output filters:
model.add(LocallyConnected2D(32, 3, 3))
# now model.output_shape == (None, 32, 28, 28)
```

__Arguments__

- __nb_filter__: 使用する畳み込みカーネルの数．
- __nb_row__: 畳み込みカーネルの行数．
- __nb_col__: 畳み込みカーネルの列数．
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
- __dim_ordering__: 'th'か'tf'．'th'モードのときはチャネルの次元 (深さ) はindex 1に， 'tf'モードではindex 3に．
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
