<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L615)</span>
### Dense

```python
keras.layers.core.Dense(output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)
```

通常の全結合ニューラルネットワークレイヤー．

__例__


```python
# シーケンシャルモデルの最初のレイヤーとして:
model = Sequential()
model.add(Dense(32, input_dim=16))
# 今，モデルは(*, 16)次元の入力配列となり，(*, 32)次元の出力配列となる

# これは上記と等価である:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))

# 最初のレイヤーの後，あなたはもはや入力サイズを特定する必要はない:
model.add(Dense(32))
```

__引数__

- __output_dim__: 正の整数 > 0.
- __init__: レイヤーの重みに対する初期化関数名([initializations](../initializations.md)を参照)，もしくは，重みを初期化するために使用するTheano関数．このパラメータは`weights`引数を与えていないときにのみ有効です．
- __activation__: 使用する活性化関数名
	([activations](../activations.md)を参照)，もしくは，要素ごとのTheano関数．
	もしあなたが何も指定しなければ，活性化は適用されない．
	(すなわち，"線形"活性化: a(x) = x)．
- __weights__: 初期重みとしてセットするnumpy配列のリスト．そのリストは重みとバイアスのそれぞれに対して`(入力次元, 出力次元)と(出力次元,)`の形の2要素持つべきである．
- __W_regularizer__: 主の重み行列に適用される[WeightRegularizer](../regularizers.md)のインスタンス
	(例えば，L1もしくはL2正則化)．
- __b_regularizer__: バイアスに適用される[WeightRegularizer](../regularizers.md)のインスタンス．
- __activity_regularizer__: ネットワーク出力に適用される[ActivityRegularizer](../regularizers.md)のインスタンス．
- __W_constraint__: 主の重み行列に適用される[constraints](../constraints.md)モジュールのインスタンス．(例えば，maxnorm，nonneg)．
- __b_constraint__: バイアスに適用される[constraints](../constraints.md)モジュールのインスタンス．
- __bias__: バイアスを含めるかどうか(すなわち，線形というよりもむしろアフィンレイヤーにさせるか)．
- __input_dim__: 入力の次元(整数)．
	この引数(もしくは，キーワード引数`input_shape`)
	は，モデルの最初のレイヤーとして使うときに必要とされる．

__入力の型__

2次元テンソルの型: `(nb_samples, input_dim)`．

__出力の型__

2次元テンソルの型: `(nb_samples, output_dim)`．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L195)</span>
### Activation

```python
keras.layers.core.Activation(activation)
```

出力に活性化関数を適用する．

__引数__

- __activation__: 使用する活性化関数名
	([activations](../activations.md)を参照),
	もしくは，TheanoかTensorFlowオペレーション．

__入力の型__

任意．モデルの最初のレイヤーとしてこのレイヤーを使う時，キーワード引数`input_shape`(整数のタプルはサンプルの軸を含まない．)を使う．

__出力の型__

入力と同じ型．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L67)</span>
### Dropout

```python
keras.layers.core.Dropout(p)
```

入力にドロップアウトを適用する．ドロップアウトは，訓練時のそれぞれの更新において入力ユニットの`p`をランダムに0にセットすることであり，それは過学習を防ぐのを助ける．

__引数__

- __p__: 0と1の間の浮動小数点数．入力ユニットをドロップする割合．

__参考文献__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L381)</span>
### Flatten

```python
keras.layers.core.Flatten()
```

入力を平坦化する．バッチサイズに影響されない．

__例__


```python
model = Sequential()
model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
# いま: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# いま: model.output_shape == (None, 65536)
```

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L225)</span>
### Reshape

```python
keras.layers.core.Reshape(target_shape)
```

ある型に出力を変形する．

__引数__

- __target_shape__: ターゲットの型．整数のタプル，
	サンプルの次元を含まない(バッチサイズ)．

__入力の型__

入力の型のすべての次元は固定されなければならないが，任意．
モデルの最初のレイヤーとしてこのレイヤーを使うとき，キーワード引数`input_shape`(整数のタプルはサンプルの軸を含まない．)を使う．


__出力の型__

`(batch_size,) + target_shape`

__例__


```python
# シーケンシャルモデルの最初のレイヤーとして
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# いま: model.output_shape == (None, 3, 4)
# 注意: `None`はバッチの次元

# シーケンシャルモデルの中間レイヤーとして
model.add(Reshape((6, 2)))
# いま: model.output_shape == (None, 6, 2)
```

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L331)</span>
### Permute

```python
keras.layers.core.Permute(dims)
```

与えられたパターンにより入力の次元を変更する．

例えば，RNNsやconvnetsの連結に対して役立ちます．

__例__


```python
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# いま: model.output_shape == (None, 64, 10)
# 注意: `None`はバッチの次元
```

__引数__

- __dims__: 整数のタプル．配列パターン，サンプルの次元を含まない．添字は1で始まる．例えば，`(2, 1)`は入力の1番目と2番目の次元を計算する．

__入力の型__

任意. モデルの最初のレイヤーとしてこのレイヤーを使う時，キーワード引数`input_shape`(整数のタプルはサンプルの軸を含まない．)を使う．

__出力の型__

入力の型と同じだが，特定のパターンにより並べ替えられた次元を持つ．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L413)</span>
### RepeatVector

```python
keras.layers.core.RepeatVector(n)
```

n回入力を繰り返す．

__例__


```python
model = Sequential()
model.add(Dense(32, input_dim=32))
# いま: model.output_shape == (None, 32)
# 注意: `None`はバッチの次元でる．

model.add(RepeatVector(3))
# いま: model.output_shape == (None, 3, 32)
```

__引数__

- __n__: 整数，繰返し因数．

__入力の型__

`(nb_samples, features)`の型の2次元テンソル．

__出力の型__

`(nb_samples, n, features)`の型の3次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/engine/topology.py#L1095)</span>
### Merge

```python
keras.engine.topology.Merge(layers=None, mode='sum', concat_axis=-1, dot_axes=-1, output_shape=None, node_indices=None, tensor_indices=None, name=None)
```

`Merge`レイヤーは，いくつかのマージ`mode`に従って，単一のテンソルにテンソルのリストをマージするために使うことができる．

__使用例__


```python
model1 = Sequential()
model1.add(Dense(32))

model2 = Sequential()
model2.add(Dense(32))

merged_model = Sequential()
merged_model.add(Merge([model1, model2], mode='concat', concat_axis=1)
- __TODO__: would this actually work? it needs to.__

# シーケンシャル内の`get_source_inputs`でこれを達成する．
```

__引数__

- __layers__: Kerasのテンソルのリストかレイヤーのインスタンスのリストであるべき．ひとつ以上のレイヤーかテンソルでなければならない．
- __mode__: 文字列かラムダ/関数．もし文字列であるなら，それは'sum', 'mul', 'concat', 'ave', 'cos', 'dot'のひとつでなければならない．もしラムダ/関数であるなら，それはテンソルのリストを入力とし，単一のテンソルを返さなければならない．
- __concat_axis__: 整数，`concat`モードで使用するための軸．
- __dot_axes__: 整数，または整数のタプル，`dot`モードで使用するための軸．
- __output_shape__: タプルの型 (整数のタプル)，もしくは`output_shape`を計算するラムダ/関数 (マージモードに限り，ラムダ/関数となる)．もし後者の場合，タプルの型のリストを入力として受け取る(入力テンソルに対して1:1にマッピングする)．また単一のタプルの型を返す．
- __node_indices__: それぞれの入力レイヤーに対する出力ノードインデックスを含む整数の追加リスト(いくつかの入力レイヤーが複数の出力ノードを持つ場合)．それはもし供給されないなら，0の配列をデフォルトとする．
- __tensor_indices__: マージのために考慮される出力のテンソルの追加リスト(いくつかの入力レイヤーノードが複数のテンソルを返す場合)．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L454)</span>
### Lambda

```python
keras.layers.core.Lambda(function, output_shape=None, arguments={})
```

前のレイヤーの出力で，任意のTheano/TensorFlow表現を評価するために使われる．

__例__


```python
# 一つのx -> x^2レイヤーを加える．
model.add(Lambda(lambda x: x ** 2))
```
```python
# 入力の正の部分と負の部分の反対の結合を返すレイヤーを加える．

def antirectifier(x):
	x -= K.mean(x, axis=1, keepdims=True)
	x = K.l2_normalize(x, axis=1)
	pos = K.relu(x)
	neg = K.relu(-x)
	return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
	shape = list(input_shape)
	assert len(shape) == 2  # only valid for 2D tensors
	shape[-1] *= 2
	return tuple(shape)

model.add(Lambda(antirectifier, output_shape=antirectifier_output_shape))
```

__引数__

- __function__: 評価される関数．一つの引数を取る: 前のレイヤーの出力
- __output_shape__: 関数からの期待される出力の型．タプルもしくは関数．
	タプルなら，入力に近いほうの次元だけを指定する，データサンプルの次元は入力と同じ:
	`output_shape = (input_shape[0], ) + output_shape`
	か入力が `None` でかつサンプル次元も`None`:
	`output_shape = (None, ) + output_shape`
	のどちらかが推測される．
  関数なら，入力型の関数として型全体を指定する: `output_shape = f(input_shape)`
- __arguments__: 関数に通されるキーワード引数の追加辞書

__入力の型__

任意．モデルの最初のレイヤーとしてこのレイヤーを使う時，キーワード引数`input_shape`(整数のタプル，それはサンプルの軸を含まない)を使う．

__出力の型__

`output_shape`引数によって特定される．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L759)</span>
### ActivityRegularization

```python
keras.layers.core.ActivityRegularization(l1=0.0, l2=0.0)
```

その変化のない入力を通過するレイヤー，しかしアクティビティに基づいたコスト関数の更新を適用する．

__引数__

- __l1__: L1正則化因子 (正の浮動小数点数)．
- __l2__: L2正則化因子 (正の浮動小数点数)．

__入力の型__

任意．モデルの最初のレイヤーとしてこのレイヤーを使う時，キーワード引数`input_shape`(整数のタプル，それはサンプルの軸を含まない)を使う．

__出力の型__

入力と同じ型．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L20)</span>
### Masking

```python
keras.layers.core.Masking(mask_value=0.0)
```

スキップされるタイムステップを特定するためのマスク値を使うことによって入力シーケンスをマスクする．

入力テンソル(テンソルの次元 #1)のそれぞれのタイムステップに対して，
もしそのタイムステップの入力テンソルのすべての値が`mask_value`に等しいなら，そのときそのタイムステップはすべてのダウンストリームレイヤー(それらがマスキングをサポートしている限り)でマスク(スキップ)されるでしょう．

もしどんなダウンストリームレイヤーが，そのような入力マスクをまだ受けておらず，マスキングをサポートしていなければ，例外が生じるだろう．


__例__

LSTMレイヤーに与えるための`(samples, timesteps, features)`の型のNumpy配列`x`を検討する．
あなたが#3と#5のタイムステップに関してデータを欠損しているので，これらのタイムステップをマスクしたい場合，あなたは以下のようにできる:

- set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
- insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
```

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L929)</span>
### Highway

```python
keras.layers.core.Highway(init='glorot_uniform', transform_bias=-2, activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)
```

密に結合されたハイウェイネットワーク，フィードフォワードネットワークへのLSTMsの自然拡張．

__引数__

- __init__: レイヤーの重みに対する初期化関数名([initializations](../initializations.md)を参照)，もしくは，重みを初期化するために使用するTheano関数．このパラメータは`weights`引数を与えていないときにのみ有効です．
- __transform_bias__: 初期に取るバイアスに対する値(デフォルト -2)
- __activation__: 使用する活性化関数名
	([activations](../activations.md)を参照)，もしくは，要素ごとのTheano関数．
	もしあなたが何も指定しなければ，活性化は適用されない．
	(すなわち，"線形"活性化: a(x) = x)．
- __weights__: 初期重みとしてセットするnumpy配列のリスト．そのリストは重みとバイアスのそれぞれに対して`(入力次元, 出力次元)と(出力次元,)`の形の2要素持つべきである．
- __W_regularizer__: 主の重み行列に適用される[WeightRegularizer](../regularizers.md)のインスタンス(例えば，L1もしくはL2正則化)．
- __b_regularizer__: バイアスに適用される[WeightRegularizer](../regularizers.md)のインスタンス．
- __activity_regularizer__: ネットワーク出力に適用される[ActivityRegularizer](../regularizers.md)のインスタンス．
- __W_constraint__: 主の重み行列に適用される[constraints](../constraints.md)モジュールのインスタンス．(例えば，maxnorm，nonneg)．
- __b_constraint__: バイアスに適用される[constraints](../constraints.md)モジュールのインスタンス．
- __bias__: バイアスを含めるかどうか(すなわち，線形というよりもむしろアフィンレイヤーにさせるか)．
- __input_dim__: 入力の次元(整数)．この引数(もしくは，キーワード引数`input_shape`)は，モデルの最初のレイヤーとして使うときに必要とされる．

__入力の型__

`(nb_samples, input_dim)`の型の2次元テンソル．

__出力の型__

`(nb_samples, output_dim)`の型の2次元テンソル．

__参考文献__

- [Highway Networks](http://arxiv.org/pdf/1505.00387v2.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L792)</span>
### MaxoutDense

```python
keras.layers.core.MaxoutDense(output_dim, nb_feature=4, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)
```

密なマックスアウトレイヤー．

`MaxoutDense`レイヤーは`nb_feature`の要素ごとの最大を取る．`Dense(input_dim, output_dim)`の線形のレイヤー．
これはそのレイヤーに入力にわたる区分的線形活性化関数の凸を学習することを許す．

これは*線形*のレイヤーであることに注意する;
もしあなたが活性化関数を適用したいのであれば，(あなたはそれをする必要はない--それらは普遍的関数近似詞である)，
`Activation`レイヤーは後で追加されなければならない．

__引数__

- __output_dim__: 正の整数 > 0.
- __nb_feature__: 内部で使われるデンスレイヤーの数．
- __init__: レイヤーの重みに対する初期化関数名([initializations](../initializations.md)を参照)，もしくは，重みを初期化するために使用するTheano関数．このパラメータは`weights`引数を与えていないときにのみ有効です．
- __weights__: 初期重みとしてセットするnumpy配列のリスト．そのリストは重みとバイアスのそれぞれに対して`(入力次元, 出力次元)と(出力次元,)`の形の2要素持つべきである．
- __W_regularizer__: 主の重み行列に適用される[WeightRegularizer](../regularizers.md)のインスタンス
	(例えば，L1もしくはL2正則化)．
- __b_regularizer__: バイアスに適用される[WeightRegularizer](../regularizers.md)のインスタンス．
- __activity_regularizer__: ネットワーク出力に適用される[ActivityRegularizer](../regularizers.md)のインスタンス．
- __W_constraint__: 主の重み行列に適用される[constraints](../constraints.md)モジュールのインスタンス．(例えば，maxnorm，nonneg)．
- __b_constraint__: バイアスに適用される[constraints](../constraints.md)モジュールのインスタンス．
- __bias__: バイアスを含めるかどうか(すなわち，線形というよりもむしろアフィンレイヤーにさせるか)．
- __input_dim__: 入力の次元(整数)．
	この引数(もしくは，キーワード引数`input_shape`)
	は，モデルの最初のレイヤーとして使うときに必要とされる．

__入力の型__

2次元テンソル: `(nb_samples, input_dim)`．

__出力の型__

2次元テンソル: `(nb_samples, output_dim)`．

__参考文献__

- [Maxout Networks](http://arxiv.org/pdf/1302.4389.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L1067)</span>
### TimeDistributedDense

```python
keras.layers.core.TimeDistributedDense(output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
```

それぞれの次元[1] (time_dimension)の入力に対して同じデンスレイヤーを適用する．
特に'return_sequence=True'でリカレントネットワークの後に役立つ．

- __注意__: このレイヤーは廃止される予定である，`TimeDistributed`を使うことがより好まれる．

ラッパー:
```python
model.add(TimeDistributed(Dense(32)))
```

__入力の型__

`(nb_sample, time_dimension, input_dim)`の型の3次元テンソル．

__出力の型__

`(nb_sample, time_dimension, output_dim)`の型の3次元テンソル．

__引数__

- __output_dim__: 正の整数 > 0.
- __init__: レイヤーの重みに対する初期化関数名([initializations](../initializations.md)を参照)，もしくは，重みを初期化するために使用するTheano関数．このパラメータは`weights`引数を与えていないときにのみ有効です．
- __activation__: 使用する活性化関数名
	([activations](../activations.md)を参照)，もしくは，要素ごとのTheano関数．
	もしあなたが何も指定しなければ，活性化は適用されない．
	(すなわち，"線形"活性化: a(x) = x)．
- __weights__: 初期重みとしてセットするnumpy配列のリスト．そのリストは重みとバイアスのそれぞれに対して`(入力次元, 出力次元)と(出力次元,)`の形の2要素持つべきである．
- __W_regularizer__: 主の重み行列に適用される[WeightRegularizer](../regularizers.md)のインスタンス
	(例えば，L1もしくはL2正則化)．
- __b_regularizer__: バイアスに適用される[WeightRegularizer](../regularizers.md)のインスタンス．
- __activity_regularizer__: ネットワーク出力に適用される[ActivityRegularizer](../regularizers.md)のインスタンス．
- __W_constraint__: 主の重み行列に適用される[constraints](../constraints.md)モジュールのインスタンス．(例えば，maxnorm，nonneg)．
- __b_constraint__: バイアスに適用される[constraints](../constraints.md)モジュールのインスタンス．
- __bias__: バイアスを含めるかどうか(すなわち，線形というよりもむしろアフィンレイヤーにさせるか)．
- __input_dim__: 入力の次元(整数)．
	この引数(もしくは，キーワード引数`input_shape`)
	は，モデルの最初のレイヤーとして使うときに必要とされる．
- __input_length__: 入力シーケンスの長さ
	(整数，もしくは可変長のシーケンスに対してNone)．
