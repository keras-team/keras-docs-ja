<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L743)</span>
### Dense

```python
keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

通常の全結合ニューラルネットワークレイヤー．

`Dense`が実行する操作：`output = activation(dot(input, kernel) + bias)`ただし，`activation`は`activation`引数として渡される要素単位の活性化関数で，`kernel`はレイヤーによって作成された重み行列であり，`bias`はレイヤーによって作成されたバイアスベクトルです.（`use_bias`が`True`の場合にのみ適用されます）．

- 注意：レイヤーへの入力のランクが2より大きい場合は，`kernel`を使用した最初のドット積の前に平坦化されます．

__例__

```python
# as first layer in a sequential model:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 32)

# after the first layer, you don't need to specify
# the size of the input anymore:
model.add(Dense(32))
```

__引数__

- __units__：正の整数，出力空間の次元数
- __activation__： 使用する活性化関数名
    （[activations](../activations.md)を参照）
    もしあなたが何も指定しなければ，活性化は適用されない．
    （すなわち，"線形"活性化： `a(x) = x`）．
- __use_bias__： 真理値，レイヤーがバイアスベクトルを使用するかどうか．
- __kernel_initializer__： `kernel`重み行列の初期化（[initializations](../initializers.md)を参照）
- __bias_initializer__： バイアスベクトルの初期化（[initializations](../initializers.md)を参照）
- __kernel_regularizer__： `kernel`重み行列に適用される正則化関数（[regularizers](../regularizers.md)を参照）
- __bias_regularizer__： バイアスベクトルに適用される正則化関数（[regularizers](../regularizers.md)を参照）
- __activity_regularizer__： レイヤーの出力に適用される正則化関数（"activation"）（[regularizers](../regularizers.md)を参照）
- __kernel_constraint__： `kernel`重み行列に適用される制約関数（[constraints](../constraints.md)を参照）
- __bias_constraint__： バイアスベクトルに適用される制約関数（[constraints](../constraints.md)を参照）

__入力のshape__

以下のshapeを持つn階テンソル： `(batch_size, ..., input_dim)`．
最も一般的なのは以下のshapeを持つ2階テンソル： `(batch_size, input_dim)`．

__出力のshape__

以下のshapeを持つn階テンソル： `(batch_size, ..., units)`．
例えば，以下のshapeを持つ2階テンソル `(batch_size, input_dim)`の入力に対して，アウトプットは以下のshapwを持つ`(batch_size, units)`．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L280)</span>
### Activation

```python
keras.layers.Activation(activation)
```

出力に活性化関数を適用する．

__引数__

- __activation__： 使用する活性化関数名
    ([activations](../activations.md)を参照)，
    もしくは，TheanoかTensorFlowオペレーション．

__入力のshape__

任意．モデルの最初のレイヤーとしてこのレイヤーを使う時，キーワード引数`input_shape`（整数のタプルはサンプルの軸（axis）を含まない．）を使う．

__出力のshape__

入力と同じshape．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L78)</span>
### Dropout

```python
keras.layers.Dropout(rate, noise_shape=None, seed=None)
```

入力にドロップアウトを適用する．

訓練時の更新においてランダムに入力ユニットを0とする割合であり，過学習の防止に役立ちます．

__引数__

- __rate__： 0と1の間の浮動小数点数．入力ユニットをドロップする割合．
- __noise_shape__： 入力と乗算されるバイナリドロップアウトマスクのshapeは1階の整数テンソルで表す．例えば入力のshapeを`(batch_size, timesteps, features)`とし，ドロップアウトマスクをすべてのタイムステップで同じにしたい場合，`noise_shape=(batch_size, 1, features)`を使うことができる.
- __seed__： random seedとして使うPythonの整数．

__参考文献__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L465)</span>
### Flatten

```python
keras.layers.Flatten()
```

入力を平滑化する．バッチサイズに影響を与えません．

__例__

```python
model = Sequential()
model.add(Conv2D(64, 3, 3,
                 border_mode='same',
                 input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# now: model.output_shape == (None, 65536)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/engine/topology.py#L1393)</span>
### Input

```python
keras.engine.topology.Input()
```

`Input()`はKerasテンソルのインスタンス化に使われます．

Kerasテンソルは下位のバックエンド（TheanoやTensorFlow，あるいはCNTK）からなるテンソルオブジェクトです．
モデルの入出力がわかっていれば，Kerasのモデルを構築するためにいくつかの属性を拡張できます．

例えばa, b, cがKerasのテンソルの場合，次のようにできます：
`model = Model(input=[a, b], output=c)`

追加されたKerasの属性：

- `_keras_shape`: Keras側の推論から伝達された整数のshapeのタプル．
- `_keras_history`: テンソルに適用される最後のレイヤー．全体のレイヤーグラフはこのレイヤーから再帰的に取り出せます．

__引数__

- __shape__: shapeのタプル（整数）で，バッチサイズを含みません．
    例えば，`shape=(32,)`は期待される入力が32次元ベクトルのバッチであることを示します．
- __batch_shape__: shapeのタプル（整数）で，バッチサイズを含みます．
    例えば，`batch_shape=(10, 32)`は期待される入力が10個の32次元ベクトルのバッチであることを示します．
    `batch_shape=(None, 32)`は任意の数の32次元ベクトルのバッチを示します．
- __name__: オプションとなるレイヤーの名前の文字列．
    モデルの中でユニークな値である必要があります（同じ名前は二回使えません）．
    指定しなければ自動生成されます．
- __dtype__: 入力から期待されるデータの型で，文字列で指定します(`float32`, `float64`, `int32`...)．
- __sparse__: 生成されるプレースホルダをスパースにするか指定する真理値．
- __tensor__: `Input`レイヤーをラップする既存のテンソル．
    設定した場合，レイヤーはプレースホルダとなるテンソルを生成しません．

__戻り値__

テンソル．

__例__

```python
# this is a logistic regression in Keras
x = Input(shape=(32,))
y = Dense(16, activation='softmax')(x)
model = Model(x, y)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L314)</span>
### Reshape

```python
keras.layers.Reshape(target_shape)
```

あるshapeに出力を変形する．

__引数__

- __target_shape__： ターゲットのshape．整数のタプル，サンプルの次元を含まない（バッチサイズ）．

__入力のshape__

入力のshapeのすべての次元は固定されなければならないが，任意．
モデルの最初のレイヤーとしてこのレイヤーを使うとき，キーワード引数`input_shape`(整数のタプルはサンプルの軸を含まない．)を使う．


__出力のshape__

`(batch_size,) + target_shape`

__例__


```python
# as first layer in a Sequential model
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# now: model.output_shape == (None, 3, 4)
# note: `None` is the batch dimension

# as intermediate layer in a Sequential model
model.add(Reshape((6, 2)))
# now: model.output_shape == (None, 6, 2)

# also supports shape inference using `-1` as dimension
model.add(Reshape((-1, 2, 2)))
# now: model.output_shape == (None, 3, 2, 2)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L413)</span>
### Permute

```python
keras.layers.Permute(dims)
```

与えられたパターンにより入力の次元を入れ替える．

例えば，RNNsやconvnetsの連結に対して役立ちます．

__例__

```python
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# now: model.output_shape == (None, 64, 10)
# note: `None` is the batch dimension
```

__引数__

- __dims__： 整数のタプル．配列パターン，サンプルの次元を含まない．添字は1で始まる．例えば，`(2, 1)`は入力の1番目と2番目の次元を入れ替える．

__入力のshape__

任意. モデルの最初のレイヤーとしてこのレイヤーを使う時，キーワード引数`input_shape`（整数のタプルはサンプルの軸を含まない）を使う．

__出力のshape__

入力のshapeと同じだが，特定のパターンにより並べ替えられた次元を持つ．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L500)</span>
### RepeatVector

```python
keras.layers.RepeatVector(n)
```

n回入力を繰り返す．

__例__

```python
model = Sequential()
model.add(Dense(32, input_dim=32))
# now: model.output_shape == (None, 32)
# note: `None` is the batch dimension

model.add(RepeatVector(3))
# now: model.output_shape == (None, 3, 32)
```

__引数__

- __n__： 整数，繰り返し因数．

__入力のshape__

`(num_samples, features)`のshapeを持つ2階テンソル．

__出力のshape__

`(num_samples, n, features)`のshapeを持つ3階テンソル．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L542)</span>
### Lambda

```python
keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None)
```

`Layer`オブジェクトのように，任意の式をラップする．

__例__

```python
# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2))
```

```python
# add a layer that returns the concatenation
# of the positive part of the input and
# the opposite of the negative part

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

model.add(Lambda(antirectifier,
                 output_shape=antirectifier_output_shape))
```

__引数__

- __function__： 評価される関数．第1引数として入力テンソルを取る
- __output_shape__： 関数からの期待される出力のshape．Theanoを使用する場合のみ関連します．タプルもしくは関数．
    タプルなら，入力に近いほうの次元だけを指定する，データサンプルの次元は入力と同じ：
    `output_shape = (input_shape[0], ) + output_shape`
    か入力が `None` でかつサンプル次元も`None`：
    `output_shape = (None, ) + output_shape`
    のどちらかが推測される．
  関数なら，入力のshapeの関数としてshape全体を指定する： `output_shape = f(input_shape)`
- __arguments__： 関数に通されるキーワード引数の追加辞書

__入力のshape__

任意．モデルの最初のレイヤーとしてこのレイヤーを使う時，キーワード引数`input_shape`（整数のタプル，サンプルの軸（axis）を含まない）を使う．

__出力のshape__

`output_shape`引数によって特定される（TensorFlowを使用していると自動推論される）．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L886)</span>
### ActivityRegularization

```python
keras.layers.ActivityRegularization(l1=0.0, l2=0.0)
```

コスト関数に基づく入力アクティビティに更新を適用するレイヤー

__引数__

- __l1__： L1正則化係数（正の浮動小数点数）．
- __l2__： L2正則化係数（正の浮動小数点数）．

__入力のshape__

任意．モデルの最初のレイヤーとしてこのレイヤーを使う時，キーワード引数`input_shape`（整数のタプル，サンプルの軸（axis）を含まない）を使う．

__出力のshape__

入力と同じshape．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L28)</span>
### Masking

```python
keras.layers.Masking(mask_value=0.0)
```

タイプステップをスキップするためのマスク値を用いてシーケンスをマスクします．

入力テンソル（テンソルの次元 #1）のそれぞれのタイムステップに対して，
もしそのタイムステップの入力テンソルのすべての値が`mask_value`に等しいなら，そのときそのタイムステップはすべての下流レイヤー（それらがマスキングをサポートしている限り）でマスク（スキップ）されるでしょう．

下流レイヤーがマスキングをサポートしていないのにそのような入力マスクを受け取ると例外が発生します．


__例__

LSTMレイヤーに与えるための`(samples, timesteps, features)`のshapeを持つのNumpy 配列`x`を考えてみましょう．
あなたが#3と#5のタイムステップに関してデータを欠損しているので，これらのタイムステップをマスクしたい場合，あなたは以下のようにできます：

- `x[:, 3, :] = 0.` と `x[:, 5, :] = 0.`をセットする．
- LSTMレイヤーの前に`mask_value=0.`の`Masking`レイヤーを追加する：

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
```
