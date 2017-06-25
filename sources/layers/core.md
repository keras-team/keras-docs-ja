<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L731)</span>
### Dense

```python
keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

通常の全結合ニューラルネットワークレイヤー．

`Dense`が実行する操作：`output = activation(dot(input, kernel) + bias)`ただし，`activation`は`activation`引数として渡される要素単位の活性化関数で，`kernel`はレイヤーによって作成された重み行列であり，`bias`はレイヤーによって作成されたバイアスベクトルです.（`use_bias`が`True`の場合にのみ適用されます）．


- Note：レイヤーへの入力のランクが2より大きい場合は，`kernel`を使用した最初のドット積の前に平坦化されます．



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
	([activations](../activations.md)を参照)
	もしあなたが何も指定しなければ，活性化は適用されない．
	(すなわち，"線形"活性化： `a(x) = x`)．
- __use_bias__： レイヤーがバイアスベクトルを使用するかどうか．
- __kernel_initializer__： `kernel`重み行列の初期化（[initializations](../initializers.md)を参照）
- __bias_initializer__： バイアスベクトルの初期化（[initializations](../initializers.md)を参照）
- __kernel_regularizer__： `kernel`重み行列に適用される正則化関数（[regularizers](../regularizers.md)を参照）
- __bias_regularizer__： バイアスベクトルに適用される正則化関数（[regularizers](../regularizers.md)を参照）
- __activity_regularizer__： レイヤーの出力に適用される正則化関数（"activation"）（[regularizers](../regularizers.md)を参照）
- __kernel_constraint__： `kernel`重み行列に適用される制約関数（[constraints](../constraints.md)を参照）
- __bias_constraint__： バイアスベクトルに適用される制約関数（[constraints](../constraints.md)を参照）


__入力のshape__

以下のshapeを持つn次元テンソル： `(batch_size, ..., input_dim)`．
最も一般的なのは以下のshapeを持つ2次元テンソル： `(batch_size, input_dim)`．

__出力のshape__

以下のshapeを持つn次元テンソル： `(batch_size, ..., units)`．
例えば，以下のshapeを持つ2次元テンソル `(batch_size, input_dim)`の入力に対して，アウトプットは以下のshapwを持つ`(batch_size, units)`．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L267)</span>
### Activation

```python
keras.layers.core.Activation(activation)
```

出力に活性化関数を適用する．

__引数__

- __activation__： 使用する活性化関数名
	([activations](../activations.md)を参照)，
	もしくは，TheanoかTensorFlowオペレーション．

__入力のshape__

任意．モデルの最初のレイヤーとしてこのレイヤーを使う時，キーワード引数`input_shape`(整数のタプルはサンプルの軸（axis）を含まない．)を使う．

__出力のshape__

入力と同じ型．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L72)</span>
### Dropout

```python
keras.layers.core.Dropout(rate, noise_shape=None, seed=None)
```

入力にドロップアウトを適用する．ドロップアウトは，訓練時のそれぞれの更新において入力ユニットの`rate`をランダムに0にセットすることであり，それは過学習を防ぐのを助ける．

__引数__

- __rate__： 0と1の間の浮動小数点数．入力ユニットをドロップする割合．
- __noise_shape__： 入力と乗算されるバイナリドロップアウトマスクのshapeは1次元の整数テンソルで表す．例えば入力のshapeを`(batch_size, timesteps, features)`とし，ドロップアウトマスクをすべてのタイムステップで同じにしたい場合，`noise_shape=(batch_size, 1, features)`を使うことができる.
- __seed__： random seedとして使うPythonの整数．

__参考文献__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L456)</span>
### Flatten

```python
keras.layers.core.Flatten()
```

入力を平滑化する．バッチサイズに影響されない．

__例__


```python
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# now: model.output_shape == (None, 65536)
```

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L298)</span>
### Reshape

```python
keras.layers.core.Reshape(target_shape)
```

ある型に出力を変形する．

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

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L404)</span>
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
# now: model.output_shape == (None, 64, 10)
# note: `None` is the batch dimension
```

__引数__

- __dims__： 整数のタプル．配列パターン，サンプルの次元を含まない．添字は1で始まる．例えば，`(2, 1)`は入力の1番目と2番目の次元を計算する．

__入力のshape__

任意. モデルの最初のレイヤーとしてこのレイヤーを使う時，キーワード引数`input_shape`（整数のタプルはサンプルの軸を含まない）を使う．

__出力のshape__

入力のshapeと同じだが，特定のパターンにより並べ替えられた次元を持つ．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L491)</span>
### RepeatVector

```python
keras.layers.core.RepeatVector(n)
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

- __n__： 整数，繰返し因数．

__入力のshape__

`(num_samples, features)`のshapeを持つ2次元テンソル．

__出力の型__

`(num_samples, n, features)`のshapeを持つ3次元テンソル．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L533)</span>
### Lambda

```python
keras.layers.core.Lambda(function, output_shape=None, mask=None, arguments=None)
```

レイヤーオブジェクトのように，任意の式をラップする．

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
  関数なら，入力型の関数として型全体を指定する： `output_shape = f(input_shape)`
- __arguments__： 関数に通されるキーワード引数の追加辞書

__入力のshape__

任意．モデルの最初のレイヤーとしてこのレイヤーを使う時，キーワード引数`input_shape`（整数のタプル，サンプルの軸（axis）を含まない）を使う．

__出力のshape__

`output_shape`引数によって特定される（TensorFlowを使用していると自動推論される）．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L874)</span>
### ActivityRegularization

```python
keras.layers.core.ActivityRegularization(l1=0.0, l2=0.0)
```

コスト関数に基づく入力アクティビティに更新を適用するレイヤー

__引数__

- __l1__： L1正則化因子（正の浮動小数点数）．
- __l2__： L2正則化因子（正の浮動小数点数）．

__入力のshape__

任意．モデルの最初のレイヤーとしてこのレイヤーを使う時，キーワード引数`input_shape`（整数のタプル，サンプルの軸（axis）を含まない）を使う．

__出力のshape__

入力と同じ型．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L25)</span>
### Masking

```python
keras.layers.core.Masking(mask_value=0.0)
```

スキップされるタイムステップを特定するためのマスク値を使うことによって入力シーケンスをマスクする．

入力テンソル（テンソルの次元 #1）のそれぞれのタイムステップに対して，
もしそのタイムステップの入力テンソルのすべての値が`mask_value`に等しいなら，そのときそのタイムステップはすべての下流レイヤー（それらがマスキングをサポートしている限り）でマスク（スキップ）されるでしょう．

下流レイヤーがマスキングをサポートしていないのにそのような入力マスクを受け取ると例外が発生します．


__例__

LSTMレイヤーに与えるための`(samples, timesteps, features)`のshapeを持つのNumpy配列`x`を考えてみましょう．
あなたが#3と#5のタイムステップに関してデータを欠損しているので，これらのタイムステップをマスクしたい場合，あなたは以下のようにできます：

- `x[:, 3, :] = 0.` と `x[:, 5, :] = 0.`をセットする．
- LSTMレイヤーの前に`mask_value=0.`の`Masking`レイヤーを追加する：

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
```

