<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L193)</span>
### Add

```python
keras.layers.Add()
```

入力のリスト同士を足し合わせるレイヤー．

入力はすべて同じshapeをもったテンソルのリストで，1つのテンソルを返す（shapeは同じ）．

__例__

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.Add()([x1, x2])  # equivalent to added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L223)</span>
### Subtract

```python
keras.layers.Subtract()
```

2つの入力の引き算をするレイヤー．

入力は同じshapeのテンソルのリストを2つで，1つのテンソルを返す(inputs[0] - inputs[1])．
返すテンソルも同じshapeです．

__例__

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# Equivalent to subtracted = keras.layers.subtract([x1, x2])
subtracted = keras.layers.Subtract()([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L260)</span>
### Multiply

```python
keras.layers.Multiply()
```

入力のリストの要素同士の積のレイヤー．

入力はすべて同じshapeをもったテンソルのリストで，1つのテンソルを返す（shapeは同じ）．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L275)</span>
### Average

```python
keras.layers.Average()
```

入力のリストを平均するレイヤー．

入力はすべて同じshapeをもったテンソルのリストで，1つのテンソルを返す（shapeは同じ）．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L290)</span>
### Maximum

```python
keras.layers.Maximum()
```

入力のリストの要素間の最大値を求めるレイヤー．

入力はすべて同じshapeをもったテンソルのリストで，1つのテンソルを返す（shapeは同じ）．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L320)</span>
### Concatenate

```python
keras.layers.Concatenate(axis=-1)
```

入力のリストをconcatenateするレイヤー．

入力は，concatenateする際のaxisを除き，すべて同じshapeをもったテンソルのリストで，全入力をconcatenateした1つのテンソルを返す．

__引数__

- __axis__: concatenateする際のaxis．
- __**kwargs__: 標準的なレイヤーのキーワード引数．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L408)</span>
### Dot

```python
keras.layers.Dot(axes, normalize=False)
```

2つのテンソルのサンプル間でdot積を計算するレイヤー．

例．もしshapeが`batch_size, n`の2つのテンソル`a`と`b`に適用する場合，出力されるテンソルのshapeは，`(batch_size, 1)`，出力の要素 `i` は，`a[i]`と`b[i]`のdot積．

__引数__

- __axes__: 整数か整数のタプル．dot積をとる際にaxisかaxesのどちらを使うか．
- __normalize__: dot積をとる前にdot積のaxisでサンプルをL2正規化するかどうか．
Trueなら，dot積の出力は，2つのサンプルのcosine．
- __**kwargs__: 標準的なレイヤーのキーワード引数．

----

### add

```python
keras.layers.add(inputs)
```

`Add`レイヤーの関数インターフェース．

__引数__

- __inputs__: 入力テンソルのリスト（最低2つ）．
- __**kwargs__: 標準的なレイヤーのキーワード引数．

__戻り値__

入力の総和のテンソル．

__例__

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

### subtract

```python
keras.layers.subtract(inputs)
```

`Subtract`レイヤーの関数インターフェース．

__引数__

- __inputs__: 入力テンソルのリスト（最低2つ）．
- __**kwargs__: 標準的なレイヤーのキーワード引数．

__戻り値__

入力の差のテンソル．

__例__

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
subtracted = keras.layers.subtract([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

### multiply

```python
keras.layers.multiply(inputs)
```

`Multiply`レイヤーの関数インターフェース．

__引数__

- __inputs__: 入力テンソルのリスト（最低2つ）．
- __**kwargs__: 標準的なレイヤーのキーワード引数．

__戻り値__

入力の要素同士の積のテンソル．

----

### average

```python
keras.layers.average(inputs)
```

`Average`レイヤーの関数インターフェース．

__引数__

- __inputs__: 入力テンソルのリスト（最低2つ）．
- __**kwargs__: 標準的なレイヤーのキーワード引数．

__戻り値__

入力の平均のテンソル．

----

### maximum

```python
keras.layers.maximum(inputs)
```

`Maximum`レイヤーの関数インターフェース．

__引数__

- __inputs__: 入力テンソルのリスト（最低2つ）．
- __**kwargs__: 標準的なレイヤーのキーワード引数．

__戻り値__

入力の要素間の最大値のテンソル．

----

### concatenate

```python
keras.layers.concatenate(inputs, axis=-1)
```
`Concatenate`レイヤーの関数インターフェース．

__引数__

- __inputs__: 入力テンソルのリスト（最低2つ）．
- __axis__: Concatenation axis．
- __**kwargs__: 標準的なレイヤーのキーワード引数．

__戻り値__

入力を`axis`の方向でconcateしたテンソル．

----

### dot

```python
keras.layers.dot(inputs, axes, normalize=False)
```
`Dot`レイヤーの関数インターフェース．

__引数__

- __inputs__: 入力テンソルのリスト（最低2つ）．
- __axes__: 整数か整数のタプル．dot積をとる際にaxisかaxesのどちらを使うか．
- __normalize__: dot積をとる前にdot積のaxisでサンプルをL2正規化するかどうか． Trueなら，dot積の出力は，2つのサンプルのcosine．
- __**kwargs__: 標準的なレイヤーのキーワード引数．

__戻り値__

入力のdot積をとったテンソル．
