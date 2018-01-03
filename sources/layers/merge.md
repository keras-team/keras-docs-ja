<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L184)</span>
### Add

```python
keras.layers.merge.Add()
```

入力のリスト同士を足し合わせるレイヤー．

入力はすべて同じshapeをもったテンソルのリストで，1つのテンソルを返す（shapeは同じ）．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L199)</span>
### Multiply

```python
keras.layers.merge.Multiply()
```

入力のリストの要素同士の積のレイヤー．

入力はすべて同じshapeをもったテンソルのリストで，1つのテンソルを返す（shapeは同じ）．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L214)</span>
### Average

```python
keras.layers.merge.Average()
```

入力のリストを平均するレイヤー．

入力はすべて同じshapeをもったテンソルのリストで，1つのテンソルを返す（shapeは同じ）．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L229)</span>
### Maximum

```python
keras.layers.merge.Maximum()
```

入力のリストの要素間の最大値を求めるレイヤー．

入力はすべて同じshapeをもったテンソルのリストで，1つのテンソルを返す（shapeは同じ）．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L244)</span>
### Concatenate

```python
keras.layers.merge.Concatenate(axis=-1)
```

入力のリストをconcatenateするレイヤー．

入力はすべて同じshapeをもったテンソルのリストで，全入力をconcatenateした1つのテンソルを返す．

__引数__

- __axis__: concatenateする際のaxis．
- __**kwargs__: 標準的なレイヤーのキーワード引数．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L335)</span>
### Dot

```python
keras.layers.merge.Dot(axes, normalize=False)
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
add(inputs)
```

`Add`レイヤーの関数インターフェース．

__引数__

- __inputs__: 入力テンソルのリスト（最低2つ）．
- __**kwargs__: 標準的なレイヤーのキーワード引数．

__戻り値__

入力の総和のテンソル．

----

### multiply

```python
multiply(inputs)
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
average(inputs)
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
maximum(inputs)
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
concatenate(inputs, axis=-1)
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
dot(inputs, axes, normalize=False)
```
`Dot`レイヤーの関数インターフェース．

__引数__

- __inputs__: 入力テンソルのリスト（最低2つ）．
- __axes__: 整数か整数のタプル．dot積をとる際にaxisかaxesのどちらを使うか．
- __normalize__: dot積をとる前にdot積のaxisでサンプルをL2正規化するかどうか． Trueなら，dot積の出力は，2つのサンプルのcosine．
- __**kwargs__: 標準的なレイヤーのキーワード引数．

__戻り値__

入力のdot積をとったテンソル．
