
## レイヤーの重み初期化方法

初期化用引数で，Kerasレイヤーの重みをランダムに初期化する確率分布を指定できます．

初期化用引数のキーワードはレイヤーにより異なりますが，大抵は単純に `kernel_initializer` 及び `bias_initializer` です:

```python
model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
```

## 利用可能な初期化方法

以下の初期化方法は全て `keras.initializers` モジュールとして定義されています．

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L9)
### Initializer

```python
keras.initializers.Initializer()
```

これは初期化クラスの基底クラスです．

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L24)
### Zeros

```python
keras.initializers.Zeros()
```

全て重みを0で初期化します．

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L32)
### Ones

```python
keras.initializers.Ones()
```

全て重みを1で初期化します．

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L40)
### Constant

```python
keras.initializers.Constant(value=0)
```

全て重みを定数で初期化します．

__Arguments__

- __value__: float またはテンソルです

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L57)
### RandomNormal

```python
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
```

正規分布に従って重みを初期化します．

__Arguments__

- __mean__: float またはスカラテンソルであって分布の平均です
- __stddev__: float またはスカラテンソルであって分布の標準偏差です
- __seed__: 整数値．乱数生成に使われます

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L85)
### RandomUniform

```python
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
```

一様分布に従って重みを初期化します．

__Arguments__

- __minval__: float またはスカラテンソル．乱数を発生する範囲の下限です
- __maxval__: float またはスカラテンソル．乱数を発生する範囲の上限です
- __seed__: 整数値．乱数生成に使われます

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L113)
### TruncatedNormal

```python
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
```

切断正規分布に従って重みを初期化します．

これは正規分布と似ていますが，平均より標準偏差の分以上離れた値は切り捨てらます．これはニューラルネットワークの重みの初期化方法として推奨されます．

__Arguments__

- __mean__: float またはスカラテンソルであって分布の平均です
- __stddev__: float またはスカラテンソルであって分布の標準偏差です
- __seed__: 整数値．乱数生成に使われます

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L146)
### VarianceScaling

```python
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
```

重みテンソルのサイズ（`shape`）に合わせてスケーリングした初期化を行います．

`distribution="normal"` としたとき，
平均を 0 とし標準偏差を
`stddev = sqrt(scale / n)`
とした切断正規分布が使われます．
ここで `n` は

- `mode="fan_in"` のとき，入力ユニットの数
- `mode="fan_out"` のとき，出力ユニットの数
- `mode="fan_avg"` のとき，入力ユニットと出力ユニットの数の平均

が使われます．

`distribution="uniform"` としたとき，
[-limit, limit] を範囲とする一様分布が用いられます．
ここで `limit = sqrt(3 * scale / n)` です．

__Arguments__

- __scale__: スケーリング値（正の実数）
- __mode__: "fan_in"，"fan_out"，"fan_avg" のいずれか
- __distribution__: 用いる確率分布．"normal"，"uniform" のいずれか
- __seed__: 整数値．乱数生成に使われます

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L219)
### Orthogonal

```python
keras.initializers.Orthogonal(gain=1.0, seed=None)
```

重みテンソルが直交行列となるように初期化されます．

__Arguments__

- __gain__: 最後に直交行列に乗ずる係数です
- __seed__: 整数値．乱数生成に使われます

__References__

Saxe et al., http://arxiv.org/abs/1312.6120

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L256)
### Identity

```python
keras.initializers.Identity(gain=1.0)
```

単位行列で初期化されます．
これは重みテンソルが2次正方行列の場合のみ使えます．

__Arguments__

- __gain__: 最後に単位行列に乗ずる係数です

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L304)
### glorot_normal

```python
glorot_normal(seed=None)
```

Glorot の正規分布（Xavier の正規分布とも呼ばれます）による初期化を返します．

これは平均を 0 ，標準偏差を
`stddev = sqrt(2 / (fan_in + fan_out))`
とする切断正規分布と同じです．
ここで `fan_in` は入力ユニット数，`fant_out` は出力ユニット数です．

__Arguments__

- __seed__: 整数値．乱数生成に使われます

__Returns__

初期化クラスが返ります

__References__

Glorot & Bengio, AISTATS 2010 -
http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L328)
### glorot_uniform

```python
glorot_uniform(seed=None)
```

Glorot の一様分布（Xavier の一様分布とも呼ばれます）による初期化を返します．

これは limit を `sqrt(6 / (fan_in + fan_out))`
としたとき [limit, -limit] を範囲とする一様分布と同じです．
ここで `fan_in` は入力ユニット数，`fant_out` は出力ユニット数です．

__Arguments__

- __seed__: 整数値．乱数生成に使われます

__Returns__

初期化クラスが返ります

__References__

Glorot & Bengio, AISTATS 2010 -
http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L352)
### he_normal

```python
he_normal(seed=None)
```

He の正規分布による初期化を返します．
これは平均を 0 ，標準偏差を
`stddev = sqrt(2 / fan_in)`
とする切断正規分布です．
ここで `fan_in` は入力ユニット数です．

__Arguments__

- __seed__: 整数値．乱数生成に使われます

__Returns__

初期化クラスが返ります

__References__

He et al., http://arxiv.org/abs/1502.01852

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L374)
### he_uniform

```python
he_uniform(seed=None)
```

He の一様分布による初期化を返します．
これは limit を
`sqrt(6 / fan_in)`
としたとき [limit, -limit] を範囲とする一様分布を用います．
ここで `fan_in` は入力ユニット数です．

__Arguments__

- __seed__: 整数値．乱数生成に使われます

__Returns__

初期化クラスが返ります

__References__

He et al., http://arxiv.org/abs/1502.01852

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/initializers.py#L281)
### lecun_uniform

```python
lecun_uniform(seed=None)
```

LeCun の一様分布による初期化を返します．
これは limit を
`sqrt(3 / fan_in)`
とするとき
[-limit, limit]
を範囲とする一様分布を用います．
ここで `fan_in` は入力ユニット数です．

__Arguments__

- __seed__: 整数値．乱数生成に使われます

__Returns__

初期化クラスが返ります

__References__

LeCun 98, Efficient Backprop, - http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

---

初期化は，文字列（上記の利用可能な初期化方法のいずれかとマッチしなければならない）かcallableとして渡される．

```python
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))

# also works; will use the default parameters.
model.add(Dense(64, kernel_initializer='random_normal'))
```

## カスタマイズ

callable なオブジェクトを渡す場合には，初期化しようとする変数の `shape` と `dtype` を引数に取るように設計して下さい．

```python
from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))
```
