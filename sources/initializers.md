
## レイヤーの重み初期化方法

初期化用引数で，Kerasレイヤーの重みをランダムに初期化する確率分布を指定できます．

初期化用引数のキーワードはレイヤーにより異なりますが，大抵は単純に`init`です:

```python
model.add(Dense(64, init='uniform'))
```

## 利用可能な初期化方法

- __uniform__
- __lecun_uniform__: input数の平方根でスケーリングした一様分布 (LeCun 98)
- __normal__
- __identity__: `shape[0] == shape[1]`の2次元のレイヤーで使えます
- __orthogonal__: `shape[0] == shape[1]`の2次元のレイヤーで使えます
- __zero__
- __glorot_normal__: fan_in + fan_outでスケーリングした正規分布 (Glorot 2010)
- __glorot_uniform__
- __he_normal__: fan_inでスケーリングした正規分布 (He et al., 2014)
- __he_uniform__

初期化は，文字列（上記の利用可能な初期化方法のいずれかとマッチしなければならない）かcallableとして渡される．callableなら，`shape` (初期化するvariableのshape) と `name` (variable名) の2つの引数を取り，variable (つまり`K.variable()` の出力) を返さなければならない:

```python
from keras import backend as K
import numpy as np

def my_init(shape, name=None):
    value = np.random.random(shape)
    return K.variable(value, name=name)

model.add(Dense(64, init=my_init))
```

次のように`keras.initializations`の関数を使うこともできる:

```python
from keras import initializations

def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

model.add(Dense(64, init=my_init))
```
