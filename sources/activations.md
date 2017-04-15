## 活性化関数の使い方

活性化関数は`Activation`レイヤー，または全てのフォワードレイヤーで使える引数`activation`で利用できます．

```python
from keras.layers.core import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

上のコードは以下と等価です：

```python
model.add(Dense(64, activation='tanh'))
```

要素ごとに適用できるTensorFlow/Theano関数を活性化関数に渡すこともできます:

```python
from keras import backend as K

def tanh(x):
    return K.tanh(x)

model.add(Dense(64, activation=tanh))
model.add(Activation(tanh))
```

## 利用可能な活性化関数

### softmax

```python
softmax(x, axis=-1)
```

Softmax関数

__Arguments__

- __x__: テンソル．
- __axis__: 整数．どの軸にsoftmaxの正規化をするか．

__Returns__

テンソル．softmax変換の出力．

__Raises__

- __ValueError__: `dim(x) == 1`のとき．

---

### elu

```python
elu(x, alpha=1.0)
```

---

### softplus

```python
softplus(x)
```

---

### softsign

```python
softsign(x)
```

---

### relu

```python
relu(x, alpha=0.0, max_value=None)
```

---

### tanh

```python
tanh(x)
```

--- 

### sigmoid

```python
sigmoid(x)
```

--- 

###hard_sigmoid

```python
hard_sigmoid(x)
```

--- 
### linear

```python
linear
```

---

## より高度な活性化関数

単純なTensorFlow/Theano関数よりも高度な活性化関数 (例: 状態を持てるlearnable activations) は，[Advanced Activation layers](layers/advanced-activations.md)として利用可能です．
これらは，`keras.layers.advanced_activations`モジュールにあり，`PReLU`や`LeakyReLU`が含まれます．
