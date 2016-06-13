
## 活性化関数(activation function)の使い方

活性化関数は`Activation`層，または全てのフォワード層で使える引数`activation`で利用できます。

```python
from keras.layers.core import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```
上のコードは以下と等価です：
```python
model.add(Dense(64, activation='tanh'))
```

element-wiseなTheano/TensorFlow関数を活性化関数として渡すこともできます:

```python
from keras import backend as K

def tanh(x):
    return K.tanh(x)

model.add(Dense(64, activation=tanh))
model.add(Activation(tanh))
```

## 利用可能な活性化関数

- __softmax__: Softmaxは入力の最後の次元に適用されます。想定入力形式は`(nb_samples, nb_timesteps, nb_dims)`または`(nb_samples, nb_dims)`です。
- __softplus__
- __softsign__
- __relu__
- __tanh__
- __sigmoid__
- __hard_sigmoid__
- __linear__

## より高度な活性化関数

単純なTheano/TensorFlow関数よりも高度な活性化関数(例：learnable activations, configurable activations, etc.)は，[Advanced Activation layers](layers/advanced-activations.md)として利用可能です。
これらは，`keras.layers.advanced_activations`モジュールに含まれています。
PReLUやLeakyReLUはここに含まれます。

