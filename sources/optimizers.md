
## オプティマイザ（最適化アルゴリズム）の利用方法

オプティマイザ（最適化アルゴリズム）はモデルをコンパイルする際に必要となるパラメータの1つです:

```python
model = Sequential()
model.add(Dense(64, init='uniform', input_dim=10))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

上記の例のように，オプティマイザのインスタンスを `model.compile()` に渡す，もしくは，オプティマイザの名前を渡すことができます．後者の場合，オプティマイザのデフォルトパラメータが利用されます．

```python
# オプティマイザを名前で指定すると，デフォルトパラメータが利用されます
model.compile(loss='mean_squared_error', optimizer='sgd')
```

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L204)</span>
### Adagrad

```python
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
```

デフォルトパラメータのまま利用することを推奨します．

__Arguments__

- __lr__: float >= 0. 学習率．
- __epsilon__: float >= 0.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L243)</span>
### Adadelta

```python
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
```

デフォルトパラメータのまま利用することを推奨します．

__Arguments__

- __lr__: float >= 0. 学習率．
	デフォルト値を推奨します．
- __rho__: float >= 0.
- __epsilon__: float >= 0. Fuzz factor.

__References__

- [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L298)</span>
### Adam

```python
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
```

デフォルトパラメータは提案論文に従います．

__Arguments__

- __lr__: float >= 0. 学習率.
- __beta_1/beta_2__: floats, 0 < beta < 1. 一般的に1に近づきます．
- __epsilon__: float >= 0. Fuzz factor.

__References__

- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L356)</span>
### Adamax

```python
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
```
 
Adamaxは，Adamの提案論文の7節で提案された，無限ノルムに基づくAdamの拡張です．

デフォルトパラメータは提案論文に従います．

__Arguments__

- __lr__: float >= 0. 学習率.
- __beta_1/beta_2__: floats, 0 < beta < 1. 一般的に1に近づきます．
- __epsilon__: float >= 0. Fuzz factor.

__References__

- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L105)</span>
### SGD

```python
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```

モーメンタム，学習率減衰，Nesterov momentumをサポートした確率的勾配降下法．

__Arguments__

- __lr__: float >= 0. 学習率.
- __momentum__: float >= 0. モーメンタム．
- __decay__: float >= 0. 各更新の学習率減衰．
- __nesterov__: boolean. Nesterov momentumを適用するかどうか.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L156)</span>
### RMSprop

```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
```

デフォルトパラメータのまま利用することを推奨します．
（ただし，学習率は除き，自由に調整することが可能です）

RMSPropは再帰型ニューラルネットワークに対して良い選択となるでしょう．

__Arguments__

- __lr__: float >= 0. 学習率.
- __rho__: float >= 0.
- __epsilon__: float >= 0. Fuzz factor.
