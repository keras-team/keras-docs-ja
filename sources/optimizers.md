
## Usage of optimizers

An optimizer is one of the two arguments required for compiling a Keras model:

```python
model = Sequential()
model.add(Dense(64, init='uniform', input_dim=10))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

You can either instantiate an optimizer before passing it to `model.compile()` , as in the above example, or you can call it by its name. In the latter case, the default parameters for the optimizer will be used.

```python
# pass optimizer by name: default parameters will be used
model.compile(loss='mean_squared_error', optimizer='sgd')
```

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L204)</span>
### Adagrad

```python
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
```

Adagrad optimizer.

It is recommended to leave the parameters of this optimizer
at their default values.

__Arguments__

- __lr__: float >= 0. Learning rate.
- __epsilon__: float >= 0.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L243)</span>
### Adadelta

```python
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
```

Adadelta optimizer.

It is recommended to leave the parameters of this optimizer
at their default values.

__Arguments__

- __lr__: float >= 0. Learning rate.
	It is recommended to leave it at the default value.
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

Adam optimizer.

Default parameters follow those provided in the original paper.

__Arguments__

- __lr__: float >= 0. Learning rate.
- __beta_1/beta_2__: floats, 0 < beta < 1. Generally close to 1.
- __epsilon__: float >= 0. Fuzz factor.

__References__

- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L356)</span>
### Adamax

```python
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
```

Adamax optimizer from Adam paper's Section 7. It is a variant
 of Adam based on the infinity norm.

Default parameters follow those provided in the paper.

__Arguments__

- __lr__: float >= 0. Learning rate.
- __beta_1/beta_2__: floats, 0 < beta < 1. Generally close to 1.
- __epsilon__: float >= 0. Fuzz factor.

__References__

- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L105)</span>
### SGD

```python
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```

Stochastic gradient descent, with support for momentum,
learning rate decay, and Nesterov momentum.

__Arguments__

- __lr__: float >= 0. Learning rate.
- __momentum__: float >= 0. Parameter updates momentum.
- __decay__: float >= 0. Learning rate decay over each update.
- __nesterov__: boolean. Whether to apply Nesterov momentum.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L156)</span>
### RMSprop

```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
```

RMSProp optimizer.

It is recommended to leave the parameters of this optimizer
at their default values
(except the learning rate, which can be freely tuned).

This optimizer is usually a good choice for recurrent
neural networks.

__Arguments__

- __lr__: float >= 0. Learning rate.
- __rho__: float >= 0.
- __epsilon__: float >= 0. Fuzz factor.
