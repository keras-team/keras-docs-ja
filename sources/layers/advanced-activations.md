<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L7)</span>
### LeakyReLU

```python
keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
```

Special version of a Rectified Linear Unit
that allows a small gradient when the unit is not active:
`f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __alpha__: float >= 0. Negative slope coefficient.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L38)</span>
### PReLU

```python
keras.layers.advanced_activations.PReLU(init='zero', weights=None)
```

Parametric Rectified Linear Unit:
`f(x) = alphas * x for x < 0`,
`f(x) = x for x >= 0`,
where `alphas` is a learned array with the same shape as x.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __init__: initialization function for the weights.
- __weights__: initial weights, as a list of a single numpy array.

__References__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L85)</span>
### ELU

```python
keras.layers.advanced_activations.ELU(alpha=1.0)
```

Exponential Linear Unit:
`f(x) =  alpha * (exp(x) - 1.) for x < 0`,
`f(x) = x for x >= 0`.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __alpha__: scale for the negative factor.

__References__

- [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/pdf/1511.07289v1.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L120)</span>
### ParametricSoftplus

```python
keras.layers.advanced_activations.ParametricSoftplus(alpha_init=0.2, beta_init=5.0, weights=None)
```

Parametric Softplus:
`alpha * log(1 + exp(beta * x))`

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __alpha_init__: float. Initial value of the alpha weights.
- __beta_init__: float. Initial values of the beta weights.
- __weights__: initial weights, as a list of 2 numpy arrays.

__References__

- [Inferring Nonlinear Neuronal Computation Based on Physiologically Plausible Inputs](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003143)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L170)</span>
### ThresholdedReLU

```python
keras.layers.advanced_activations.ThresholdedReLU(theta=1.0)
```

Thresholded Rectified Linear Unit:
`f(x) = x for x > theta`
`f(x) = 0 otherwise`.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __theta__: float >= 0. Threshold location of activation.

__References__

- [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/pdf/1402.3337.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L203)</span>
### SReLU

```python
keras.layers.advanced_activations.SReLU(t_left_init='zero', a_left_init='glorot_uniform', t_right_init='glorot_uniform', a_right_init='one')
```

S-shaped Rectified Linear Unit.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __t_left_init__: initialization function for the left part intercept
- __a_left_init__: initialization function for the left part slope
- __t_right_init__: initialization function for the right part intercept
- __a_right_init__: initialization function for the right part slope

__References__

- [Deep Learning with S-shaped Rectified Linear Activation Units](http://arxiv.org/abs/1512.07030)
