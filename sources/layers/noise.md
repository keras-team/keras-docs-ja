<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/noise.py#L6)</span>
### GaussianNoise

```python
keras.layers.noise.GaussianNoise(sigma)
```

Apply to the input an additive zero-centred gaussian noise with
standard deviation `sigma`. This is useful to mitigate overfitting
(you could see it as a kind of random data augmentation).
Gaussian Noise (GS) is a natural choice as corruption process
for real valued inputs.

As it is a regularization layer, it is only active at training time.

__Arguments__

- __sigma__: float, standard deviation of the noise distribution.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/noise.py#L44)</span>
### GaussianDropout

```python
keras.layers.noise.GaussianDropout(p)
```

Apply to the input an multiplicative one-centred gaussian noise
with standard deviation `sqrt(p/(1-p))`.

As it is a regularization layer, it is only active at training time.

__Arguments__

- __p__: float, drop probability (as with `Dropout`).

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

__References__

- __[Dropout__: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
