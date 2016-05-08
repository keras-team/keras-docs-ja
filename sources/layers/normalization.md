<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/normalization.py#L6)</span>
### BatchNormalization

```python
keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')
```

Normalize the activations of the previous layer at each batch,
i.e. applies a transformation that maintains the mean activation
close to 0 and the activation standard deviation close to 1.

__Arguments__

- __epsilon__: small float > 0. Fuzz parameter.
- __mode__: integer, 0 or 1.
	- 0: feature-wise normalization.
	Each feature map in the input will
	be normalized separately. The axis on which
	to normalize is specified by the `axis` argument.
	Note that if the input is a 4D image tensor
	using Theano conventions (samples, channels, rows, cols)
	then you should set `axis` to `1` to normalize along
	the channels axis.
	- 1: sample-wise normalization. This mode assumes a 2D input.
- __axis__: integer, axis along which to normalize in mode 0. For instance,
	if your input tensor has shape (samples, channels, rows, cols),
	set axis to 1 to normalize per feature map (channels axis).
- __momentum__: momentum in the computation of the
	exponential average of the mean and standard deviation
	of the data, for feature-wise normalization.
- __weights__: Initialization weights.
	List of 2 numpy arrays, with shapes:
	`[(input_shape,), (input_shape,)]`
- __beta_init__: name of initialization function for shift parameter
	(see [initializations](../initializations.md)), or alternatively,
	Theano/TensorFlow function to use for weights initialization.
	This parameter is only relevant if you don't pass a `weights` argument.
- __gamma_init__: name of initialization function for scale parameter (see
	[initializations](../initializations.md)), or alternatively,
	Theano/TensorFlow function to use for weights initialization.
	This parameter is only relevant if you don't pass a `weights` argument.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

__References__

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/pdf/1502.03167v3.pdf)
