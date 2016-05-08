<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L507)</span>
### Dense

```python
keras.layers.core.Dense(output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)
```

Just your regular fully connected NN layer.

__Example__


```python
# as first layer in a sequential model:
model = Sequential()
model.add(Dense(32, input_dim=16))
# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 32)

# this is equivalent to the above:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))

# after the first layer, you don't need to specify
# the size of the input anymore:
model.add(Dense(32))
```

__Arguments__

- __output_dim__: int > 0.
- __init__: name of initialization function for the weights of the layer
	(see [initializations](../initializations.md)),
	or alternatively, Theano function to use for weights
	initialization. This parameter is only relevant
	if you don't pass a `weights` argument.
- __activation__: name of activation function to use
	(see [activations](../activations.md)),
	or alternatively, elementwise Theano function.
	If you don't specify anything, no activation is applied
	(ie. "linear" activation: a(x) = x).
- __weights__: list of numpy arrays to set as initial weights.
	The list should have 2 elements, of shape `(input_dim, output_dim)`
	and (output_dim,) for weights and biases respectively.
- __W_regularizer__: instance of [WeightRegularizer](../regularizers.md)
	(eg. L1 or L2 regularization), applied to the main weights matrix.
- __b_regularizer__: instance of [WeightRegularizer](../regularizers.md),
	applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
	applied to the network output.
- __W_constraint__: instance of the [constraints](../constraints.md) module
	(eg. maxnorm, nonneg), applied to the main weights matrix.
- __b_constraint__: instance of the [constraints](../constraints.md) module,
	applied to the bias.
- __bias__: whether to include a bias (i.e. make the layer affine rather than linear).
- __input_dim__: dimensionality of the input (integer).
	This argument (or alternatively, the keyword argument `input_shape`)
	is required when using this layer as the first layer in a model.

__Input shape__

2D tensor with shape: `(nb_samples, input_dim)`.

__Output shape__

2D tensor with shape: `(nb_samples, output_dim)`.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L96)</span>
### Activation

```python
keras.layers.core.Activation(activation)
```

Applies an activation function to an output.

__Arguments__

- __activation__: name of activation function to use
	- __(see__: [activations](../activations.md)),
	or alternatively, a Theano or TensorFlow operation.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L67)</span>
### Dropout

```python
keras.layers.core.Dropout(p)
```

Applies Dropout to the input. Dropout consists in randomly setting
a fraction `p` of input units to 0 at each update during training time,
which helps prevent overfitting.

__Arguments__

- __p__: float between 0 and 1. Fraction of the input units to drop.

__References__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L282)</span>
### Flatten

```python
keras.layers.core.Flatten()
```

Flattens the input. Does not affect the batch size.

__Example__


```python
model = Sequential()
model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# now: model.output_shape == (None, 65536)
```

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L126)</span>
### Reshape

```python
keras.layers.core.Reshape(target_shape)
```

Reshapes an output to a certain shape.

__Arguments__

- __target_shape__: target shape. Tuple of integers,
	does not include the samples dimension (batch size).

__Input shape__

Arbitrary, although all dimensions in the input shaped must be fixed.
Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

`(batch_size,) + target_shape`

__Example__


```python
# as first layer in a Sequential model
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# now: model.output_shape == (None, 3, 4)
# note: `None` is the batch dimension

# as intermediate layer in a Sequential model
model.add(Reshape((6, 2)))
# now: model.output_shape == (None, 6, 2)
```

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L232)</span>
### Permute

```python
keras.layers.core.Permute(dims)
```

Permutes the dimensions of the input according to a given pattern.

Useful for e.g. connecting RNNs and convnets together.

__Example__


```python
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# now: model.output_shape == (None, 64, 10)
# note: `None` is the batch dimension
```

__Arguments__

- __dims__: Tuple of integers. Permutation pattern, does not include the
	samples dimension. Indexing starts at 1.
	For instance, `(2, 1)` permutes the first and second dimension
	of the input.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same as the input shape, but with the dimensions re-ordered according
to the specified pattern.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L314)</span>
### RepeatVector

```python
keras.layers.core.RepeatVector(n)
```

Repeats the input n times.

__Example__


```python
model = Sequential()
model.add(Dense(32, input_dim=32))
# now: model.output_shape == (None, 32)
# note: `None` is the batch dimension

model.add(RepeatVector(3))
# now: model.output_shape == (None, 3, 32)
```

__Arguments__

- __n__: integer, repetition factor.

__Input shape__

2D tensor of shape `(nb_samples, features)`.

__Output shape__

3D tensor of shape `(nb_samples, n, features)`.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/engine/topology.py#L1034)</span>
### Merge

```python
keras.engine.topology.Merge(layers=None, mode='sum', concat_axis=-1, dot_axes=-1, output_shape=None, node_indices=None, tensor_indices=None, name=None)
```

A `Merge` layer can be used to merge a list of tensors
into a single tensor, following some merge `mode`.

__Example usage__


```python
model1 = Sequential()
model1.add(Dense(32))

model2 = Sequential()
model2.add(Dense(32))

merged_model = Sequential()
merged_model.add(Merge([model1, model2], mode='concat', concat_axis=1)
- ____TODO__: would this actually work? it needs to.__

# achieve this with get_source_inputs in Sequential.
```

__Arguments__

- __layers__: can be a list of Keras tensors or
	a list of layer instances. Must be more
	than one layer/tensor.
- __mode__: string or lambda/function. If string, must be one
	- __of__: 'sum', 'mul', 'concat', 'ave', 'cos', 'dot'.
	If lambda/function, it should take as input a list of tensors
	and return a single tensor.
- __concat_axis__: integer, axis to use in mode `concat`.
- __dot_axes__: integer or tuple of integers, axes to use in mode `dot`.
- __output_shape__: shape tuple (tuple of integers), or lambda/function
	to compute output_shape (only if merge mode is a lambda/function).
	If the latter case, it should take as input a list of shape tuples
	- __(1__:1 mapping to input tensors) and return a single shape tuple.
- __node_indices__: optional list of integers containing
	the output node index for each input layer
	(in case some input layers have multiple output nodes).
	will default to an array of 0s if not provided.
- __tensor_indices__: optional list of indices of output tensors
	to consider for merging
	(in case some input layer node returns multiple tensors).

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L355)</span>
### Lambda

```python
keras.layers.core.Lambda(function, output_shape=None, arguments={})
```

Used for evaluating an arbitrary Theano / TensorFlow expression
on the output of the previous layer.

__Examples__


```python
# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2))
```
```python
# add a layer that returns the concatenation
# of the positive part of the input and
# the opposite of the negative part

def antirectifier(x):
	x -= K.mean(x, axis=1, keepdims=True)
	x = K.l2_normalize(x, axis=1)
	pos = K.relu(x)
	neg = K.relu(-x)
	return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
	shape = list(input_shape)
	assert len(shape) == 2  # only valid for 2D tensors
	shape[-1] *= 2
	return tuple(shape)

model.add(Lambda(antirectifier, output_shape=antirectifier_output_shape))
```

__Arguments__

- __function__: The function to be evaluated.
	Takes one argument: the output of previous layer
- __output_shape__: Expected output shape from function.
	Could be a tuple or a function of the shape of the input
- __arguments__: optional dictionary of keyword arguments to be passed
	to the function.

__Input shape__

Arbitrary. Use the keyword argument input_shape
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Specified by `output_shape` argument.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L651)</span>
### ActivityRegularization

```python
keras.layers.core.ActivityRegularization(l1=0.0, l2=0.0)
```

Layer that passes through its input unchanged, but applies an update
to the cost function based on the activity.

__Arguments__

- __l1__: L1 regularization factor (positive float).
- __l2__: L2 regularization factor (positive float).

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L20)</span>
### Masking

```python
keras.layers.core.Masking(mask_value=0.0)
```

Masks an input sequence by using a mask value to
identify timesteps to be skipped.

For each timestep in the input tensor (dimension #1 in the tensor),
if all values in the input tensor at that timestep
are equal to `mask_value`, then the timestep will masked (skipped)
in all downstream layers (as long as they support masking).

If any downstream layer does not support masking yet receives such
an input mask, an exception will be raised.

__Example__


Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
to be fed to a LSTM layer.
You want to mask timestep #3 and #5 because you lack data for
these timesteps. You can:

- set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
- insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
```

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L826)</span>
### Highway

```python
keras.layers.core.Highway(init='glorot_uniform', transform_bias=-2, activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)
```

Densely connected highway network,
a natural extension of LSTMs to feedforward networks.

__Arguments__

- __init__: name of initialization function for the weights of the layer
	(see [initializations](../initializations.md)),
	or alternatively, Theano function to use for weights
	initialization. This parameter is only relevant
	if you don't pass a `weights` argument.
- __transform_bias__: value for the bias to take on initially (default -2)
- __activation__: name of activation function to use
	(see [activations](../activations.md)),
	or alternatively, elementwise Theano function.
	If you don't specify anything, no activation is applied
	(ie. "linear" activation: a(x) = x).
- __weights__: list of numpy arrays to set as initial weights.
	The list should have 2 elements, of shape `(input_dim, output_dim)`
	and (output_dim,) for weights and biases respectively.
- __W_regularizer__: instance of [WeightRegularizer](../regularizers.md)
	(eg. L1 or L2 regularization), applied to the main weights matrix.
- __b_regularizer__: instance of [WeightRegularizer](../regularizers.md),
	applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
	applied to the network output.
- __W_constraint__: instance of the [constraints](../constraints.md) module
	(eg. maxnorm, nonneg), applied to the main weights matrix.
- __b_constraint__: instance of the [constraints](../constraints.md) module,
	applied to the bias.
- __bias__: whether to include a bias (i.e. make the layer affine rather than linear).
- __input_dim__: dimensionality of the input (integer).
	This argument (or alternatively, the keyword argument `input_shape`)
	is required when using this layer as the first layer in a model.

__Input shape__

2D tensor with shape: `(nb_samples, input_dim)`.

__Output shape__

2D tensor with shape: `(nb_samples, input_dim)`.

__References__

- [Highway Networks](http://arxiv.org/pdf/1505.00387v2.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L684)</span>
### MaxoutDense

```python
keras.layers.core.MaxoutDense(output_dim, nb_feature=4, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)
```

A dense maxout layer.

A `MaxoutDense` layer takes the element-wise maximum of
`nb_feature` `Dense(input_dim, output_dim)` linear layers.
This allows the layer to learn a convex,
piecewise linear activation function over the inputs.

Note that this is a *linear* layer;
if you wish to apply activation function
(you shouldn't need to --they are universal function approximators),
an `Activation` layer must be added after.

__Arguments__

- __output_dim__: int > 0.
- __nb_feature__: number of Dense layers to use internally.
- __init__: name of initialization function for the weights of the layer
	(see [initializations](../initializations.md)),
	or alternatively, Theano function to use for weights
	initialization. This parameter is only relevant
	if you don't pass a `weights` argument.
- __activation__: name of activation function to use
	(see [activations](../activations.md)),
	or alternatively, elementwise Theano function.
	If you don't specify anything, no activation is applied
	(ie. "linear" activation: a(x) = x).
- __weights__: list of numpy arrays to set as initial weights.
	The list should have 2 elements, of shape `(input_dim, output_dim)`
	and (output_dim,) for weights and biases respectively.
- __W_regularizer__: instance of [WeightRegularizer](../regularizers.md)
	(eg. L1 or L2 regularization), applied to the main weights matrix.
- __b_regularizer__: instance of [WeightRegularizer](../regularizers.md),
	applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
	applied to the network output.
- __W_constraint__: instance of the [constraints](../constraints.md) module
	(eg. maxnorm, nonneg), applied to the main weights matrix.
- __b_constraint__: instance of the [constraints](../constraints.md) module,
	applied to the bias.
- __bias__: whether to include a bias (i.e. make the layer affine rather than linear).
- __input_dim__: dimensionality of the input (integer).
	This argument (or alternatively, the keyword argument `input_shape`)
	is required when using this layer as the first layer in a model.

__Input shape__

2D tensor with shape: `(nb_samples, input_dim)`.

__Output shape__

2D tensor with shape: `(nb_samples, output_dim)`.

__References__

- [Maxout Networks](http://arxiv.org/pdf/1302.4389.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L964)</span>
### TimeDistributedDense

```python
keras.layers.core.TimeDistributedDense(output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
```

Apply a same Dense layer for each dimension[1] (time_dimension) input.
Especially useful after a recurrent network with 'return_sequence=True'.

- __Note__: this layer is deprecated, prefer using the `TimeDistributed` wrapper:
```python
model.add(TimeDistributed(Dense(32)))
```

__Input shape__

3D tensor with shape `(nb_sample, time_dimension, input_dim)`.
__Output shape__

3D tensor with shape `(nb_sample, time_dimension, output_dim)`.
__Arguments__

- __output_dim__: int > 0.
- __init__: name of initialization function for the weights of the layer
	(see [initializations](../initializations.md)),
	or alternatively, Theano function to use for weights
	initialization. This parameter is only relevant
	if you don't pass a `weights` argument.
- __activation__: name of activation function to use
	(see [activations](../activations.md)),
	or alternatively, elementwise Theano function.
	If you don't specify anything, no activation is applied
	(ie. "linear" activation: a(x) = x).
- __weights__: list of numpy arrays to set as initial weights.
	The list should have 2 elements, of shape `(input_dim, output_dim)`
	and (output_dim,) for weights and biases respectively.
- __W_regularizer__: instance of [WeightRegularizer](../regularizers.md)
	(eg. L1 or L2 regularization), applied to the main weights matrix.
- __b_regularizer__: instance of [WeightRegularizer](../regularizers.md),
	applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
	applied to the network output.
- __W_constraint__: instance of the [constraints](../constraints.md) module
	(eg. maxnorm, nonneg), applied to the main weights matrix.
- __b_constraint__: instance of the [constraints](../constraints.md) module,
	applied to the bias.
- __bias__: whether to include a bias (i.e. make the layer affine rather than linear).
- __input_dim__: dimensionality of the input (integer).
	This argument (or alternatively, the keyword argument `input_shape`)
	is required when using this layer as the first layer in a model.
- __input_length__: length of inputs sequences
	(integer, or None for variable-length sequences).
