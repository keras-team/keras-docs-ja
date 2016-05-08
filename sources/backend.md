# Keras backends

## What is a "backend"?

Keras is a model-level library, providing high-level building blocks for developing deep learning models. It does not handle itself low-level operations such as tensor products, convolutions and so on. Instead, it relies on a specialized, well-optimized tensor manipulation library to do so, serving as the "backend engine" of Keras. Rather than picking one single tensor library and making the implementation of Keras tied to that library, Keras handles the problem in a modular way, and several different backend engines can be plugged seamlessly into Keras.

At this time, Keras has two backend implementations available: the **Theano** backend and the **TensorFlow** backend.

- [Theano](http://deeplearning.net/software/theano/) is an open-source symbolic tensor manipulation framework developed by LISA/MILA Lab at Université de Montréal.
- [TensorFlow](http://www.tensorflow.org/) is an open-source symbolic tensor manipulation framework developed by Google, Inc.

----

## Switching from one backend to another

If you have run Keras at least once, you will find the Keras configuration file at:

`~/.keras/keras.json`

If it isn't there, you can create it.

It probably looks like this:

`{"epsilon": 1e-07, "floatx": "float32", "backend": "theano"}`

Simply change the field `backend` to either `"theano"` or `"tensorflow"`, and Keras will use the new configuration next time you run any Keras code.

You can also define the environment variable ``KERAS_BACKEND`` and this will
override what is defined in your config file :

```bash
KERAS_BACKEND=tensorflow python -c "from keras import backend; print backend._BACKEND"
Using TensorFlow backend.
tensorflow
```

----

## Using the abstract Keras backend to write new code

If you want the Keras modules you write to be compatible with both Theano and TensorFlow, you have to write them via the abstract Keras backend API. Here's an intro.

You can import the backend module via:
```python
from keras import backend as K
```

The code below instantiates an input placeholder. It's equivalent to `tf.placeholder()` or `T.matrix()`, `T.tensor3()`, etc.

```python
input = K.placeholder(shape=(2, 4, 5))
# also works:
input = K.placeholder(shape=(None, 4, 5))
# also works:
input = K.placeholder(ndim=3)
```

The code below instantiates a shared variable. It's equivalent to `tf.variable()` or `theano.shared()`.

```python
val = np.random.random((3, 4, 5))
var = K.variable(value=val)

# all-zeros variable:
var = K.zeros(shape=(3, 4, 5))
# all-ones:
var = K.ones(shape=(3, 4, 5))
```

Most tensor operations you will need can be done as you would in TensorFlow or Theano:

```python
a = b + c * K.abs(d)
c = K.dot(a, K.transpose(b))
a = K.sum(b, axis=2)
a = K.softmax(b)
a = concatenate([b, c], axis=-1)
# etc...
```

----

## Backend functions


### learning_phase


```python
learning_phase()
```


Returns the learning phase flag.

The learning phase flag is an integer tensor (0 = test, 1 = train)
to be passed as input to any Keras function
that uses a different behavior at train time and test time.

----

### floatx


```python
floatx()
```


Returns the default float type, as a string
(e.g. 'float16', 'float32', 'float64').

----

### cast_to_floatx


```python
cast_to_floatx(x)
```


Cast a Numpy array to floatx.

----

### shape


```python
shape(x)
```


Returns the symbolic shape of a tensor.

----

### variable


```python
variable(value, dtype='float32', name=None)
```


Instantiates a tensor.

__Arguments__

- __value__: numpy array, initial value of the tensor.
- __dtype__: tensor type.
- __name__: optional name string for the tensor.

__Returns__

Tensor variable instance.

----

### placeholder


```python
placeholder(shape=None, ndim=None, dtype='float32', name=None)
```


Instantiates a placeholder.

__Arguments__

- __shape__: shape of the placeholder
(integer tuple, may include None entries).
- __ndim__: number of axes of the tensor.
At least one of {`shape`, `ndim`} must be specified.
If both are specified, `shape` is used.
- __dtype__: placeholder type.
- __name__: optional name string for the placeholder.

__Returns__

Placeholder tensor instance.

----

### int_shape


```python
int_shape(x)
```


Returns the shape of a tensor as a tuple of
integers or None entries.

----

### ndim


```python
ndim(x)
```


Returns the number of axes in a tensor, as an integer.

----

### dtype


```python
dtype(x)
```


Returns the dtype of a tensor, as a string.

----

### eval


```python
eval(x)
```


Evaluates the value of a tensor.
Returns a Numpy array.

----

### zeros


```python
zeros(shape, dtype='float32', name=None)
```


Instantiates an all-zeros tensor variable.

----

### ones


```python
ones(shape, dtype='float32', name=None)
```


Instantiates an all-ones tensor variable.

----

### eye


```python
eye(size, dtype='float32', name=None)
```


Instantiate an identity matrix.

----

### zeros_like


```python
zeros_like(x, name=None)
```


Instantiates an all-zeros tensor
of the same shape as another tensor.

----

### ones_like


```python
ones_like(x, name=None)
```


Instantiates an all-ones tensor
of the same shape as another tensor.

----

### count_params


```python
count_params(x)
```


Returns the number of scalars in a tensor.

----

### cast


```python
cast(x, dtype)
```


Casts a tensor to a different dtype.

----

### dot


```python
dot(x, y)
```


Multiplies 2 tensors.
When attempting to multiply a ND tensor
with a ND tensor, reproduces the Theano behavior
(e.g. (2, 3).(4, 3, 5) = (2, 4, 5))

----

### batch_dot


```python
batch_dot(x, y, axes=None)
```


Batchwise dot product.

batch_dot results in a tensor with less dimensions than the input.
If the number of dimensions is reduced to 1, we use `expand_dims` to
make sure that ndim is at least 2.

__Example__

Assume x = [[1, 2]   and y = [[5, 6]
	[3, 4]]   [7, 8]]
batch_dot(x, y, axes=1) = [[17, 53]] which is the main diagonal
of x.dot(y.T), although we never have to calculate the off-diagonal
elements.


__Arguments__

x, y: tensors with ndim >= 2
- __axes__: list (or single) int with target dimensions

__Returns__

Tensor with ndim >= 2

----

### transpose


```python
transpose(x)
```


Transposes a matrix.

----

### gather


```python
gather(reference, indices)
```


Retrieves the vectors of indices `indices`
in the 2D tensor `reference`.

__Arguments__

- __reference__: a 2D tensor.
- __indices__: an int tensor of indices.

__Returns__

A 3D tensor of same type as `reference`.

----

### max


```python
max(x, axis=None, keepdims=False)
```


Maximum value in a tensor.

----

### min


```python
min(x, axis=None, keepdims=False)
```


Minimum value in a tensor.

----

### sum


```python
sum(x, axis=None, keepdims=False)
```


Sum of the values in a tensor, alongside the specified axis.

----

### prod


```python
prod(x, axis=None, keepdims=False)
```


Multiplies the values in a tensor, alongside the specified axis.

----

### std


```python
std(x, axis=None, keepdims=False)
```


Standard deviation of a tensor, alongside the specificied axis.

----

### mean


```python
mean(x, axis=None, keepdims=False)
```


Mean of a tensor, alongside the specificied axis.

----

### any


```python
any(x, axis=None, keepdims=False)
```


Bitwise reduction (logical OR).

Returns an uint8 tensor (0s and 1s).

----

### argmax


```python
argmax(x, axis=-1)
```


Returns the index of the maximum value
along a tensor axis.

----

### argmin


```python
argmin(x, axis=-1)
```


Returns the index of the minimum value
along a tensor axis.

----

### square


```python
square(x)
```


Element-wise square.

----

### abs


```python
abs(x)
```


Element-wise absolute value.

----

### sqrt


```python
sqrt(x)
```


Element-wise square root.

----

### exp


```python
exp(x)
```


Element-wise exponential.

----

### log


```python
log(x)
```


Element-wise log.

----

### round


```python
round(x)
```


Element-wise rounding to the closest integer.

----

### sign


```python
sign(x)
```


Element-wise sign.

----

### pow


```python
pow(x, a)
```


Element-wise exponentiation.

----

### clip


```python
clip(x, min_value, max_value)
```


Element-wise value clipping.

----

### equal


```python
equal(x, y)
```


Element-wise equality between two tensors.
Returns a bool tensor.

----

### not_equal


```python
not_equal(x, y)
```


Element-wise inequality between two tensors.
Returns a bool tensor.

----

### maximum


```python
maximum(x, y)
```


Element-wise maximum of two tensors.

----

### minimum


```python
minimum(x, y)
```


Element-wise minimum of two tensors.

----

### sin


```python
sin(x)
```


Computes sin of x element-wise.

----

### cos


```python
cos(x)
```


Computes cos of x element-wise.

----

### concatenate


```python
concatenate(tensors, axis=-1)
```


Concantes a list of tensors alongside the specified axis.

----

### reshape


```python
reshape(x, shape)
```


Reshapes a tensor to the specified shape.

----

### permute_dimensions


```python
permute_dimensions(x, pattern)
```


Permutes axes in a tensor.

__Arguments__

- __pattern__: should be a tuple of
dimension indices, e.g. (0, 2, 1).

----

### resize_images


```python
resize_images(X, height_factor, width_factor, dim_ordering)
```


Resizes the images contained in a 4D tensor of shape
- [batch, channels, height, width] (for 'th' dim_ordering)
- [batch, height, width, channels] (for 'tf' dim_ordering)
by a factor of (height_factor, width_factor). Both factors should be
positive integers.

----

### repeat_elements


```python
repeat_elements(x, rep, axis)
```


Repeats the elements of a tensor along an axis, like np.repeat

If x has shape (s1, s2, s3) and axis=1, the output
will have shape (s1, s2 * rep, s3)

----

### repeat


```python
repeat(x, n)
```


Repeats a 2D tensor:

if x has shape (samples, dim) and n=2,
the output will have shape (samples, 2, dim)

----

### batch_flatten


```python
batch_flatten(x)
```


Turn a n-D tensor into a 2D tensor where
the first dimension is conserved.

----

### expand_dims


```python
expand_dims(x, dim=-1)
```


Adds a 1-sized dimension at index "dim".

----

### squeeze


```python
squeeze(x, axis)
```


Removes a 1-dimension from the tensor at index "axis".

----

### temporal_padding


```python
temporal_padding(x, padding=1)
```


Pads the middle dimension of a 3D tensor
with "padding" zeros left and right.

----

### spatial_2d_padding


```python
spatial_2d_padding(x, padding=(1, 1), dim_ordering='th')
```


Pads the 2nd and 3rd dimensions of a 4D tensor
with "padding[0]" and "padding[1]" (resp.) zeros left and right.

----

### get_value


```python
get_value(x)
```


Returns the value of a tensor variable,
as a Numpy array.

----

### batch_get_value


```python
batch_get_value(xs)
```


Returns the value of more than one tensor variable,
as a list of Numpy arrays.

----

### set_value


```python
set_value(x, value)
```


Sets the value of a tensor variable,
from a Numpy array.

----

### batch_set_value


```python
batch_set_value(tuples)
```


Sets the values of many tensor variables at once.

__Arguments__

- __tuples__: a list of tuples `(tensor, value)`.
`value` should be a Numpy array.

----

### function


```python
function(inputs, outputs, updates=[])
```


Instantiates a Keras function.

__Arguments__

- __inputs__: list of placeholder/variable tensors.
- __outputs__: list of output tensors.
- __updates__: list of update tuples (old_tensor, new_tensor).

----

### gradients


```python
gradients(loss, variables)
```


Returns the gradients of `variables` (list of tensor variables)
with regard to `loss`.

----

### rnn


```python
rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```


Iterates over the time dimension of a tensor.

__Arguments__

- __inputs__: tensor of temporal data of shape (samples, time, ...)
(at least 3D).
- __step_function__:
- __Parameters__:
	- __input__: tensor with shape (samples, ...) (no time dimension),
	representing input for the batch of samples at a certain
	time step.
	- __states__: list of tensors.
- __Returns__:
	- __output__: tensor with shape (samples, ...) (no time dimension),
	- __new_states__: list of tensors, same length and shapes
	as 'states'.
- __initial_states__: tensor with shape (samples, ...) (no time dimension),
containing the initial values for the states used in
the step function.
- __go_backwards__: boolean. If True, do the iteration over
the time dimension in reverse order.
- __mask__: binary tensor with shape (samples, time, 1),
with a zero for every element that is masked.
- __constants__: a list of constant values passed at each step.
- __unroll__: with TensorFlow the RNN is always unrolled, but with Theano you
can use this boolean flag to unroll the RNN.
- __input_length__: not relevant in the TensorFlow implementation.
Must be specified if using unrolling with Theano.

__Returns__

A tuple (last_output, outputs, new_states).

- __last_output__: the latest output of the rnn, of shape (samples, ...)
- __outputs__: tensor with shape (samples, time, ...) where each
entry outputs[s, t] is the output of the step function
at time t for sample s.
- __new_states__: list of tensors, latest states returned by
the step function, of shape (samples, ...).

----

### switch


```python
switch(condition, then_expression, else_expression)
```


Switches between two operations depending on a scalar value (int or bool).
Note that both `then_expression` and `else_expression`
should be symbolic tensors of the *same shape*.

__Arguments__

- __condition__: scalar tensor.
- __then_expression__: TensorFlow operation.
- __else_expression__: TensorFlow operation.

----

### in_train_phase


```python
in_train_phase(x, alt)
```


Selects `x` in train phase, and `alt` otherwise.
Note that `alt` should have the *same shape* as `x`.

----

### in_test_phase


```python
in_test_phase(x, alt)
```


Selects `x` in test phase, and `alt` otherwise.
Note that `alt` should have the *same shape* as `x`.

----

### relu


```python
relu(x, alpha=0.0, max_value=None)
```


Rectified linear unit

__Arguments__

- __alpha__: slope of negative section.
- __max_value__: saturation threshold.

----

### softmax


```python
softmax(x)
```


Softmax of a tensor.

----

### softplus


```python
softplus(x)
```


Softplus of a tensor.

----

### categorical_crossentropy


```python
categorical_crossentropy(output, target, from_logits=False)
```


Categorical crossentropy between an output tensor
and a target tensor, where the target is a tensor of the same
shape as the output.

----

### sparse_categorical_crossentropy


```python
sparse_categorical_crossentropy(output, target, from_logits=False)
```


Categorical crossentropy between an output tensor
and a target tensor, where the target is an integer tensor.

----

### binary_crossentropy


```python
binary_crossentropy(output, target, from_logits=False)
```


Binary crossentropy between an output tensor and a target tensor.

----

### sigmoid


```python
sigmoid(x)
```


Element-wise sigmoid.

----

### hard_sigmoid


```python
hard_sigmoid(x)
```


Segment-wise linear approximation of sigmoid.
Faster than sigmoid.

----

### tanh


```python
tanh(x)
```


Element-wise tanh.

----

### dropout


```python
dropout(x, level, seed=None)
```


Sets entries in `x` to zero at random,
while scaling the entire tensor.

__Arguments__

- __x__: tensor
- __level__: fraction of the entries in the tensor
that will be set to 0
- __seed__: random seed to ensure determinism.

----

### l2_normalize


```python
l2_normalize(x, axis)
```


Normalizes a tensor wrt the L2 norm alonside the specified axis.

----

### conv2d


```python
conv2d(x, kernel, strides=(1, 1), border_mode='valid', dim_ordering='th', image_shape=None, filter_shape=None)
```


2D convolution.

__Arguments__

- __kernel__: kernel tensor.
- __strides__: strides tuple.
- __border_mode__: string, "same" or "valid".
- __dim_ordering__: "tf" or "th". Whether to use Theano or TensorFlow dimension ordering
in inputs/kernels/ouputs.

----

### pool2d


```python
pool2d(x, pool_size, strides=(1, 1), border_mode='valid', dim_ordering='th', pool_mode='max')
```


2D Pooling.

__Arguments__

- __pool_size__: tuple of 2 integers.
- __strides__: tuple of 2 integers.
- __border_mode__: one of "valid", "same".
- __dim_ordering__: one of "th", "tf".
- __pool_mode__: one of "max", "avg".






