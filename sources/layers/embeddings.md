<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/embeddings.py#L8)</span>
### Embedding

```python
keras.layers.embeddings.Embedding(input_dim, output_dim, init='uniform', input_length=None, W_regularizer=None, activity_regularizer=None, W_constraint=None, mask_zero=False, weights=None, dropout=0.0)
```

Turn positive integers (indexes) into dense vectors of fixed size.
eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

This layer can only be used as the first layer in a model.

__Example__


```python
  model = Sequential()
  model.add(Embedding(1000, 64, input_length=10))
  # the model will take as input an integer matrix of size (batch, input_length).
  # the largest integer (i.e. word index) in the input should be no larger than 1000 (vocabulary size).
  # now model.output_shape == (None, 10, 64), where None is the batch dimension.

  input_array = np.random.randint(1000, size=(32, 10))

  model.compile('rmsprop', 'mse')
  output_array = model.predict(input_array)
  assert output_array.shape == (32, 10, 64)
```

__Arguments__

  - __input_dim__: int >= 0. Size of the vocabulary, ie.
  1 + maximum integer index occurring in the input data.
  - __output_dim__: int >= 0. Dimension of the dense embedding.
  - __init__: name of initialization function for the weights
  of the layer (see: [initializations](../initializations.md)),
  or alternatively, Theano function to use for weights initialization.
  This parameter is only relevant if you don't pass a `weights` argument.
  - __weights__: list of numpy arrays to set as initial weights.
  The list should have 1 element, of shape `(input_dim, output_dim)`.
  - __W_regularizer__: instance of the [regularizers](../regularizers.md) module
(eg. L1 or L2 regularization), applied to the embedding matrix.
  - __W_constraint__: instance of the [constraints](../constraints.md) module
  (eg. maxnorm, nonneg), applied to the embedding matrix.
  - __mask_zero__: Whether or not the input value 0 is a special "padding"
  value that should be masked out.
  This is useful for [recurrent layers](recurrent.md) which may take
  variable length input. If this is `True` then all subsequent layers
  in the model need to support masking or an exception will be raised.
  - __input_length__: Length of input sequences, when it is constant.
  This argument is required if you are going to connect
  `Flatten` then `Dense` layers upstream
  (without it, the shape of the dense outputs cannot be computed).
  - __dropout__: float between 0 and 1. Fraction of the embeddings to drop.

__Input shape__

2D tensor with shape: `(nb_samples, sequence_length)`.

__Output shape__

3D tensor with shape: `(nb_samples, sequence_length, output_dim)`.

__References__

- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
