### to_categorical


```python
to_categorical(y, nb_classes=None)
```


Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.

__Arguments__

- __y__: class vector to be converted into a matrix
- __nb_classes__: total number of classes

__Returns__

A binary matrix representation of the input.

----

### convert_kernel


```python
convert_kernel(kernel, dim_ordering='default')
```


Converts a kernel matrix (Numpy array)
from Theano format to TensorFlow format
(or reciprocally, since the transformation
is its own inverse).
