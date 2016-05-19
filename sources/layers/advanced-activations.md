<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L7)</span>
### LeakyReLU

```python
keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
```

ユニットがアクティブでないときに微少な勾配を可能とするRectified Linear Unitの特別なバージョン．
`f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`.

__入力形状__

任意．このレイヤーをモデルの1番目のレイヤーとして利用する場合，
`input_shape`というキーワード引数を利用してください（サンプル数の軸を含まない整数のタプル）．

__出力形状__

入力形状と同じ．

__引数__

- __alpha__: 0以上の浮動小数点数．Negative slope coefficient.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L38)</span>
### PReLU

```python
keras.layers.advanced_activations.PReLU(init='zero', weights=None)
```

Parametric Rectified Linear Unit:
`f(x) = alphas * x for x < 0`,
`f(x) = x for x >= 0`,
`alphas`はxと同じ形状の配列から学習されます．

__入力形状__

任意．このレイヤーをモデルの1番目のレイヤーとして利用する場合，
`input_shape`というキーワード引数を利用してください（サンプル数の軸を含まない整数のタプル）．

__出力形状__

入力形状と同じ．

__引数__

- __init__: 重みを初期化する関数．
- __weights__: 重みの初期値．1つのnumpy配列を持つリスト．

__参考文献__

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

__入力形状__

任意．このレイヤーをモデルの1番目のレイヤーとして利用する場合，
`input_shape`というキーワード引数を利用してください（サンプル数の軸を含まない整数のタプル）．

__出力形状__

入力形状と同じ．

__引数__

- __alpha__: scale for the negative factor.

__参考文献__

- [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/pdf/1511.07289v1.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L120)</span>
### ParametricSoftplus

```python
keras.layers.advanced_activations.ParametricSoftplus(alpha_init=0.2, beta_init=5.0, weights=None)
```

Parametric Softplus:
`alpha * log(1 + exp(beta * x))`

__入力形状__

任意．このレイヤーをモデルの1番目のレイヤーとして利用する場合，
`input_shape`というキーワード引数を利用してください（サンプル数の軸を含まない整数のタプル）．

__出力形状__

入力形状と同じ．

__引数__

- __alpha_init__: 浮動小数点数．alphaの重みの初期値．
- __beta_init__: 浮動小数点数．betaの重みの初期値．
- __weights__: 重みの初期値．2つのnumpy配列から構成されるリスト．

__参考文献__

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

__入力形状__

任意．このレイヤーをモデルの1番目のレイヤーとして利用する場合，
`input_shape`というキーワード引数を利用してください（サンプル数の軸を含まない整数のタプル）．

__出力形状__

入力形状と同じ．

__引数__

- __theta__: 0以上の浮動小数点数．活性化する閾値．

__参考文献__

- [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/pdf/1402.3337.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L203)</span>
### SReLU

```python
keras.layers.advanced_activations.SReLU(t_left_init='zero', a_left_init='glorot_uniform', t_right_init='glorot_uniform', a_right_init='one')
```

S-shaped Rectified Linear Unit.

__入力形状__

任意．このレイヤーをモデルの1番目のレイヤーとして利用する場合，
`input_shape`というキーワード引数を利用してください（サンプル数の軸を含まない整数のタプル）．

__出力形状__

入力形状と同じ．

__引数__

- __t_left_init__: initialization function for the left part intercept
- __a_left_init__: initialization function for the left part slope
- __t_right_init__: initialization function for the right part intercept
- __a_right_init__: initialization function for the right part slope

__参考文献__

- [Deep Learning with S-shaped Rectified Linear Activation Units](http://arxiv.org/abs/1512.07030)
