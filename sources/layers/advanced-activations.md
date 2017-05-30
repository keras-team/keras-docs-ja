<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L13)
</span>
### LeakyReLU

```python
keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
```

ユニットがアクティブでないときに微少な勾配を可能とするRectified Linear Unitの特別なバージョン：
`f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`.

__入力shape__

任意．このレイヤーをモデルの最初のレイヤーとして利用する場合，
`input_shape`というキーワード引数（サンプル数の軸を含まない整数のタプル）を指定してください．

__出力shape__

入力shapeと同じ．

__引数__

- __alpha__：0以上の浮動小数点数．負の部分の傾き．

__参考文献__

- [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L49)</span>
### PReLU

```python
keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
```

Parametric Rectified Linear Unit：
`f(x) = alphas * x for x < 0`,
`f(x) = x for x >= 0`,
`alphas`はxと同じshapeを持つ学習対象の配列です．

__入力shape__

任意．このレイヤーをモデルの最初のレイヤーとして利用する場合，
`input_shape`というキーワード引数（サンプル数の軸を含まない整数のタプル）を指定してください．

__出力shape__

入力shapeと同じ．

__引数__

- __alpha_initializer__：重みを初期化する関数．
- __alpha_regularizer__：重みを正則化する関数．
- __alpha_constraint__：重みに対する制約．
- __shared_axes__：活性化関数で共有する学習パラメータの軸．
	例えば，incoming feature mapsが，出力shapeとして`(batch, height, width, channels)`を持つ，2Dコンボリューションからなるもので，空間全体で各フィルターごとに一組しかパラメータを持たないたない場合にそのパラメータを共有したければ，`shared_axes=[1, 2]`とセットして下さい．

__参考文献__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L141)</span>
### ELU

```python
keras.layers.advanced_activations.ELU(alpha=1.0)
```

Exponential Linear Unit:
`f(x) =  alpha * (exp(x) - 1.) for x < 0`,
`f(x) = x for x >= 0`.

__入力shape__

任意．このレイヤーをモデルの最初のレイヤーとして利用する場合，
`input_shape`というキーワード引数（サンプル数の軸を含まない整数のタプル）を指定してください．

__出力shape__

入力shapeと同じ．

__引数__

- __alpha__：負の部分のscale．

__参考文献__

- [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289v1)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L177)</span>
### ThresholdedReLU

```python
keras.layers.advanced_activations.ThresholdedReLU(theta=1.0)
```

Thresholded Rectified Linear Unit:
`f(x) = x for x > theta`
`f(x) = 0 otherwise`.

__入力shape__

任意．このレイヤーをモデルの最初のレイヤーとして利用する場合，
`input_shape`というキーワード引数（サンプル数の軸を含まない整数のタプル）を指定してください．

__出力shape__

入力shapeと同じ．

__引数__

- __theta__：0以上の浮動小数点数．活性化する閾値．

__参考文献__

- [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/pdf/1402.3337.pdf)
