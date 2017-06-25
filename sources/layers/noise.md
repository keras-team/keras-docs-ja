<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/noise.py#L7)</span>
### GaussianNoise

```python
keras.layers.noise.GaussianNoise(stddev)
```

平均値0，ガウシアンノイズを加えます．

これはオーバーフィッティングの軽減に有効です（random data augmentationの一種）．
ガウシアンノイズ (GS) は実数値の入力におけるノイズ付与として一般的です．

regularization layerは訓練時のみ有効です．

__引数__

- __stddev__: 浮動小数点数，ノイズ分布の標準偏差値．

__入力のshape__

任意．
モデルの最初のレイヤーで使う場合は，`input_shape`キーワードで指定してください．
（整数のタプル（サンプルのaxisは含まない））

__出力のshape__

入力と同じ．

----
<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/noise.py#L7)</span>
### AlphaDropout

```python
keras.layers.noise.AlphaDropout(rate, noise_shape=None, seed=None)
```

入力にAlpha Dropoutを適用します．

Alpha Dropoutは，dropoutの適用後でもself-normalizingの性質を担保するために入力のもともとの値の平均と分散を保持しつつ、`Dropout`を行います．
Alpha Dropoutは，活性化値にランダムに負の飽和値をセットするために、Scaled Exponential Linear Unitsと相性が良いです．

__引数__

- __rate__: 浮動小数点数，drop probability (`Dropout`同様)．平均1，標準偏差値`sqrt(rate / (1 - rate))`のノイズを乗じます．
- __seed__: 整数．乱数のシードに使います．

__入力のshape__

任意．
モデルの最初のレイヤーで使う場合は，`input_shape`キーワードで指定してください．
（整数のタプル（サンプルのaxisは含まない））

__出力のshape__

入力と同じ．

__参考文献__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/noise.py#L45)</span>
### GaussianDropout

```python
keras.layers.noise.GaussianDropout(rate)
```

平均値1，ガウシアンノイズを乗じます．

regularization layerは訓練時のみ有効です．

__引数__

- __rate__: 浮動小数点数，drop probability（`Dropout`同様）．平均1，標準偏差値`sqrt(rate / (1 - rate))`のノイズを乗じます．

__入力のshape__

任意．
モデルの最初のレイヤーで使う場合は，`input_shape`キーワードで指定してください．
（整数のタプル（サンプルのaxisは含まない））

__出力のshape__

入力と同じ．

__参考文献__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
