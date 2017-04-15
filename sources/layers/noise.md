<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/noise.py#L7)</span>
### GaussianNoise

```python
keras.layers.noise.GaussianNoise(stddev)
```

平均値0，ガウシアンノイズを加えます．

これはオーバーフィッティングの軽減に有効です (random data augmentationの一種)．
ガウシアンノイズ (GS) は実数値の入力におけるノイズ付与として一般的です．

regularization layerは訓練時のみ有効です．

__Arguments__

- __stddev__: float，ノイズ分布の標準偏差値．

__Input shape__

任意．
モデルの最初のレイヤーで使う場合は，`input_shape`キーワードで指定してください．
(整数のタプル(サンプルのaxisは含まない))

__Output shape__

入力と同じ．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/noise.py#L45)</span>
### GaussianDropout

```python
keras.layers.noise.GaussianDropout(rate)
```

平均値1，ガウシアンノイズを乗じます．

regularization layerは訓練時のみ有効です．

__Arguments__

- __rate__: float，drop probability (`Dropout`同様)．平均1，標準偏差値`sqrt(rate / (1 - rate))`のノイズを乗じます．

__Input shape__

任意．
モデルの最初のレイヤーで使う場合は，`input_shape`キーワードで指定してください．
(整数のタプル(サンプルのaxisは含まない))

__Output shape__

入力と同じ．

__References__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
