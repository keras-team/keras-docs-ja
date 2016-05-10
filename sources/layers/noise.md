<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/noise.py#L6)</span>
### GaussianNoise

```python
keras.layers.noise.GaussianNoise(sigma)
```

入力に平均0，標準偏差`sigma`のガウシアンノイズを加えます。
これはオーバーフィッティングの低減に有効です(random data augmentationの一種)。
ガウシアンノイズは入力が実数値のときのノイズ付与として一般的です。

regularization layerは訓練時のみ有効です。

__Arguments__

- __sigma__: float, ノイズ分布の標準偏差

__Input shape__

任意。
モデルの最初のレイヤーで`input_shape`キーワードで指定してください。
(整数のタプル(データ数の軸は含まない))

__Output shape__

入力と同じ。

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/noise.py#L44)</span>
### GaussianDropout

```python
keras.layers.noise.GaussianDropout(p)
```

入力に平均1，標準偏差`sqrt(p/(1-p))`のガウシアンノイズを乗じます。

regularization layerは訓練時のみ有効です。

__Arguments__

- __p__: float, 制御パラメータ (`Dropout`同様).

__Input shape__

任意。
モデルの最初のレイヤーで`input_shape`キーワードで指定してください。
(整数のタプル(データ数の軸は含まない))

__Output shape__

入力と同じ。

__References__

- __[Dropout__: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

