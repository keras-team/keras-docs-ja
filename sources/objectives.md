
## 目的関数の利用方法

目的関数（ロス関数や最適スコア関数）はモデルをコンパイルする際に必要となるパラメータの1つです:

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

既存の目的関数の名前を引数に与えるか，各データ点に対してスカラを返し，以下の2つの引数を取るTheano/TensorFlowのシンボリック関数を与えることができます:

- __y_true__: 正解ラベル．Theano/TensorFlow テンソル
- __y_pred__: 予測．y_trueと同じ形状のTheano/TensorFlow テンソル

実際の目的関数は全データ点における出力の平均です．

このような関数の実装例に関しては，[objectives source](https://github.com/fchollet/keras/blob/master/keras/objectives.py)を参照してください．

## 利用可能な目的関数

- __mean_squared_error__ / __mse__
- __mean_absolute_error__ / __mae__
- __mean_absolute_percentage_error__ / __mape__
- __mean_squared_logarithmic_error__ / __msle__
- __squared_hinge__
- __hinge__
- __binary_crossentropy__: loglossとしても知られています．
- __categorical_crossentropy__: マルチクラスloglossとしても知られています． __Note__: この目的関数を使うには，ラベルがバイナリ配列であり，その形状が`(nb_samples, nb_classes)`であることが必要です．
- __sparse_categorical_crossentropy__: categorical_crossentropyと同じですが，スパースラベルを取る点で違います． __Note__: ラベルの次元と出力の次元が同じである必要があります．例えば，ラベル形状を拡張するために，`np.expand_dims(y, -1)`を用いて新しく次元を追加する必要があるかもしれません．
 - __kullback_leibler_divergence__ / __kld__: 予測した確率分布Qから真の確率分布Pへの情報ゲイン．2つの分布の異なりの度合いを得る．
- __poisson__: `(予測 - 正解 * log(予測))`の平均
- __cosine_proximity__: 予測と正解間のコサイン類似度の負の平均．
