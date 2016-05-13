## 正則化の利用方法

正則化によって，最適化中にレイヤーパラメータあるいはレイヤーの出力に制約を課すことができます．これらの制約はネットワークが最適化するロス関数に組み込まれます．

この制約はレイヤー毎に適用されます．厳密なAPIはレイヤーに依存しますが，`Dense`，`TimeDistributedDense`，`MaxoutDense`，`Convolution1D`，そして，`Convolution2D`レイヤーは統一的なAPIを持っています．

これらのレイヤーは3つの引数を取ります:

- `W_regularizer`: `keras.regularizers.WeightRegularizer` のインスタンス
- `b_regularizer`: `keras.regularizers.WeightRegularizer` のインスタンス
- `activity_regularizer`: `keras.regularizers.ActivityRegularizer` のインスタンス


## 例

```python
from keras.regularizers import l2, activity_l2
model.add(Dense(64, input_dim=64, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
```

## 利用可能な制約

```python
keras.regularizers.WeightRegularizer(l1=0., l2=0.)
```

```python
keras.regularizers.ActivityRegularizer(l1=0., l2=0.)
```

## ショートカット

`keras.regularizers` の中に利用可能なショートカット関数があります．

- __l1__(l=0.01): 重みのL1正則化．LASSOとしても知られています．
- __l2__(l=0.01): 重みのL2正則化．荷重減衰やRidgeとしても知られています．
- __l1l2__(l1=0.01, l2=0.01): 重みのL1+L2正則化．ElasticNetとしても知られています．
- __activity_l1__(l=0.01): 出力のL1正則化．
- __activity_l2__(l=0.01): 出力のL2正則化．
- __activity_l1l2__(l1=0.01, l2=0.01): 出力のL1+L2正則化．
