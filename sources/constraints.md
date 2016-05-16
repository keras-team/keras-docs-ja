## 制約の利用方法

`constraints`モジュールの関数により，最適化中のネットワークパラメータに制約（例えば非負の制約）を設定することができます．

この制約はレイヤー毎に適用されます．厳密なAPIはレイヤーに依存しますが，`Dense`，`TimeDistributedDense`，`MaxoutDense`，`Convolution1D`，そして，`Convolution2D`レイヤーは統一的なAPIを持っています．

これらのレイヤーは2つの引数を取ります:

- `W_constraint` 重み行列の制約．
- `b_constraint` バイアスの制約.

## 例

```python
from keras.constraints import maxnorm
model.add(Dense(64, W_constraint = maxnorm(2)))
```

## 利用可能な制約

- __maxnorm__(m=2): 最大値ノルム制約
- __nonneg__(): 非負値制約
- __unitnorm__(): ノルム正規化制約．指定した軸でノルムを正規化します．