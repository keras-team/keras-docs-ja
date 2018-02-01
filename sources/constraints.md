## 制約の利用方法

`constraints`モジュールの関数により，最適化中のネットワークパラメータに制約（例えば非負の制約）を設定することができます．

この制約はレイヤー毎に適用されます．厳密なAPIはレイヤーに依存しますが，`Dense`，`Conv1D`，`Conv2D`，`Conv3D`レイヤーは統一的なAPIを持っています．

これらのレイヤーは2つの引数を取ります:

- `kernel_constraint` 重み行列の制約．
- `bias_constraint` バイアスの制約．

```python
from keras.constraints import max_norm
model.add(Dense(64, kernel_constraint=max_norm(2.)))
```

## 利用可能な制約

- __maxnorm(max_value=2, axis=0)__: 最大値ノルム制約
- __non_neg()__: 非負値制約
- __unit_norm(axis=0)__: ノルム正規化制約
- __min_max_norm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)__: 最小/最大値ノルム制約
