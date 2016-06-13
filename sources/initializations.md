
## レイヤーの重み初期化方法

初期化用引数で，Kerasレイヤーの重みをランダムに初期化する確率分布を指定できます．

初期化用引数のキーワードはレイヤーにより異なりますが，大抵は単純に`init`です:

```python
model.add(Dense(64, init='uniform'))
```

## 利用可能な初期化方法

- __uniform__
- __lecun_uniform__: input数の平方根でスケーリングした一様分布 (LeCun 98)
- __normal__
- __identity__: `shape[0] == shape[1]`の2次元のレイヤーで使えます
- __orthogonal__: `shape[0] == shape[1]`の2次元のレイヤーで使えます
- __zero__
- __glorot_normal__: fan_in + fan_outでスケーリングした正規分布 (Glorot 2010)
- __glorot_uniform__
- __he_normal__: fan_inでスケーリングした正規分布 (He et al., 2014)
- __he_uniform__

