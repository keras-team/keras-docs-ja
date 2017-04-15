<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/normalization.py#L6)</span>
### BatchNormalization

```python
keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
```

Batch normalization layer (Ioffe and Szegedy, 2014)．

各バッチ毎に前の層の出力（このレイヤーへの入力）を正規化します．
つまり，平均を0，標準偏差値を1に近づける変換を適用します．

__Arguments__

- __axis__: 整数．正規化する軸 (典型的には，特徴量の軸)．例えば，`data_format="channels_first"`の`Conv2D`の後では，`axis=1`にします．
- __momentum__: 移動平均のためのMomentum．
- __epsilon__: 0除算を避けるために分散加算する小さなfloat値．
- __center__: Trueなら，正規化されたテンソルに`beta`のオフセットを加算します．Falseなら, `beta`は無視します．
- _scale_: Trueなら, `gamma`をかけます．Falseなら, `gamma`は使われません．次のレイヤーがlinear (例えば `nn.relu` も)ならば，次のレイヤーによってスケーリングされるので無効にできます．
- __beta_initializer__: betaの重みのためのInitializer．
- __gamma_initializer__: gammaの重みのためのInitializer．
- __moving_mean_initializer__: 移動平均のためのInitializer．
- __moving_variance_initializer__: 移動分散のためのInitializer．
- __beta_regularizer__: betaの重みのためのオプショナルなRegularizer．
- __gamma_regularizer__: gammaの重みのためのオプショナルなRegularizer．
- __beta_constraint__: betaの重みのためのオプショナルなConstraint．
- __gamma_constraint__: gammaの重みのためのオプショナルなConstraint．

__Input shape__

任意．
このレイヤーがモデルの最初のレイヤーとなる場合は，`input_shape`引数（サンプル軸を含まない整数のタプル）を与える必要があります．

__Output shape__

Input shapeと同じです．

__参考文献__

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://jmlr.org/proceedings/papers/v37/ioffe15.html)
