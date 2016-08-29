<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/normalization.py#L6)</span>
### BatchNormalization

```python
keras.layers.normalization.BatchNormalization(epsilon=1e-05, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one')
```

各バッチ毎に前の層の出力（このレイヤーへの入力）を正規化します．
つまり，平均を0，標準偏差を1に近づけるような変換を適用します．

__引数__

- __epsilon__: small float > 0. 微少な値．
- __mode__: 0，1または2の整数．
    - 0: feature-wise正規化．
       入力の各特徴マップが別々に正規化されます．どの軸で正規化するかは`axis`引数で指定します．
       もし入力がTheanoの慣例である4次元の画像テンソル (samples, channels, rows, cols) の場合，
       チャンネル軸を正規化するためには`axis=1`とする必要があることに注意してください．
       学習時はデータの正規化にバッチごとの統計量を使用し， テスト時は学習時の移動平均を使います．
    - 1: sample-wise正規化．このモードは2次元の入力を想定します．
    - 2: modeが0の場合と同様，feature-wise正規化をしますが，学習時とテスト時の両方ともデータの正規化にバッチごとの統計量を使用します．
- __axis__: 整数．modeが0のときに，どの軸を正規化するかを指定します．
    例えば，(samples, channels, rows, cols) の形状を持つテンソルを入力するとき，
    特徴マップ（チャンネル軸）毎に正規化するためには，`axis=1`とします．
- __momentum__: feature-wise正規化に対して，データの平均と標準偏差の指数平均を計算する際のモーメンタム．
- __weights__: 重みの初期値．
    2つのNumpy配列を要素に持つリスト: `[(input_shape,), (input_shape,)]`
    このリストは， [gamma, beta, mean, std] の順であることに注意してください．
- __beta_init__: シフトパラメータのための初期化関数名 ([initializations](../initializations.md))，
    または，重みの初期化のためにTheano/TensorFlow関数を指定します．
    このパラメータは，`weights`引数を与えていない場合のみ有効です．
- __gamma_init__: スケールパラメータのための初期化関数名 ([initializations](../initializations.md))，
    または，重みの初期化のためにTheano/TensorFlow関数を指定します．
    このパラメータは，`weights`引数を与えていない場合のみ有効です．

__入力形状__

任意．
このレイヤーがモデルの1番目のレイヤーとなる場合は，`input_shape`引数（サンプル軸を含まない整数のタプル）を与える必要があります．

__出力形状__

入力形状と同じです．

__参考文献__

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://jmlr.org/proceedings/papers/v37/ioffe15.html)
