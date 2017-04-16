<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L42)</span>
### Recurrent

```python
keras.layers.recurrent.Recurrent(return_sequences=False, go_backwards=False, stateful=False, unroll=False, implementation=0)
```

Recurrentレイヤーに対する抽象的な基底クラス．
モデルの中では利用しないでください -- これは直接利用可能なレイヤーではありません!
代わりに子クラスである`LSTM`, `GRU`, `SimpleRNN`を利用してください．

すべてのRecurrentレイヤー (`LSTM`, `GRU`, `SimpleRNN`) はこのクラスの仕様に従い，下に列挙したキーワード引数が使用可能です．

__Examples__


```python
# Sequentialモデルの最初のレイヤーとして
model = Sequential()
model.add(LSTM(32, input_shape=(10, 64)))
# ここで model.output_shape == (None, 32)
# 注: `None`はバッチ次元．

# 2層目以降のレイヤーに対しては，入力サイズを指定する必要はありません:
model.add(LSTM(16))

# to stack recurrent layers, you must use return_sequences=True
# on any recurrent layer that feeds into another recurrent layer.
# note that you only need to specify the input size on the first layer.
model = Sequential()
model.add(LSTM(64, input_dim=64, input_length=10, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(10))
```

__Arguments__

- __weights__: 重みの初期値に用いるnumpy配列のリスト．
  リストは次のshapeを持つ3つの要素からなります:
  `[(input_dim, output_dim), (output_dim, output_dim), (output_dim,)]`.
- __return_sequences__: Boolean．出力系列の最後の出力を返すか，完全な系列を返すか．
- __go_backwards__: Boolean（デフォルトはFalse）．Trueなら，入力系列の逆向きから処理します．
- __stateful__: Boolean（デフォルトはFalse）．Trueなら，バッチ内のインデックスiの各サンプル
    に対する最後の状態が次のバッチ内のインデックスiのサンプルに対する初期状態として使われます．
- __unroll__: Boolean（デフォルトはFalse）．Trueなら，ネットワークは展開され，
    そうでなければシンボリックループが使われます．
    展開はよりメモリ集中傾向になりますが，RNNをスピードアップできます．
    展開は短い系列にのみ適しています．
- __implementation__: {0, 1, 2}のいずれか．
0なら，RNNは，少なく大きな行列積を計算するため，CPU上では速くなりますが，メモリの消費は大きいです．
1なら，小さい行列に対して多く行列積を計算するため，より遅くなります（GPUでは速いかもしれません）が，メモリの消費は小さいです．
 (LSTM/GRU のみで) 2なら，入力ゲートと忘却ゲートを組み合わせます，忘却ゲートと出力ゲートは1つの行列になり，GPU並列時の時間効率をあげます．
    - __Note__: RNN dropout は全ゲートで共有しなければなりません，結果として僅かに汎化性能が低下します．
- __input_dim__: 入力の次元（整数）．
    この引数（または代わりのキーワード引数`input_shape`）は，
    このレイヤーをモデルの最初のレイヤーとして利用するときに必要です．
- __input_length__: 入力系列の長さ．
    この引数はこのレイヤーの後に`Flatten`から`Dense`レイヤーへ接続する際に必要です (これがないと，denseの出力のshapeを計算できません)．
    Recurrentレイヤーがモデルの最初のレイヤーでなければ，
    最初のレイヤーのレベルで入力系列の長さを指定する必要があります
    （例えば`input_shape`引数を通じて）．

__Input shape__

shapeが`(batch_size, timesteps, input_dim)`の3Dテンソル．（オプション）shapeが`(batch_size, output_dim)`の2Dテンソル．

__Output shape__

- `return_sequences`がTrue: shapeが`(batch_size, timesteps, output_dim)`の3Dテンソル．
- そうでないとき，shapeが`(batch_size, output_dim)`の2Dテンソル．

__マスキング__

このレイヤーはタイムステップの変数を持つ入力データに対するマスキングをサポートします．
あなたのデータにマスクを導入するためには，
`mask_zero`パラメータに`True`を渡した[Embedding](embeddings.md)レイヤーを利用してください．

__RNNで状態管理を利用するときの注意点__

状態管理を可能にするためには:
  - レイヤーコンストラクタにおいて`stateful=True`を指定してください．
  - もしsequentialモデルなら:
      `batch_input_shape=(...)`を最初のレイヤーに
    1つ以上の入力層をもったfunctionalモデルなら:
      `batch_input_shape=(...)`をモデルのすべての最初のレイヤーに
    渡すことで固定系列長のバッチサイズを指定してください．
    これは*バッチサイズを含む*入力の期待されるshapeです．
    これは整数のタプルであるべきです，例えば`(32, 10, 100)`．
 - `fit()`を呼ぶときは，`stateful=False` を指定してください．

モデルの状態を再設定するには，指定したレイヤーもしくはモデル全体で`.reset_states()`を呼び出してください．

__RNNの初期状態を指定するときの注意点__

`initial_state`のキーワード引数を渡してRNNを呼び出すことで，内部状態の初期値を指定できます．
`initial_state`の値は，RNNの初期値を表現したテンソルかテンソルのリストです．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L255)</span>
### SimpleRNN

```python
keras.layers.recurrent.SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```

出力が入力にフィードバックされる全結合RNN．

__Arguments__

- __units__: 正の整数値，出力の次元数．
- __activation__: 活性化関数([activations](../activations.md))．`None`を渡すと活性化関数は適用されません (例．"linear" activation: `a(x) = x`)．
- __use_bias__: Boolean，biasベクトルを使うかどうか．
- __kernel_initializer__: 入力の線形変換に使われる`kernel`の重み行列のためのInitializer．([initializers](../initializers.md))．
- __recurrent_initializer__: 再帰の線形変換に使われる`recurrent_kernel`の重み行列のInitializer．([initializers](../initializers.md))．
- __bias_initializer__: biasベクトルのInitializer．([initializers](../initializers.md))．
- __kernel_regularizer__: `kernel`の重み行列に適用するRegularizer関数 ([regularizer](../regularizers.md))．
- __recurrent_regularizer__: `recurrent_kernel`の重み行列に適用するRegularizer関数 ([regularizer](../regularizers.md))．
- __bias_regularizer__: biasベクトルに適用するRegularizer関数 ([regularizer](../regularizers.md))．
- __activity_regularizer__: 出力 (そのactivation) に適用するRegularizer関数 ([regularizer](../regularizers.md))．
- __kernel_constraint__: `kernel`の重み行列に適用するConstraint関数 ([constraints](../constraints.md))．
- __recurrent_constraint__: `recurrent_kernel`の重み行列に適用するConstraint関数 ([constraints](../constraints.md))．
- __recurrent_constraint__: biasベクトルに適用するConstraint関数 ([constraints](../constraints.md))．
- __dropout__: 0から1の間のfloat．入力の線形変換においてdropするユニットの割合．
- __recurrent_dropout__: 0から1の間のfloat．再帰の線形変換においてdropするユニットの割合．

__References__

- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L404)</span>
### GRU

```python
keras.layers.recurrent.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```

ゲートのあるリカレントユニット - Cho et al. 2014.

__Arguments__

- __units__: 正の整数値，出力の次元数．
- __activation__: 活性化関数([activations](../activations.md))．`None`を渡すと活性化関数は適用されません (例．"linear" activation: `a(x) = x`)．
- __recurrent_activation__: 再帰計算時に使う活性化関数([activations](../activations.md))．
- __use_bias__: Boolean，biasベクトルを使うかどうか．
- __kernel_initializer__: 入力の線形変換に使われる`kernel`の重み行列のためのInitializer．([initializers](../initializers.md))．
- __recurrent_initializer__: 再帰の線形変換に使われる`recurrent_kernel`の重み行列のInitializer．([initializers](../initializers.md))．
- __bias_initializer__: biasベクトルのInitializer．([initializers](../initializers.md))．
- __kernel_regularizer__: `kernel`の重み行列に適用するRegularizer関数 ([regularizer](../regularizers.md))．
- __recurrent_regularizer__: `recurrent_kernel`の重み行列に適用するRegularizer関数 ([regularizer](../regularizers.md))．
- __bias_regularizer__: biasベクトルに適用するRegularizer関数 ([regularizer](../regularizers.md))．
- __activity_regularizer__: 出力 (そのactivation) に適用するRegularizer関数 ([regularizer](../regularizers.md))．
- __kernel_constraint__: `kernel`の重み行列に適用するConstraint関数 ([constraints](../constraints.md))．
- __recurrent_constraint__: `recurrent_kernel`の重み行列に適用するConstraint関数 ([constraints](../constraints.md))．
- __recurrent_constraint__: biasベクトルに適用するConstraint関数 ([constraints](../constraints.md))．
- __dropout__: 0から1の間のfloat．入力の線形変換においてdropするユニットの割合．
- __recurrent_dropout__: 0から1の間のfloat．再帰の線形変換においてdropするユニットの割合．

__References__

- [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L623)</span>
### LSTM

```python
keras.layers.recurrent.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```

長短期記憶ユニット - Hochreiter 1997.

アルゴリズムの段階的な記述については，
[このチュートリアル](http://deeplearning.net/tutorial/lstm.html)を参照してください．

__Arguments__

- __units__: 正の整数値，出力の次元数．
- __activation__: 活性化関数([activations](../activations.md))．`None`を渡すと活性化関数は適用されません (例．"linear" activation: `a(x) = x`)．
- __recurrent_activation__: 再帰計算時に使う活性化関数([activations](../activations.md))．
- __use_bias__: Boolean，biasベクトルを使うかどうか．
- __kernel_initializer__: 入力の線形変換に使われる`kernel`の重み行列のためのInitializer．([initializers](../initializers.md))．
- __recurrent_initializer__: 再帰の線形変換に使われる`recurrent_kernel`の重み行列のInitializer．([initializers](../initializers.md))．
- __bias_initializer__: biasベクトルのInitializer．([initializers](../initializers.md))．
- __unit_forget_bias__: Boolean．Trueなら，初期化時に忘却ゲートのbiasに1加算．また，trueの場合は強制的に`bias_initializer="zeros"`になります．これは[Jozefowicz et al.](http://proceedings.mlr.press/v37/jozefowicz15.pdf)で推奨されています．
- __kernel_regularizer__: `kernel`の重み行列に適用するRegularizer関数 ([regularizer](../regularizers.md))．
- __recurrent_regularizer__: `recurrent_kernel`の重み行列に適用するRegularizer関数 ([regularizer](../regularizers.md))．
- __bias_regularizer__: biasベクトルに適用するRegularizer関数 ([regularizer](../regularizers.md))．
- __activity_regularizer__: 出力 (そのactivation) に適用するRegularizer関数 ([regularizer](../regularizers.md))．
- __kernel_constraint__: `kernel`の重み行列に適用するConstraint関数 ([constraints](../constraints.md))．
- __recurrent_constraint__: `recurrent_kernel`の重み行列に適用するConstraint関数 ([constraints](../constraints.md))．
- __recurrent_constraint__: biasベクトルに適用するConstraint関数 ([constraints](../constraints.md))．
- __dropout__: 0から1の間のfloat．入力の線形変換においてdropするユニットの割合．
- __recurrent_dropout__: 0から1の間のfloat．再帰の線形変換においてdropするユニットの割合．

__References__

- [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
- [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
- [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
