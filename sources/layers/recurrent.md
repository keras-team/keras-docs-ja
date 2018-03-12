<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L212)</span>
### RNN

```python
keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

Recurrentレイヤーに対する基底クラス．

__引数__

- __cell__: RNN cellインスタンス．RNN cellは以下の属性を持つクラスです．
    - `call(input_at_t, states_at_t)`メソッド，これは`(output_at_t, states_at_t_plus_1)`を返します．
        cellのメソッド呼び出しはオプションの引数`constants`も使えます．
        下記の「外部定数を渡す場合の注意」を参照してください．
    - `state_size`属性．
        これは1つの整数（1つの状態）でもよく，その場合はrecurrent stateのサイズになります（これはcellの出力のサイズと同じである必要があります）．
        （1つ状態につき1つのサイズが対応するように）整数やリストやタプルもとれます．
        この場合は最初のエントリ(`state_size[0]`)がcellの出力のサイズと同じである必要があります．
        `cell`をRNN cellインスタンスのリストとすることも可能です．
        この場合，cellはRNNの中で他のcellの後にスタックされているいれば，効率的なstacked RNNが実現されます．
- __return_sequences__: 真理値．出力系列の最後の出力を返すか，完全な系列を返すか．
- __return_state__: 真理値．出力とともに，最後の状態を返すかどうか．
- __go_backwards__: 真理値（デフォルトはFalse）．Trueなら，入力系列を逆向きから処理し，逆順の系列を返します．
- __stateful__: 真理値（デフォルトはFalse）．Trueなら，バッチ内のインデックスiの各サンプル
    に対する最後の状態が次のバッチ内のインデックスiのサンプルに対する初期状態として使われます．
- __unroll__: 真理値（デフォルトはFalse）．Trueなら，ネットワークは展開され，
    そうでなければシンボリックループが使われます．
    展開はよりメモリ集中傾向になりますが，RNNをスピードアップできます．
    展開は短い系列にのみ適しています．
- __input_dim__: 入力の次元（整数）．
    この引数（または代わりのキーワード引数`input_shape`）は，
    このレイヤーをモデルの最初のレイヤーとして利用するときに必要です．
- __input_length__: 入力系列の長さ．
    この引数はこのレイヤーの後に`Flatten`から`Dense`レイヤーへ接続する際に必要です (これがないと，denseの出力のshapeを計算できません)．
    Recurrentレイヤーがモデルの最初のレイヤーでなければ，
    最初のレイヤーのレベルで入力系列の長さを指定する必要があります
    （例えば`input_shape`引数を通じて）．

__入力のshape__

shapeが`(batch_size, timesteps, input_dim)`の3階テンソル．

__出力のshape__

- `return_state`の場合：テンソルのリスト．最初のテンソルが出力になります．残りのテンソルは最終状態で，それぞれのshapeは`(batch_size, units)`です．
- `return_sequences`の場合：shapeが`(batch_size, timesteps, input_dim)`の3階テンソル．
- それ以外の場合：shapeが`(batch_size, input_dim)`の2階テンソル．

__マスキング__

このレイヤーはタイムステップの変数を持つ入力データに対するマスキングをサポートします．
あなたのデータにマスクを導入するためには，
`mask_zero`パラメータに`True`を渡した[Embedding](embeddings.md)レイヤーを利用してください．

__RNNで状態管理を利用するときの注意点__

RNNレイヤーを'stateful'にすることができます．
これはあるバッチでサンプルに対して計算された状態が次のバッチのサンプルの初期状態として再利用されるという意味です．
これは別々の連続したバッチ内のサンプルが一対一対応することを仮定します．

状態管理を可能にするためには:
  - レイヤーコンストラクタにおいて`stateful=True`を指定してください．
  - モデルに一定のバッチサイズを指定してください．
    もしsequentialモデルなら:
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

__RNNに外部定数を渡すときの注意__

（`RNN.call`のように）`RNN.__call__`メソッドの`constants`キーワード引数を使うことによって「外部」定数を渡せます．
`cell.call`メソッドが`constants`と同じキーワード変数を受け入れる必要があります．
そのような定数は，アテンションメカニズムで知られるような，追加の固定入力（時間変動しない）におけるcellの変化の状態として使われます．

__例__

```python
# First, let's define a RNN Cell, as a layer subclass.

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

# Let's use this cell in a RNN layer:

cell = MinimalRNNCell(32)
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)

# Here's how to use the cell to build a stacked RNN:

cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L941)</span>
### SimpleRNN

```python
keras.layers.SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

出力が入力にフィードバックされる全結合RNN．

__引数__

- __units__: 正の整数値，出力の次元数．
- __activation__: 活性化関数（[activations](../activations.md)を参照）．
    デフォルト：ハイパボリックタンジェント（`tanh`）．
    `None`を渡すと活性化関数は適用されません (例．"linear" activation: `a(x) = x`)．
- __use_bias__: 真理値，biasベクトルを使うかどうか．
- __kernel_initializer__: 入力の線形変換に使われる`kernel`の重み行列のためのInitializer（[initializers](../initializers.md)を参照）．
- __recurrent_initializer__: 再帰の線形変換に使われる`recurrent_kernel`の重み行列のInitializer（[initializers](../initializers.md)を参照）．
- __bias_initializer__: biasベクトルのInitializer（[initializers](../initializers.md)を参照）．
- __kernel_regularizer__: `kernel`の重み行列に適用するRegularizer関数（[regularizer](../regularizers.md)を参照）．
- __recurrent_regularizer__: `recurrent_kernel`の重み行列に適用するRegularizer関数（[regularizer](../regularizers.md)を参照）．
- __bias_regularizer__: biasベクトルに適用するRegularizer関数（[regularizer](../regularizers.md)を参照）．
- __activity_regularizer__: 出力 (そのactivation) に適用するRegularizer関数（[regularizer](../regularizers.md)を参照）．
- __kernel_constraint__: `kernel`の重み行列に適用するConstraint関数（[constraints](../constraints.md)を参照）．
- __recurrent_constraint__: `recurrent_kernel`の重み行列に適用するConstraint関数（[constraints](../constraints.md)を参照）．
- __bias_constraint__: biasベクトルに適用するConstraint関数（[constraints](../constraints.md)を参照）．
- __dropout__: 0から1の間の浮動小数点数．入力の線形変換においてdropするユニットの割合．
- __recurrent_dropout__: 0から1の間の浮動小数点数．再帰の線形変換においてdropするユニットの割合．
- __return_sequences__: 真理値．出力系列の最後の出力を返すか，完全な系列を返すか．
- __return_state__: 真理値．出力とともに，最後の状態を返すかどうか．
- __go_backwards__: 真理値（デフォルトはFalse）．Trueなら，入力系列の後ろから処理し，逆順の系列を返します．
- __stateful__: 真理値（デフォルトはFalse）．Trueなら，バッチ内のインデックスiの各サンプル
    に対する最後の状態が次のバッチ内のインデックスiのサンプルに対する初期状態として使われます．
- __unroll__: 真理値（デフォルトはFalse）．Trueなら，ネットワークは展開され，
    そうでなければシンボリックループが使われます．
    展開はよりメモリ集中傾向になりますが，RNNをスピードアップできます．
    展開は短い系列にのみ適しています．

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1465)</span>
### GRU

```python
keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)
```

ゲートのあるリカレントユニット - Cho et al. 2014.

2つの異なる変種があります．デフォルトは1406.1078v3を基にしたもので，行列の乗算の前に隠れ状態にリセットゲートを適用します．もう1つはオリジナルである1406.1078v1をベースにしているもので，処理の順番が逆です．

2つ目の変種は（GPU限定の）CuDNNGRUに互換があり，CPUでの推論も可能です．結果として`kernel`と`recurrent_kernel`に対して異なるバイアスがあります．`'reset_after'=True`と`recurrent_activation='sigmoid'`を使用してください．

__引数__

- __units__: 正の整数値，出力の次元数．
- __activation__: 活性化関数（[activations](../activations.md)を参照）．
    デフォルト：ハイパボリックタンジェント（`tanh`）．
    `None`を渡すと活性化関数は適用されません (例．"linear" activation: `a(x) = x`)．
- __recurrent_activation__: 再帰計算時に使う活性化関数（[activations](../activations.md)を参照）．
- __use_bias__: 真理値，biasベクトルを使うかどうか．
- __kernel_initializer__: 入力の線形変換に使われる`kernel`の重み行列のためのInitializer（[initializers](../initializers.md)を参照）．
- __recurrent_initializer__: 再帰の線形変換に使われる`recurrent_kernel`の重み行列のInitializer（[initializers](../initializers.md)を参照）．
- __bias_initializer__: biasベクトルのInitializer（[initializers](../initializers.md)を参照）．
- __kernel_regularizer__: `kernel`の重み行列に適用するRegularizer関数（[regularizer](../regularizers.md)を参照）．
- __recurrent_regularizer__: `recurrent_kernel`の重み行列に適用するRegularizer関数（[regularizer](../regularizers.md)を参照）．
- __bias_regularizer__: biasベクトルに適用するRegularizer関数（[regularizer](../regularizers.md)を参照）．
- __activity_regularizer__: 出力 (そのactivation) に適用するRegularizer関数（[regularizer](../regularizers.md)を参照）．
- __kernel_constraint__: `kernel`の重み行列に適用するConstraint関数（[constraints](../constraints.md)を参照）．
- __recurrent_constraint__: `recurrent_kernel`の重み行列に適用するConstraint関数（[constraints](../constraints.md)を参照）．
- __bias_constraint__: biasベクトルに適用するConstraint関数（[constraints](../constraints.md)を参照）．
- __dropout__: 0から1の間の浮動小数点数．入力の線形変換においてdropするユニットの割合．
- __recurrent_dropout__: 0から1の間の浮動小数点数．再帰の線形変換においてdropするユニットの割合．
- __implementation__: 実装モードで，1か2．モード1は
より小さいドット積や加算で大きな方の数を処理する構造となる一方，モード2は少ない方にし，より大きな処理．
これらのモードはハードウェアやアプリケーションによって異なるパフォーマンスプロファイルとなるでしょう．
mode, either 1 or 2. Mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different performance profiles on different hardware and for different applications.
- __return_sequences__: 真理値．出力系列の最後の出力を返すか，完全な系列を返すか．
- __return_state__: 真理値．出力とともに，最後の状態を返すかどうか．
- __go_backwards__: 真理値（デフォルトはFalse）．Trueなら，入力系列の後ろから処理し，逆順の系列を返します．
- __stateful__: 真理値（デフォルトはFalse）．Trueなら，バッチ内のインデックスiの各サンプル
    に対する最後の状態が次のバッチ内のインデックスiのサンプルに対する初期状態として使われます．
- __unroll__: 真理値（デフォルトはFalse）．Trueなら，ネットワークは展開され，
    そうでなければシンボリックループが使われます．
    展開はよりメモリ集中傾向になりますが，RNNをスピードアップできます．
    展開は短い系列にのみ適しています．
- __reset_after__: GRUの慣習（行列の乗算の前後のどちらでリセットゲートの適用を行うか）．False = "before" (デフォルト), True = "after" (CuDNN互換).

__参考文献__

- [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1996)</span>
### LSTM

```python
keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

長短期記憶ユニット - Hochreiter 1997.

__引数__

- __units__: 正の整数値，出力の次元数．
- __activation__: 活性化関数（[activations](../activations.md)を参照）．
    デフォルト：ハイパボリックタンジェント（`tanh`）．
    `None`を渡すと活性化関数は適用されません (例．"linear" activation: `a(x) = x`)．
- __recurrent_activation__: 再帰計算時に使う活性化関数（[activations](../activations.md)を参照）．
    デフォルト：ハードシグモイド（`hard_sigmoid`）．
    `None`を渡すと活性化関数は適用されません (例．"linear" activation: `a(x) = x`)．
- __use_bias__: 真理値，biasベクトルを使うかどうか．
- __kernel_initializer__: 入力の線形変換に使われる`kernel`の重み行列のためのInitializer（[initializers](../initializers.md)を参照）．
- __recurrent_initializer__: 再帰の線形変換に使われる`recurrent_kernel`の重み行列のInitializer（[initializers](../initializers.md)を参照）．
- __bias_initializer__: biasベクトルのInitializer（[initializers](../initializers.md)を参照）．
- __unit_forget_bias__: 真理値．Trueなら，初期化時に忘却ゲートのbiasに1加算．また，trueの場合は強制的に`bias_initializer="zeros"`になります．これは[Jozefowicz et al.](http://proceedings.mlr.press/v37/jozefowicz15.pdf)で推奨されています．
- __kernel_regularizer__: `kernel`の重み行列に適用するRegularizer関数（[regularizer](../regularizers.md)を参照）．
- __recurrent_regularizer__: `recurrent_kernel`の重み行列に適用するRegularizer関数（[regularizer](../regularizers.md)を参照）．
- __bias_regularizer__: biasベクトルに適用するRegularizer関数（[regularizer](../regularizers.md)を参照）．
- __activity_regularizer__: 出力（そのactivation）に適用するRegularizer関数（[regularizer](../regularizers.md)を参照）．
- __kernel_constraint__: `kernel`の重み行列に適用するConstraint関数（[constraints](../constraints.md)を参照）．
- __recurrent_constraint__: `recurrent_kernel`の重み行列に適用するConstraint関数（[constraints](../constraints.md)を参照）．
- __bias_constraint__: biasベクトルに適用するConstraint関数（[constraints](../constraints.md)を参照）．
- __dropout__: 0から1の間の浮動小数点数．入力の線形変換においてdropするユニットの割合．
- __recurrent_dropout__: 0から1の間の浮動小数点数．再帰の線形変換においてdropするユニットの割合．
- __implementation__: Implementation mode, either 1 or 2. Mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different performance profiles on different hardware and for different applications.
- __return_sequences__: 真理値．出力系列の最後の出力を返すか，完全な系列を返すか．
- __return_state__: 真理値．出力とともに，最後の状態を返すかどうか．
- __go_backwards__: 真理値（デフォルトはFalse）．Trueなら，入力系列の後ろから処理し，逆順の系列を返します．
- __stateful__: 真理値（デフォルトはFalse）．Trueなら，バッチ内のインデックスiの各サンプル
    に対する最後の状態が次のバッチ内のインデックスiのサンプルに対する初期状態として使われます．
- __unroll__: 真理値（デフォルトはFalse）．Trueなら，ネットワークは展開され，
    そうでなければシンボリックループが使われます．
    展開はよりメモリ集中傾向になりますが，RNNをスピードアップできます．
    展開は短い系列にのみ適しています．

__参考文献__

- [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
- [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
- [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
