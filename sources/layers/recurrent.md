<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L42)</span>
### Recurrent

```python
keras.layers.recurrent.Recurrent(weights=None, return_sequences=False, go_backwards=False, stateful=False, unroll=False, consume_less='cpu', input_dim=None, input_length=None)
```

Recurrentレイヤーに対する抽象的な基底クラス．
モデルの中では利用しないでください -- これは直接利用可能なレイヤーではありません！
代わりに子クラスである`LSTM`, `GRU`, `SimpleRNN`を利用してください．

すべてのRecurrentレイヤー (`LSTM`, `GRU`, `SimpleRNN`) はこのクラスの仕様に従い，下に列挙したキーワード引数が使用可能です．

__例__


```python
# Sequentialモデルの最初のレイヤーとして
model = Sequential()
model.add(LSTM(32, input_shape=(10, 64)))
# ここで model.output_shape == (None, 32)
# 注: `None`はバッチ次元．note: `None` is the batch dimension.

# 以下は同様の意味です:
model = Sequential()
model.add(LSTM(32, input_dim=64, input_length=10))

# 2層目以降のレイヤーに対しては，入力サイズを指定する必要はありません:
model.add(LSTM(16))
```

__引数__

- __weights__: 重みの初期値として設定するnumpy arrayのリスト．
  リストは次のshapeを持つ3つの要素からなります:
  `[(input_dim, output_dim), (output_dim, output_dim), (output_dim,)]`.
- __return_sequences__: 論理型．出力シーケンスの最後の出力を返すか，
    完全なシーケンスを返すか．
- __go_backwards__: 論理型（デフォルトはFalse）．
  Trueであれば，入力シーケンスを逆向きに進みます．
- __stateful__: 論理型（デフォルトはFalse）．Trueであれば，バッチ内の添字iの各サンプル
    に対する最後の状態が次のバッチ内の添字iのサンプルに対する初期状態として使われます．
- __unroll__: 論理型（デフォルトはFalse）．Trueであれば，ネットワークは展開され，
    そうでなければシンボリックループが使われます．TensorFlowを利用するとき，ネットワークは
    常に展開されるので，この引数は何もしません．
    展開はよりメモリ集中傾向になりますが，RNNをスピードアップできます．
    展開は短いシーケンスにのみ適しています．
- __consume_less__: "cpu", "mem", "gpu"（LSTM/GRUのみ）のいずれか．
    "cpu"に設定すれば，RNNは少ない代わりに大きい行列積を用いた実装を利用します．
    CPUでは速く動作する一方，多くのメモリを消費します．
    "mem"に設定すれば，RNNは多い代わりに小さい行列積を用いた実装を利用します．
    CPUでは遅く動作する一方（GPUではむしろ高速になるかもしれません），
    メモリの消費は少なくなります．
    "gpu"に設定すれば（LSTM/GRUのみ），RNNは入力ゲート，
    忘却ゲート，出力ゲートを1つの行列に結び付け，
    GPUにおいてより計算時間の効率の良い並列化を可能にします．
    注意: RNNのドロップアウトはすべてのゲートに対して共有化されている必要があり，
    結果として僅かに正則化の効果が低減します．

- __input_dim__: 入力の次元（整数）
    この引数（または代わりのキーワード引数`input_shape`）は
    このレイヤーをモデルの最初のレイヤーとして利用するときに必要となります．

- __input_length__: 入力シーケンスの長さ．
    この引数は`Flatten`ひいては`Dense`なレイヤーを上流に結びつけるときに必要となります．
    （これなしでは，密な出力のshapeを計算できません）．
    注意: Recurrentレイヤーがあなたのモデルの最初のレイヤーでなければ，
    最初のレイヤーのレベルで入力の長さを指定する必要があります
    （例えば`input_shape`引数を通じて）．

__入力のshape__

shape `(nb_samples, timesteps, input_dim)`を持つ3次元テンソル．

__出力のshape__

- `return_sequences`のとき: shape `(nb_samples, timesteps, output_dim)`を持つ3次元テンソル．
- そうでないとき，shape `(nb_samples, output_dim)`を持つ2次元テンソル．

__マスキング__

このレイヤーはタイムステップの変数を持つ入力データに対するマスキングをサポートします．
あなたのデータにマスクを導入するためには，
`True`に設定された`mask_zero`パラメータを持つ[埋め込み](embeddings.md)レイヤーを利用してください．

__パフォーマンスの注意点__

RNNのパフォーマンスはTensorFlowよりもTheanoのほうが遥かに優れています．
加えて，TensorFlowを使う際は，`unroll=True`がよりパフォーマンスが良いです．

__Recurrentレイヤーで状態管理を利用するときの注意点__

Recurrentレイヤーが状態管理されるように設定できます．
これは1つのバッチのサンプルに対して計算される状態が
次のバッチのサンプルに対する初期状態として再利用されることを意味します．
これは異なる連続したバッチのサンプル間の1対1対応を仮定します．

状態管理を可能にするためには:
  - レイヤーコンストラクタにおいて`stateful=True`を指定してください．
  - もしsequentialモデルなら:
      `batch_input_shape=(...)`を最初のレイヤーに
    1つ以上の入力層をもったfunctionalモデルなら:
      `batch_shape=(...)`をモデルのすべての最初のレイヤーに
    渡すことで固定長のバッチサイズを指定してください．
    これは*バッチサイズを含む*あなたの入力の期待されるshapeです．
    これは整数のタプルであるべきです，例えば`(32, 10, 100)`．

あなたのモデルの状態を再設定するためには，指定したレイヤーもしくは
あなたの全体のモデル上で`.reset_states()`を呼び出してください．

__TensorFlowでドロップアウトを利用するときの注意__

TensorFlowバックエンドを利用するとき，状態管理されたRecurrentレイヤーについての注意に従って，
あなたのモデルに対する固定されたバッチサイズを指定してください．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L255)</span>
### SimpleRNN

```python
keras.layers.recurrent.SimpleRNN(output_dim, init='glorot_uniform', inner_init='orthogonal', activation='tanh', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)
```

出力が入力にフィードバックされる全結合RNN．

__引数__

- __output_dim__: 内部の射影と最終的な出力の次元
- __init__: 重み初期化関数
        既存の関数やTheano関数の名前(str)をとることができます
        (参照: [初期化](../initializations.md))．
- __inner_init__: 内部のセルの初期化関数．
- __activation__: 活性化関数．
        既存の関数やTheano関数の名前(str)をとることができます
        (参照: [活性化](../activations.md))．
- __W_regularizer__: [重み正則化](../regularizers.md)の例
        (例えばL1もしくはL2正則化)，入力の重み行列に適用されます．
- __U_regularizer__: [重み正則化](../regularizers.md)の例
        (例えばL1もしくはL2正則化)，リカレント重み行列に適用されます．
- __b_regularizer__: [重み正則化](../regularizers.md)の例，バイアスに適用されます．
- __dropout_W__: 0と1の間のfloat．入力ゲートに対してドロップする入力ユニットの割合．
- __dropout_U__: 0と1の間のfloat．リカレント結合に対してドロップする入力ユニットの割合．

__参考文献__

- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L404)</span>
### GRU

```python
keras.layers.recurrent.GRU(output_dim, init='glorot_uniform', inner_init='orthogonal', activation='tanh', inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)
```

ゲートのあるリカレントユニット - Cho et al. 2014.

__引数__

- __output_dim__: 内部の射影と最終的な出力の次元
- __init__: 重み初期化関数
        既存の関数やTheano関数の名前(str)をとることができます
        (参照: [初期化](../initializations.md)).
- __inner_init__: 内部のセルの初期化関数．
- __activation__: 活性化関数．
        既存の関数やTheano関数の名前(str)をとることができます
        (参照: [活性化](../activations.md))．
- __inner_activation__: 内部セルに対する活性化関数．
- __W_regularizer__: [重み正則化](../regularizers.md)の例
        (例えばL1もしくはL2正則化)，入力の重み行列に適用されます．
- __U_regularizer__: [重み正則化](../regularizers.md)の例
        (例えばL1もしくはL2正則化)，リカレント重み行列に適用されます．
- __b_regularizer__: [重み正則化](../regularizers.md)の例，バイアスに適用されます．
- __dropout_W__: 0と1の間のfloat．入力ゲートに対してドロップする入力ユニットの割合．
- __dropout_U__: 0と1の間のfloat．リカレント結合に対してドロップする入力ユニットの割合．


__参考文献__

- [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L623)</span>
### LSTM

```python
keras.layers.recurrent.LSTM(output_dim, init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)
```

長短期記憶ユニット - Hochreiter 1997.

アルゴリズムの段階的な記述については，
[このチュートリアル](http://deeplearning.net/tutorial/lstm.html)を参照してください．

__引数__

- __output_dim__: 内部の射影と最終的な出力の次元
- __init__: 重み初期化関数
        既存の関数やTheano関数の名前(str)をとることができます
        (参照: [初期化](../initializations.md)).
- __inner_init__: 内部のセルの初期化関数．
- __forget_bias_init__: 忘却ゲートのバイアスに対する初期化関数．
        [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        は1で初期化することを推奨しています．
- __activation__: 活性化関数．
        既存の関数やTheano関数の名前(str)をとることができます
        (参照: [活性化](../activations.md))．
- __inner_activation__: 内部セルに対する活性化関数．
- __W_regularizer__: [重み正則化](../regularizers.md)の例
        (例えばL1もしくはL2正則化)，入力の重み行列に適用されます．
- __U_regularizer__: [重み正則化](../regularizers.md)の例
        (例えばL1もしくはL2正則化)，リカレント重み行列に適用されます．
- __b_regularizer__: [重み正則化](../regularizers.md)の例，バイアスに適用されます．
- __dropout_W__: 0と1の間のfloat．入力ゲートに対してドロップする入力ユニットの割合．
- __dropout_U__: 0と1の間のfloat．リカレント結合に対してドロップする入力ユニットの割合．


__参考文献__

- [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
- [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
- [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
