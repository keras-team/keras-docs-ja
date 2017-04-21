<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/embeddings.py#L8)</span>
### Embedding

```python
keras.layers.embeddings.Embedding(input_dim, output_dim, init='uniform', input_length=None, W_regularizer=None, activity_regularizer=None, W_constraint=None, mask_zero=False, weights=None, dropout=0.0)
```

正の整数（インデックス）を固定次元の密ベクトルに変換します．
例）[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

このレイヤーはモデルの最初のレイヤーとしてのみ利用できます．

__例__

```python
  model = Sequential()
  model.add(Embedding(1000, 64, input_length=10))
  # このモデルは入力として次元が (batch, input_length) である整数行列を取ります．
  # 最大の整数（つまり，単語インデックス）は1000です（語彙数）．
  # ここで，model.output_shape == (None, 10, 64) となります．Noneはバッチ次元です．

  input_array = np.random.randint(1000, size=(32, 10))

  model.compile('rmsprop', 'mse')
  output_array = model.predict(input_array)
  assert output_array.shape == (32, 10, 64)
```

__引数__
  - __input_dim__: 正の整数．語彙数．入力データの最大インデックス + 1．
  - __output_dim__: 0以上の整数．出力ベクトルの次元数．
  - __init__: 重みの初期化関数名 ([initializations](../initializations.md)) ，または，重みを初期化するTheano関数．
    このパラメータは`weights`引数を与えていないときにのみ有効です．
  - __weights__: 重みの初期値として利用するNumpy配列のリスト．
    このリストは形が`(input_dim, output_dim)`である1つの要素のみを持ちます．
  - __W_regularizer__: 埋め込み行列に適用する[regularizers](../regularizers.md)モジュールのインスタンス（例えば，L1やL2正則化）．
  - __W_constraint__: 埋め込み行列に適用する[constraints](../constraints.md)モジュールのインスタンス（例えば，最大ノルム制約や非負制約）．
  - __mask_zero__: 真理値．0という入力値をパディングのための特別値として扱うかどうか．
    これは入力長が可変となる可能性がある[Recurrentレイヤー](recurrent.md)に対して有効です．
    この引数が`True`のとき，以降のレイヤーは全てこのマスクをサポートする必要があり，
    そうしなければ，例外が発生するでしょう．
    mask_zeroがTrueのとき，index 0は語彙の中で使われません（input_dim は語彙数+2と等しくなるべきです）．
  - __input_length__: 入力シーケンスの長さ（定数）．
    この引数はこのレイヤーの後に`Flatten`そして`Dense`レイヤーを接続する場合に必要となります．
  - __dropout__: 0より大きく1未満の浮動小数点数．ランダムに欠損させる埋め込みの割合．

__入力形状__

形が`(nb_samples, sequence_length)`である2階テンソル．

__出力形状__

形が`(nb_samples, sequence_length, output_dim)`である3階テンソル．

__参考文献__

- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
