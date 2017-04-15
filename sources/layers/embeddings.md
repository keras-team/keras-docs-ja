<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/embeddings.py#L8)</span>
### Embedding

```python
keras.layers.embeddings.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
```

正の整数（インデックス）を固定次元の密ベクトルに変換します．
例）[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

このレイヤーはモデルの最初のレイヤーとしてのみ利用できます．

__例__

```python
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
```

__引数__

  - __input_dim__: 正の整数．語彙数．入力データの最大インデックス + 1．
  - __output_dim__: 0以上の整数．密なembeddingsの次元数．
  - __embeddings_initializer__: `embeddings`行列の[Initializers](../initializers.md)．
  - __embeddings_regularizer__: `embeddings`行列に適用する[Regularizers](../regularizers.md)．
  - __embeddings_constraint__: `embeddings`行列に適用する[Constraints](../constraints.md)．
  - __mask_zero__: 真理値．入力の0をパディングのための特別値として扱うかどうか．
    これは入力の系列長が可変長となりうる変数を入力にもつ[Recurrentレイヤー](recurrent.md)に対して有効です．
    この引数が`True`のとき，以降のレイヤーは全てこのマスクをサポートする必要があり，
    そうしなければ，例外が起きます．
    mask_zeroがTrueのとき，index 0は語彙の中で使えません（input_dim は`語彙数+2`と等しくなるべきです）．
  - __input_length__: 入力の系列長（定数）．
    この引数はこのレイヤーの後に`Flatten`から`Dense`レイヤーへ接続する際に必要です (これがないと，denseの出力のshapeを計算できません)．

__Input shape__

shapeが`(batch_size, sequence_length)`の2階テンソル．

__Output shape__

shapeが`(batch_size, sequence_length, output_dim)`の3階テンソル．

__参考文献__

- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
