<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/wrappers.py#L43)</span>
### TimeDistributed

```python
keras.layers.wrappers.TimeDistributed(layer)
```

このラッパーにより，入力のすべての時間スライスにレイヤーを適用できます．

入力は少なくとも3次元である必要があります．
インデックスの次元は時間次元と見なされます．

例えば，32個のサンプルを持つバッチを考えます．各サンプルは16次元で構成される10個のベクトルを持ちます．
このバッチの入力のshapeは`(32, 10, 16)`となります（`input_shape`はサンプル数の次元を含まないため，`(10, 16)`となります）．

このとき，10個のタイムスタンプのレイヤーそれぞれに`Dense`を適用するために，`TimeDistributed`を利用できます:

```python
# as the first layer in a model
model = Sequential()
model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
# now model.output_shape == (None, 10, 8)

# subsequent layers: no need for input_shape
model.add(TimeDistributed(Dense(32)))
# now model.output_shape == (None, 10, 32)
```

出力のshapeは`(32, 10, 8)`です．

`TimeDistributed`は`Dense`だけでなく任意のレイヤーに使えます．
例えば，`Conv2D`に対して:

```python
model = Sequential()
model.add(TimeDistributed(Conv2D(64, (3, 3)),
              input_shape=(10, 299, 299, 3)))
```

__Arguments__

- __layer__: レイヤーインスタンス．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/wrappers.py#L134)</span>
### Bidirectional

```python
keras.layers.wrappers.Bidirectional(layer, merge_mode='concat', weights=None)
```

RNNのBidirectionalなラッパー．

__Arguments__

- __layer__: `Recurrent`のインスタンス．
- __merge_mode__: RNNのforwardとbackwardの出力同士を組み合わせる際のモード．{'sum', 'mul', 'concat', 'ave', None}のいずれか．Noneの場合，出力はリストになります．

__Raises__

- __ValueError__: `merge_mode`引数が不正な場合．

__Examples__

```python
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```
