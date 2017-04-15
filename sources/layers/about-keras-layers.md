# About Keras layers

全てのKerasレイヤーは次のいくつかの共通したメソッドを持っています．

- `layer.get_weights()`: レイヤーの重みをNumpy 配列のリストとして返す．
- `layer.set_weights(weights)`: Numpy 配列(`get_weights`で得られる重みと同じshapeをもつ)のリストでレイヤーの重みをセットする．
- `layer.get_config()`: レイヤーの設定をもつ辞書を返す．レイヤーは次のように，それ自身の設定から再インスタンス化できます:

```python
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)
```

あるいは，

```python
from keras import layers

config = layer.get_config()
layer = layers.deserialize({'class_name': layer.__class__.__name__,
                            'config': config})
```

もし，レイヤーがシングルノードを持つなら(i.e. もし共有レイヤーでないなら)，入力テンソル，出力テンソル，入力のshape，出力のshapeを得ることができます:

- `layer.input`
- `layer.output`
- `layer.input_shape`
- `layer.output_shape`

もし，レイヤーが複数ノードを持つなら，(see: [the concept of layer node and shared layers](/getting-started/functional-api-guide/#the-concept-of-layer-node))，次のメソッドが使えます．

- `layer.get_input_at(node_index)`
- `layer.get_output_at(node_index)`
- `layer.get_input_shape_at(node_index)`
- `layer.get_output_shape_at(node_index)`
