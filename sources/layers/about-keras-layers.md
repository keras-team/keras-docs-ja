# About Keras layers


全てのKerasレイヤーは次のようないくつかの共通のメソッドを持っています．

- `layer.get_weights()`: レイヤーの重みをNumpy arraysのリストとして返す．
- `layer.set_weights(weights)`: Numpy arrays(`get_weights`で得られる重みと同じshapeをもつ)のリストからレイヤーの重みをセットする．
- `layer.get_config()`: レイヤーの設定をもつ辞書を返す．レイヤーは次のように，それ自身の設定から再インスタンス化できます．

```python
from keras.utils.layer_utils import layer_from_config

config = layer.get_config()
layer = layer_from_config(config)
```
もし，レイヤーがシングルノードを持つなら(i.e. もし共有レイヤーでないなら)，インプットテンソル，アウトプットテンソル，インプットshape，アウトプットshapeを得ることができます．

- `layer.input`
- `layer.output`
- `layer.input_shape`
- `layer.output_shape`

もし，レイヤーが複数ノードを持つなら，(see: [the concept of layer node and shared layers](/getting-started/functional-api-guide/#the-concept-of-layer-node)), 次のメソッドが使えます．

- `layer.get_input_at(node_index)`
- `layer.get_output_at(node_index)`
- `layer.get_input_shape_at(node_index)`
- `layer.get_output_shape_at(node_index)`