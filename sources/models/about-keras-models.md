# About Keras models

Kerasには2つの利用可能なモデルがあります．1つは[シーケンシャルモデル](/models/sequential)，そしてもう1つは[functional APIとともに用いるモデルクラス](/models/model).

これらのモデルには，いくつかの共通のメソッドがあります．

- `model.summary()`: モデルの要約を出力します．
- `model.get_config()`: モデルの設定を持つ辞書を返します．下記のように，モデルはそれ自身の設定から再インスタンス化することができます．
```python
config = model.get_config()
model = Model.from_config(config)
# or, for Sequential:
model = Sequential.from_config(config)
```

- `model.get_weights()`: モデルの全ての重みテンソル(Numpy arrays)のリスト返します．
- `model.set_weights(weights)`: Numpy arraysのリストからモデルの重みの値をセットします． リスト中のNumpy arraysのshapeは`get_weights()`で得られるリスト中のNumpy arraysのshapeと同じである必要があります.
- `model.to_json()`: モデルの表現をJSON文字列として返します．このモデルの表現は，重みを含まないアーキテクチャのみであることに注意してください．下記の様に，JSON文字列から同じアーキテクチャのモデル(重みについては初期化されます)を再インスタンス化することができます．

```python
from models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)
```
- `model.to_yaml()`: モデルの表現をYAML文字列として返します． このモデルの表現は，重みを含まないアーキテクチャのみであることに注意してください．下記の様に，YAML文字列から同じアーキテクチャのモデル(重みについては初期化されます)を再インスタンス化することができます．
```python
from models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
```
- `model.save_weights(filepath)`: モデルの重みをHDF5形式のファイルとして保存します．
- `model.load_weights(filepath)`: モデルの重みをHDF5形式のファイル(`save_weights`によって作られた)から読み込みます．






