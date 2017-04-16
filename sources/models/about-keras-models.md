# About Keras models

Kerasには2つの利用可能なモデルがあります: 1つは[Sequentialモデル](/models/sequential)，そしてもう1つは[functional APIとともに用いるモデルクラス](/models/model)．

これらのモデルには，共通のメソッドがあります．

- `model.summary()`: モデルの要約を出力します．
- `model.get_config()`: モデルの設定を持つ辞書を返します．下記のように，モデルはそれ自身の設定から再インスタンス化できます．

```python
config = model.get_config()
model = Model.from_config(config)
# or, for Sequential:
model = Sequential.from_config(config)
```

- `model.get_weights()`: モデルの全ての重みテンソルをNumpy配列を要素にもつリスト返します．
- `model.set_weights(weights)`: Numpy配列のリストからモデルの重みの値をセットします．リスト中のNumpy配列のshapeは`get_weights()`で得られるリスト中のNumpy配列のshapeと同じ必要があります．
- `model.to_json()`: モデルの表現をJSON文字列として返します．このモデルの表現は，重みを含まないアーキテクチャのみであることに注意してください．下記の様に，JSON文字列から同じアーキテクチャのモデル(重みについては初期化されます)を再インスタンス化できます．

```python
from models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)
```

- `model.to_yaml()`: モデルの表現をYAML文字列として返します． このモデルの表現は，重みを含まないアーキテクチャのみであることに注意してください．下記の様に，YAML文字列から同じアーキテクチャのモデル(重みについては初期化されます)を再インスタンス化できます．

```python
from models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
```

- `model.save_weights(filepath)`: モデルの重みをHDF5形式のファイルに保存します．
- `model.load_weights(filepath, by_name=False)`: (`save_weights`によって作られた)モデルの重みをHDF5形式のファイルから読み込みます．デフォルトでは，アーキテクチャは不変であることが望まれます．(いくつかのレイヤーが共通した)異なるアーキテクチャに重みを読み込む場合，`by_name=True`を使うことで，同名のレイヤーにのみ読み込み可能です．
