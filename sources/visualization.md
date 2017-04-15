
## モデルの可視化

`keras.utils.vis_util`モジュールは（graphvizを用いて）Kerasモデルの可視化するためのユーティリティ関数を提供します．

以下の例は，モデルのグラフ構造をプロットし，それをファイルに保存します:

```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

`plot_model`は2つのオプショナルな引数を取ります:

- `show_shapes`（デフォルト: False）グラフ中に出力形状を出力するかどうか．
- `show_layer_names` （デフォルト: True） グラフ中にレイヤー名を出力するかどうか．

また，`pydot.Graph`オブジェクトを直接操作して可視化もできます．
IPython Notebook内で可視化する例:

```python
from IPython.display import SVG
from keras.utils.vis_util import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```
