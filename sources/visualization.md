
## モデルの可視化

`keras.utils.visualize_util`モジュールは（graphvizを用いて）Kerasモデルの可視化するためのユーティリティ関数を提供します．

以下の例は，モデルのグラフ構造をプロットし，それをファイルに保存します:

```python
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')
```

`plot`は2つのオプショナルな引数を取ります:

- `show_shapes`（デフォルト: False）グラフ中に出力形状を出力するかどうかを制御します．
- `show_layer_names` （デフォルト: True） グラフ中にレイヤー名を出力するかどうかを制御します．

また，`pydot.Graph`オブジェクトを直接操作して可視化することもできます．
IPython Notebook内で可視化する例:

```python
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```
