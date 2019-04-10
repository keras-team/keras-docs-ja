## モデルの可視化

Kerasは（`graphviz`を用いて）Kerasモデルの可視化するためのユーティリティ関数を提供します．

以下の例は，モデルのグラフ構造をプロットし，それをファイルに保存します：

```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

`plot_model`は4つのオプショナルな引数を取ります:

- `show_shapes`（デフォルトはFalse）グラフ中に出力のshapeを出力するかどうかを制御します．
- `show_layer_names` （デフォルトはTrue）グラフ中にレイヤー名を出力するかどうかを制御します．
- `expand_nested` (デフォルトはFalse) グラフ中にネストしたモデルをクラスタに展開するかどうかを制御します．
- `dpi` (デフォルトは96) 画像のdpiを制御します．

また，`pydot.Graph`オブジェクトを直接操作して可視化もできます．
IPython Notebook内での可視化例:

```python
from IPython.display import SVG
from keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```

## 訓練の履歴の可視化

Keras`Model`の`fit()`は`History`オブジェクトを返します．この`History.history`属性は一連のエポックの訓練時の損失やメトリクスの値と（該当する場合は）検証時の損失やメトリクスの値を記録した辞書です．以下に`matplotlib`を用いて訓練時と評価時の損失と精度を生成する例を示します：

```python
import matplotlib.pyplot as plt

history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```
