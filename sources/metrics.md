## 評価関数の利用方法

評価関数はモデルの性能を測るのに使われます．
次のコードのように，モデルをコンパイルする際に `metrics` パラメータとして評価関数を渡して指定します．

```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])
```

```python
from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
```

評価関数は [目的関数](/objectives) とよく似ていますが，
評価結果の値が訓練に直接使われることはありません．

渡す `metrics` パラメータには既存の評価関数の名前を引数に与えるか，
自分で作った評価関数を渡す事ができます ([カスタマイズ](#_3) を参照して下さい)．

## 利用可能な評価関数

- __binary_accuracy__
- __categorical_accuracy__
- __sparse_categorical_accuracy__
- __top_k_categorical_accuracy__

## カスタマイズ

`(y_true, y_pred)` を引数とし，各データ点に対してスカラを返す関数を評価関数として利用できます:

- __y_true__: 正解ラベル．Theano/TensorFlow テンソル
- __y_pred__: 予測．y_trueと同じ形状のTheano/TensorFlow テンソル


```python
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```
