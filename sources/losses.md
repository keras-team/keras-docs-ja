## ロス関数の利用方法

ロス関数（ロス関数や最適スコア関数）はモデルをコンパイルする際に必要なパラメータの1つです:

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

```python
from keras import losses

model.compile(loss=losses.mean_squared_error, optimizer='sgd')
```

既存の目的関数の名前を引数に与えるか，各データ点に対してスカラを返し，以下の2つの引数を取るTensorFlow/Theanoのシンボリック関数を与えることができます:

- __y_true__: 正解ラベル．TensorFlow/Theano テンソル
- __y_pred__: 予測値．y_trueと同じshapeのTensorFlow/Theano テンソル

実際に最適化される目的関数値は全データ点における出力の平均です．

このような関数の実装例に関しては，[losses source](https://github.com/fchollet/keras/blob/master/keras/losses.py)を参照してください．

## 利用可能な目的関数

### mean_squared_error

```python
mean_squared_error(y_true, y_pred)
```

---

### mean_absolute_error

```python
mean_absolute_error(y_true, y_pred)
```

---

### mean_absolute_percentage_error

```python
mean_absolute_percentage_error(y_true, y_pred)
```

---

### mean_squared_logarithmic_error

```python
mean_squared_logarithmic_error(y_true, y_pred)
```

---

### squared_hinge

```python
squared_hinge(y_true, y_pred)
```

---

### hinge
```python
hinge(y_true, y_pred)
```

---

### categorical_crossentropy

```python
categorical_crossentropy(y_true, y_pred)
```
---

### sparse_categorical_crossentropy

```python
sparse_categorical_crossentropy(y_true, y_pred)
```
---

### binary_crossentropy

```python
binary_crossentropy(y_true, y_pred)
```
---

### kullback_leibler_divergence

```python
kullback_leibler_divergence(y_true, y_pred)
```
---

### poisson

```python
poisson(y_true, y_pred)
```
---

### cosine_proximity

```python
cosine_proximity(y_true, y_pred)
```
---

__NOTE__: `categorical_crossentropy`を使う場合，目的値はカテゴリカルにしなければいけません．(例. もし10クラスなら，サンプルに対する目的値は，サンプルのクラスに対応する次元の値が1，それ以外が0の10次元のベクトルです)．
*整数の目的値からカテゴリカルな目的値に*変換するためには，Keras utilityの`to_categorical`を使えます．

```
from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(int_labels, num_classes=None)
```
