### to_categorical


```python
to_categorical(y, nb_classes=None)
```


クラスベクトル（0からnb_classesまでの整数）を
categorical_crossentropyとともに用いるためのバイナリのクラス行列に変換します．

__引数__

- __y__: 行列に変換するクラスベクトル
- __nb_classes__: 総クラス数

__戻り値__

入力のバイナリ行列表現

----

### convert_kernel


```python
convert_kernel(kernel, dim_ordering='default')
```


カーネル行列（Numpyの配列）をTheano形式からTensorFlow形式に変換します．
（この変換は逆変換と同一なので，TensorFlow形式からTheano形式への変換も可能です）
