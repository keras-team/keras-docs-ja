# データセット

## CIFAR10 画像分類

10のクラスにラベル付けされた，50000枚の32x32訓練用カラー画像，10000枚のテスト用画像のデータセット．

### 使い方:

```python
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

- __返り値:__
    - 2つのタプル:
        - __X_train, X_test__: shape (nb_samples, 3, 32, 32) のRGB画像データのuint8配列．
        - __y_train, y_test__: shape (nb_samples,) のカテゴリラベル(0-9の範囲のinteger)のuint8配列．

---

## CIFAR100 画像分類

100のクラスにラベル付けされた，50000枚の32x32訓練用カラー画像，10000枚のテスト用画像のデータセット．

### 使い方:

```python
from keras.datasets import cifar100

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
```

- __返り値:__
    - 2つのタプル:
        - __X_train, X_test__: shape (nb_samples, 3, 32, 32) のRGB画像データのuint8配列．
        - __y_train, y_test__: shape (nb_samples,) のカテゴリラベルのuint8配列．

- __引数:__

    - __label_mode__: "fine" または "coarse".

---

## IMDB映画レビュー感情分類

感情 (肯定/否定) のラベル付けをされた，25,000のIMDB映画レビューのデータセット．レビューは前処理済みで，各レビューは単語のインデックス (整数値) の[シーケンス](preprocessing/sequence.md)としてエンコードされている．便宜上，単語はデータセットにおいての出現頻度によってインデックスされている．そのため例えば，整数値"3"はデータの中で3番目に頻度が多い単語にエンコードされる．これによって"上位20個の頻出語を除いた，上位10,000個の頻出語についてのみ考える"というようなフィルタリング作業を高速に行うことができる．

慣例として，"0"は特定の単語を表さずに，未知語にエンコードされる．

### 使い方:

```python
from keras.datasets import imdb


(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
```
- __返り値:__
    - 2つのタプル:
        - __x_train, x_test__: シーケンスのリスト，リストはインデックス(整数値)．引数num_wordsに具体的な整数値が与えられた場合，取り得るインデックスの最大値はnum_words-1となる．引数maxlenに具体的な数値が与えられた場合，シーケンスの最大長はmaxlenとなる．
        - __y_train, y_test__: integer型ラベル(1または0)のリスト．

- __引数:__
    - __path__: データをローカルに持っている場合(`'~/.keras/datasets/' + path`)，cPickleフォーマットではこの位置にダウンロードされる．
    - __num_words__: integer型 または None． 指定された数値だけ上位の頻出語が対象となる．指定された数値より下位の頻出語はシーケンスデータにおいて0と表される．
    - __skip_top__: integer型．指定された数値だけ上位の頻出語が無視される(シーケンスデータにおいて0と表される)．
    - __maxlen__: int型．シーケンスの最大長．最大長より長いシーケンスは切り捨てられる．
    - __seed__: int型．再現可能なデータシャッフルのためのシード．
    - __start_char__: この文字が系列の開始記号として扱われる．
        0は通常パディング用の文字であるため，1以上からセットしてください．
    - __oov_char__: `num_words`か`skip_top`によって削除された単語を置換します．
    - __index_from__: 単語のインデックスはこのインデックス以上の数値が与えられます．

---

## ロイターのニュースワイヤー トピックス分類
46のトピックにラベル付けされた，11228個のロイターのニュースワイヤーのデータセット．IMDBデータセットと同様，各ワイヤーが一連の単語インデックスとしてエンコードされる(同じ慣例に基づく)．

### 使い方:

```python
from keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)
```

仕様はIMDBデータセットのものに加えて，次のパラメータが追加される:

- __test_split__: float．テストデータとして使用するデータセットの割合．

このデータセットはシーケンスをエンコードに使われている単語インデックスを利用できる．

```python
word_index = reuters.get_word_index(path="reuters_word_index.npz")
```

- __返り値:__ キーが単語(str型)，値がインデックス(integer型)の辞書．例，`word_index["giraffe"]`は`1234`が返る．

- __引数:__

    - __path__: データをローカルに持っていない場合(`'~/.keras/datasets/' + path`)，この位置にダウンロードされる．

## MNIST 手書き数字データベース

60,000枚の28x28，10個の数字の白黒画像と10,000枚のテスト用画像データセット．

### 使い方:

```python
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

- __返り値:__
    - 2つのタプル:
        - __x_train, x_test__: shape (num_samples, 28, 28) の白黒画像データのuint8配列．
        - __y_train, y_test__: shape (num_samples,) のカテゴリラベル(0-9のinteger)のuint8配列．

- __引数:__
    - __path__: データをローカルに持っていない場合(`'~/.keras/datasets/' + path`)，この位置にダウンロードされる．

## ボストンの住宅価格回帰データセット

Carnegie Mellon大学のStatLib ライブラリのデータセット．

サンプルは，1970年代後半におけるボストン近郊の異なる地域の住宅に関する13の属性値を含む．
予測値は，その地域での住宅価格の中央値 (単位はk$) ．

### 使い方:

```python
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
```

- __引数__:
    - __path__: ローカルに保存するパス．(~/.keras/datasets)．
    - __seed__: テストデータに分ける前にデータをシャッフルするためのシード．
    - __test_split__: テストデータとして使用するデータセットの割合．

- __返り値__: Numpy配列のタプル: (x_train, y_train), (x_test, y_test)．
