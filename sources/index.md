# Keras: Pythonの深層学習ライブラリー

Keras は最小限で記述できる,モジュール構造に対応しているニューラルネットワークのライブラリです。Pythonによって記述されており、TensorflowやTheanoに対応しています。
革新的な研究、開発を行うためにはアイデアから結果まで最小限の時間で行うことが求められます。そこでKerasはより早い実装、改良を行うことを目的として開発されました。

Kerasは以下の深層学習のライブラリを必要とするときに使用できます。

- 簡潔で素早いプロトタイプの作成が可能となります。（全てモジュール構成可能、最小限の記述、 拡張性高い）
- CNN (Convolutional Neural Network)とRNN (Recurrent Neural Network)共に実装が可能です。また、お互いを組み合わせた実装も可能となっています。
- CPUやGPUでも動作します。

Keras.ioの文章も参照してください。 KerasはPython 2.7-3.5に対応しています

------------------


## ガイドライン

- __モジュール性__：あらゆるモデルはスタンドアローンのモジュールの組み合わせで実装することが出来ます。各モジュールも簡単に組み合わせて使うことが出来ます。特に、ニューラルネットワークの各階層、目的関数、パラメータ最適化、初期化、活性化関数、正則化、それぞれがスタンドアローンのモジュールで、新しいモデルを作るときにそれらを組み合わせて実装することが出来ます。
- __ミニマリズム__：それぞれのモジュールが短く、簡潔に構成されています。それぞれのコードが分かりやすく、ブラックボックス化されている部分がありません。これによって作業スピードが上がり、革新的な事をしやすくなるでしょう。
- __拡張性__：新しいモジュールを追加するのがとても簡単です。（新しいクラスや関数に関しては）また、それぞれのモジュールには多くの実装例があります。新しいモジュールを簡潔に作成できるのであらゆる事を表現することが可能になっています。これによってKerasは先進的な研究に適するモデルとなりました。
- __Pythonで実装しましょう。__ 各モデルはPythonによって実装されています。それはコンパクトでデバッグしやすく、簡単に拡張することが出来ます。


------------------



## 30秒でkerasに入門しましょう。

Kerasの中心はネットワーク層を構築するモデル(__model__)にあります。主なモデルとして線形に階層された逐次モデル([`Sequential`](http://keras.io/getting-started/sequential-model-guide))があります。
更に複雑な構造を実装する場合、[Keras functional API](http://keras.io/getting-started/functional-api-guide).を用いる必要があります。

逐次モデルの一例を見てみましょう。

```python
from keras.models import Sequential

model = Sequential()
```

階層（レイヤー）をスタックするのは次のように簡単です:

```python
from keras.layers import Dense, Activation

model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
```

実装したモデルが良さそうに見えたら`.compile()`で学習過程を確認しましょう。

```python
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

もし必要なら、最適化ルーチンを設定することも出来ます。Kerasの設計主義としてあらゆるものをシンプルに実装できるというものがあります。これはユーザーが必要なときにそれを直ぐに調整できるよう設計されているからです。

```python
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```

ここでトレーニングデータのバッチサイズを指定してエポック数だけ実行できます。

```python
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
```

以上の代わりに、バッチサイズを別に規定することも出来ます。

```python
model.train_on_batch(X_batch, Y_batch)
```

また、たったの一行で学習精度を表示することも出来ます。

```python
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
```

また、新しいテストデータに対してモデルを適用することも出来ます。

```python
classes = model.predict_classes(X_test, batch_size=32)
proba = model.predict_proba(X_test, batch_size=32)
```

質疑応答システムをや画像分類、外部記憶を持ったニューラルネットワーク（neural turning machine）,word2vecモデル、その他多くのモデルを高速かつシンプルに実装することが可能となりました。深層学習の根底にあるアイデアはとてもシンプルです。実装もシンプルであるべきではないでしょうか？

Kerasについてもっと詳しい情報が知りたければ以下のチュートリアルを参照してください。

- [Getting started with the Sequential model](http://keras.io/getting-started/sequential-model-guide)
- [Getting started with the functional API](http://keras.io/getting-started/functional-api-guide)

examples folderのレポジトリにはもっと高度なモデルが保存してあります。
メモリーネットワークを用いた質疑応答システムや積層LSTMを用いた文章生成等です。

------------------


## インストール

Kerasは以下のライブラリに依存関係があります。

- numpy, scipy
- pyyaml
- HDF5 h5py (もし必要なら。関数をセーブしたりロードしたりして呼び出したい場合)
- Optional but recommended if you use CNNs: cuDNN.

*Theanoをバックエンドで使用したい場合:*

- Theano
    - [See installation instructions](http://deeplearning.net/software/theano/install.html#install).

*Tensorflowをバックエンドで使用したい場合:*

- TensorFlow
    - [See installation instructions](https://github.com/tensorflow/tensorflow#download-and-setup).

Kerasをインストールするには、まず、端末上でcdコマンドでkerasのフォルダに行って以下のインストールコマンドを入力してください。

```
sudo python setup.py install
```

PyPIからもインストール出来ます。

```
sudo pip install keras
```

------------------


## TheanoからTensorflowに変更する方法

初期ではKerasはTheanoをテンソル計算ライブラリとしています。気になる方はKerasのバックエンドについての以下の導入事項を確認ください。[Follow these instructions](http://keras.io/backend/)

------------------


## サポート

もし、質問や開発についての議論に参加したい場合は [Keras Google group](https://groups.google.com/forum/#!forum/keras-users) まで。

また、バグ報告やgithub上のfeature requestがあればよろしくお願いします。また、その場合、先に [our guidelines](https://github.com/fchollet/keras/blob/master/CONTRIBUTING.md) を読まれる事をお願いします。


------------------
