# Keras: TensorFlowとTheanoのための深層学習ライブラリ

## あなたはKerasを見つけました．

Kerasは，Pythonで書かれた，[TensorFlow](https://github.com/tensorflow/tensorflow)または[Theano](https://github.com/Theano/Theano)上で実行可能な高水準のニューラルネットワークライブラリです．
Kerasは，迅速な実験を可能にすることに重点を置いて開発されました．
*可能な限り遅れなくアイデアから結果に進められることは，良い研究をする上で重要です．* 

次のような深層学習ライブラリが必要な場合は，Kerasを使用します:

- 簡単で早くプロトタイプ作成が可能 (全体的なモジュール性，ミニマリズム，および拡張性による)
- CNNとRNNの両方，およびこれらの2つの組み合わせをサポート
- 任意の接続方式 (複数入力および複数出力の学習を含む) をサポート
- CPUとGPUでシームレスな実行

[Keras.io](https://keras.io)のドキュメントを読んでください．

Kerasは**Python 2.7-3.5**と互換性があります．


------------------


## ガイドライン

- __モジュール性__: モデルとは，できるだけ制約を少なく接続可能で，完全構成可能な独立したモジュールのシーケンスまたはグラフとして理解されています． 
特に，ニューラルネットワークの層，損失関数，最適化，初期化，活性化関数，正規化はすべて新しいモデルを作成するために組み合わせられる独立したモジュールです．
- __ミニマリズム__: それぞれのモジュールが短く，簡潔に構成されています．それぞれのコードが分かりやすく，ブラックボックス化されている部分がありません．これによって作業スピードが上がり，革新的な事をしやすくなるでしょう．
- __拡張性__: 新しいモジュールを追加するのがとても簡単です（新しいクラスや関数として）．また，それぞれのモジュールには多くの実装例があります．新しいモジュールを簡潔に作成できるのであらゆる事を表現することが可能になっています．これによってKerasは先進的な研究に適しています．
- __Pythonで実装__: 宣言形式の別個のモデル設定ファイルはありません．モデルはPythonコードで記述されます，これは，コンパクトでデバッグと拡張が容易です．


------------------


## 30秒でkerasに入門しましょう．

Kerasの中心的なデータ構造は__model__で，層を構成する方法です．
主なモデルはSequentialモデル([`Sequential`](http://keras.io/getting-started/sequential-model-guide))で，層の線形スタックです．
更に複雑なアーキテクチャの場合は，[Keras functional API](http://keras.io/getting-started/functional-api-guide)を使用する必要があります．

`Sequential` モデルの一例を見てみましょう．

```python
from keras.models import Sequential

model = Sequential()
```

`.add()`で簡単に層を積めます: 

```python
from keras.layers import Dense, Activation

model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
```

実装したモデルがよさそうなら`.compile()`で学習プロセスを設定しましょう．

```python
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

必要に応じて，最適化アルゴリズムも設定できます．Kerasの中心的な設計思想は，ユーザーが必要なときに完全にコントロールできるようにしながら (ソースコードの容易な拡張性を実現する究極のコントロール) ，適度に単純にすることです．

```python
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```

学習データをミニバッチで繰り返し処理できます．

```python
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
```

代わりに，バッチサイズを別に規定できます．

```python
model.train_on_batch(X_batch, Y_batch)
```

また，1行でモデルの評価．

```python
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
```

また，新しいデータに対して予測:

```python
classes = model.predict_classes(X_test, batch_size=32)
proba = model.predict_proba(X_test, batch_size=32)
```

質問応答システムや画像分類，ニューラルチューリングマシン，word2vecやその他多くのモデルは高速かつシンプルに実装可能です．深層学習の根底にあるアイデアはとてもシンプルです．実装もシンプルであるべきではないでしょうか？

Kerasについてのより詳細なチュートリアルについては，以下を参照してください．

- [Getting started with the Sequential model](http://keras.io/getting-started/sequential-model-guide)
- [Getting started with the functional API](http://keras.io/getting-started/functional-api-guide)

レポジトリの[examples folder](https://github.com/fchollet/keras/tree/master/examples)にはさらに高度なモデルがあります．
メモリーネットワークを用いた質問応答システムや積層LSTMを用いた文章生成などです．


------------------


## インストール

Kerasは以下のライブラリに依存関係があります．

- numpy, scipy
- pyyaml
- HDF5 h5py (モデルの保存や読み込み関数を使う場合のみ)
- cuDNN: オプションですが，CNNを使用する場合は推奨


*Tensorflowをバックエンドで使用する場合:*

- TensorFlow
    - [See installation instructions](https://github.com/tensorflow/tensorflow#download-and-setup).

*Theanoをバックエンドで使用する場合:*

- Theano
    - [See installation instructions](http://deeplearning.net/software/theano/install.html#install).

Kerasをインストールするには，まず，ターミナル上で`cd`コマンドでkerasのフォルダに移動してから以下のインストールコマンドを実行してください．

```sh
sudo python setup.py install
```

PyPIからもインストールできます．

```sh
sudo pip install keras
```


------------------


## TensorFlowからTheanoへの変更

デフォルトでは，KerasはTensorFlowをテンソル計算ライブラリとしています．Kerasバックエンドを設定するには，[この手順](http://keras.io/backend/)に従ってください．


------------------


## サポート

質問と開発に関するディスカッションに参加できます:

- [Keras Google group](https://groups.google.com/forum/#!forum/keras-users)
- [Keras Gitter channel](https://gitter.im/Keras-io/Lobby)

Githubの問題にバグレポートや機能リクエストを投稿できます． まず[ガイドライン](https://github.com/fchollet/keras/blob/master/CONTRIBUTING.md)を必ず読んでください．

------------------
