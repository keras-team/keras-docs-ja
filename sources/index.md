# Keras: TensorFlowとTheanoのための深層学習ライブラリ

## Kerasとは

Kerasは，Pythonで書かれた，[TensorFlow](https://github.com/tensorflow/tensorflow)または[Theano](https://github.com/Theano/Theano)上で実行可能な高水準のニューラルネットワークライブラリです．
Kerasは，迅速な実験を可能にすることに重点を置いて開発されました．
*アイデアから結果に到達するまでのリードタイムをできるだけ小さくすることが，良い研究をするための鍵になります．* 

次のような場合で深層学習ライブラリが必要なら，Kerasを使用してください:

- 容易に素早くプロトタイプの作成が可能 (ユーザーフレンドリー，モジュール性，および拡張性による)
- CNNとRNNの両方，およびこれらの2つの組み合わせをサポート
- CPUとGPU上でシームレスな動作

[Keras.io](https://keras.io)のドキュメントを読んでください．

Kerasは**Python 2.7-3.5**に対応しています．


------------------


## ガイドライン

- __ユーザーフレンドリー__: Kerasは機械向けでなく，人間向けに設計されたライブラリです．ユーザーエクスペリエンスを前面と中心においています．Kerasは，認知負荷を軽減するためのベストプラクティスをフォローします．一貫したシンプルなAPI群を提供し，一般的な使用事例で要求されるユーザーアクションを最小限に抑え，ユーザーエラー時に明確で実用的なフィードバックを提供します．
- __モジュール性__: モデルとは，できるだけ制約の少ない接続が可能で，独立した，完全に設定可能なモジュールの，シーケンスまたはグラフとして理解されています．
特に，ニューラルネットワークの層，損失関数，最適化，初期化，活性化関数，正規化はすべて，新しいモデルを作成するための組み合わせ可能な，独立したモジュールです．
- __拡張性__: 新しいモジュールが（新しいクラスや関数として）簡単に追加できます．また，既存のモジュールには多くの実装例があります．新しいモジュールを容易に作成できるため，あらゆる表現が可能になっています．このことからKerasは先進的な研究に適しています．
- __Pythonで実装__: 宣言形式の設定ファイルを持ったモデルはありません．モデルはPythonコードで記述されています．このPythonコードは，コンパクトで，デバッグが容易で，簡単に拡張できます．


------------------


## 30秒でkerasに入門しましょう．

Kerasの中心的なデータ構造は__model__で，層を構成する方法です．
主なモデルは[`Sequential`](http://keras.io/getting-started/sequential-model-guide)モデルで，層の線形スタックです．
更に複雑なアーキテクチャの場合は，[Keras functional API](http://keras.io/getting-started/functional-api-guide)を使用する必要があります．これで層のなす任意のグラフが構築可能になります．

`Sequential` モデルの一例を見てみましょう．

```python
from keras.models import Sequential

model = Sequential()
```

`.add()`で簡単に層を積み重ねることができます: 

```python
from keras.layers import Dense, Activation

model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))
```

実装したモデルがよさそうなら`.compile()`で学習プロセスを設定しましょう．

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

必要に応じて，最適化アルゴリズムも設定できます．Kerasの中心的な設計思想は，ユーザーが必要なときに完全にコントロール (ソースコードの容易な拡張性を実現する究極のコントロール) できる一方で，適度に単純にすることです．

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

学習データをミニバッチで繰り返し処理できます．

```python
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

代わりに，バッチサイズを別に規定できます．

```python
model.train_on_batch(x_batch, y_batch)
```

また，1行でモデルの評価．

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

また，新しいデータに対して予測:

```python
classes = model.predict(x_test, batch_size=128)
```

質問応答システムや画像分類，ニューラルチューリングマシン，word2vecやその他多くのモデルは高速かつシンプルに実装可能です．深層学習の根底にあるアイデアはとてもシンプルです．実装もシンプルであるべきではないでしょうか？

Kerasについてのより詳細なチュートリアルについては，以下を参照してください．

- [Getting started with the Sequential model](http://keras.io/getting-started/sequential-model-guide)
- [Getting started with the functional API](http://keras.io/getting-started/functional-api-guide)

レポジトリの[examples folder](https://github.com/fchollet/keras/tree/master/examples)にはさらに高度なモデルがあります．
メモリーネットワークを用いた質問応答システムや積層LSTMを用いた文章生成などです．


------------------


## インストール

Kerasは以下のライブラリに依存しています．

- numpy, scipy
- yaml
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

デフォルトでは，KerasはTensorFlowをテンソル計算ライブラリとしています．Kerasのバックエンドを設定するには，[この手順](http://keras.io/backend/)に従ってください．


------------------


## サポート

質問をしたり，開発に関するディスカッションに参加することができます:

- [Keras Google group](https://groups.google.com/forum/#!forum/keras-users)上で
- [Keras Slack channel](https://kerasteam.slack.com)上で．チャンネルへのリクエストするには[このリンク](https://keras-slack-autojoin.herokuapp.com/)を使って下さい．

 [Githubのissues](https://github.com/fchollet/keras/issues)に**バグレポートや機能リクエスト**を投稿できます．まず[ガイドライン](https://github.com/fchollet/keras/blob/master/CONTRIBUTING.md)を必ず読んでください．

------------------


## どうしてこのライブラリにKerasという名前を付けたのですか？

Keras (κέρας) はギリシア語で**角**を意味します．古代ギリシア文学およびラテン文学における文学上の想像がこの名前の由来です．最初にこの想像が見つかったのは_Odyssey_で，夢の神 (_Oneiroi_，単数形 _Oneiros_)は，象牙の門を通って地上に訪れて偽りのビジョンで人々を騙す神と， 角の門を通って地上に訪れて起こるはずの未来を知らせる神とに分かれているそうです．これは κέρας (角) / κραίνω (遂行)と ἐλέφας (象牙) / ἐλεφαίρομαι (欺瞞) の似た響きを楽しむ言葉遊びです．

Kerasは当初プロジェクトONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System) の研究の一環として開発されました．

>_"Oneiroi are beyond our unravelling --who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ Homer, Odyssey 19. 562 ff (Shewring translation).

------------------
