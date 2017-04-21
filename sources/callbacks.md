## コールバックの使い方

コールバックはトレーニング手順の段階で適用される関数セットです．トレーニング中にモデル内部の状態と統計を可視化する際に，コールバックを使います．`Sequential`モデルの`.fit()`メソッドに(キーワード引数`callbacks`として)コールバックのリストを渡すことができます．コールバックに関連するメソッドは，トレーニングの各段階で呼び出されます．

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L78)</span>
### Callback

```python
keras.callbacks.Callback()
```

この抽象基底クラスは新しいコールバックを構築するために使用されます．

__プロパティ__

- __params__: 辞書型．トレーニングのパラメータ(例: 冗長性,バッチサイズ,エポック数...)．
- __model__: `keras.models.Model`のインスタンス.トレーニングされたモデルのリファレンス．

コールバック関数が引数としてとる辞書型の`logs`は，現在のバッチ数かエポック数に関連したデータのキーを含みます．

現在，`Sequential`モデルクラスの`.fit()`メソッドは，そのコールバックに渡す`logs`に次のデータが含まれます．

- __on_epoch_end__: ログは`acc`と`loss`を含み，オプションとして(`fit`内のバリデーションが有効になっている場合は)`val_loss`，(バリデーションと精度の監視が有効になっている場合は)`val_acc`を含みます．
- __on_batch_begin__: ログは現在のバッチのサンプル数`size`を含みます．
- __on_batch_end__: ログは`loss`と(精度の監視が有効になっている場合は)オプションとして`acc`を含みます．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L132)</span>
### BaseLogger

```python
keras.callbacks.BaseLogger()
```

監視されているメトリクスのエポック平均を蓄積するコールバックです．

このコールバックはすべてのKerasモデルに自動的に適用されます．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L160)</span>
### ProgbarLogger

```python
keras.callbacks.ProgbarLogger()
```

標準出力にメトリクスを出力するコールバックです．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L199)</span>
### History

```python
keras.callbacks.History()
```

`History`オブジェクトにイベントを記録するコールバックです．

このコールバックはすべてのKersモデルに自動的に適用されます．`History`オブジェクトはモデルの`fit`メソッドで返り値を取得します．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L217)</span>
### ModelCheckpoint

```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
```

各エポック終了後にモデルを保存します．

`filepath`は，(`on_epoch_end`で渡された)`epoch`の値と`logs`のキーで埋められた書式設定オプションを含むことができます．

例えば，`filepath`が`weights.{epoch:02d}-{val_loss:.2f}.hdf5`の場合，複数のファイルがエポック数とバリデーションロスの値を付与して保存されます．

__引数__

- __filepath__: 文字列型, モデルファイルを保存するパス．
- __monitor__: 監視するデータ．
- __verbose__: 冗長モード, 0 または 1．
- __save_best_only__: `save_best_only=True`の場合，監視しているデータによって最新の最良モデルが上書きされることはありません．
- __mode__: {auto, min, max}の内，一つが選択されます．`save_best_only=True`ならば，現在保存されているファイルを上書きするかは，監視されている値を最大化もしくは最小化によって決定されます．`val_acc`の場合，この引数は`max`となり，`val_loss`の場合は`min`になります．`auto`モードでは，この傾向は自動的に監視されているデータから推測されます．
- __save_weights_only__: Trueなら，モデルの重みが保存されます (`model.save_weights(filepath)`)，そうでないなら，モデルの全体が保存されます(`model.save(filepath)`)．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L310)</span>
### EarlyStopping

```python
keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
```

監視されているデータの変化が停止した時にトレーニングを終了します．

__引数__

- __monitor__: 監視されるデータ．
- __patience__: トレーニングが停止し，値が改善しなくなった時のエポック数．
- __verbose__: 冗長モード．
- __mode__: {auto, min, max}の内，一つが選択されます．`min`モードでは，監視されているデータの減少が停止した際に，トレーニングを終了します．また，`max`モードでは，監視されているデータの増加が停止した際に，トレーニングを終了します．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L369)</span>
### RemoteMonitor

```python
keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data')
```

このコールバックはサーバーにイベントをストリームするときに使用されます．

`requests`ライブラリが必要となります．

__引数__

- __root__: イベントのルートURLは(すべてのエポックの終わりに)送信されます．イベントはデフォルトで`root + '/publish/epoch/end/'`に送信されます．コールすることによって，イベントデータをJSONエンコードした辞書型の`data`引数をHTTP POSTされます．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L405)</span>
### LearningRateScheduler

```python
keras.callbacks.LearningRateScheduler(schedule)
```

学習率のスケジューラ．

__引数__

- __schedule__: この関数はエポックのインデックス(数値型, 0から始まるインデックス)を入力とし，新しい学習率(float)を返します．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L425)</span>
### TensorBoard

```python
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)
```

 Tensorboardによる基本的な可視化．

 このコールバックはTensorBoardのログを出力します．TensorBoardでは，異なる層への活性化ヒストグラムと同様に，トレーニングとテストのメトリクスを動的にグラフ化し，可視化することができます．

 TensorBoardはTensorFlowによって提供されている可視化ツールです．

 pipからTensorFlowをインストールしているならば，コマンドラインからTensorBoardを起動することができます．
```
tensorboard --logdir=/full_path_to_your_logs
```
TensorBoardに関する詳細な情報は以下を参照してください．
- __[here](https__://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html).

__引数__

- __log_dir__: tensorfboardによって解析されたログファイルを保存するディレクトリのパス
- __histogram_freq__: モデルの層の活性化ヒストグラムを計算する(エポック中の)頻度．この値を0に設定するとヒストグラムが計算されることはありません．
- __write_graph__: tensorboardのグラフを可視化するか．write_graphがTrueに設定されている場合，ログファイルが非常に大きくなることがあります．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L549)</span>
### ReduceLROnPlateau

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
```

評価関数の改善が止まった時に学習率を減らします．

モデルは学習が停滞した時に学習率を2〜10で割ることで恩恵を受けることがあります．
このコールバックは評価関数を監視し，patienceで指定されたエポック数の間改善が見られなかった場合，学習率を減らします．

__例__

```python
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
				  patience=5, min_lr=0.001)
	model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

__引数__

- __monitor__: 監視するデータ．
- __factor__: 学習率を減らす割合．new_lr = lr * factor
- __patience__: 何エポックエポック改善が見られなかったら学習率の削減を行うか．
- __verbose__: 整数．0: 何も表示しない．1: 学習率削減時メッセージを表示．
- __mode__: `auto`，`min`，`max`のいずれか．
    `min`の場合，評価関数の減少が止まった時に学習率を更新します．
    `max`の場合，評価関数の増加が止まった時に学習率を更新します．
    `auto`の場合，monitorの名前から自動で判断します．
- __epsilon__: 改善があったと判断する閾値．重要な変化だけに注目するために用います．
- __cooldown__: 学習率を減らした後，通常の学習を再開するまで待機するエポック数．
- __min_lr__: 学習率の下限．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L654)</span>
### CSVLogger

```python
keras.callbacks.CSVLogger(filename, separator=',', append=False)
```

各エポックの結果をcsvファイルに保存するコールバックです．
np.ndarrayのような1次元イテラブルを含む，文字列表現可能な値をサポートしています．

__例__

```python
	csv_logger = CSVLogger('training.log')
	model.fit(X_train, Y_train, callbacks=[csv_logger])
```

__引数__

- __filename__: csvファイルの名前．例えば'run/log.csv'．
- __separator__: csvファイルで各要素を区切るために用いられる文字．
- __append__: True: ファイルが存在する場合，追記します．（学習を続ける場合に便利です）
    False: 既存のファイルを上書きします．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L708)</span>
### LambdaCallback

```python
keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
```

シンプルな自作コールバックを急いで作るためのコールバックです．

このコールバックは，適切なタイミングで呼び出される無名関数で構築されます．
以下のような位置引数が必要であることに注意してください:
 - `on_epoch_begin` と `on_epoch_end` は2つの位置引数が必要です: `epoch`，`logs`
 - `on_batch_begin` と `on_batch_end` は2つの位置引数が必要です: `batch`，`logs`
 - `on_train_begin` と `on_train_end` は1つの位置引数が必要です: `logs`

__引数__

- __on_epoch_begin__: すべてのエポックの開始時に呼ばれます．
- __on_epoch_end__: すべてのエポックの終了時に呼ばれます．
- __on_batch_begin__: すべてのバッチの開始時に呼ばれます．
- __on_batch_end__: すべてのバッチの終了時に呼ばれます．
- __on_train_begin__: 学習の開始時に呼ばれます．
- __on_train_end__: 学習の終了時に呼ばれます．

__例__

```python
# すべてのバッチの開始時にバッチ番号を表示
batch_print_callback = LambdaCallback(on_batch_begin=lambda batch, logs: print(batch))

# すべてのエポックの終了時に損失をプロット
import numpy as np
import matplotlib.pyplot as plt
plot_loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch), logs['loss']))

# 学習の終了時にいくつかのプロセスを終了
processes = ...
cleanup_callback = LambdaCallback(on_train_end=lambda logs: [p.terminate() for p in processes if p.is_alive()])

model.fit(..., callbacks=[batch_print_callback, plot_loss_callback, cleanup_callback])
```

---


# コールバックを作成する

基本クラスの`keras.callbacks.Callback`を拡張することで，カスタムコールバックを作成することができます．コールバックは，`self.model`プロパティによって，関連したモデルにアクセスすることができます．

トレーニング中の各バッチの損失のリストを保存する簡単な例は，以下のようになります．
```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```

---

### 例: 損失の履歴を記録する

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Dense(10, input_dim=784, init='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=0, callbacks=[history])

print history.losses
# 出力
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
'''
```

---

### 例: モデルのチェックポイント

```python
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(10, input_dim=784, init='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
バリデーションロスが減少した場合に，各エポック終了後，モデルの重みを保存します
'''
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])

```
