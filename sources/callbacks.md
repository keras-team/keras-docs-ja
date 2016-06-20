## コールバックの使い方

コールバックはトレーニング手順の段階で適用される関数セットです。トレーニング中にモデル内部の状態と統計を可視化する際に、コールバックを使います。`Sequential`モデルの`.fit()`メソッドに(キーワード引数`callbacks`として)コールバックのリストを渡すことができます。コールバックに関連するメソッドは、トレーニングの各段階で呼び出されます。

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L359)</span>
### RemoteMonitor

```python
keras.callbacks.RemoteMonitor(root='http://localhost:9000')
```

このコールバックはサーバーにイベントをストリームするときに使用されます。

`requests`ライブラリが必要となります。

__引数__

- __root__: イベントのルートURLは(すべてのエポックの終わりに)送信されます。イベントは`root + '/publish/epoch/end/'`に送信されます。コールすることによって、イベントデータをJSONエンコードした辞書型の`data`引数をHTTP POSTされます。

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L389)</span>
### LearningRateScheduler

```python
keras.callbacks.LearningRateScheduler(schedule)
```

学習率のスケジューラ。

__引数__

- __schedule__: この関数はエポックのインデックス(数値型, 0から始まるインデックス)を入力とし、新しい学習率(float)を返します。

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L409)</span>
### TensorBoard

```python
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)
```

 Tensorboardによる基本的な可視化。

 このコールバックはTensorBoardのログを出力します。TensorBoardでは、異なる層への活性化ヒストグラムと同様に、トレーニングとテストのメトリクスを動的にグラフ化し、可視化することができます。

 TensorBoardはTensorFlowによって提供されている可視化ツールです。

 pipからTensorFlowをインストールしているならば、コマンドラインからTensorBoardを起動することができます。
```
tensorboard --logdir=/full_path_to_your_logs
```
TensorBoardに関する詳細な情報は以下を参照してください。
- __[here](https__://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html).

__引数__

- __log_dir__: tensorfboardによって解析されたログファイルを保存するディレクトリのパス
- __histogram_freq__: モデルの層の活性化ヒストグラムを計算する(エポック中の)頻度。この値を0に設定するとヒストグラムが計算されることはありません。
- __write_graph__: tensorboardのグラフを可視化するか。write_graphがTrueに設定されている場合、ログファイルが非常に大きくなることがあります。

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L77)</span>
### Callback

```python
keras.callbacks.Callback()
```

この抽象基底クラスは新しいコールバックを構築するために使用されます。

__プロパティ__

- __params__: 辞書型。トレーニングのパラメータ(例: 冗長性,バッチサイズ,エポック数...)。
- __model__: `keras.models.Model`のインスタンス.トレーニングされたモデルのリファレンス。

コールバック関数が引数としてとる辞書型の`logs`は、現在のバッチ数かエポック数に関連したデータのキーを含みます。

現在、`Sequential`モデルクラスの`.fit()`メソッドは、そのコールバックに渡す`logs`に次のデータが含まれます。

- __on_epoch_end__: ログは`acc`と`loss`を含み、オプションとして(`fit`内のバリデーションが有効になっている場合は)`val_loss`、(バリデーションと精度の監視が有効になっている場合は)`val_acc`を含みます。
- __on_batch_begin__: ログは現在のバッチのサンプル数`size`を含みます。
- __on_batch_end__: ログは`loss`と(精度の監視が有効になっている場合は)オプションとして`acc`を含みます。

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L131)</span>
### BaseLogger

```python
keras.callbacks.BaseLogger()
```

監視されているメトリクスのエポック平均を蓄積するコールバックです。

このコールバックはすべてのKerasモデルに自動的に適用されます。

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L159)</span>
### ProgbarLogger

```python
keras.callbacks.ProgbarLogger()
```

標準出力にメトリクスを出力するコールバックです。

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L198)</span>
### History

```python
keras.callbacks.History()
```

`History`オブジェクトにイベントを記録するコールバックです。

このコールバックはすべてのKersモデルに自動的に適用されます。`History`オブジェクトはモデルの`fit`メソッドで返り値を取得します。

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L218)</span>
### ModelCheckpoint

```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
```

全エポック終了後にモデルを保存します。

`filepath`は、(`on_epoch_end`で渡された)`epoch`の値と`logs`のキーで埋められた書式設定オプションを含むことができます。

例えば、`filepath`が`weights.{epoch:02d}-{val_loss:.2f}.hdf5`の場合、複数のファイルがエポック数とバリデーションロスの値を付与して保存されます。

__引数__

- __filepath__: 文字列型, モデルファイルを保存するパス。
- __monitor__: 監視するデータ。
- __verbose__: 冗長モード, 0 または 1。
- __save_best_only__: `save_best_only=True`の場合、バリデーションロスによって最新の最良モデルが上書きされることはありません。
- __mode__: {auto, min, max}の内、一つが選択されます。`save_best_only=True`ならば、現在保存されているファイルを上書きするかは、監視されている最大化もしくは最小化によって判定されます。`val_acc`の場合、この引数は`max`となり、`val_loss`の場合は`min`になります。`auto`モードでは、この傾向は自動的に監視されているデータから推測されます。

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L301)</span>
### EarlyStopping

```python
keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
```

監視されているデータの変化が停止した時にトレーニングを終了します。

__引数__

- __monitor__: 監視されるデータ。
- __patience__: トレーニングが停止し、値が改善しなくなった時のエポック数。
- __verbose__: 冗長モード。
- __mode__: {auto, min, max}の内、一つが選択されます。`min`モードでは、監視されているデータの減少が停止した際に、トレーニングを終了します。また、`max`モードでは、監視されているデータの増加が停止した際に、トレーニングを終了します。

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L14)</span>
### CallbackList

```python
keras.callbacks.CallbackList(callbacks=[], queue_length=10)
```


---


# コールバックを作成する

基本クラスの`keras.callbacks.Callback`を拡張することで、カスタムコールバックを作成することができます。コールバックは、`self.model`プロパティによって、関連したモデルにアクセスすることができます。

トレーニング中の各バッチの損失のリストを保存する簡単な例は、以下のようになります。
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
バリデーションロスが減少した場合に、各エポック終了後、モデルの重みを保存します
'''
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])

```
