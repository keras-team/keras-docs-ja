## コールバックの使い方

コールバックは学習手順の段階で適用される関数集合です．学習中にモデル内部の状態と統計量を可視化する際に，コールバックを使います．`Sequential`と`Model`クラスの`.fit()`メソッドに（キーワード引数`callbacks`として）コールバックのリストを渡すことができます．コールバックに関連するメソッドは，学習の各段階で呼び出されます．

---

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L201)</span>
### BaseLogger

```python
keras.callbacks.BaseLogger()
```

監視されているメトリクスのエポック平均を蓄積するコールバックです．

このコールバックは全Kerasモデルに自動的に適用されます．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L146)</span>
### Callback

```python
keras.callbacks.Callback()
```

この抽象基底クラスは新しいコールバックを構築するために使用されます．

__プロパティ__

- __params__: 辞書型．学習のパラメータ(例: 冗長性，バッチサイズ，エポック数...)．
- __model__: `keras.models.Model`のインスタンス．学習されたモデルへの参照．

コールバック関数が引数としてとる辞書型の`logs`は，現在のバッチ数かエポック数に関連したデータのキーを含みます．

現在，`Sequential`モデルクラスの`.fit()`メソッドは，そのコールバックに渡す`logs`に以下のデータが含まれます．

- __on_epoch_end__: ログは`acc`と`loss`を含み，オプションとして(`fit`内のバリデーションが有効になっている場合は)`val_loss`，(バリデーションと精度の監視が有効になっている場合は)`val_acc`を含みます．
- __on_batch_begin__: ログは現在のバッチのサンプル数`size`を含みます．
- __on_batch_end__: ログは`loss`と(精度の監視が有効になっている場合は)オプションとして`acc`を含みます．

----



<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L160)</span>
### ProgbarLogger

```python
keras.callbacks.ProgbarLogger(count_mode='samples')
```

標準出力にメトリクスを出力するコールバックです．

__引数__

- __count_mode__: "steps"か"samples"の一方．サンプルかステップ（バッチ）のどちらをプログレスバーの集計に使うか．

__Raises__

- __ValueError__: `count_mode`の値が不正のとき．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L199)</span>
### History

```python
keras.callbacks.History()
```

`History`オブジェクトにイベントを記録するコールバックです．

このコールバックは全Kersモデルに自動的に適用されます．`History`オブジェクトはモデルの`fit`メソッドの返り値として取得します．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L217)</span>
### ModelCheckpoint

```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

各エポック終了後にモデルを保存します．

`filepath`は，(`on_epoch_end`で渡された)`epoch`の値と`logs`のキーで埋められた書式設定オプションを含むことができます．

例えば，`filepath`が`weights.{epoch:02d}-{val_loss:.2f}.hdf5`の場合，複数のファイルがエポック数とバリデーションロスの値を付与して保存されます．

__引数__

- __filepath__: 文字列，モデルファイルを保存するパス．
- __monitor__: 監視する値．
- __verbose__: 冗長モード, 0 または 1．
- __save_best_only__: `save_best_only=True`の場合，監視しているデータによって最新の最良モデルが上書きされません．
- __mode__: {auto, min, max}の内の一つが選択されます．`save_best_only=True`ならば，現在保存されているファイルを上書きするかは，監視されている値の最大化か最小化によって決定されます．`val_acc`の場合，この引数は`max`となり，`val_loss`の場合は`min`になります．`auto`モードでは，この傾向は自動的に監視されている値から推定します．
- __save_weights_only__: Trueなら，モデルの重みが保存されます (`model.save_weights(filepath)`)，そうでないなら，モデルの全体が保存されます(`model.save(filepath)`)．
- __period__: チェックポイント間の間隔(エポック数)．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L310)</span>
### EarlyStopping

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
```

監視する値の変化が停止した時に学習を終了します．

__引数__

- __monitor__: 監視する値．
- __min_delta__: 監視する値について改善として判定される最小変化値．つまり，min_deltaよりも絶対値の変化が小さければ改善していないとみなします．
- __patience__: 学習が停止し，値が改善しなくなってからのエポック数．
- __verbose__: 冗長モード．
- __mode__: {auto, min, max}の内，一つが選択されます．`min`モードでは，監視する値の減少が停止した際に，学習を終了します．また，`max`モードでは，監視する値の増加が停止した際に，学習を終了します．`auto`モードでは，この傾向は自動的に監視されている値から推定します．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L369)</span>
### RemoteMonitor

```python
keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None)
```

このコールバックはサーバーにイベントをストリームするときに使用されます．

`requests`ライブラリが必要です．イベントはデフォルトで`root + '/publish/epoch/end/'`に送信されます．
コールすることによって，イベントデータをJSONエンコードした辞書型の`data`引数をHTTP POSTされます．

__引数__

- __root__: 文字列；対象サーバのルートURL．
- __path__: 文字列；イベントを送るサーバへの相対的な`path`．
- __field__: 文字列；データを保存するJSONのフィールド．
- __headers__: 辞書型; オプションでカスタムできるHTTPヘッダー．デフォルト: 
    - `{'Accept': 'application/json', 'Content-Type': 'application/json'}`


----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L405)</span>
### LearningRateScheduler

```python
keras.callbacks.LearningRateScheduler(schedule)
```

学習率のスケジューラ．

__引数__

- __schedule__: この関数はエポックのインデックス(整数型, 0から始まるインデックス)を入力とし，新しい学習率(float)を返します．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L425)</span>
### TensorBoard

```python
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
```

Tensorboardによる基本的な可視化．

このコールバックはTensorBoardのログを出力します．TensorBoardでは，異なる層への活性化ヒストグラムと同様に，トレーニングとテストのメトリクスを動的にグラフ化し，可視化できます．

TensorBoardはTensorFlowによって提供されている可視化ツールです．

pipからTensorFlowをインストールしているならば，コマンドラインからTensorBoardを起動できます．

```
tensorboard --logdir=/full_path_to_your_logs
```

TensorBoardに関する詳細な情報は[ここ](https://www.tensorflow.org/get_started/summaries_and_tensorboard)を参照してください．

__引数__

- __log_dir__: TensorfBoardによって解析されたログファイルを保存するディレクトリのパス
- __histogram_freq__: モデルの層の活性化ヒストグラムを計算する(エポック中の)頻度．この値を0に設定するとヒストグラムが計算されません．
- __write_graph__: TensorBoardのグラフを可視化するか．`write_graph`がTrueの場合，ログファイルが非常に大きくなることがあります．
- __write_grads__: TensorBoardに勾配のヒストグラフを可視化するかどうか．`histogram_freq`は0より大きくしなければなりません．
- __write_images__: TensorfBoardで可視化するモデルの重みを画像として書き出すかどうか．
- __embeddings_freq__: 選択したembeddingsレイヤーを保存する(エポックに対する)頻度．
- __embeddings_layer_names__: 
観察するレイヤー名のリスト．もしNoneか空リストなら全embeddingsレイヤーを観察します．
- __embeddings_metadata__: 
レイヤー名からembeddingsレイヤーに関するメタデータの保存ファイル名へマップする辞書．
メタデータのファイルフォーマットの[詳細](https://www.tensorflow.org/get_started/embedding_viz#metadata_optional)．
全embeddingsレイヤーに対して同じメタデータファイルを使う場合は文字列を渡します．

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L549)</span>
### ReduceLROnPlateau

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
```

メトリクスの改善が止まった時に学習率を減らします．

モデルは学習が停滞した時に学習率を2〜10で割ることで恩恵を受けることがあります．
このコールバックはメトリクスを監視し，patienceで指定されたエポック数の間改善が見られなかった場合，学習率を減らします．

__例__

```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(x_train, y_train, callbacks=[reduce_lr])
```

__引数__

- __monitor__: 監視する値．
- __factor__: 学習率を減らす割合．new_lr = lr * factor
- __patience__: 何エポック改善が見られなかったら学習率の削減を行うか．
- __verbose__: 整数．0: 何も表示しない．1: 学習率削減時メッセージを表示．
- __mode__: `auto`，`min`，`max`のいずれか．
    `min`の場合，監視する値の減少が停止した際に，学習率を更新します．
    `max`の場合，監視する値の増加が停止した時に，学習率を更新します．
    `auto`の場合，監視する値の名前から自動で判断します．
- __epsilon__: 改善があったと判断する閾値．有意な変化だけに注目するために用います．
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
model.fit(x_train, y_train, callbacks=[csv_logger])
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
batch_print_callback = LambdaCallback(
    on_batch_begin=lambda batch,logs: print(batch))

# すべてのエポックの終了時に損失値をプロット
import numpy as np
import matplotlib.pyplot as plt
plot_loss_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch),
                      logs['loss']))

# 学習の終了時にいくつかのプロセスを終了
processes = ...
cleanup_callback = LambdaCallback(
    on_train_end=lambda logs: [
    p.terminate() for p in processes if p.is_alive()])

model.fit(...,
      callbacks=[batch_print_callback,
         plot_loss_callback,
         cleanup_callback])
```

---


# コールバックを作成

基底クラスの`keras.callbacks.Callback`を拡張することで，カスタムコールバックを作成できます．
コールバックは，`self.model`プロパティによって，関連したモデルにアクセスできます．

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
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

print(history.losses)
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
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
バリデーションロスが減少した場合に，各エポック終了後，モデルの重みを保存します
'''
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, validation_data=(x_test, y_test), callbacks=[checkpointer])
```
