# SequentialモデルAPI

はじめに，[KerasのSequentialモデルのガイド](/getting-started/sequential-model-guide) をご覧下さい．

## モデルの有用な属性

- `model.layers`は，モデルに加えたレイヤーのリストです．

----

## Sequentialモデルのメソッド

### compile


```python
compile(self, optimizer, loss, metrics=None, sample_weight_mode=None)
```


学習過程の設定．

__引数__

- __optimizer__: 文字列型 (optimizer 名) あるいは optimizer のオブジェクト．
	[optimizers](/optimizers) をご覧下さい．
- __loss__: 文字列型 (objective 関数名) あるいは objective 関数．
	[losses](/losses) をご覧下さい．
- __metrics__: 訓練やテストの際にモデルを評価するためのメトリックのリスト．
	典型的には `metrics=['accuracy']`を使用するでしょう．[metrics](/metrics) をご覧下さい．
- __sample_weight_mode__: もし時間ごとのサンプルの重み付け (2次元の重み) を行う必要があれば
	"temporal" と設定して下さい．
	"None" の場合，サンプルへの (1次元) 重み付けをデフォルトとしています．
- __kwargs__: Theano がバックエンドの場合， これらは K.function に渡されます．
	Tensorflow がバックエンドの場合は無視されます．

__例__

```python
model = Sequential()
model.add(Dense(32, input_shape=(500,)))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
```

----

### fit


```python
fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
```


固定のエポック数でモデルを学習する．

__引数__

- __x__: 入力データ，Numpy 配列，あるいは Numpy 配列のリスト
	(モデルに複数の入力がある場合)．
- __y__: ラベル，Numpy 配列．
- __batch_size__: 整数値．設定したサンプル数ごとに勾配の更新を行います．
- __epochs__: 整数値で，モデルを学習するエポック数．
- __verbose__: 0 とすると標準出力にログを出力しません．
	1 の場合はログをプログレスバーで標準出力，2 の場合はエポックごとに 1 行のログを出力します．
- __callbacks__: `keras.callbacks.Callback` にあるインスタンスのリスト．
	トレーニングの間 callbacks のリストを適用します．
	[callbacks](/callbacks)をご覧下さい．
- __validation_split__: float (0. < x < 1) で，
	ホールドアウト検証のデータとして使うデータの割合．
- __validation_data__: ホールドアウト検証用データとして使うデータのタプル (x_val, y_val) か (x_val, y_val, val_sample_weights)．
	設定すると validation_split を無視します．
- __shuffle__: boolean か str (for 'batch')．
	各エポックにおいてサンプルをシャッフルするかどうか．
	'batch' は HDF5 データだけに使える特別なオプションです．バッチサイズのチャンクの中においてシャッフルします．
- __class_weight__: dictionary で，クラス毎の重みを格納します．
	(学習の間だけ) 損失関数をスケーリングするために使います．
- __sample_weight__: 入力サンプルと同じ長さの1次元の Numpy 配列で，学習のサンプルに対する重みを格納します．
	これは損失関数をスケーリングするために (学習の間だけ) 使用します．
	- (重みとサンプルの間における1:1 の写像),
	あるいは系列データの場合において，2次元配列の (samples, sequence_length) という形式で，
	すべてのサンプルの各時間において異なる重みを適用できます．
	この場合，compile() の中で sample_weight_mode="temporal" と確実に明記すべきです．
- __initial_epoch__: 学習開始時のepoch (前の学習から再開する際に便利です)．

__返り値__

`History` オブジェクト． `History.history` 属性は
実行に成功したエポックにおける学習の損失値とメトリックの値の記録と，(適用可能ならば) 検証における損失値とメトリックの値も記録しています．

__Raises__

- __RuntimeError__: モデルが1度もcompileされていないとき．

----

### evaluate

```python
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
```

バッチごとにある入力データにおける損失値を計算します．

__引数__

- __x__: 入力データ，Numpy 配列あるいは Numpy 配列のリスト
	(モデルに複数の入力がある場合)．
- __y__: ラベル，Numpy 配列．
- __batch_size__: 整数値で，指定したサンプル数で勾配の更新を行います．
- __verbose__: 進行状況メッセージ出力モードで，0 か 1．
- __sample_weight__: サンプルの重み，Numpy 配列．

__返り値__

スカラーで，テストデータの損失値 (モデルのメトリックを設定していない場合)
あるいはスカラーのリスト (モデルが他のメトリックを計算している場合) ．
属性 `model.metrics_names` により，スカラーの出力でラベルを表示します．

__Raises__

- __RuntimeError__: モデルが1度もcompileされていないとき．

----

### predict


```python
predict(self, x, batch_size=32, verbose=0)
```

入力サンプルに対する予測値の出力を生成します．

入力サンプルごとにバッチごとに処理します．

__引数__

- __x__: 入力データで，Numpy 配列の形式．
- __batch_size__: 整数値．
- __verbose__: 進行状況メッセージ出力モード，0 または 1．

__返り値__

予測値を格納した Numpy 配列．

----

### predict_classes


```python
predict_classes(self, x, batch_size=32, verbose=1)
```

バッチごとに入力サンプルに対するクラスの予測を生成します．

入力サンプルごとにバッチごとに処理します．

__引数__

- __x__: 入力データ，Numpy 配列または Numpy 配列のリスト (モデルに複数の入力がある場合)．
- __batch_size__: 整数値．
- __verbose__: 進行状況メッセージ出力モード，0 または 1．

__返り値__

属するクラスの予測を格納した Numpy 配列．

----

### predict_proba


```python
predict_proba(self, x, batch_size=32, verbose=1)
```


入力サンプルの各々のクラスの所属確率を生成します．

入力サンプルごとにバッチごとに処理します．

__引数__

- __x__: 入力データ，Numpy 配列または Numpy 配列のリスト (モデルに複数の入力がある場合)．
- __batch_size__: 整数値．
- __verbose__: 進行状況メッセージ出力モード，0 または 1．

__返り値__

確率の予測値を格納した Numpy 配列．

----

### train_on_batch


```python
train_on_batch(self, x, y, class_weight=None, sample_weight=None)
```

サンプル中の1つのバッチで勾配を更新します．

__引数__

- __x__: 入力データ，Numpy 配列または Numpy 配列のリスト (モデルに複数の入力がある場合)．
- __y__: ラベル，Numpy 配列．
- __class_weight__: dictionary 型で，クラス毎の重みを格納します．
	(学習の間だけ) 損失関数をスケーリングするために使います．
- __sample_weight__: サンプルの重み，Numpy 配列．

__返り値__

スカラーでトレーニングの損失値 (モデルにメトリックが設定されていない場合)
あるいはスカラーのリスト (モデルが他のメトリックを計算している場合)．
`model.metrics_names` 属性により，スカラーの出力でラベルを表示する．

__Raises__

- __RuntimeError__: モデルが1度もcompileされていないとき．

----

### test_on_batch


```python
test_on_batch(self, x, y, sample_weight=None)
```

サンプルの単一バッチにおけるモデルの評価を行います．

__引数__

- __x__: 入力データ，Numpy 配列または Numpy 配列のリスト (モデルに複数の入力がある場合)．
- __y__: ラベル，Numpy 配列の形式．
- __sample_weight__: サンプルの重み，Numpy 配列の形式．

__返り値__

スカラーで，テストの損失値 (モデルにメトリックが設定されていない場合)
あるいはスカラーのリスト (モデルが他のメトリックを計算している場合)．
`model.metrics_names` 属性により，スカラーの出力でラベルを表示する．

__Raises__

- __RuntimeError__: モデルが1度もcompileされていないとき．

----

### predict_on_batch


```python
predict_on_batch(self, x)
```

サンプルの単一のバッチに対する予測値を返します．

__引数__

- __x__: 入力データ，Numpy 配列または Numpy 配列のリスト (モデルに複数の入力がある場合)．

__返り値__

予測値を格納した Numpy 配列．

----

### fit_generator


```python
fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)
```

Python のジェネレータにより，バッチごとに生成されるデータでモデルを学習させます．

ジェネレータは効率化のために，モデルを並列に実行します．
たとえば，これを使えば CPU 上でリアルタイムに画像データを拡大しながら，それと並行して GPU 上でモデルを学習できます．

__引数__

- __generator__: ジェネレータ．
	ジェネレータの出力は以下のいずれかでなければならず，どの配列も同数のサンプルを含まなければなりません．
	- タプル (inputs, targets)
	- タプル (inputs, targets, sample_weights)．
	ジェネレータは永遠にそのデータを繰り返すことを期待されています．
	`samples_per_epoch` 数のサンプルが，モデルによって確認されたときにエポックが終了します．
- __steps_per_epoch__: 1エポックを宣言してから次のエポックの開始前までに`generator`から生成されるサンプル (サンプルのバッチ) の総数．
典型的には，データにおけるユニークなサンプル数をバッチサイズで割った値です．
- __epochs__: 整数値で，イテレーションの総数．
- __verbose__: 進行状況メッセージ出力モードで，0，1，あるいは 2．
- __callbacks__: callbacks のリストで，学習の際に呼び出されます．
- __validation_data__: 以下のいずれかです．
	- 検証用データのジェネレータ
	- タプル (inputs, targets)
	- タプル (inputs, targets, sample_weights)．
- __validation_steps__: `validation_data` がジェネレータである場合だけ関係があります．
	各エポックの終わりに検証用ジェネレータから使用するステップ数です．典型的には，検証用データにおけるユニークなサンプル数をバッチサイズで割った値です．
- __class_weight__: dictionary 型で，クラス毎の重みを格納します．
	(学習の間だけ) 損失関数をスケーリングするために使います．
- __max_q_size__: ジェネレータのキューの最大サイズ．
- __workers__: スレッドベースのプロセス使用時の最大プロセス数
- __pickle_safe__: Trueならスレッドベースのプロセスを使います．実装がmultiprocessingに依存しているため，子プロセスに簡単に渡すことができないものとしてPickableでない引数をgeneratorに渡すべきではないことに注意してください．
- __initial_epoch__: 学習開始時のepoch (前の学習から再開する際に便利です)．

__返り値__

`History` オブジェクト．

__Raises__

- __RuntimeError__: モデルが1度もcompileされていないとき．

__例__

```python
def generate_arrays_from_file(path):
    while 1:
        f = open(path)
        for line in f:
            # create numpy arrays of input data
            # and labels, from each line in the file
            x, y = process_line(line)
            yield (x, y)
        f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
        samples_per_epoch=10000, epochs=10)
```

----

### evaluate_generator


```python
evaluate_generator(self, generator, steps, max_q_size=10, workers=1, pickle_safe=False)
```

ジェネレータのデータによってモデルを評価します．

ジェネレータは `test_on_batch` が受け取るデータと同じ種類のデータを返却するべきです．

__引数__

- __generator__:
	(inputs, targets)あるいは(inputs, targets, sample_weights)のタプルを生成するジェネレーター．
- __steps__:
	`generator`が停止するまでに生成するサンプル (サンプルのバッチ) の総数
- __max_q_size__: ジェネレータのキューの最大サイズ
- __nb_worker__: スレッドベースのプロセス使用時の最大プロセス数
- __pickle_safe__: Trueならスレッドベースのプロセスを使います．実装がmultiprocessingに依存しているため，子プロセスに簡単に渡すことができないものとしてPickableでない引数をgeneratorに渡すべきではないことに注意してください．

__返り値__

スカラーで，テストの損失値 (モデルにメトリックが設定されていない場合)
あるいはスカラーのリスト (モデルが他のメトリックを計算している場合)．
`model.metrics_names` 属性により，スカラーの出力でラベルを表示する．

__Raises__

- __RuntimeError__: モデルが1度もcompileされていないとき．

----

### predict_generator


```python
predict_generator(self, generator, steps, max_q_size=10, workers=1, pickle_safe=False, verbose=0)
```

ジェネレータから生成されたデータに対して予測します．

ジェネレータは `predict_on_batch` が受け取るデータと同じ種類のデータを返すべきです．

__引数__

- __generator__:
	入力サンプルのバッチを生成するジェネレータ．
- __steps__: 
	`generator`が停止するまでに生成するサンプル (サンプルのバッチ) の総数
- __max_q_size__: ジェネレータのキューの最大サイズ
- __workers__: スレッドベースのプロセス使用時の最大プロセス数
- __pickle_safe__: Trueならスレッドベースのプロセスを使います．実装がmultiprocessingに依存しているため，子プロセスに簡単に渡すことができないものとしてPickableでない引数をgeneratorに渡すべきではないことに注意してください．
- __verbose__: 進行状況メッセージ出力モード，0 または 1．

__返り値__

予測値の Numpy 配列．
