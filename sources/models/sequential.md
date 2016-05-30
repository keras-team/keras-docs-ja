# Sequentialモデルの API

はじめに、 [KerasのSequentialモデルのガイド](/getting-started/sequential-model-guide) をご覧下さい。

## モデルの有用なアトリビュート

- `model.layers` は、モデルに加えたレイヤーのリストです。


----

## Sequentialモデルのメソッド

### compile


```python
compile(self, optimizer, loss, metrics=[], sample_weight_mode=None)
```


学習過程の設定。

__引数__

- __optimizer__: 文字列型str(optimizer　の名前) あるいは optimizer のオブジェクト。
	[optimizers](/optimizers) をご覧下さい。
- __loss__: 文字列型 str(objective 関数の名前) あるいは objective 関数。
	[objectives](/objectives) をご覧下さい。
- __metrics__: 訓練や検証の際にモデルを評価するためのメトリックのリスト
	典型的には `metrics=['accuracy']`を使用するでしょう。
- __sample_weight_mode__: もし時間ごとのサンプルの重み付け(2次元重み付け)　を行う必要があれば
	"temporal" と設定して下さい。
	"None" の場合、サンプルへの重み(1次元重み)　付けを既定値としています。
- __kwargs__: Theano がバックエンドの場合、 これらはK.function に渡されます。
	Tensorflow がバックエンドの場合は無視されます。

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
fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
```


固定回のエポック数でモデルをトレーニングする。

__引数__

- __x__: 入力データ, 形式は Numpy の配列、あるいはNumpy 配列のリスト
	(モデルに複数の入力がある場合).
- __y__: ラベル, 形式は Numpy 配列.
- __batch_size__: 整数で、 設定したサンプル数ごとに勾配の更新を行う。
- __nb_epoch__: 整数で、 モデルをトレーニングするエポック数。
- __verbose__: 0 とすると標準出力にログを出力しない。
	1 の場合ログを進捗バーで標準出力する、 2 の場合はエポックごとに1 行のログを出力する。 
- __callbacks__: `keras.callbacks.Callback` にあるインスタンスのリスト。
	トレーニングの間callbacks のリストを適用する。
	[callbacks](/callbacks)をご覧下さい。
- __validation_split__: float (0. < x < 1)型で、
	ホールドアウト検証のデータとして使うデータの割合。
- __validation_data__: ホールドアウト検証のデータとして使うデータのタプル (X, y)。
	validation_split　を無視する。
- __shuffle__: boolean　型もしくはstr 型(for 'batch')。
	各エポックにおいてサンプルをシャッフルするかどうか。
	'batch' はHDF5 データだけに使える特別なオプションである。バッチサイズのチャンクの中においてシャッフルする。
- __class_weight__: dictionary 型で、重みの値でクラスを分ける、
	損失関数をスケーリングするために(トレーニングの間だけ)使う。
- __sample_weight__: Numpy 配列で、トレーニングのサンプルに対する重みを格納する。
	これは損失関数をスケーリングするために(トレーニングの間だけ)使用する。
	入力サンプルと同じ長さの1次元のNumpy 配列、
	- __(重みとサンプルの間における1__:1 の写像),
	あるいは時間的なデータの場合において、2次元配列の(samples, sequence_length) という形式で、
	すべてのサンプルの各時間において異なる重みを適用できる。
	この場合、compile() の中でsample_weight_mode="temporal" と確実に明記すべきです。

__返却値__

`History` オブジェクト。 `History.history` アトリビュートは
実行に成功したエポックにおける訓練の損失値とメトリックの値の記録と、(適用可能ならば)検証における損失値とメトリックの値も記録している。

----

### evaluate


```python
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
```


バッチごとにある入力データにおける損失を計算する。

__引数__

- __x__: 入力データ、Numpy 配列あるいはNumpy 配列のリスト
	(モデルに複数の入力がある場合).
- __y__: ラベル、 Numpy 配列の形式。
- __batch_size__: 整数で、指定したサンプル数で勾配の更新を行う。
- __verbose__: 進行状況メッセージ出力モードで、 0 または 1。
- __sample_weight__: サンプルの重み、 Numpy 配列の形式。

__返却値__

スカラーで、テストデータの損失(モデルのメトリックを設定していない場合)
あるいはスカラーのリスト(モデルが他のメトリックを計算している場合)。
アトリビュート`model.metrics_names` により、スカラーの出力でラベルを表示する。

----

### predict


```python
predict(self, x, batch_size=32, verbose=0)
```

入力サンプルに対する予測値の出力を生成し、サンプルを加工してバッチ処理する。

__引数__

- __x__: 入力データで、Numpy 配列の形式。
- __batch_size__: 整数。
- __verbose__: 進行状況メッセージ出力モード、 0 または 1。

__返却値__

予測値を格納した Numpy 配列 

----

### predict_classes


```python
predict_classes(self, x, batch_size=32, verbose=1)
```


バッチごとに入力サンプルに対するクラスの予測を生成する。

__引数__

- __x__: 入力データ、 Numpy 配列または Numpy 配列のリスト(モデルに複数の入力がある場合)。
- __batch_size__: 整数。
- __verbose__: 進行状況メッセージ出力モード、 0 または 1。

__返却値__

属するクラスの予測を格納したnumpy 配列。

----

### predict_proba


```python
predict_proba(self, x, batch_size=32, verbose=1)
```


バッチごとに入力サンプルに対する各々のクラスに所属する確率の予測値を生成する。

__引数__

- __x__: 入力データ、Numpy 配列または Numpy 配列のリスト(モデルに複数の入力がある場合)。
- __batch_size__: 整数。
- __verbose__: 進行状況メッセージ出力モード、 0 または 1。

__返却値__

確率の予測値を格納した Numpy 配列。

----

### train_on_batch


```python
train_on_batch(self, x, y, class_weight=None, sample_weight=None)
```

サンプルのある1つのバッチにおける単一の勾配更新

__引数__

- __x__: 入力データ、Numpy 配列または Numpy 配列のリスト(モデルに複数の入力がある場合)。
- __y__: ラベル、 Numpy 配列の形式。
- __class_weight__: dictionary 型で、重みの値でクラスを分ける、
	損失関数をスケーリングするために(トレーニングの間だけ)使う。
- __sample_weight__: サンプルの重み、Numpy 配列の形式。

__返却値__

スカラーでトレーニングの損失値(モデルにメトリックが設定されていない場合)
あるいはスカラーのリスト(モデルが他のメトリックを計算している場合)。
アトリビュート`model.metrics_names` により、スカラーの出力でラベルを表示する。

----

### test_on_batch


```python
test_on_batch(self, x, y, sample_weight=None)
```

サンプルの単一バッチにおけるモデルの評価

__引数__

- __x__: 入力データ、Numpy 配列または Numpy 配列のリスト(モデルに複数の入力がある場合)。
- __y__: ラベル、 Numpy 配列の形式。
- __sample_weight__: サンプルの重み、Numpy 配列の形式。

__返却値__

スカラーで、テストの損失値(モデルにメトリックが設定されていない場合)
あるいはスカラーのリスト(モデルが他のメトリックを計算している場合)。
アトリビュート`model.metrics_names` により、スカラーの出力でラベルを表示する。

----

### predict_on_batch


```python
predict_on_batch(self, x)
```

サンプルの単一のバッチに対する予測値を返却する

----

### fit_generator


```python
fit_generator(self, generator, samples_per_epoch, nb_epoch, verbose=1, callbacks=[], validation_data=None, nb_val_samples=None, class_weight=None, max_q_size=10)
```

Python のジェネレータにより、バッチごとに生成されるデータにおけるモデルに適合させる。
ジェネレータは効率化のために、モデルを平行に実行する。
たとえば、これのおかげでCPU 上で画像におけるリアルタイムなデータの増加と並行してGPU でモデルをトレーニングできる。

__引数__

- __generator__: ジェネレータ。
	ジェネレータの出力は以下のいずれかでなければならない、
	- タプル  (inputs, targets)
	- タプル (inputs, targets, sample_weights)。
	どの配列も同数のサンプルを含まなければならない。ジェネレータは漠然とそのデータを繰り返すことを期待している。
	`samples_per_epoch`数のサンプルが、そのモデルによって確認されたときにエポックが終了する。
- __samples_per_epoch__: 整数で、 次のエポックに進む前に加工されるサンプル数
- __nb_epoch__: 整数で、 データにおけるイテレーションの総数。
- __verbose__: 進行状況メッセージ出力モードで、 0、 1、 あるいは 2。
- __callbacks__: callbacks のリストで、トレーニングの際に呼び出される。
- __validation_data__: 以下のいずれかである。
	- 検証データに対するジェネレータ
	- タプル (inputs, targets)
	- タプル (inputs, targets, sample_weights)。
- __nb_val_samples__: `validation_data` がジェネレータである場合だけ関係がある。
	各エポックの終わりに検証用ジェネレータから使用するサンプルの数
- __class_weight__: dictionary 型で、重みの値でクラスを分ける、
	損失関数をスケーリングするために(トレーニングの間だけ)使う。
- __max_q_size__: ジェネレータのキューの最大サイズ

__Returns__

`History` オブジェクト。

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
		samples_per_epoch=10000, nb_epoch=10)
```

----

### evaluate_generator


```python
evaluate_generator(self, generator, val_samples, max_q_size=10)
```

ジェネレータのデータにおけるモデルを評価する。ジェネレータは`test_on_batch`で受け取ったデータと同じ種類のデータを返却するべきである。

- __Arguments__:
- __generator__:
	ジェネレータが生成するタプルで (inputs, targets)
	あるいは (inputs, targets, sample_weights)
- __val_samples__:
	返却する前に`generator`　により生成されるサンプルの総数
- __max_q_size__: ジェネレータのキューの最大サイズ
