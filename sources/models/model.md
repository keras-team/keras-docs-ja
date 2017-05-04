# ModelクラスAPI

functional APIでは，テンソルの入出力が与えられると，`Model`を以下のようにインスタンス化できます．

```python
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(input=a, output=b)
```

このモデルは，`a`を入力として`b`を計算する際に必要となるあらゆる層を含むことになります．

また，マルチ入力またはマルチ出力のモデルの場合は，リストを使うこともできます．

```python
model = Model(input=[a1, a2], output=[b1, b3, b3])
```

`Model`の詳しい解説は，[Keras functional API](/getting-started/functional-api-guide)をご覧下さい．

## モデルの便利な属性

- `model.layers` はモデルのグラフで構成される層を平坦化したリストです．
- `model.inputs` はテンソル入力のリストです．
- `model.outputs` はテンソル出力のリストです．



## メソッド

### compile


```python
compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None)
```


学習のためのモデルを設定します．

__引数__

- __optimizer__: 文字列(optimizer名)またはoptimizerのオブジェクト．詳細は[optimizers](/optimizers)を参照．
- __loss__: 文字列(目的関数の名前)または目的関数．詳細は[losses](/losses)を参照．モデルが複数の出力を持つ場合は，オブジェクトの辞書かリストを渡すことで，各出力に異なる損失を用いることができます．
- __metrics__: 学習時とテスト時にモデルにより評価されるメトリクスのリスト．一般的には`metrics=['accuracy']`を使うことになります．マルチ出力モデルの各出力のための各メトリクスを指定するために，`metrics={'output_a': 'accuracy'}`のような辞書を渡すこともできます．
- __loss_weights__: 異なるモデルの出力における損失寄与度に重み付けをするためのスカラー係数（Python の float値）を表すオプションのリスト，または辞書．
- __sample_weight_mode__: タイムステップ毎にサンプルを重み付け（2次元の重み）する場合は，この値を`temporal`に設定してください．`None`はデフォルト値で，サンプル毎の重み（1次元の重み）です．モデルが複数の出力をする時，modeの辞書かリストを渡すことで，各出力に異なる`sample_weight_mode`を使うことができます．
- __kwargs__: バックエンドにTheanoを用いる時は，これら引数はK.functionに渡されます．Tensorflowバックエンドの場合は無視されます．

__Raises__

- __ValueError__: `optimizer`，`loss`，`metrics`，または`sample_weight_mode`に対して無効な引数が与えられた場合．

----

### fit


```python
fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
```


固定回数（データセットの反復）の試行でモデルを学習させます．

__引数__

- __x__: 学習データのNumpy配列，もしくはモデルが複数の入力を持つ場合はNumpy配列のリスト．モデル内のあらゆる入力が名前を当てられている場合，入力の名前とNumpy配列をマップした辞書を渡すことも可能です．
- __y__: 教師（targets）データのNumpy配列，もしくはモデルが複数の出力を持つ場合はNumpy配列のリスト．モデル内のあらゆる出力が名前を当てられている場合，出力の名前とNumpy配列をマップした辞書を渡すことも可能です．
- __batch_size__: 勾配更新毎のサンプル数を示す整数．
- __epochs__: 学習データ配列の反復回数を示す整数．
	- __verbose__: 0，1，2のいずれかを指定．冗長モード．0 = 表示なし，1 = 冗長，2 = 各試行毎に一行の出力．
- __callbacks__: 学習時に呼ばれるコールバックのリスト．詳細は[callbacks](/callbacks)を参照．
- __validation_split__: 0から1の間のfloat値: バリデーションデータとして使われる学習データの割合．モデルはこの割合の学習データを区別し，それらでは学習を行わず，各試行の終わりにこのデータにおける損失とモデルメトリクスを評価します．
- __validation_data__: 各試行の最後に損失とモデルメトリクスを評価するためのデータ．モデルはこのデータで学習を行いません．この値は(x_val, y_val)のタプル，または(val_x, val_y, val_sample_weights)のタプルとなり得ます．
- __shuffle__: 学習データを各試行の前にシャッフルするかどうかを示すboolean．
- __class_weight__: クラスのインデックスと重み（float）をマップするオプションの辞書で，学習時に各クラスのサンプルに関するモデルの損失に適用します．これは過小評価されたクラスのサンプルに「より注意を向ける」ようモデルに指示するために有用です．
- __sample_weight__: xと同じサイズのオプションの配列で，各サンプルに関してモデルの損失に割り当てる重みを含みます．時間データの場合に，(samples, sequence_length)の形式の2次元配列を渡すことができ，各サンプルの各タイムステップに異なる重みを割り当てられます．この場合，compile()内で，sample_weight_mode="temporal"と指定するようにします．
- __initial_epoch__: 学習を開始するエポック（前回の学習を再開するのに便利です）．


__戻り値__

`History`インスタンス．本インスタンスの`history`属性は学習時に得られた全ての情報を含みます．

__Raises__

- __ValueError__: 与えられた入力データとモデルが期待するものとが異なる場合．

----

### evaluate


```python
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
```


テストモードにおいて，モデルの損失値と評価値を返します．

その計算はバッチ処理で行われます．

__引数__

- __x__: テストデータのNumpy配列，もしくはモデルが複数の入力を持つ場合はNumpy配列のリスト．モデル内のあらゆる入力が名前を当てられている場合，入力の名前とNumpy配列をマップした辞書を渡すことも可能です．
- __y__: 教師データのNumpy配列，もしくはモデルが複数の出力を持つ場合はNumpy配列のリスト．モデル内のあらゆる出力が名前を当てられている場合，出力の名前とNumpy配列をマップした辞書を渡すことも可能です．
- __batch_size__: 勾配更新毎のサンプル数を示す整数．
- __verbose__: 冗長モードで，0または1．
- __sample_weight__: 異なるサンプルが損失，または評価に寄与度を重み付けするための値の配列．

__戻り値__

テストの損失を表すスカラー値（モデルが単一の出力を持ち，かつメトリクスがない場合），またはスカラー値のリスト（モデルが複数の出力やメトリクスを持つ場合）．`model.metrics_names`属性はスカラー出力の表示ラベルを提示します．

----

### predict


```python
predict(self, x, batch_size=32, verbose=0)
```

入力サンプルに対する予測の出力を生成します．

その計算はバッチ処理で行われます．

__引数__

- __x__: Numpy配列の入力データ（もしくはモデルが複数の出力を持つ場合はNumpy配列のリスト）．
- __batch_size__: 整数値．
- __verbose__: 冗長モードで，0または1．

__戻り値__

予測結果のNumpy配列．

__Raises__

- __ValueError__: 与えられた入力データとモデルが期待するものが異なる場合，またはステートフルなモデルがバッチサイズの倍数でないサンプル数を受け取った場合．

----

### train_on_batch


```python
train_on_batch(self, x, y, sample_weight=None, class_weight=None)
```


単一バッチデータにつき一度の勾配更新を行います．

__引数__

- __x__: 学習データのNumpy配列，もしくはモデルが複数の入力を持つ場合はNumpy配列のリスト．
  モデル内のあらゆる入力が名前を当てられている場合，入力の名前とNumpy配列をマップした辞書を渡すことも可能です．
- __y__: 教師データのNumpy配列，もしくはモデルが複数の出力を持つ場合はNumpy配列のリスト．
	モデル内のあらゆる出力が名前を当てられている場合，出力の名前とNumpy配列をマップした辞書を渡すことも可能です．
- __sample_weight__: xと同じサイズのオプションの配列で，各サンプルに関してモデルの損失に割り当てる重みを含みます．
	時間データの場合に，(samples, sequence_length)の形式の2次元配列を渡すことができ，各サンプルの各タイムステップに異なる重みを割り当てられます．
	この場合，compile()内で，sample_weight_mode="temporal"と指定するようにします．
- __class_weight__: クラスのインデックスと重み（float）をマップするオプションの辞書で，学習時に各クラスのサンプルに関するモデルの損失に適用します．
	これは過小評価されたクラスのサンプルに「より注意を向ける」ようモデルに指示するために有用です．

__戻り値__

学習の損失を表すスカラー値（モデルが単一の出力を持ち，かつメトリクスがない場合），またはスカラー値のリスト（モデルが複数の出力やメトリクスを持つ場合）．`model.metrics_names`属性はスカラー出力の表示ラベルを提示します．

----

### test_on_batch


```python
test_on_batch(self, x, y, sample_weight=None)
```


サンプルの単一バッチでモデルをテストします．

__引数__

- __x__: テストデータのNumpy配列，もしくはモデルが複数の入力を持つ場合はNumpy配列のリスト．
  モデル内のあらゆる入力が名前を当てられている場合，入力の名前とNumpy配列をマップした辞書を渡すことも可能です．
- __y__: 教師データのNumpy配列，もしくはモデルが複数の出力を持つ場合はNumpy配列のリスト．
	モデル内のあらゆる出力が名前を当てられている場合，出力の名前とNumpy配列をマップした辞書を渡すことも可能です．
- __sample_weight__: xと同じサイズのオプションの配列で，各サンプルに関してモデルの損失に割り当てる重みを含みます．
	時間データの場合に，(samples, sequence_length)の形式の2次元配列を渡すことができ，各サンプルの各タイムステップに異なる重みを割り当てられます．
	この場合，compile()内で，sample_weight_mode="temporal"と指定するようにします．

__戻り値__

テストの損失を表すスカラー値（モデルが単一の出力を持ち，かつメトリクスがない場合），またはスカラー値のリスト（モデルが複数の出力やメトリクスを持つ場合）．`model.metrics_names`属性はスカラー出力の表示ラベルを提示します．

----

### predict_on_batch


```python
predict_on_batch(self, x)
```


サンプルの単一バッチに関する予測を返します．

__引数__

- __x__: 入力データ，Numpy配列．

__返り値__

予測値を格納したNumpy配列．

----

### fit_generator


```python
fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)
```


Pythonジェネレータによりバッチ毎に生成されたデータでモデルを学習します．

本ジェネレータは効率性のためモデルに並列して実行されます．例えば，モデルをGPUで学習させながらCPU上で画像のリアルタイムデータ拡張を行うことができるようになります．

__引数__

- __generator__: ジェネレータ．本ジェネレータの出力は，以下のいずれかです．
	- (inputs, targets)のタプル．
	- (inputs, targets, sample_weights)のタプル．すべての配列は同じ数のサンプルを含む必要があります．本ジェネレータは無期限にそのデータをループさせるようになっています．`steps_per_epoch`数のサンプルがモデルに与えられると1度の試行が終了します．
- __steps_per_epoch__: ある一つのエポックが終了し，次のエポックが始まる前に`generator`から使用する総ステップ数（サンプルのバッチ数）．もし，データサイズをバッチサイズで割った時，通常ユニークなサンプル数に等しくなります．
- __epochs__: データの反復回数の合計を示す整数．
- __verbose__: 冗長モードで，0，1，または2．
- __callbacks__: 学習時に呼ばれるコールバックのリスト．
- __validation_data__: これは以下のいずれかです．
	- バリデーションデータ用のジェネレータ．
	- (inputs, targets)のタプル．
	- (inputs, targets, sample_weights)のタプル．
- __validation_steps__: `validation_data`がジェネレータの場合にのみ関係します．終了する前に`generator`から使用する総ステップ数（サンプルのバッチ数）．
- __class_weight__: クラスインデックスと各クラスの重みをマップする辞書です．
- __max_q_size__: ジェネレータのキューの最大サイズです．
- __workers__: スレッドベースのプロセス使用時の最大プロセス数．
- __pickle_safe__: Trueならスレッドベースのプロセスを使います．実装がmultiprocessingに依存しているため，子プロセスに簡単に渡すことができないものとしてPickableでない引数をgeneratorに渡すべきではないことに注意してください．
- __initial_epoch__: 学習を開始するエポック（前回の学習を再開するのに便利です）．

__戻り値__

`History`オブジェクト

__例__


```python
def generate_arrays_from_file(path):
    while 1:
    f = open(path)
    for line in f:
        # create numpy arrays of input data
        # and labels, from each line in the file
        x1, x2, y = process_line(line)
        yield ({'input_1': x1, 'input_2': x2}, {'output': y})
    f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
        steps_per_epoch=10000, epochs=10)
```

__Raises__

- __ValueError__: ジェネレータが無効なフォーマットのデータを使用した場合．

----

### evaluate_generator


```python
evaluate_generator(self, generator, steps, max_q_size=10, workers=1, pickle_safe=False)
```


データジェネレータでモデルを評価します．

ジェネレータは`test_on_batch`で受け取られたのと同じ種類のデータを返します．

__引数__:
- __generator__: ジェネレータは(inputs, targets)もしくは(inputs, targets, sample_weights)のタプルを使用します．
- __steps__: 終了する前に`generator`から使用する総ステップ数（サンプルのバッチ数）．
- __max_q_size__: ジェネレータのキューのための最大サイズ．
- __workers__: スレッドベースのプロセス使用時の最大プロセス数．
- __pickle_safe__: Trueならスレッドベースのプロセスを使います．実装がmultiprocessingに依存しているため，子プロセスに簡単に渡すことができないものとしてPickableでない引数をgeneratorに渡すべきではないことに注意してください．

__戻り値__

テストの損失を表すスカラー値（モデルが単一の出力を持ち，かつメトリクスがない場合），またはスカラー値のリスト（モデルが複数の出力やメトリクスを持つ場合）．`model.metrics_names`属性はスカラー出力の表示ラベルを提示します．

__Raises__

- __ValueError__: ジェネレータが無効なフォーマットのデータを使用した場合．

----

### predict_generator

ジェネレータのデータに対して予測します．

ジェネレータは `predict_on_batch` が受け取るデータと同じ種類のデータを返します．

```python
predict_generator(self, generator, steps, max_q_size=10, workers=1, pickle_safe=False, verbose=0)
```

__引数__

- __generator__: ジェネレータは入力サンプルのバッチ数を使用します．
- __steps__: 終了する前に`generator`から使用する総ステップ数（サンプルのバッチ数）．
- __max_q_size__: ジェネレータのキューの最大サイズ．
- __workers__: スレッドベースのプロセス使用時の最大プロセス数．
- __pickle_safe__: Trueならスレッドベースのプロセスを使います．実装がmultiprocessingに依存しているため，子プロセスに簡単に渡すことができないものとしてPickableでない引数をgeneratorに渡すべきではないことに注意してください．
- __verbose__: 冗長モードで，0または1．

__返り値__

予測値のNumpy配列．

__Raises__

- __ValueError__: ジェネレータが無効なフォーマットのデータを使用した場合．

----

### get_layer


```python
get_layer(self, name=None, index=None)
```


層の（ユニークな）名前，またはインデックスに基づき層を返します．

インデックスはボトムアップの幅優先探索の順番に基づきます．

__引数__

- __name__: 層の名前を表す文字列．
- __index__: 層のインデックスを表す整数．

__戻り値__

層のインスタンス．

__Raises__

- __ValueError__: 無効なレイヤーの名前，またはインデックスの場合．
