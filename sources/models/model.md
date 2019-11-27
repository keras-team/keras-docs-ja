# ModelクラスAPI

functional APIでは，テンソルの入出力が与えられると，`Model`を以下のようにインスタンス化できます．

```python
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)
```

このモデルは，`a`を入力として`b`を計算する際に必要となるあらゆる層を含むことになります．

また，マルチ入力またはマルチ出力のモデルの場合は，リストを使うこともできます．

```python
model = Model(inputs=[a1, a2], outputs=[b1, b2, b3])
```

`Model`の詳しい解説は，[Keras functional API](/getting-started/functional-api-guide)をご覧ください．

## メソッド

### compile

```python
compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
```

学習のためのモデルを設定します．

__引数__

- __optimizer__: 文字列（optimizer名）またはoptimizerのオブジェクト．詳細は[optimizers](/optimizers)を参照してください．
- __loss__: 文字列（目的関数名）または目的関数．詳細は[losses](/losses)を参照してください．モデルが複数の出力を持つ場合は，オブジェクトの辞書かリストを渡すことで，各出力に異なる損失を用いることができます．モデルによって最小化される損失値は全ての個々の損失の合計になります．
- __metrics__: 訓練時とテスト時にモデルにより評価される評価関数のリスト．一般的には`metrics=['accuracy']`を使うことになります．マルチ出力モデルの各出力のための各評価関数を指定するために，`metrics={'output_a': 'accuracy'}`のような辞書を渡すこともできます．
- __loss_weights__: 異なるモデルの出力における損失寄与度に重み付けをするためのスカラ係数（Pythonの浮動小数点数）を表すオプションのリスト，または辞書．モデルによって最小化される損失値は，`loss_weights`係数で重み付けされた個々の損失の*加重合計*です．リストの場合，モデルの出力と1:1対応している必要があります．テンソルの場合，出力の名前（文字列）がスカラー係数に対応している必要があります．
- __sample_weight_mode__: タイムステップ毎にサンプルを重み付け（2次元の重み）する場合は，この値を`"temporal"`に設定してください．`None`はデフォルト値で，サンプル毎の重み（1次元の重み）です．モデルに複数の出力がある場合，モードとして辞書かリストを渡すことで，各出力に異なる`sample_weight_mode`を使うことができます．
- __weighted_metrics__: 訓練やテストの際にsample_weightまたはclass_weightにより評価と重み付けされるメトリクスのリスト．
- __target_tensors__: Kerasはデフォルトでモデルのターゲットためのプレースホルダを作成します．これは訓練中にターゲットデータが入力されるものです．代わりの自分のターゲットテンソルを利用したい場合（訓練時にKerasはこれらのターゲットに対して外部のNumpyデータを必要としません）は，それらを`target_tensors`引数で指定することができます．これは単一のテンソル（単一出力モデルの場合），テンソルのリスト，または出力名をターゲットのテンソルにマッピングした辞書になります．
- __**kwargs__: バックエンドにTheano/CNTKを用いる時は，これら引数は`K.function`に渡されます．Tensorflowバックエンドの場合は`tf.Session.run`に渡されます．

__Raises__

- __ValueError__: `optimizer`，`loss`，`metrics`，または`sample_weight_mode`に対して無効な引数が与えられた場合．

----

### fit

```python
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
```

固定回数（データセットの反復）の試行でモデルを学習させます．

__引数__

- __x__: モデルが単一の入力を持つ場合は訓練データのNumpy配列，もしくはモデルが複数の入力を持つ場合はNumpy配列のリスト．モデル内のあらゆる入力に名前を当てられている場合，入力の名前とNumpy配列をマップした辞書を渡すことも可能です．フレームワーク固有のテンソル（例えばTensorFlowデータテンソル）からフィードする場合は`x`を`None`にすることもできます．
- __y__: モデルが単一の入力を持つ場合は教師（targets）データのNumpy配列，もしくはモデルが複数の出力を持つ場合はNumpy配列のリスト．モデル内のあらゆる出力が名前を当てられている場合，出力の名前とNumpy配列をマップした辞書を渡すことも可能です．フレームワーク固有のテンソル（例えばTensorFlowデータテンソル）からフィードする場合は`y`を`None`にすることもできます．
- __batch_size__: 整数または`None`．勾配更新毎のサンプル数を示す整数．指定しなければ`batch_size`はデフォルトで32になります．
- __epochs__: 整数．訓練データ配列の反復回数を示す整数．エポックは，提供される`x`および`y`データ全体の反復です． `initial_epoch`と組み合わせると，`epochs`は"最終エポック"として理解されることに注意してください．このモデルは`epochs`で与えられた反復回数だの訓練をするわけではなく，単に`epochs`という指標に試行が達するまで訓練します．
- __verbose__: 整数．0，1，2のいずれか．進行状況の表示モード．0 = 表示なし，1 = プログレスバー，2 = 各試行毎に一行の出力．
- __callbacks__: `keras.callbacks.Callback`インスタンスのリスト．訓練時に呼ばれるコールバックのリスト．詳細は[callbacks](/callbacks)を参照．
- __validation_split__: 0から1の間の浮動小数点数．バリデーションデータとして使われる訓練データの割合．モデルはこの割合の訓練データを区別し，それらでは学習を行わず，各試行の終わりにこのデータにおける損失とモデル評価関数を評価します．このバリデーションデータは，シャッフルを行う前に，与えられた`x`と`y`のデータの後ろからサンプリングされます．
- __validation_data__: 各試行の最後に損失とモデル評価関数を評価するために用いられる`(x_val, y_val)`のタプル，または`(val_x, val_y, val_sample_weights)`のタプル．モデルはこのデータで学習を行いません．`validation_data`は`validation_split`を上書きします．
- __shuffle__: 真理値（訓練データを各試行の前にシャッフルするかどうか）または文字列（'batch'の場合）．'batch'はHDF5データの限界を扱うための特別なオプションです．バッチサイズのチャンクでシャッフルします．`steps_per_epoch`が`None`でない場合には効果がありません．
- __class_weight__: クラスのインデックスと重み（浮動小数点数）をマップするオプションの辞書で，訓練時に各クラスのサンプルに関するモデルの損失に適用します．これは過小評価されたクラスのサンプルに「より注意を向ける」ようモデルに指示するために有用です．
- __sample_weight__: オプションのNumpy配列で訓練サンプルの重みです．（訓練時のみ）損失関数への重み付けに用いられます．（重みとサンプルが1:1対応するように）入力サンプルと同じ長さの1次元Numpy配列を渡すこともできますし，時系列データの場合には，`(samples, sequence_length)`の形式の2次元配列を渡すことができ，各サンプルの各タイムステップに異なる重みを割り当てられます．この場合，`compile()`内で，`sample_weight_mode="temporal"`と指定するようにします．
- __initial_epoch__: 整数．訓練を開始するエポック（前回の学習を再開するのに便利です）．
- __steps_per_epoch__: 整数または`None`．終了した1エポックを宣言して次のエポックを始めるまでのステップ数の合計（サンプルのバッチ）．TensorFlowのデータテンソルのような入力テンソルを使用して訓練する場合，デフォルトの`None`はデータセットのサンプル数をバッチサイズで割ったものに等しくなります．それが決定できない場合は1になります．
- __validation_steps__: `steps_per_epoch`を指定している場合のみ関係します．停止する前にバリデーションするステップの総数（サンプルのバッチ）．

__戻り値__

`History` オブジェクト．`History.history`属性は
実行に成功したエポックにおける訓練の損失値と評価関数値の記録と，（適用可能ならば）検証における損失値と評価関数値も記録しています．

__Raises__

- __RuntimeError__: モデルがコンパイルされていない場合．
- __ValueError__: 与えられた入力データとモデルが期待するものとが異なる場合．

----

### evaluate

```python
evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None)
```

テストモードにおいて，モデルの損失値と評価値を返します．

その計算はバッチ処理で行われます．

__引数__

- __x__: モデルが単一の入力を持つ場合は訓練データのNumpy配列，もしくはモデルが複数の入力を持つ場合はNumpy配列のリスト．モデル内のあらゆる入力に名前を当てられている場合，入力の名前とNumpy配列をマップした辞書を渡すことも可能です．フレームワーク固有のテンソル（例えばTensorFlowデータテンソル）からフィードする場合は`x`を`None`にすることもできます．
- __y__: モデルが単一の入力を持つ場合は教師（targets）データのNumpy配列，もしくはモデルが複数の出力を持つ場合はNumpy配列のリスト．モデル内のあらゆる出力が名前を当てられている場合，出力の名前とNumpy配列をマップした辞書を渡すことも可能です．フレームワーク固有のテンソル（例えばTensorFlowデータテンソル）からフィードする場合は`y`を`None`にすることもできます．
- __batch_size__: 整数または`None`．勾配更新毎のサンプル数を示す整数．指定しなければ`batch_size`はデフォルトで32になります．
- __verbose__: 0または1．進行状況の表示モード．0 = 表示なし，1 = プログレスバー．
- __sample_weight__: オプションのNumpy配列で訓練サンプルの重みです．（訓練時のみ）損失関数への重み付けに用いられます．（重みとサンプルが1:1対応するように）入力サンプルと同じ長さの1次元Numpy配列を渡すこともできますし，時系列データの場合には，`(samples, sequence_length)`の形式の2次元配列を渡すことができ，各サンプルの各タイムステップに異なる重みを割り当てられます．この場合，`compile()`内で，`sample_weight_mode="temporal"`と指定するようにします．
- __steps__: 整数または`None`．評価ラウンド終了を宣言するまでの総ステップ数（サンプルのバッチ）．デフォルト値の`None`ならば無視されます．

__戻り値__

テストの損失を表すスカラ値（モデルが単一の出力を持ち，かつ評価関数がない場合），またはスカラ値のリスト（モデルが複数の出力や評価関数を持つ場合）．`model.metrics_names`属性はスカラ出力の表示ラベルを提示します．

----

### predict

```python
predict(x, batch_size=None, verbose=0, steps=None)
```

入力サンプルに対する予測の出力を生成します．

その計算はバッチ処理で行われます．

__引数__

- __x__: Numpy配列の入力データ（もしくはモデルが複数の出力を持つ場合はNumpy配列のリスト）．
- __batch_size__: 整数値．指定しなければデフォルトで32になります．
- __verbose__: 進行状況の表示モードで，0または1．
- __steps__: 予測ラウンド終了を宣言するまでの総ステップ数（サンプルのバッチ）．デフォルト値の`None`ならば無視されます．

__戻り値__

予測結果のNumpy配列．

__Raises__

- __ValueError__: 与えられた入力データとモデルが期待するものが異なる場合，またはステートフルなモデルがバッチサイズの倍数でないサンプル数を受け取った場合．

----

### train_on_batch

```python
train_on_batch(x, y, sample_weight=None, class_weight=None)
```

単一バッチデータにつき一度の勾配更新を行います．

__引数__

- __x__: モデルが単一の入力を持つ場合は訓練データのNumpy配列，もしくはモデルが複数の入力を持つ場合はNumpy配列のリスト．モデル内のあらゆる入力に名前を当てられている場合，入力の名前とNumpy配列をマップした辞書を渡すことも可能です．
- __y__: モデルが単一の入力を持つ場合は教師（targets）データのNumpy配列，もしくはモデルが複数の出力を持つ場合はNumpy配列のリスト．モデル内のあらゆる出力が名前を当てられている場合，出力の名前とNumpy配列をマップした辞書を渡すことも可能です．
- __sample_weight__: オプションのNumpy配列で訓練サンプルの重みです．（訓練時のみ）損失関数への重み付けに用いられます．（重みとサンプルが1:1対応するように）入力サンプルと同じ長さの1次元Numpy配列を渡すこともできますし，時系列データの場合には，`(samples, sequence_length)`の形式の2次元配列を渡すことができ，各サンプルの各タイムステップに異なる重みを割り当てられます．この場合，`compile()`内で，`sample_weight_mode="temporal"`と指定するようにします．
- __class_weight__: クラスのインデックスと重み（浮動小数点数）をマップするオプションの辞書で，訓練時に各クラスのサンプルに関するモデルの損失に適用します．これは過小評価されたクラスのサンプルに「より注意を向ける」ようモデルに指示するために有用です．

__戻り値__

学習の損失を表すスカラ値（モデルが単一の出力を持ち，かつ評価関数がない場合），またはスカラ値のリスト（モデルが複数の出力や評価関数を持つ場合）．`model.metrics_names`属性はスカラ出力の表示ラベルを提示します．

----

### test_on_batch

```python
test_on_batch(x, y, sample_weight=None)
```

サンプルの単一バッチでモデルをテストします．

__引数__

- __x__: テストデータのNumpy配列，もしくはモデルが複数の入力を持つ場合はNumpy配列のリスト．
  モデル内のあらゆる入力が名前を当てられている場合，入力の名前とNumpy配列をマップした辞書を渡すことも可能です．
- __y__: 教師データのNumpy配列，もしくはモデルが複数の出力を持つ場合はNumpy配列のリスト．
    モデル内のあらゆる出力が名前を当てられている場合，出力の名前とNumpy配列をマップした辞書を渡すことも可能です．
- __sample_weight__: オプションのNumpy配列で訓練サンプルの重みです．（訓練時のみ）損失関数への重み付けに用いられます．（重みとサンプルが1:1対応するように）入力サンプルと同じ長さの1次元Numpy配列を渡すこともできますし，時系列データの場合には，`(samples, sequence_length)`の形式の2次元配列を渡すことができ，各サンプルの各タイムステップに異なる重みを割り当てられます．この場合，`compile()`内で，`sample_weight_mode="temporal"`と指定するようにします．

__戻り値__

テストの損失を表すスカラ値（モデルが単一の出力を持ち，かつ評価関数がない場合），またはスカラ値のリスト（モデルが複数の出力や評価関数を持つ場合）．`model.metrics_names`属性はスカラ出力の表示ラベルを提示します．

----

### predict_on_batch

```python
predict_on_batch(x)
```

サンプルの単一バッチに関する予測を返します．

__引数__

- __x__: 入力データ，Numpy配列．

__戻り値__

予測値を格納したNumpy配列．

----

### fit_generator

```python
fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
```

Pythonジェネレータ（または`Sequence`のインスタンス）によりバッチ毎に生成されたデータでモデルを訓練します．

本ジェネレータは効率性のためモデルに並列して実行されます．例えば，モデルをGPUで学習させながらCPU上で画像のリアルタイムデータ拡張を行うことができるようになります．

`use_multiprocessing=True`のときに，`keras.utils.Sequence`を使うことで順序とエポックごとに全入力を1度だけ使用することを保証します．

__引数__

- __generator__: ジェネレータかマルチプロセッシング時にデータの重複を防ぐための`Sequence`（`keras.utils.Sequence`）オブジェクトのインスタンス．本ジェネレータの出力は，以下のいずれかです．
    - `(inputs, targets)`のタプル．
    - `(inputs, targets, sample_weights)`のタプル．このタプル（単一出力のジェネレータ）は単一のバッチを作ります．つまり，このタプルにある全ての配列は全て同じ長さ（バッチサイズと等しい）でなければなりません．バッチによってサイズが異なる場合もあります．例えば，データセットのサイズがバッチサイズで割り切れない場合，一般的にエポックの最後のバッチはそれ以外よりも小さくなります．このジェネレータはデータが無限にループすることを期待します．`steps_per_epoch`数のサンプルがモデルに与えられると1度の試行が終了します．
- __steps_per_epoch__: ある一つのエポックが終了し，次のエポックが始まる前に`generator`から使用する総ステップ数（サンプルのバッチ数）．もし，データサイズをバッチサイズで割った時，通常ユニークなサンプル数に等しくなります．`Sequence`のオプション：指定されていない場合は，`len(generator)`をステップ数として使用します．
- __epochs__: 整数．モデルを訓練させるエポック数．
    エポックは与えられたデータ全体の反復で，`steps_per_epoch`で定義されます．
    `initial_epoch`と組み合わせると，`epochs`は「最終エポック」として理解されることに注意してください．このモデルは`epochs`で与えられた反復回数の訓練をするわけではなく，単に`epochs`という指標に試行が達するまで訓練します．
- __verbose__: 整数．0，1，2のいずれか．進行状況の表示モード．0 = 表示なし，1 = プログレスバー，2 = 各試行毎に一行の出力．
- __callbacks__: `keras.callbacks.Callback`インスタンスのリスト．訓練時に呼ばれるコールバックのリスト．詳細は[callbacks](/callbacks)を参照．
- __validation_data__: これは以下のいずれかです．
    - バリデーションデータ用のジェネレータ．
    - (inputs, targets)のタプル．
    - (inputs, targets, sample_weights)のタプル．各エポックの最後に損失関数やモデルの評価関数の評価に用いられます．このデータは学習には使われません．
- __validation_steps__: `validation_data`がジェネレータの場合にのみ関係します．終了する前に`generator`から使用する総ステップ数（サンプルのバッチ数）．`Sequence`のオプション：指定されていない場合は，`len(validation_data)`をステップ数として使用します．
- __class_weight__: クラスインデックスと各クラスの重みをマップする辞書です．
    （訓練のときだけ）損失関数の重み付けに使われます．
    過小評価されたクラスのサンプルに「より注意を向ける」場合に有用です．
- __max_queue_size__: 整数．ジェネレータのキューのための最大サイズ．
    指定しなければ`max_queue_size`はデフォルトで10になります．
- __workers__: 整数．スレッドベースのプロセス使用時の最大プロセス数．指定しなければ`workers`はデフォルトで1になります．もし0ならジェネレータはメインスレッドで実行されます．
- __use_multiprocessing__: 真理値．`True`ならスレッドベースのプロセスを使います．指定しなければ`workers`はデフォルトでFalseになります．実装がmultiprocessingに依存しているため，子プロセスに簡単に渡すことができないものとしてPickableでない引数をジェネレータに渡すべきではないことに注意してください．
- __shuffle__: 真理値．各試行の初めにバッチの順番をシャッフルするかどうか．`Sequence`(`keras.utils.Sequence`)の時のみ使用されます．
- __initial_epoch__: 整数．学習を開始するエポック（前回の学習を再開するのに便利です）．

__戻り値__

`History`オブジェクト．`History.history` 属性は
実行に成功したエポックにおける訓練の損失値と評価関数値の記録と，（適用可能ならば）検証における損失値と評価関数値も記録しています．

__Raises__

- __ValueError__: ジェネレータが無効なフォーマットのデータを使用した場合．

__例__

```python
def generate_arrays_from_file(path):
    while True:
        with open(path) as f:
            for line in f:
                # create numpy arrays of input data
                # and labels, from each line in the file
                x1, x2, y = process_line(line)
                yield ({'input_1': x1, 'input_2': x2}, {'output': y})

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                    steps_per_epoch=10000, epochs=10)
```

----

### evaluate_generator

```python
evaluate_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
```

データジェネレータでモデルを評価します．

ジェネレータは`test_on_batch`で受け取られたのと同じ種類のデータを返します．

__引数__:

- __generator__: ジェネレータは(inputs, targets)タプルもしくは(inputs, targets, sample_weights)タプルかマルチプロセッシング時にデータの重複を防ぐためのSequence (keras.utils.Sequence) オブジェクトのインスタンスを使用します．
- __steps__: 終了する前に`generator`から使用する総ステップ数（サンプルのバッチ数）．`Sequence`のオプション：指定されていない場合は，`len(generator)`をステップ数として使用します．
- __max_queue_size__: ジェネレータのキューのための最大サイズ．
- __workers__: 整数．スレッドベースのプロセス使用時の最大プロセス数．指定しなければ`workers`はデフォルトで1になります．もし0ならジェネレータはメインスレッドで実行されます．
- __use_multiprocessing__: `True`ならスレッドベースのプロセスを使います．実装がmultiprocessingに依存しているため，子プロセスに簡単に渡すことができないものとしてPickableでない引数をジェネレータに渡すべきではないことに注意してください．
- __verbose__: 進行状況の表示モードで，0または1．

__戻り値__

テストの損失を表すスカラ値（モデルが単一の出力を持ち，かつ評価関数がない場合），またはスカラ値のリスト（モデルが複数の出力や評価関数を持つ場合）．`model.metrics_names`属性はスカラ出力の表示ラベルを提示します．

__Raises__

- __ValueError__: ジェネレータが無効なフォーマットのデータを使用した場合．

----

### predict_generator

```python
predict_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
```

データジェネレータから得た入力サンプルに対する予測を生成します．

ジェネレータは`predict_on_batch`が受け取るデータと同じ種類のデータを返します．

__引数__

- __generator__: 入力サンプルのバッチかマルチプロセッシング時にデータの重複を防ぐためのSequence (keras.utils.Sequence) オブジェクトのインスタンスを生成するジェネレータ．
- __steps__: 終了する前に`generator`から使用する総ステップ数（サンプルのバッチ数）．`Sequence`のオプション：指定されていない場合は，`len(generator)`をステップ数として使用します．
- __max_queue_size__: ジェネレータのキューの最大サイズ．
- __workers__: 整数．スレッドベースのプロセス使用時の最大プロセス数．指定しなければ`workers`はデフォルトで1になります．もし0ならジェネレータはメインスレッドで実行されます．
- __use_multiprocessing__: `True`ならスレッドベースのプロセスを使います．実装がmultiprocessingに依存しているため，子プロセスに簡単に渡すことができないものとしてPickableでない引数をジェネレータに渡すべきではないことに注意してください．
- __verbose__: 進行状況の表示モードで，0または1．

__戻り値__

予測値のNumpy配列．

__Raises__

- __ValueError__: ジェネレータが無効なフォーマットのデータを使用した場合．

----

### get_layer

```python
get_layer(name=None, index=None)
```

（ユニークな）名前，またはインデックスに基づきレイヤーを探します．

`name`と`index`の両方が与えられた場合，`index`が優先されます．

インデックスはボトムアップの幅優先探索の順番に基づきます．

__引数__

- __name__: レイヤーの名前を表す文字列．
- __index__: レイヤーのインデックスを表す整数．

__戻り値__

レイヤーのインスタンス．

__Raises__

- __ValueError__: 無効なレイヤーの名前，またはインデックスの場合．
