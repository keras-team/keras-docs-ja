# Keras backends

## "バックエンド"とは?

Kerasはモデルレベルのライブラリーで，深層学習モデルを開発するための高水準のモデル構築ブロックを与えます．
テンソル積，畳み込みなどのような低水準の操作を自身で扱うことはありません．
代わりに，Kerasの"バックエンドエンジン"としての役割を果たす，そのような操作を行うために特別に良く最適化されたテンソル操作ライブラリに依存します．
一つの単一のテンソルライブラリーを取り上げたり，そのライブラリに束縛されたKerasの実装を行うのではなく，
Kerasはモジュール方式でこの問題を扱い，いくつかの異なるバックエンドエンジンをKerasにシームレスに付加できます．

今回から，Kerasは二つのバックエンド，即ち**Theano**バックエンドと**TensorFlow**バックエンド，が利用可能になりました．

- [Theano](http://deeplearning.net/software/theano/) はモントリオール大学のLISA/MILA Labにより開発されたオープンソースのシンボリックなテンソル操作のフレームワークです．
- [TensorFlow](http://www.tensorflow.org/) はGoogle, Inc.により開発されたオープンソースのシンボリックなテンソル操作のフレームワークです．

----

## あるバックエンドから別のバックエンドへの移行

少なくとも一度Kerasを実行したら，Kerasの設定ファイルを

`~/.keras/keras.json`

で見つけるでしょう．

そこになければ，あなたはこの設定ファイルを作成することができます．

おそらくこのように見えるでしょう:

`{"epsilon": 1e-07, "floatx": "float32", "backend": "theano"}`

単にフィールド`backend`を`"theano"`もしくは`"tensorflow"`に変えると，次回あなたが任意のKerasコードを実行するときに新しい設定を利用します．

あなたは環境変数``KERAS_BACKEND``も定義することができ，これはあなたの設定ファイルで定義されているものを上書きします:

```bash
KERAS_BACKEND=tensorflow python -c "from keras import backend; print backend._BACKEND"
Using TensorFlow backend.
tensorflow
```

----

## 新しいコードを書くための抽象的なKerasバックエンドの利用

TheanoとTesorFlowの両方で互換性があるように書くKerasモジュールが欲しいときは，
抽象的なKerasバックエンドAPIを通じて書く必要があります．

あなたは以下を通じてバックエンドモジュールをインポートできます:
```python
from keras import backend as K
```

以下のコードは入力のプレースホルダーのインスタンスを作成します．
これは`tf.placeholder()`や`T.matrix()`, `T.tensor3()`, などと同じことです．

```python
input = K.placeholder(shape=(2, 4, 5))
# 以下も動作します:
input = K.placeholder(shape=(None, 4, 5))
# 以下も動作します:
input = K.placeholder(ndim=3)
```

以下のコードは共有変数のインスタンスを作成します．
これは`tf.variable()`や`theano.shared()`と同じことです．

```python
val = np.random.random((3, 4, 5))
var = K.variable(value=val)

# すべて0の変数:
var = K.zeros(shape=(3, 4, 5))
# すべて1の変数:
var = K.ones(shape=(3, 4, 5))
```

あなたが必要とする大抵のテンソル操作はTensorFlowやTheanoにおいて行うように実行できます:

```python
a = b + c * K.abs(d)
c = K.dot(a, K.transpose(b))
a = K.sum(b, axis=2)
a = K.softmax(b)
a = concatenate([b, c], axis=-1)
# etc...
```

----

## バックエンド関数


### learning_phase


```python
learning_phase()
```


学習フェーズのフラグを返します．

学習フェーズのフラグは
学習期間とテスト期間で異なる振る舞いをする任意のKeras関数への入力として渡される
整数のテンソル (0 = test, 1 = train) です．

----

### floatx


```python
floatx()
```


デフォルトのfloat型を文字列で返します
(e.g. 'float16', 'float32', 'float64')．

----

### cast_to_floatx


```python
cast_to_floatx(x)
```


Numpy配列をfloatxにキャストします．

----

### shape


```python
shape(x)
```


テンソルのシンボリックなshapeを返します．

----

### variable


```python
variable(value, dtype='float32', name=None)
```


テンソルのインスタンスを作成します．

__引数__

- __value__: Numpy配列, テンソルの初期値．
- __dtype__: テンソルの型．
- __name__: このテンソルに対する任意の名前を表す文字列．

__返り値__

テンソル変数のインスタンス．

----

### placeholder


```python
placeholder(shape=None, ndim=None, dtype='float32', name=None)
```


プレースホルダーのインスタンスを作成します．

__引数__

- __shape__: プレースホルダーのshape
（整数のタプル，Noneのエントリーを含んでも構いません）．
- __ndim__: テンソルの軸の数．
少なくとも{`shape`, `ndim`}から一つ指定する必要があります．
両方が指定されると，`shape`が利用されます．
- __dtype__: プレースホルダー型．
- __name__: このプレースホルダーに対する任意の名前を表す文字列．

__返り値__

プレースホルダーのテンソルインスタンス．

----

### int_shape


```python
int_shape(x)
```


整数もしくはNoneのエントリーからなるタプルとしてテンソルのshapeを返します．

----

### ndim


```python
ndim(x)
```


テンソルの軸の数を整数として返します．

----

### dtype


```python
dtype(x)
```


テンソルのdtypeを文字列として返します．

----

### eval


```python
eval(x)
```


テンソルの値を評価します．
Numpy配列を返します．

----

### zeros


```python
zeros(shape, dtype='float32', name=None)
```


すべて0のテンソル変数のインスタンスを作成します．

----

### ones


```python
ones(shape, dtype='float32', name=None)
```


すべて1のテンソル変数のインスタンスを作成します．

----

### eye


```python
eye(size, dtype='float32', name=None)
```


単位行列のインスタンスを作成します．

----

### zeros_like


```python
zeros_like(x, name=None)
```


別のテンソルと同じshapeを持つすべて0のテンソルのインスタンスを作成します．

----

### ones_like


```python
ones_like(x, name=None)
```


別のテンソルと同じshapeを持つすべて1のテンソルのインスタンスを作成します．

----

### count_params


```python
count_params(x)
```


テンソルにおけるスカラーの数を返します．

----

### cast


```python
cast(x, dtype)
```


テンソルを異なる型にキャストします．

----

### dot


```python
dot(x, y)
```


二つのテンソルを掛け合わせます．
NDテンソルにNDテンソルを掛けようとすると，Theanoの振る舞いを再現します．
(e.g. (2, 3).(4, 3, 5) = (2, 4, 5))

----

### batch_dot


```python
batch_dot(x, y, axes=None)
```


バッチごとのdot積．

batch_dotの結果は入力より小さい次元を持つテンソルになります．
次元数が1になれば，ndimが少なくとも2であることを確認するために`expand_dims`を利用します．

__例__

x = [[1, 2], [3,4]], y = [[5, 6], [7, 8]]
と仮定すると，
非対角成分を計算しなくても，x.dot(y.T)の主対角成分である
batch_dot(x, y, axes=1) = [[17, 53]]が得られます．

__引数__

x, y: ndim >= 2であるようなテンソル
- __axes__: 目標となる次元を持つ整数のリスト（もしくは整数単体）

__返り値__

ndim >= 2であるようなテンソル

----

### transpose


```python
transpose(x)
```


行列を転置します．

----

### gather


```python
gather(reference, indices)
```


2Dテンソル`reference`の中で添字ベクトル`indices`を探索します．

__Arguments__

- __reference__: 2Dテンソル．
- __indices__: 添字の整数テンソル．

__返り値__

`reference`と同じ型を持つ3Dテンソル．

----

### max


```python
max(x, axis=None, keepdims=False)
```


テンソル内の最大値．

----

### min


```python
min(x, axis=None, keepdims=False)
```


テンソル内の最小値．

----

### sum


```python
sum(x, axis=None, keepdims=False)
```


テンソル内の値の，指定した軸に沿った和．

----

### prod


```python
prod(x, axis=None, keepdims=False)
```


テンソル内の値の，指定した軸に沿った積．

----

### std


```python
std(x, axis=None, keepdims=False)
```


テンソル内の値の，指定した軸に沿った標準偏差．

----

### mean


```python
mean(x, axis=None, keepdims=False)
```


テンソル内の値の，指定した軸に沿った平均．

----

### any


```python
any(x, axis=None, keepdims=False)
```


ビットごとの縮約（論理OR）．

（0と1からなる）unit8テンソルを返します．

----

### argmax


```python
argmax(x, axis=-1)
```


テンソルの軸に沿った最大値の添字を返します．

----

### argmin


```python
argmin(x, axis=-1)
```


テンソルの軸に沿った最小値の添字を返します．

----

### square


```python
square(x)
```


成分ごとの二乗．

----

### abs


```python
abs(x)
```


成分ごとの絶対値．

----

### sqrt


```python
sqrt(x)
```


成分ごとの平方根．

----

### exp


```python
exp(x)
```


指数関数における成分ごとの値．

----

### log


```python
log(x)
```


成分ごとの対数．

----

### round


```python
round(x)
```


成分ごとの最も近い整数への丸め．

----

### sign


```python
sign(x)
```


成分ごとの符号．

----

### pow


```python
pow(x, a)
```


成分ごとの指数乗．

----

### clip


```python
clip(x, min_value, max_value)
```


成分ごとの値のクリッピング．

----

### equal


```python
equal(x, y)
```


二つのテンソル間の成分ごとの等値性．
ブール値からなるテンソルを返します．

----

### not_equal


```python
not_equal(x, y)
```


二つのテンソル間の成分ごとの不等性．
ブール値からなるテンソルを返します．

----

### maximum


```python
maximum(x, y)
```


二つのテンソルの成分ごとの最大値．

----

### minimum


```python
minimum(x, y)
```


二つのテンソルの成分ごとの最小値．

----

### sin


```python
sin(x)
```


成分ごとにxのsinを計算します．

----

### cos


```python
cos(x)
```


成分ごとにxのcosを計算します．

----

### concatenate


```python
concatenate(tensors, axis=-1)
```


指定した軸に沿ってテンソルのリストを連結します．

----

### reshape


```python
reshape(x, shape)
```


指定したshapeにテンソルを整形します．

----

### permute_dimensions


```python
permute_dimensions(x, pattern)
```


テンソルにおける軸を置換します．

__引数__

- __pattern__: 次元の添字かなるタプルであるべきです，e.g. (0, 2, 1)．

----

### resize_images


```python
resize_images(X, height_factor, width_factor, dim_ordering)
```


次のshapeを持つ4Dテンソルに含まれるように(height_factor, width_factor)の因子についてイメージのサイズを変更します
- [batch, channels, height, width] ('th' dim_orderingに対して)
- [batch, height, width, channels] ('tf' dim_orderingに対して)
両方の因子は正の整数であるべきです．

----

### repeat_elements


```python
repeat_elements(x, rep, axis)
```


np.repeatのように，軸に沿ってテンソルの要素を繰り返します．

xがshape (s1, s2, s3)を持ち，axis=1であれば，この出力はshape (s1, s2 * rep, s3)を持ちます．

----

### repeat


```python
repeat(x, n)
```


2Dテンソルを繰り返します:

xがshape (samples, dim)を持ちn=2であれば，
この出力はshape (samples, 2, dim)を持ちます．

----

### batch_flatten


```python
batch_flatten(x)
```


n-Dテンソルを最初の次元が保たれるように2Dテンソルに変換します．

----

### expand_dims


```python
expand_dims(x, dim=-1)
```


添字"dim"でのサイズ1の次元を加えます．

----

### squeeze


```python
squeeze(x, axis)
```


テンソルから添字"axis"での1次元を除きます．

----

### temporal_padding


```python
temporal_padding(x, padding=1)
```


左と右に"padding"個の0を増やすことで，3Dテンソルの真ん中の次元を増やします．

----

### spatial_2d_padding


```python
spatial_2d_padding(x, padding=(1, 1), dim_ordering='th')
```


左と右にそれぞれ"padding[0]"個と"padding[1]"の0を増やすことで，4Dテンソルの二番目と三番目の次元を増やします．

----

### get_value


```python
get_value(x)
```


Numpy配列のようにテンソル変数の値を返します．

----

### batch_get_value


```python
batch_get_value(xs)
```


Numpy配列のリストのように，一つ以上のテンソル変数の値を返します．

----

### set_value


```python
set_value(x, value)
```


Numpy配列から，テンソル変数の値を設定します．

----

### batch_set_value


```python
batch_set_value(tuples)
```


多くのテンソル変数の値を一度に設定します．

__引数__

- __tuples__: タプルのリスト `(tensor, value)`．
`value`はNumpy配列であるべきです．

----

### function


```python
function(inputs, outputs, updates=[])
```


Keras関数のインスタンスを作成します．

__引数__

- __inputs__: プレースホルダー/変数のテンソルのリスト．
- __outputs__: 出力のテンソルのリスト．
- __updates__: 更新するタプルのリスト (old_tensor, new_tensor)．

----

### gradients


```python
gradients(loss, variables)
```


`variables`の`loss`についての勾配 (テンソル変数のリスト) を返します．

----

### rnn


```python
rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```


テンソルの時間次元にわたって反復します．

__引数__

- __inputs__: shape (samples, time, ...) を持つ時間データのテンソル（少なくとも3D）．
- __step_function__:
- __Parameters__:
	- __input__: ある時間ステップでのサンプルのバッチに対する入力を表す，
	shape (samples, ...) を持つテンソル（時間軸を持たない）．
	- __states__: テンソルのリスト．
- __返り値__:
	- __output__: shape (samples, ...) を持つテンソル（時間軸を持たない）．
	- __new_states__: 'states'と同じ長さとshapeを持つテンソルのリスト．
- __initial_states__: ステップ関数で利用される状態に対する初期値を含む，
shape (samples, ...) を持つテンソル（時間軸を持たない）．
- __go_backwards__: ブール値．真ならば，逆順で時間軸にわたって反復します．
- __mask__: マスクされたすべての要素に対して0となるような，
shape (samples, time, 1)を持つバイナリテンソル．
- __constants__: 各ステップで渡される定数値のリスト．
- __unroll__: TensorFlowではRNNは常にアンロールされますが，
TheanoではRNNをアンロールするブール値のフラグを利用することができます．
- __input_length__: TensorFlowの実装では関係ありません．
Theanoでアンロールを利用するときは指定する必要があります．

__返り値__

タプル (last_output, outputs, new_states)．

- __last_output__: shape (samples, ...)を持つRNNの最新の出力．
- __outputs__: 各エントリーの出力[s, t]がサンプルsに対する時刻tでのステップ関数の出力であるような，
shape (samples, time, ...) を持つテンソル
- __new_states__: shape (samples, ...)を持つ，ステップ関数で返される最新の状態を表すテンソルのリスト．

----

### switch


```python
switch(condition, then_expression, else_expression)
```


スカラー値に依存した二つの操作間を入れ替えます（整数もしくはブール値）．
`then_expression`と`else_expression`はともに*同じshape*を持つシンボリックなテンソルであるべきであることに注意してください．

__Arguments__

- __condition__: スカラーテンソル．
- __then_expression__: TensorFlow操作．
- __else_expression__: TensorFlow操作．

----

### in_train_phase


```python
in_train_phase(x, alt)
```


訓練フェーズでは`x`を選択し，それ以外では`alt`を選択します．
`alt`は`x`と*同じshape*を持つべきであることに注意してください．

----

### in_test_phase


```python
in_test_phase(x, alt)
```


テストフェーズでは`x`を選択し，それ以外では`alt`を選択します．
`alt`は`x`と*同じshape*を持つべきであることに注意してください．

----

### relu


```python
relu(x, alpha=0.0, max_value=None)
```


修正線形ユニット

__引数__

- __alpha__: 負のセクションの傾き．
- __max_value__: 飽和度の閾値．

----

### softmax


```python
softmax(x)
```


テンソルのソフトマックス．

----

### softplus


```python
softplus(x)
```


テンソルのソフトプラス．

----

### categorical_crossentropy


```python
categorical_crossentropy(output, target, from_logits=False)
```


出力テンソルと目標テンソルの間のカテゴリカルクロスエントロピー．
目標テンソルは出力と同じshapeを持つ必要があります．

----

### sparse_categorical_crossentropy


```python
sparse_categorical_crossentropy(output, target, from_logits=False)
```


出力テンソルと目標テンソルの間のカテゴリカルクロスエントロピー．
目標テンソルは整数のテンソルである必要があります．

----

### binary_crossentropy


```python
binary_crossentropy(output, target, from_logits=False)
```


出力テンソルと目標テンソルの間のバイナリクロスエントロピー．

----

### sigmoid


```python
sigmoid(x)
```


成分ごとのシグモイド．

----

### hard_sigmoid


```python
hard_sigmoid(x)
```


セグメントごとのシグモイドの線形近似．
シグモイドよりも高速．

----

### tanh


```python
tanh(x)
```


成分ごとのtanh．

----

### dropout


```python
dropout(x, level, seed=None)
```


`x`のエントリーにランダムに0を設定し，一方でテンソル全体をスケーリングします．

__引数__

- __x__: テンソル
- __level__: 0に設定されるテンソルにおけるエントリーの割合
- __seed__: 決定論を保証するランダムシード．

----

### l2_normalize


```python
l2_normalize(x, axis)
```


指定した軸に沿って，L2ノルムについてテンソルを正規化します．

----

### conv2d


```python
conv2d(x, kernel, strides=(1, 1), border_mode='valid', dim_ordering='th', image_shape=None, filter_shape=None)
```


2Dの畳み込み．

__引数__

- __kernel__: カーネルテンソル．
- __strides__: ストライドのタプル．
- __border_mode__: 文字列，"same"もしくは"valid"．
- __dim_ordering__: "tf"もしくは"th"．入力/カーネル/出力でTheanoもしくはTensorFlowの次元順序を利用するかどうか．

----

### pool2d


```python
pool2d(x, pool_size, strides=(1, 1), border_mode='valid', dim_ordering='th', pool_mode='max')
```


2Dのプーリング．

__引数__

- __pool_size__: 二つの整数からなるタプル．
- __strides__: 二つの整数からなるタプル．
- __border_mode__: "valid"もしくは"same"の一つ．
- __dim_ordering__: "th"もしくは"tf"の一つ．
- __pool_mode__: "max"，"avg"の一つ．
