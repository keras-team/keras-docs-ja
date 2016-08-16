# Keras FAQ:

- [Kerasを引用するには？](#keras)
- [KerasをGPUで動かすには？](#kerasgpu)
- [Keras modelの保存](#keras-model)
- [なぜtraining lossがtesting lossよりもずっと大きいのですか？](#training-losstesting-loss)
- [中間層の出力を可視化するには？](#_1)
- [メモリに乗らない大きさのデータを扱うには？](#_2)
- [validation lossが減らなくなったときに学習を中断するには？](#validation-loss)
- [validation splitはどのように実行されますか？](#validation-split)
- [訓練時にデータはシャッフルされますか？](#_3)
- [各epochのtraining/validation lossやaccuracyを記録するには？](#epochtrainingvalidation-lossaccuracy)
- [層を "freeze" するには？](#freeze)
- [stateful RNNを利用するには？](#stateful-rnn)
- [Sequentialモデルから層を取り除くには？](#sequential)
- [Kerasで事前学習したモデルを使うには？](#keras_1)

---

### Kerasを引用するには？

Kerasがあなたのお役に立てたら，ぜひ著書のなかでKerasを引用してください。BibTexの例は以下の通りです。

```
@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois},
  year={2015},
  publisher={GitHub},
  howpublished={\url{https://github.com/fchollet/keras}},
}
```

### KerasをGPUで動かすには？

バックエンドでTensorFlowを使っている場合，利用可能なGPUがあれば自動的にGPUが使われます。
バックエンドがTheanoの場合，以下の方法があります:

方法1: Theanoフラグを使う:
```bash
THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
```

'gpu'の部分はデバイス識別子に合わせて変更してください(例: gpu0，gpu1など)。

方法2: `.theanorc`を使う: [使い方](http://deeplearning.net/software/theano/library/config.html)

方法3: コードの先頭で，`theano.config.device`，`theano.config.floatX`を設定する:
```python
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
```

---

### Keras modelの保存

*Kerasのモデルを保存するのに，pickleやcPickleを使うことは推奨されません。*

`model.save(filepath)`を使うことで，以下の含む単一のHDF5ファイルにKerasのモデルを保存できます．

- 再構築可能なモデルの構造
- モデルの重み
- 学習時の設定 (損失関数，最適化アルゴリズム)
- 最適化アルゴリズムの状態，学習を終えた時点から正確に継続して学習可能

`keras.models.load_model(filepath)`でモデルを再インスタンス化できます．
`load_model` は，保存してモデルの設定を使い，コンパイルも行います(ただし最初にモデルを定義した際に一度もコンパイルされなかった場合を除く)．

例:

```python
from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
```

**モデルのアーキテクチャ** のみ(weightパラメータを含まない)を保存する場合は以下の通りです:

```python
# save as JSON
json_string = model.to_json()

# save as YAML
yaml_string = model.to_yaml()
```

生成されたJSON / YAMLファイルは，人間可読であり，必要に応じて人手で編集可能です．

保存したデータから，以下のように新しいモデルを作成できます:

```python
# model reconstruction from JSON:
from keras.models import model_from_json
model = model_from_json(json_string)

# model reconstruction from YAML
model = model_from_yaml(yaml_string)
```

**モデルの重み** を保存する場合，以下のようにHDF5を使います。

注: HDF5とPythonライブラリの h5pyがインストールされている必要があります(Kerasには同梱されていません)。

```python
model.save_weights('my_model_weights.h5')
```

モデルのインスタンス作成後，同じモデルアーキテクチャのweightパラメータを以下のようにロードします:

```python
model.load_weights('my_model_weights.h5')
```

---

### なぜtraining lossがtesting lossよりもずっと大きいのですか？

Kerasモデルにはtrainingとtestingの二つのモードがあります。DropoutやL1/L2正則化はtesting時には機能しません。

さらに，training lossは訓練データの各バッチのlossの平均です。モデルは変化していくため，各epochの最初のバッチのlossは最後のバッチのlossよりもかなり大きくなります。一方，testing lossは各epochの最後に計算されるため，lossが小さくなります。

---

### 中間層の出力を可視化するには？

以下のように，ある入力を与えたときの，ある層の出力を返すKeras functionを記述できます:

```python
from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([X])[0]
```

直接TheanoやTensorFlowのfunctionを利用することもできます。

注: 訓練時とテスト時でモデルの振る舞いが異なる場合(例: `Dropout`や`BatchNormalization`利用時など)，以下のようにlearning phaseフラグを利用してください:

```python
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

# output in test mode = 0
layer_output = get_3rd_layer_output([X, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([X, 1])[0]
```

その他の中間層の出力を取得する方法については，[functional API](/getting-started/functional-api-guide)を参照してください。例えばMNISTに対してautoencoderを作成したとします:

```python
inputs = Input(shape=(784,))
encoded = Dense(32, activation='relu')(inputs)
decoded = Dense(784)(encoded)
model = Model(input=inputs, output=decoded)
```

コンパイルと学習後のモデルから，次のようにしてencoderからの出力を得ることができます:

```python
encoder = Model(input=inputs, output=encoded)
X_encoded = encoder.predict(X)
```

---

### メモリに乗らない大きさのデータを扱うには？

`model.train_on_batch(X, y)`と`model.test_on_batch(X, y)`でバッチ学習ができます。詳細は[models documentation](/models/sequential)を参照してください。

その他に，ジェネレータを使うこともできます。 `model.fit_generator(data_generator, samples_per_epoch, nb_epoch)`

実際のバッチ学習の方法については，[CIFAR10 example](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py)を参照してください。

---

### validation lossが減らなくなったときに学習を中断するには？

コールバック関数の`EarlyStopping`を利用してください:

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])
```

詳細は[callbacks documentation](/callbacks)を参照してください。

---

### validation splitはどのように実行されますか？

`model.fit`の引数`validation_split`を例えば0.1に設定すると，データの*最後の10％*が検証用に利用されます(0.25なら最後の25％)。


---

### 訓練時にデータはシャッフルされますか？

`model.fit`の引数`shuffle`が`True`であればシャッフルされます(初期値はTrueです)。各epochでデータはランダムにシャッフルされます。

検証用データ(validation data)はシャッフルされません。

---


### 各epochのtraining/validation lossやaccuracyを記録するには？

`model.fit`が返す`History`コールバックの`history`を参照してください。

```python
hist = model.fit(X, y, validation_split=0.2)
print(hist.history)
```

---

### 層を "freeze" するには？

層を "freeze" することは学習から層を除外することを意味します，すなわちその重みは決して更新されません。
このことはモデルのfine-tuningやテキスト入力のための固定されたembeddingsを使用するという文脈において有用です。

層のコンストラクタの`trainable`に真偽値をに渡すことで，層を学習しないようにできます。

```python
frozen_layer = Dense(32, trainable=False)
```

加えて，インスタンス化後に層の`trainable`propertyに`True`か`False` を与えることができます。このことによる影響としては，`trainable`propertyの変更後のモデルで`compile()`を呼ぶ必要があります。以下にその例を示します:

```python
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# in the model below, the weights of `layer` will not be updated during training
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# with this model the weights of the layer will be updated during training
# (which will also affect the above model since it uses the same layer instance)
trainable_model.compile(optimizer='rmsprop', loss='mse')

frozen_model.fit(data, labels)  # this does NOT update the weights of `layer`
trainable_model.fit(data, labels)  # this updates the weights of `layer`
```

---

### stateful RNNを利用するには？

stateful RNNでは各バッチの内部状態を次のバッチの初期状態として再利用します。

sateful RNNを利用する際は，以下を仮定します:

- 全てのバッチのサンプル数が同じ
- `X1`と`X2`が連続するバッチであるとき，各`i`について`X2[i]`は`X1[i]`のfollow-upシーケンスになっている

stateful RNNを利用するには:

- 最初の層の引数`batch_input_shape`で明示的にバッチサイズを指定してください。バッチサイズはタプルで，例えば32 samples，10 timesteps，特徴量16次元の場合，`(32, 10, 16)`となります
- RNN層で`stateful=True`を指定してください

積算された状態をリセットするには:

- 全層の状態をリセットするには，`model.reset_states()`を利用してください
- 特定のstateful RNN層の状態をリセットするには，`layer.reset_states()`を利用してください

例:

```python

X  # this is our input data, of shape (32, 21, 16)
# we will feed it to our model in sequences of length 10

model = Sequential()
model.add(LSTM(32, batch_input_shape=(32, 10, 16), stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# we train the network to predict the 11th timestep given the first 10:
model.train_on_batch(X[:, :10, :], np.reshape(X[:, 10, :], (32, 16)))

# the state of the network has changed. We can feed the follow-up sequences:
model.train_on_batch(X[:, 10:20, :], np.reshape(X[:, 20, :], (32, 16)))

# let's reset the states of the LSTM layer:
model.reset_states()

# another way to do it in this case:
model.layers[0].reset_states()
```

注: `predict`, `fit`, `train_on_batch`, `predict_classes`メソッドなどは全て，stateful層の状態を更新します。そのため，訓練だけでなく，statefulな予測も可能となります。

---

### Sequentialモデルから層を取り除くには？
`.pop()`を使うことで，Sequentialモデルへ最後に追加した層を削除できます。

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"
```

---

### Kerasで事前学習したモデルを使うには？

以下の画像分類用のモデルのコードと事前学習した重みが利用可能です．

- VGG-16
- VGG-19
- ResNet50
- Inception v3

コードと重みは[このリポジトリ](https://github.com/fchollet/deep-learning-models)にあります．

特徴量抽出やfine-tuningのために事前学習したモデルを使う例は，[このブログ記事](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)をみてください．

また，VGG16はいくつかのKerasのサンプルスクリプトの基礎になっています．

- [Style transfer](https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py)
- [Feature visualization](https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py)
- [Deep dream](https://github.com/fchollet/keras/blob/master/examples/deep_dream.py)
