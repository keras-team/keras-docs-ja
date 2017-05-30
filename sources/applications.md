# Applications

Kerasの応用は事前学習した重みを利用可能な深層学習のモデルです．
これらのモデルは予測，特徴量抽出そしてfine-tuningのために利用できます．

モデルをインスタンス化すると重みは自動的にダウンロードされます．重みは`~/.keras/models/`に格納されます．

## 利用可能なモデル

### ImageNetで学習した重みをもつ画像分類のモデル:

- [Xception](#xception)
- [VGG16](#vgg16)
- [VGG19](#vgg19)
- [ResNet50](#resnet50)
- [InceptionV3](#inceptionv3)

（Xceptionを除く）これらすべてのアーキテクチャは，TensorFlowとTheanoの両方に対応しており，`~/.keras/keras.json`の設定にしたがってモデルはインスタンス化されます．
例えば，`image_dim_ordering=channels_last`とした際は，このリポジトリからロードされるモデルは，TensorFlowの次元の順序"Width-Height-Depth"にしたがって構築されます．

`SeparableConvolution`を用いているため，XceptionモデルはTensorFlowでのみ使用可能です．

-----

## 画像分類モデルの使用例

### Classify ImageNet classes with ResNet50

```python
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
```

### Extract features with VGG16

```python
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
```

### Extract features from an arbitrary intermediate layer with VGG19

```python
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
```

### Fine-tune InceptionV3 on a new set of classes

```python
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(...)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(...)
```


### Build InceptionV3 over a custom input tensor

```python
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
```

-----

# Documentation for individual models

- [Xception](#xception)
- [VGG16](#vgg16)
- [VGG19](#vgg19)
- [ResNet50](#resnet50)
- [InceptionV3](#inceptionv3)

-----


## Xception


```python
keras.applications.xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
```

ImageNetで事前学習した重みを利用可能なXception V1モデル．

ImageNetにおいて，このモデルのtop-1のvalidation accuracyは0.790で，top-5のvalidation accuracyは0.945です．

`SeparableConvolution`を用いているため，XceptionモデルはTensorFlowでのみ使用可能であることに注意してください．
さらにデータフォーマットは"channels_last" (height, width, channels)のみサポートしています．

デフォルトの入力サイズは299x299．

### Arguments

- include_top: ネットワークの出力層側にある全結合層を含むかどうか．
- weights: `None` (ランダム初期化) か "imagenet" (ImageNetで学習した重み) のどちらか一方．
- input_tensor: モデルの入力画像として利用するためのオプションのKerasテンソル (つまり，`layers.Input()`の出力)
- input_shape: オプショナルなshapeのタプル，`include_top`がFalseの場合のみ指定可能 (そうでないときは入力のshapeは`(299, 299, 3)`)．正確に3つの入力チャンネルをもつ必要があり，width と height は71以上にする必要があります．例えば`(150, 150, 3)`は有効な値です．
- pooling: 特徴量抽出のためのオプショナルなpooling mode，`include_top`が`False`の場合のみ指定可能．
    - `None`：モデルの出力が，最後のconvolutional layerの4Dテンソルであることを意味しています．
    - `avg`：最後のconvolutional layerの出力にglobal average poolingが適用されることで，モデルの出力が2Dテンソルになることを意味しています．
    - `max`：global max poolingが適用されることを意味します．
- classes: 画像のクラス分類のためのオプショナルなクラス数，`include_top`がTrueかつ`weights`が指定されていない場合のみ指定可能．

### Returns

Kerasのモデルインスタンス．

### References

- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

### License

この重みは私達自身が学習したもので，MITライセンスの下で公開されています．

-----

## VGG16

```python
keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
```

ImageNetで事前学習した重みを利用可能なVGG16モデル．

このモデルは，TheanoとTensorFlowの両方のbackendで利用でき，
"channels_first" データフォーマット (channels, height, width) か "channels_last" データフォーマット (height, width, channels)の両方で構築可能です．

デフォルトの入力サイズは224x224．

### Arguments

- include_top: ネットワークの出力層側にある3つの全結合層を含むかどうか．
- weights: `None` (ランダム初期化) か "imagenet" (ImageNetで学習した重み) のどちらか一方．
- input_tensor: モデルの入力画像として利用するためのオプションのKerasテンソル (つまり，`layers.Input()`の出力)
- input_shape: オプショナルなshapeのタプル，`include_top`がFalseの場合のみ指定可能 (そうでないときは入力のshapeは`(224, 224, 3)` (`tf`のとき) か `(3, 224, 224)` (`th`のとき) )．正確に3つの入力チャンネルをもつ必要があり，width と height は48以上にする必要があります．例えば`(200, 200, 3)`は有効値．
- pooling: 特徴量抽出のためのオプショナルなpooling mode，`include_top`が`False`の場合のみ指定可能．
    - `None`：モデルの出力が，最後のconvolutional layerの4Dテンソルであることを意味しています．
    - `avg`：最後のconvolutional layerの出力にglobal average poolingが適用されることで，モデルの出力が2Dテンソルになることを意味しています．
    - `max`：global max poolingが適用されることを意味します．
- classes: 画像のクラス分類のためのオプショナルなクラス数，`include_top`がTrueかつ`weights`が指定されていない場合のみ指定可能．

### Returns

Kerasのモデルインスタンス．

### References

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556): please cite this paper if you use the VGG models in your work.

### License

この重みは[Oxford大学のVGG](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)により[Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/)の下で公開されたものを移植しています．

-----

## VGG19


```python
keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
```

ImageNetで事前学習した重みを利用可能なVGG19モデル．

このモデルは，TheanoとTensorFlowの両方のbackendで利用でき，
"channels_first" データフォーマット (channels, height, width) か "channels_last" データフォーマット (height, width, channels)の両方で構築可能です．

デフォルトの入力サイズは224x224．

### Arguments

- include_top: ネットワークの出力層側にある3つの全結合層を含むかどうか．
- weights: `None` (ランダム初期化) か "imagenet" (ImageNetで学習した重み) の一方．
- input_tensor: モデルの入力画像として利用するためのオプションのKerasテンソル (つまり，`layers.Input()`の出力)
- input_shape: オプショナルなshapeのタプル，`include_top`がFalseの場合のみ指定可能 (そうでないときは入力のshapeは`(224, 224, 3)` (`channels_last`データフォーマットのとき) か `(3, 224, 224)` (`channels_first`データフォーマットのとき) )．正確に3つの入力チャンネルをもつ必要があり，width と height は48以上にする必要があります．例えば`(200, 200, 3)`は有効値．
- pooling: 特徴量抽出のためのオプショナルなpooling mode，`include_top`が`False`の場合のみ指定可能．
    - `None`：モデルの出力が，最後のconvolutional layerの4Dテンソルであることを意味しています．
    - `avg`：最後のconvolutional layerの出力にglobal average poolingが適用されることで，モデルの出力が2Dテンソルになることを意味しています．
    - `max`：global max poolingが適用されることを意味します．
- classes: 画像のクラス分類のためのオプショナルなクラス数，`include_top`がTrueかつ`weights`が指定されていない場合のみ指定可能．

### Returns

Kerasのモデルインスタンス．


### References

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

### License

この重みは[Oxford大学のVGG](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)により[Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/)の下で公開されたものを移植しています．

-----

## ResNet50


```python
keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
```

ImageNetで事前学習した重みを利用可能なResNet50モデル．

このモデルは，TheanoとTensorFlowの両方のbackendで利用でき，
"channels_first" データフォーマット (channels, height, width) か "channels_last" データフォーマット (height, width, channels)の両方で構築可能です．

デフォルトの入力サイズは224x224．

### Arguments

- include_top: ネットワークの出力層側にある全結合層を含むかどうか．
- weights: `None` (ランダム初期化) か "imagenet" (ImageNetで学習した重み) の一方．
- input_tensor: モデルの入力画像として利用するためのオプションのKerasテンソル (つまり，`layers.Input()`の出力)
- input_shape: オプショナルなshapeのタプル，`include_top`がFalseの場合のみ指定可能 (そうでないときは入力のshapeは`(224, 224, 3)` (`channels_last`データフォーマットのとき) か `(3, 224, 224)` (`channels_first`データフォーマットのとき) )．正確に3つの入力チャンネルをもつ必要があり，width と height は197以上にする必要があります．例えば`(200, 200, 3)`は有効値．
- pooling: 特徴量抽出のためのオプショナルなpooling mode，`include_top`が`False`の場合のみ指定可能．
    - `None`：モデルの出力が，最後のconvolutional layerの4Dテンソルであることを意味しています．
    - `avg`：最後のconvolutional layerの出力にglobal average poolingが適用されることで，モデルの出力が2Dテンソルになることを意味しています．
    - `max`：global max poolingが適用されることを意味します．
- classes: 画像のクラス分類のためのオプショナルなクラス数，`include_top`がTrueかつ`weights`が指定されていない場合のみ指定可能．

### Returns

Kerasのモデルインスタンス．

### References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

### License

この重みは[Kaiming He](https://github.com/KaimingHe/deep-residual-networks)により[MITライセンス](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE)の下で公開されたものを移植しています．

-----

## InceptionV3


ImageNetで事前学習した重みを利用可能なInception V3モデル．

このモデルは，TheanoとTensorFlowの両方のbackendで利用でき，
"channels_first" データフォーマット (channels, height, width) か "channels_last" データフォーマット (height, width, channels)の両方で構築可能です．

デフォルトの入力サイズは299x299．

```python
keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None)
```

### Arguments

- include_top: ネットワークの出力層側にある全結合層を含むかどうか．
- weights: `None` (ランダム初期化) か "imagenet" (ImageNetで学習した重み) の一方．
- input_tensor: モデルの入力画像として利用するためのオプションのKerasテンソル (つまり，`layers.Input()`の出力)
- input_shape: オプショナルなshapeのタプル，`include_top`がFalseの場合のみ指定可能 (そうでないときは入力のshapeは`(299, 299, 3)` (`channels_last`データフォーマットのとき) か `(3, 299, 299)` (`channels_first`データフォーマットのとき) )．正確に3つの入力チャンネルをもつ必要があり，width と height は139以上にする必要があります．例えば`(150, 150, 3)`は有効値．
- pooling: 特徴量抽出のためのオプショナルなpooling mode，`include_top`が`False`の場合のみ指定可能．
    - `None`：モデルの出力が，最後のconvolutional layerの4Dテンソルであることを意味しています．
    - `avg`：最後のconvolutional layerの出力にglobal average poolingが適用されることで，モデルの出力が2Dテンソルになることを意味しています．
    - `max`：global max poolingが適用されることを意味します．
- classes: 画像のクラス分類のためのオプショナルなクラス数，`include_top`がTrueかつ`weights`が指定されていない場合のみ指定可能．

### Returns

Kerasのモデルインスタンス．

### References

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

### License

この重みは [Apacheライセンス](https://github.com/tensorflow/models/blob/master/LICENSE)の下で公開されています．
