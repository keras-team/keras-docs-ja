## ImageDataGenerator

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    dim_ordering=K.image_dim_ordering())
```

リアルタイムにデータ拡張しながら，テンソル画像データのバッチを生成します．また，このジェネレータは，データを無限にループするので，無限にバッチを生成します．

- __Arguments__:
    - __featurewise_center__: 真理値．データセット全体で，入力の平均を0にします．
    - __samplewise_center__: 真理値． 各サンプルの平均を0にします．
    - __featurewise_std_normalization__: 真理値． 入力をデータセットの標準偏差で正規化します．
    - __samplewise_std_normalization__: 真理値．各入力をその標準偏差で正規化します．
    - __zca_whitening__: 真理値．ZCA白色化を適用します．
    - __rotation_range__: 整数．画像をランダムに回転する回転範囲．
    - __width_shift_range__: 浮動小数点数（横幅に対する割合）．ランダムに水平シフトする範囲．
    - __height_shift_range__: 浮動小数点数（縦幅に対する割合）．ランダムに垂直シフトする範囲．
    - __shear_range__: 浮動小数点数．シアー強度（反時計回りのシアー角度（ラジアン））．
    - __zoom_range__: 浮動小数点数または[lower，upper]．ランダムにズームする範囲．浮動小数点数が与えられた場合，`[lower, upper] = [1-zoom_range, 1+zoom_range]`となります．
    - __channel_shift_range__: 浮動小数点数．ランダムにチャンネルをシフトする範囲．
    - __fill_mode__: {"constant", "nearest", "reflect", "wrap"}のいずれか．指定されたモードに応じて，入力画像の境界周りを埋めます．
    - __cval__: 浮動小数点数または整数．`fill_mode = "constant"`のときに利用される値．
    - __horizontal_flip__: 真理値．水平方向に入力をランダムに反転します．
    - __vertical_flip__: 真理値．垂直方向に入力をランダムに反転します．
    - __rescale__: rescale factor. デフォルトはNone．Noneか0ならば，適用しない．それ以外であれば，(他の変換を行う前に) 与えられた値をデータに積算する．
    - __preprocessing_function__: 各入力に適用される関数です．この関数は他の変更が行われる前に実行されます．この関数は三次元のnumpyテンソルを引数にとり，同じshapeのテンソルを出力するように定義する必要があります．
    - __data_format__: `"channels_last"`(デフォルト)か`"channels_first"`を指定します. `"channels_last"`の場合，入力の型は`(samples, height, width, channels)`となり，"channels_first"の場合は`(samples, channels, height, width)`となります．デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_data_format`の値です．一度も値を変更していなければ，"channels_last"になります．

- __Methods__:
    - __fit(x)__: 与えられたサンプルデータに基づいて，データに依存する統計量を計算します．
    featurewise_center，featurewise_std_normalization，または，zca_whiteningが指定されたときに必要になります．
        - __引数__:
            - __x__: サンプルデータ．
            - __augment__: 真理値（デフォルト: False）．ランダムにサンプルを拡張するかどうか．
            - __rounds__: 整数（デフォルト: 1）．augumentが与えられたときに，利用するデータに対して何回データ拡張を行うか．
            - __seed__: 整数 (デフォルト: None). ランダムシード.
    - __flow(x, y)__: numpyデータとラベルのarrayを受け取り，拡張/正規化したデータのバッチを生成する．無限ループ内で，無限にバッチを生成します．
        - __引数__:
            - __x__: データ．4次元データである必要があります．RGBではチャネルを3に, グレースケールではチャネルを1にしてください．
            - __y__: ラベル．
            - __batch_size__: 整数（デフォルト: 32）．
            - __shuffle__: 真理値（デフォルト: False）．
            - __save_to_dir__: Noneまたは文字列（デフォルト: None）．生成された拡張画像を保存するディレクトリを指定できます（行ったことの可視化に有用です）．
            - __save_prefix__: 文字列（デフォルト`''`）．画像を保存する際にファイル名に付けるプレフィックス（`set_to_dir`に引数が与えられた時のみ有効）．
            - __save_format__: "png"または"jpeg"（`set_to_dir`に引数が与えられた時のみ有効）．デフォルトは"jpeg"．
        - __yields__: `(x, y)`からなるタプルで`x`は画像データのnumpy配列で`y`はラベルのnumpy配列．ジェネレーターは無限ループする．
    - __flow_from_directory(directory)__: ディレクトリへのパスを受け取り，拡張/正規化したデータのバッチを生成する．データを無限にループするので，無限にバッチを生成します．
        - __引数__:
            - __directory__: ディレクトリへのパス．クラスごとに1つのサブディレクトリを含み，サブディレクトリはPNGかJPG形式の画像を含まなければならない．詳細は[このスクリプト](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)を．
            - __target_size__: 整数のタプル．デフォルトは`(256, 256)`．この値に全画像はリサイズされる．
            - __color_mode__: "grayscale"か"rbg"の一方．デフォルトは"rgb"．画像を1か3チャンネルの画像に変換するかどうか．
            - __classes__: クラスサブディレクトリのリスト．（例えば，`['dogs', 'cats']`）．デフォルトはNone．与えられなければ，クラスのリスト自動的に推論される（そして，ラベルのインデックスと対応づいたクラスの順序はalphanumericになる）．
            - __class_mode__: "categorical"か"binary"か"sparse"か"None"のいずれか1つ．デフォルトは"categorical"．返すラベルの配列の型を決定する: "categorical"は2次元のone-hotにエンコード化されたラベル，"binary"は1次元の2値ラベル，"sparse"は1次元の整数ラベル．Noneであれば，ラベルを返さない (ジェネレーターは画像のバッチのみ生成するため，`model.predict_generator()`や`model.evaluate_generator()`などを使う際に有用)．
            - __batch_size__: 整数（デフォルト: 32）．
            - __shuffle__: 真理値（デフォルト: True）．
            - __seed__: shuffleのための乱数seed
            - __save_to_dir__: Noneまたは文字列（デフォルト: None）．生成された拡張画像を保存するディレクトリを指定できます（行ったことの可視化に有用です）．
            - __save_prefix__: 文字列．画像を保存する際にファイル名に付けるプレフィックス（`set_to_dir`に引数が与えられた時のみ有効）．
            - __save_format__: "png"または"jpeg"（`set_to_dir`に引数が与えられた時のみ有効）．デフォルトは"jpeg"．
            - __follow_links__: サブディレクトリ内のシンボリックリンクに従うかどうか．デフォルトはFalse．

- __Examples__:

`.flow(x, y)`の使用例:

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train), epochs=epochs)

# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.train(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```

`.flow_from_directory(directory)`の使用例:

```python
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=800)
```

画像とマスクに対して，同時に変更を加える例．

```python
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
```
