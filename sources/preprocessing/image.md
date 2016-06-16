
## ImageDataGenerator (画像データジェネレータ)

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
    dim_ordering='th')
```

リアルタイムにデータ拡張しながら，テンソル画像データのバッチを生成します．また，このジェネレータは，データを無限にループするので，無限にバッチを生成します．

- __引数__:
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
    - __dim_ordering__: {"th", "tf"}のいずれか．
        "tf"モードは入力の形状が`(samples, width, height, channels)`であることを想定します．
        "th"モードは入力の形状が`(samples, channels, width, height)`であることを想定します．
        デフォルトはKerasの設定ファイル`~/.keras/keras.json`の`image_dim_ordering`の値です．値を設定していなければ，"th"になります．

- __メソッド__:
    - __fit(X)__: featurewise_center，featurewise_std_normalization，または，zca_whiteningが指定されたときに必要になります．いくつかのサンプルに対して必要な値を計算します．
        - __引数__:
            - __X__: サンプルデータ．
            - __augment__: 真理値（デフォルト: False）．ランダムにサンプルを拡張するかどうか．
            - __rounds__: 整数（デフォルト: 1）．augumentが与えられたときに，利用するデータに対して何回データ拡張を行うか．
    - __flow(X, y)__:
        - __引数__:
            - __X__: データ．
            - __y__: ラベル．
            - __batch_size__: 整数（デフォルト: 32）．
            - __shuffle__: 真理値（デフォルト: False）．
            - __save_to_dir__: Noneまたは文字列．生成された拡張画像を保存するディレクトリ（どのような処理を行ったか確認するのに役立つでしょう）．
            - __save_prefix__: 文字列．画像を保存する際にファイル名に付けるプレフィックス．
            - __save_format__: "png"または"jpeg"．

- __例__:

```python
(X_train, y_train), (X_test, y_test) = cifar10.load_data(test_split=0.1)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                    samples_per_epoch=len(X_train), nb_epoch=nb_epoch)

# here's a more "manual" example
for e in range(nb_epoch):
    print 'Epoch', e
    batches = 0
    for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=32):
        loss = model.train(X_batch, Y_batch)
        batches += 1
        if batches >= len(X_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```
