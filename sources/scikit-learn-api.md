# Scikit-Learn APIのためのラッパー

`keras.wrappers.scikit_learn.py`にあるラッパーを通して，Kerasの`Sequential`モデル（1つの入力のみ）をScikit-Learnワークフローの一部として利用できます．

2つのラッパーが利用可能です:

`keras.wrappers.sk_learn.KerasClassifier(build_fn=None, **sk_params)`, これはScikit-Learnのclassifierインターフェースを実装します．

`keras.wrappers.sk_learn.KerasRegressor(build_fn=None, **sk_params)`, これはScikit-Learnのregressorインターフェースを実装します．

### 引数

- __build_fn__: 呼び出し可能な関数，または，クラスインスタンス
- __sk_params__: モデルパラメータとfittingパラメータ

`build_fn`は，Kerasモデルを構成し，コンパイルし，返します．
このモデルは，fit/predictのために利用されます．以下の3つの値のうち
1つをbuild_fnに渡すことができます:

1. 関数
2. __call__ メソッドを実装したクラスのインスタンス
3. None．これは`KerasClassifier`または`KerasRegressor`を継承したクラスを意味します．この __call__ メソッドはbuild_fnのデフォルト
として扱われます．

`sk_params`はモデルパラメータとfittingパラメータの両方を取ります．
モデルパラメータは`build_fn`の引数です．`sk_params`に何も与えなくとも予測器が作れるように，
scikit-learnの他の予測器と同様に，`build_fn`はその引数にデフォルトパラメータを取ります．

また，`sk_params`は`fit`，`predict`，`predict_proba`，および，`score`メソッドを
呼ぶためのパラメータも取ります（例えば，`epochs`, `batch_size`）．
fitting (predicting) パラメータは以下の順番で選択されます:

1. `fit`，`predict`，`predict_proba`，および，`score`メソッドの辞書引数に与えられた値
2. `sk_params`に与えられた値
3. `keras.models.Sequential`，`fit`，`predict`，`predict_proba`，および，`score`メソッドのデフォルト値

scikit-learnの`grid_search`APIを利用するとき，チューニングパラメータは`sk_params`に渡したものになります．
これには，fittingパラメータも含まれます．つまり，最適なモデルパラメータだけでなく，最適な`batch_size`や
`epochs`の探索に，`grid_search`を利用できます．
