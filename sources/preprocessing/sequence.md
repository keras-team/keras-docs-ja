## pad_sequences

```python
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32')
```

`nb_samples sequences`から構成されるリスト（スカラのリスト）を形状が`(nb_samples, nb_timesteps)`である2次元のnumpy配列に変換します．`nb_timesteps`は`maxlen`引数が与えられれば`maxlen`，あるいは，最大のシーケンス長になります．`nb_timesteps`より短いシーケンスはパディングされます．

- __戻り値__: 形状が`(nb_samples, nb_timesteps)`である2次元のnumpy配列．

- __引数__:
    - __sequences__: 整数または浮動小数点数を含むリストのリスト．
    - __maxlen__: Noneまたは整数．シーケンスの最大長．この値より長いシーケンスはカットされ，短いシーケンスはパディングされます．
    - __dtype__: 戻り値のデータタイプ．
    - __padding__: 'pre'まはた'post'．各シーケンスの前後どちらを埋めるか．
    - __truncating__: 'pre'まはた'post'．maxlenより長いシーケンスの前後どちらをカットするか．
    - __value__: 浮動小数点数．パディングする値．

---

## skipgrams

```python
keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size, 
    window_size=4, negative_samples=1., shuffle=True, 
    categorical=False, sampling_table=None)
```

単語インデックスのシーケンス（整数のリスト）を以下の2つの形式に変換します:

- (word, word in the same window), with label 1 (positive samples).
- (word, random word from the vocabulary), with label 0 (negative samples).

詳細はMikolovらの論文を参照してください: [Efficient Estimation of Word Representations in
Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)

- __戻り値__: `(couples, labels)`のタプル．
    - `couples`は2つの整数リスト `[word_index, other_word_index]` から構成されるリストです．
    - `labels`は0と1からなるリストです．1は同じウィンドウに`other_word_index`が`word_index`として存在すること意味し，0は`other_word_index`がランダムであることを意味します．
    - categoricalがTrueの場合，ラベルはカテゴリカルになります．つまり，1は[0, 1]になり，0は[1, 0]になります．

- __引数__:
    - __sequence__: インデックスのリスト．sampling_tableを使う場合，単語インデックスはデータセット中のそのランク（1から始まる）になります．
    - __vocabulary_size__: 整数．
    - __window_size__: 整数．positive couple中の2つの単語間の最大距離．
    - __negative_samples__: 0以上の浮動小数点数．0はネガティブサンプル数が0になります．1はネガティブサンプル数がポジティブサンプルと同じ数になります．
    - __shuffle__: 真理値．サンプルをシャッフルするかどうか．
    - __categorical__: 真理値．戻り値のラベルをカテゴリカルにするかどうか．
    - __sampling_table__: 形状が`(vocabulary_size,)`であるnumpy配列．`sampling_table[i]`はインデックスiを持つ単語（データセット中でi番目に頻出する単語であることを想定します）のサンプリング確率です．

---

## make_sampling_table

```python
keras.preprocessing.sequence.make_sampling_table(size, sampling_factor=1e-5)
```

`skipgrams`の`sampling_table`引数を生成するために利用します．`sampling_table[i]`はデータセット中でi番目に頻出する単語をサンプリングする確率です（バランスを保つために，より頻出する語はこれより低い頻度でサンプリングされます）．

- __戻り値__: 形状が`(size,)`であるnumpy配列．

- __引数__:
    - __size__: 語彙数．
    - __sampling_factor__: この値が小さければ小さいほど，頻出語のサンプリング頻度が低くなります．1が与えられたとき，サブサンプリングは行われません（全てのサンプリング確率が1になります）．