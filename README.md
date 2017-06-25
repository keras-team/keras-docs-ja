# Japanese translation of the Keras documentation

This is the repository for the translated `.md` sources files of [keras.io](http://keras.io/). The translation project is currently in progress for __keras-2__.
To contribute, please grab a task on [this TODO spreadsheet](https://docs.google.com/spreadsheets/d/14foDtxrWUzJVIKGC0dgGFH4faNMlMyMDrLXzR02duEQ/edit?usp=sharing) then send a Pull Request to this repository.

# TODO

1. Translate for keras v2: __current state__
  - You can ignore correct `source` link (please put a dummy link, e.g. `<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L1)</span>` in `recurrent.md`) since master branch may move before this project completes
1. Decide [termification](https://github.com/fchollet/keras-docs-ja/issues/40)
  - I think that Japanese terminology should follow English terminology as much as possible for searchability.
1. Fix links to keras source code (I made a private script half a year ago, so I may improve it)

---

# Keras documentationの日本語訳化

このリポジトリは，[keras.io](http://keras.io/)の`.md`ファイルを日本語訳するリポジトリです．
現在 __keras-2__ の作業中です．（なぜなら[keras-docs-ja](https://keras.io/ja/)はversion1で止まっているためです）
contributeいただける方は，[this TODO spreadsheet](https://docs.google.com/spreadsheets/d/14foDtxrWUzJVIKGC0dgGFH4faNMlMyMDrLXzR02duEQ/edit?usp=sharing)の中から担当したい作業を見つけて名前を書き込んでから，このリポジトリにPRをください．

# TODO

1. keras v2の翻訳: __作業中__
    - kerasのmasterブランチが作業中に変更するため，`source`リンクは無視できます，(この場合，ダミーのリンクを置いてください，例えば`recurrent.md`では`<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L1)</span>`がダミーリンクになります) 
1. 表記ゆれのために語彙を決める (see also: [termification](https://github.com/fchollet/keras-docs-ja/issues/40))
   - 検索性のために，英語にできるだけ従うべきだと思っています
1. masterブランチのソースコードへのリンクを更新するスクリプト作成 (作りかけがあります)

# 翻訳ガイドライン
- 翻訳対象は本文とコード中のコメントとする．
- 本文は敬体（です・ます調）とする．
- 句読点は「，．」を用いる．
- 引用符（'，"）は基本的にそのままとする．強調の意味で使用されている場合のみ「」に変換する．（bckend.md3行目）
- 記号「，．（）？！：；」は全角とする．
- 文中のシンタックスハイライト（syntax highlight）の前後に空白は入れない． 
- 用語の訳は対訳表に従う．

※ 翻訳は英語から日本語へのただの変換作業ではなく，英文の意味を読み取り，日本語として表現する創作作業です．
英語の言い回しに引きずられることなく自然な日本語で表現しましょう．

# 対訳表
- 構文キーワードなどはそのまま英語表記とする．
- 検索性のため，python/numpy/keras特有の単語はそのまま英語表記とする．

| English | 日本語
|:---|:---
| arguments | 引数
| data augumentation | データ拡張
| layer | レイヤー
| loss function | 損失関数
| return | 戻り値
| recurrent  | recurrent
| shape | shape
| target | ターゲット
| testing | テスト
| training | 学習

※ 見つけやすいようにアルファベット順で列挙しています．
必要に応じて追記してください．
