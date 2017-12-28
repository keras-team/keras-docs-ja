# なぜKerasを使うか?

今日，数え切れない数の深層学習フレームワークが存在します．なぜ他のライブラリではなくて，Kerasを使うのでしょうか？ここでは，Kerasが既存の選択肢に引けを取らない理由のいくつかを紹介します．

---

## Kerasはシンプルかつ柔軟に使用できます
- Kerasは，機械ではなく，人間のために設計されたAPIです．[Kerasは認知的負荷を軽減するためのベストプラクティスに従っています](https://blog.keras.io/user-experience-design-for-apis.html): 一貫性のあるシンプルなAPIを提供し，一般的なユースケースで必要なユーザーの操作を最小限に抑え，エラー時には明確で実用的なフィードバックを提供します．
- これにより，Kerasは簡単に学ぶことが出来て，簡単に使う事が出来ます．Kerasユーザーは，生産性が高く，競争相手よりも，より多くのアイデアを試す事が出来ます． -- これにより，[機械学習のコンテストで勝つのに役立ちます](https://www.quora.com/Why-has-Keras-been-so-successful-lately-at-Kaggle-competitions).
- 手軽さがあっても，柔軟性がなければいけません: Kerasは低レベルな深層学習言語（特にTensorFlow）と統合しているので，基本の深層学習言語で構築されたものを実装する事が出来ます．特に，`tf.keras`として，Keras APIはTensorFlowワークフローとシームレスに統合されています．

---

## Kerasは事業と研究コミュニティの両方で幅広く使用されています
2017年11月現在，Kerasは20万以上の個人ユーザーがおり，TensorFlow自体を除いて，他の深層学習フレームワークよりも事業と研究コミュニティの両方で多く採用されています（KerasはTensorFlowとの組み合わせでよく使用されます）．

あなたはすでにKerasで構築された機能を日常で使用しています -- Netflix，Uber，Yelp，Instacart，Zocdoc，Squareなど多くの企業がKerasを使用しています．特に，自社製品の核となる部分を深層学習で用いているようなスタートアップ企業で人気があります．

また，Kerasは，深層学習研究者の間でも人気があり，プレプリント・サーバ[arXiv.org](https://arxiv.org/archive/cs)にアップロードされた，科学技術論文で言及されているフレームワークの中で二番目に使用されています。

<img src='https://keras.io/img/arxiv-mentions.png' style='width:500px; display: block; margin: 0 auto;'/>

また，Kerasは，大規模な科学機関，例えば，CERNやNASAの研究者によって採用されています．

---

## Kerasは簡単にモデルを製品化できます
Kerasのモデルは，他の深層学習フレームワークよりも多くのプラットフォームで，簡単にデプロイできます．

- iOS（[Apple’s CoreML](https://developer.apple.com/documentation/coreml)経由，Kerasのサポートは正式にAppleから提供されています）
- Android（TensorFlow Androidランタイム経由）
- ブラウザ（[Keras.js](https://transcranial.github.io/keras-js/#/)や，[WebDNN](https://mil-tokyo.github.io/webdnn/)などのGPU利用が可能なJavaScriptランタイム経由）
- Google Cloud（[TensorFlow-Serving](https://www.tensorflow.org/serving/)経由）
- Pythonのウェブアプリのバックエンド（例えば，Flaskアプリ）
- JVM（[SkyMindによって提供されたDL4J モデル](https://deeplearning4j.org/model-import-keras)経由）
- ラズベリーパイ

---

## Kerasは複数のバックエンドをサポートし，一つのエコシステムに縛られません
Kerasは複数の[バックエンドエンジン](https://keras.io/ja/backend/)をサポートしています．重要な事に，組み込みレイヤーのみで構成されるKerasモデルは，全てのバックエンド間で移植可能です．: 一つのバックエンドを使用して学習したモデルを用いて，別のバックエンドを使用してモデルをロードする事が出来ます．（例えば，デプロイなどで用いる事が出来ます）
使用可能なバックエンドは以下のとおりです．

- TensorFlow バックエンド (from Google)
- CNTK バックエンド (from Microsoft)
- Theano バックエンド

Amazonも現在，KerasのMXNetバックエンドの開発にも取り組んでいます．

また，KerasモデルはCPU以外の様々なハードウェアプラットフォームで学習する事が出来ます．

- [NVIDIA GPUs](https://developer.nvidia.com/deep-learning)
- [Google TPUs](https://cloud.google.com/tpu/)（TensorFlowバックエンドかつ，Google Cloud経由）
- OpenGLが使用出来るAMDのようなGPU（[the PlaidML Kerasバックエンド](https://github.com/plaidml/plaidml)経由）

---

## Kerasは複数のGPU，分散学習のサポートが強力です
- Kerasは[複数GPU並列処理のための組み込みサポート](/utils/#multi_gpu_model)もあります．
- Uberの[Horovod](https://github.com/uber/horovod)は，Kerasモデルを最もサポートしています．
- Kerasモデルは[TensorFlow Estimatorsに変換する事](https://www.tensorflow.org/versions/master/api_docs/python/tf/keras/estimator/model_to_estimator)が出来ます．また，[Google CloudのGPUクラスターを用いて](https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine)学習が出来ます．
- Kerasは[Dist-Keras](https://github.com/cerndb/dist-keras) (from CERN)と [Elephas](https://github.com/maxpumperla/elephas)経由でSpark上で走らせる事が出来ます．

---

## Kerasの開発は深層学習の主要企業によってサポートされています
Kerasの開発は主にGoogleによってサポートされ，Keras APIはTensorFlowに `tf.keras`としてパッケージ化されています． 加えて，MicrosoftはCNTK Kerasバックエンドを管理しています． Amazon AWSはMXNetサポートを開発中です．
その他，NVIDIA，Uber，Apple（CoreML）によって，サポートされています．

<img src='https://keras.io/img/google-logo.png' style='width:200px; margin-right:15px;'/>
<img src='https://keras.io/img/microsoft-logo.png' style='width:200px; margin-right:15px;'/>
<img src='https://keras.io/img/nvidia-logo.png' style='width:200px; margin-right:15px;'/>
<img src='https://keras.io/img/aws-logo.png' style='width:110px; margin-right:15px;'/>
