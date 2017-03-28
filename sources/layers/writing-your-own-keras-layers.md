# オリジナルのKerasレイヤーを作成する

シンプルで状態を持たない独自演算では，`layers.core.Lambda`を用いるべきでしょう．しかし，学習可能な重みを持つような独自演算は，自身でレイヤーを実装する必要があります．

以下にKerasレイヤーの枠組みを示します．ここでは3つのメソッドを実装する必要があります．

- `build(input_shape)`: これは重みを定義するメソッドです．学習可能な重みはリスト`self.trainable_weights`に追加されます．他の注意すべき属性は以下の通りです．`self.updates`(更新されるタプル(tensor, new_tensor)のリスト)．`non_trainable_weights`と`updates`の使用例は，`BatchNormalization`レイヤーのコードを参照してください．
- `call(x)`: ここではレイヤーのロジックを記述します．オリジナルのレイヤーでマスキングをサポートしない限り，第一引数の入力テンソルが`call`に渡されることに気を付けてください．
- `get_output_shape_for(input_shape)`: 作成したレイヤーの内部で入力の形状を変更する場合には，ここで形状変換のロジックを指定する必要があります．こうすることでKerasは，自動的に形状を推定できます．

```python
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        initial_weight_value = np.random.random((input_dim, output_dim))
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
```

既存のKerasレイヤーは何を実装するにしても十分なサンプルを提供しています．なので，躊躇せずソースコードを読んでください!
