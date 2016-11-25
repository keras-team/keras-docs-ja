### layer_from_config


```python
layer_from_config(config, custom_objects={})
```



__引数__

- __config__: {'class_name': str, 'config': dict}の形式の辞書
- __custom_objects__: 自作の（Kerasのものではない）オブジェクトのクラス名（もしくは関数名）と
そのクラスや関数を対応付けた辞書

__戻り値__

Layerインスタンス（Model，Sequential，Graph，Layerなど）
