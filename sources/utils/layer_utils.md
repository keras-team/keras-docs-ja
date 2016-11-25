### layer_from_config


```python
layer_from_config(config, custom_objects={})
```



__Arguments__

- __config__: dict of the form {'class_name': str, 'config': dict}
- __custom_objects__: dict mapping class names (or function names)
of custom (non-Keras) objects to class/functions

__Returns__

Layer instance (may be Model, Sequential, Graph, Layer...)
