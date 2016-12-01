### get_file


```python
get_file(fname, origin, untar=False, md5_hash=None, cache_subdir='datasets')
```


既にキャッシュされていないならURLからファイルをダウンロードします．

MD5ハッシュ値を渡せば，既にキャッシュされていたとしてもファイルをダウンロードし，同一性を検証します．

__引数__

- __fname__: ファイル名
- __origin__: ファイルの置かれているURL
- __untar__: 真偽値．ファイルを解凍するかどうか
- __md5_hash__: 同一性検証のためのMD5ハッシュ値
- __cache_subdir__: キャッシュ先のディレクトリ

__戻り値__

ダウンロードされたファイルへのパス