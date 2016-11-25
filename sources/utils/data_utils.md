### get_file


```python
get_file(fname, origin, untar=False, md5_hash=None, cache_subdir='datasets')
```


Downloads a file from a URL if it not already in the cache.

Passing the MD5 hash will verify the file after download as well as if it is already present in the cache.

__Arguments__

- __fname__: name of the file
- __origin__: original URL of the file
- __untar__: boolean, whether the file should be decompressed
- __md5_hash__: MD5 hash of the file for verification
- __cache_subdir__: directory being used as the cache

__Returns__

Path to the downloaded file
