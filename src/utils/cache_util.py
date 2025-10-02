_cache = None
_cache_key = None

def cache_set(key, value):
  global _cache_key, _cache
  if _cache_key != key:
    _cache_key = key
    _cache = value

def cache_get(key, default=None):
  global _cache_key, _cache
  if _cache_key == key:
    return _cache
  return default

def cache_delete(key):
  global _cache_key, _cache
  if _cache_key == key:
    _cache = None
    _cache_key = None
