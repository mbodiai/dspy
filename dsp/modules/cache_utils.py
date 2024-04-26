import json
import os
from functools import wraps
from pathlib import Path
from typing import Any, Dict

from joblib import Memory

from dsp.utils import dotdict

cache_turn_on = os.environ.get('DSP_CACHEBOOL', 'True').lower() != 'false'


def noop_decorator(arg=None, *noop_args, **noop_kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    if callable(arg):
        return decorator(arg)
    else:
        return decorator


cachedir = os.environ.get('DSP_CACHEDIR') or os.path.join(Path.home(), 'cachedir_joblib')
CacheMemory = Memory(location=cachedir, verbose=0)

cachedir2 = os.environ.get('DSP_NOTEBOOK_CACHEDIR')
NotebookCacheMemory = dotdict()
NotebookCacheMemory.cache = noop_decorator

if cachedir2:
    NotebookCacheMemory = Memory(location=cachedir2, verbose=0)


if not cache_turn_on:
    CacheMemory = dotdict()
    CacheMemory.cache = noop_decorator

    NotebookCacheMemory = dotdict()
    NotebookCacheMemory.cache = noop_decorator

def make_hashable(item):
    """ Recursively convert items to a hashable form suitable for use as cache keys. """
    if isinstance(item, (tuple, list)):
        return tuple(make_hashable(x) for x in item)
    elif isinstance(item, dict):
        # Sort items to ensure consistent order
        return tuple((k, make_hashable(v)) for k, v in sorted(item.items()))
    return item

def filter_kwargs(kwargs, ignore_fields):
    """ Recursively filter kwargs to remove specified ignore_fields even if they are nested. """
    result = {}
    for key, value in kwargs.items():
        if key in ignore_fields:
            continue
        if isinstance(value, dict):
            nested_ignore_fields = [field.split('.', 1)[1] for field in ignore_fields if field.startswith(key + '.')]
            if nested_ignore_fields:
                value = filter_kwargs(value, nested_ignore_fields)
        result[key] = value
    return result

def selective_cache(ignore_fields=None):
    ignore_fields = set(ignore_fields or [])
    cache_store = {}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Filter kwargs to exclude ignored fields, even if nested
            filtered_kwargs = filter_kwargs(kwargs, ignore_fields)
            # Make the filtered_kwargs hashable
            key = make_hashable(filtered_kwargs)
            if key in cache_store:
                print("Cache hit for key:", key)
                return cache_store[key]
            else:
                print("Cache miss for key:", key)
                result = func(*args, **kwargs)
                cache_store[key] = result
                return result
        return wrapper
    return decorator
    
# Usage example adjusted for the setup
@selective_cache(ignore_fields=['image', 'user.profile_picture'])
def process_request(**kwargs) -> Dict[str, Any]:
    if "stringify_request" in kwargs:
        kwargs = json.loads(kwargs.pop("stringify_request"))
    print("Processing request with:", kwargs)
    return {"status": "success", "data": kwargs}

def main():
    user_data = {"name": "Alice", "profile_picture": "profile.jpg"}
    image_data = {"path": "image.jpg", "size": 1024}
    other_data = {"text": "Hello, World!"}
    response = process_request(user=user_data, image=image_data, other=other_data)
    print("Response:", response)
    another_user_data = {"name": "Alice", "profile_picture": "other.jpg"}
    another_response = process_request(user=another_user_data, image=image_data, other=other_data)
    print("Another response:", another_response)

if __name__ == "__main__":
    main()
