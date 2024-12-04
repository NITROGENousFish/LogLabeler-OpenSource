import os
import pickle


GLOBAL_CACHE_DIR = "my_cache"

cache_folder = GLOBAL_CACHE_DIR
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

def cache_save(content, path):
    cache_folder = GLOBAL_CACHE_DIR
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    with open(os.path.join(cache_folder,path), "wb") as fp:
        pickle.dump(content, fp)


def cache_load(path):
    _ = None
    if not os.path.exists(os.path.join(cache_folder, path)):
        raise OSError(f"Cache file not found in {path}")
    with open(os.path.join(cache_folder, path), "rb") as fp:
        _ = pickle.load(fp)
    return _

def is_have_cache(path):
    return os.path.exists(os.path.join(cache_folder, path))


# def cache_tool(F, path: str):


#     full_path = os.path.join(cache_folder, path)
#     # Check if the cache file exists
#     if os.path.exists(full_path):
#         # Load the result from the cache file
#         with open(full_path, "rb") as cache_file:
#             result = pickle.load(cache_file)
#     else:
#         # Execute the function and cache its result
#         result = F()
#         with open(full_path, "wb") as cache_file:
#             pickle.dump(result, cache_file)
#     return result


def remove_all_cache():
    # Remove all files in the .my_cache folder
    for filename in os.listdir(GLOBAL_CACHE_DIR):
        file_path = os.path.join(GLOBAL_CACHE_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def split_into_chunks(obj, n):
    # Calculate the length of each chunk
    k, m = divmod(len(obj), n)
    return [
        obj[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(n)
    ]


from contextlib import contextmanager
import sys
import uuid


@contextmanager
def globalized(nested_subprocess_func):
    """
    When you want to define a function inside a function and use it in a multiprocessing pool (for code management)
    you can use this decorator to make the function global.
    This can solve AttributeError: Can't pickle local object 'multi.<locals>.nested_subprocess_func'
    See https://gist.github.com/EdwinChan/3c13d3a746bb3ec5082f

    Usage:
        def wrapper_func():
            def nested_subprocess_func():
                ...

            with globalized(nested_subprocess_func), multiprocessing.Pool(NUMPROCESS) as pool:
                pool.map(nested_subprocess_func,...)

        if __name__ == "__main__":
            wrapper_func()
    """
    namespace = sys.modules[nested_subprocess_func.__module__]
    name, qualname = (
        nested_subprocess_func.__name__,
        nested_subprocess_func.__qualname__,
    )
    nested_subprocess_func.__name__ = nested_subprocess_func.__qualname__ = (
        f"_{name}_{uuid.uuid4().hex}"
    )
    setattr(namespace, nested_subprocess_func.__name__, nested_subprocess_func)
    try:
        yield
    finally:
        delattr(namespace, nested_subprocess_func.__name__)
        nested_subprocess_func.__name__, nested_subprocess_func.__qualname__ = (
            name,
            qualname,
        )


def split_length_to_start_and_chunklen(input_length, num_process):
    """
    Given length L and split into N chunks,
    return a list of tuples: [(start_id, length of this chunk)]

    Example:
        split_length_to_start_and_chunklen(10, 3) -> [(0, 4), (4, 3), (7, 3)]

    Usage:

        LARGE_OBJ = [...]
        def nested_subprocess_func(start, chunklen):
            for item in LARGE_OBJ[start:start+chunklen]
                ... # do something with this chunk of LARGE_OBJ

        with globalized(nested_subprocess_func), multiprocessing.Pool(NUMPROCESS) as pool:
            for result in pool.starmap(nested_subprocess_func, split_length_to_start_and_chunklen(len(LARGE_OBJ,NUMPROCESS),NUMPROCESS), chunksize=1):
                ... # do something with result

    """
    # Calculate the base size of each chunk and the remainder
    k, m = divmod(input_length, num_process)
    # Create the list of tuples
    ret = []
    start_id = 0
    for i in range(num_process):
        # Each chunk will have a length of k + 1 if i < m, otherwise k
        chunk_length = k + 1 if i < m else k
        ret.append((start_id, chunk_length))
        start_id += chunk_length
    return ret
