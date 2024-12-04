import warnings
import functools

import warnings

class MainWarningWaning(Warning): ...

def MainWarning(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn("WARN: This function can be only called as main, please mind the usage", MainWarningWaning)
        return func(*args, **kwargs)
    return wrapper