import os
import sys
import shutil
from typing import NamedTuple

fastpathnametuple = NamedTuple("fastpathnametuple", [
    ("raw", str),
    ("absdir",str),
    ("dir", str),
    ("filename_extention", str),
    ("f_e", str),
    ("filename", str),
    ("f", str),
    ("dot_extention", str),
    ("dote", str),
    ("extention", str),
    ("e", str)
])

def fastpath(p: str):
    absp = os.path.abspath(p)
    f_e = os.path.basename(p)
    f, e = os.path.splitext(f_e)
    return fastpathnametuple(
        p,
        os.path.dirname(absp),
        os.path.dirname(p),
        f_e,   # with extention
        f_e,   # with extention
        f,
        f,
        e,
        e,
        e[1:],
        e[1:]
    )

def get_temp_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__),"../tmp"))
