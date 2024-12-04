import json
import pickle
import functools
import os
import shutil
from .path_tools import get_temp_path
from .csv_tools import iterable_to_csv_single_line_with_escape,csv_to_innerlist_single_line_with_escape
import pandas as pd
from itertools import islice

class AdvancedJSONEncoder(json.JSONEncoder):
    """usage
    1. implement __jsonencode__ method to let the curren class be serializable
    2. add param when json.dump: cls=AdvancedJSONEncoder    
    """
    def default(self, obj):
        if hasattr(obj, '__jsonencode__'):
            return obj.__jsonencode__()

        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

def jsonl_save(content:object,path:str):
    with open(path,'w',encoding='utf-8') as fp:
        for line in content:
            json.dump(line,fp,ensure_ascii=False,cls=AdvancedJSONEncoder)
            fp.write('\n')

def jsonl_loadyeild(path:str):
    _ = None
    with open(path,'r',encoding='utf-8') as fp:
        for line in fp:
            data = json.loads(line)
            yield data

def jsonl_load(path:str):
    _ = []
    with open(path,'r',encoding='utf-8') as fp:
        for line in fp:
            _.append(json.loads(line))
    return _       

def json_save(content:object,path:str):
    with open(path,'w',encoding='utf-8') as fp:
        json.dump(content,fp,ensure_ascii=False,cls=AdvancedJSONEncoder)

def json_load(path:str):
    _ = None
    with open(path,'r',encoding='utf-8') as fp:
        _ = json.load(fp)
    return _


def csvl_save(content:object,path:str):
    with open(path,'w',encoding='utf-8') as fp:
        for line in content:
            fp.write(iterable_to_csv_single_line_with_escape(line))
            fp.write('\n')

def csvl_load(path:str):
    _ = []
    with open(path,'r',encoding='utf-8') as fp:
        for line in fp:
            _.append(csv_to_innerlist_single_line_with_escape(line))
    return _


def pickle_save(content,path):
    with open(path,'wb') as fp:
        pickle.dump(content,fp)

def pickle_load(path):
    _ = None
    with open(path,'rb') as fp:
        _ = pickle.load(fp)
    return _

def txtline_load(path):
    _ = None
    with open(path,'r') as fp:
        _ = fp.read().splitlines()
    return _


def txtline_load_with_bound(file_path, start_line, end_line):
    _ = []
    with open(file_path, "r") as fp:
        for line in islice(fp, start_line , end_line):            
            _.append(line.strip())  # 打印从第4行开始的内容
    return _

def txtline_save(content,path):
    with open(path,'w') as fp:
        for line in content:
            fp.write(str(line))
            fp.write('\n')


def tmp_pickle_save(content:object,name:str):
    pickle_save(content,os.path.join(get_temp_path(),name))

def tmp_pickle_load(name:str):
    return pickle_load(os.path.join(get_temp_path(),name))

def tmp_clear():
    for file in os.listdir(get_temp_path()):
        file_path = os.path.join(get_temp_path(), file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


# * annotation, mock print to a file
from contextlib import redirect_stdout
from io import StringIO
def print_to_file(file_path):
    """ annotation, mock print to a file

    Args:
        file_path (_type_): _description_
    Usage:
        @print_to_file(file_path)
        def func():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with StringIO() as buffer, redirect_stdout(buffer):
                func(*args, **kwargs)
                with open(file_path, 'w') as f:
                    f.write(buffer.getvalue())
        return wrapper
    return decorator


if __name__ == "__main__":
    # txtline_save([1,2,3,4],"tmp_test.txt")
    ...
