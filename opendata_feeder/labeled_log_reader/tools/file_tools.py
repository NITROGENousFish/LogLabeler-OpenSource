import json
import pickle
import functools
import os
import shutil
from src.tools.path_tools import get_temp_path
import pandas as pd
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

#*** json

def json_save(content:object,path:str,append=False):
    with open(path,'a+' if append else 'w',encoding='utf-8') as fp:
        json.dump(content,fp,ensure_ascii=False,cls=AdvancedJSONEncoder)

def json_load(path:str):
    _ = None
    with open(path,'r',encoding='utf-8') as fp:
        _ = json.load(fp)
    return _

#*** jsonline    
 
def jsonline_save(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False,cls=AdvancedJSONEncoder)
            f.write(json_record + '\n')

def jsonline_load(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data

def jsonline_load_iter(input_path):
    """
    Read list of objects from a JSON lines file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line.rstrip('\n|\r'))



#*** pickle

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






#* annotation, mock print to a file
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