import sys

import os, gc, math, abc, pickle, argparse
import random
import time
import hashlib
import numpy as np
import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from tqdm import *
import regex as re
import logging
from collections import Counter
from torch.nn.parameter import Parameter
from multiprocessing import Manager, Pool
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
seed = 6
random.seed(seed)  # Python random module.
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Device configuration

SESSION = hashlib.md5(
    time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 8 * 60 * 60)).encode('utf-8')).hexdigest()
SESSION = 'SESSION_' + SESSION

def GET_PROJECT_ROOT():
    return os.path.dirname(os.path.abspath(__file__))
    # _pythonpath = os.getenv("PYTHONPATH")
    # if _pythonpath != None:
    #     return _pythonpath
    # else:
    #     raise EnvironmentError('Please set PYTHONPATH in your environment variables.')


def GET_LOGS_ROOT():
    log_file_root = os.path.join(GET_PROJECT_ROOT(), 'logs')
    if not os.path.exists(log_file_root):
        os.makedirs(log_file_root)
    return log_file_root


LOG_ROOT = GET_LOGS_ROOT()
PROJECT_ROOT = GET_PROJECT_ROOT()
pretrained_mode_path = os.path.join(PROJECT_ROOT, 'outputs/models/pretrain')
if not os.path.exists(pretrained_mode_path):
    os.makedirs(pretrained_mode_path)
os.environ['TRANSFORMERS_CACHE'] = pretrained_mode_path
os.environ['TRANSFORMERS_OFFLINE'] = '1'
