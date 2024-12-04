import os,sys

BGL_DATASET_FOLDER = config.BGL_DATASET_FOLDER

def is_anomaly(s:str):
    return not s.startswith("-")

class BGL_reader():
    def __init__(self,BGL_dir_path:str=BGL_DATASET_FOLDER,) -> None:
        assert os.path.isdir(BGL_dir_path)
        self.logfile = os.path.join(BGL_dir_path,"BGL.log")
        self.drain_miner
    def line_reader