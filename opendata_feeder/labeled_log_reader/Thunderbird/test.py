import ptvsd
ptvsd.enable_attach(address = ('0.0.0.0', 35678))
ptvsd.wait_for_attach()

import os,sys,time
from labeled_log_reader.tools import file_tools
from labeled_log_reader.parsing import drain_wrapper

from labeled_log_reader import config

import json
import pickle
from tqdm import tqdm
import csv
ENABLE_FAST_TEST = False
import multiprocessing
import re

import flinkdrain3


def step2_drain_one_file(filename:str,drain_config_path):
    if True:
    # try:
        #* init miner
        config = flinkdrain3.template_miner_config.TemplateMinerConfig_ini_modified()
        config.load(drain_config_path)
        miner = flinkdrain3.TemplateMiner(None,config)
        
        # all_log_res = [] #* each with a dict {'log':log_line,'cluster_id':cluster_id}
        print(f'Process Start: {os.getpid()} {filename}')
        
        process_date_id = re.compile(r" \.\. \w*? [A-z]{3,4}  :: (?:(\w*?)\/\1|.*?@.*?) ")
        with open(filename,"r", encoding='utf-8', errors='replace') as file:
            for line in tqdm(file):
                #! TBD
                line = process_date_id.sub("",line)
                result = miner.add_log_message(line,flagSkipMask=True)
                # cluster_id = result["cluster_id"]
                # all_log_res.append({'log':line,'cluster_id':cluster_id})
        # rescontent = get_template_from_drain(miner)
        # respipe.put([i['template'] for i in rescontent])
        print(f'Process Finish: {os.getpid()} {filename}')
    # except Exception as e:
    #     print(f'Process Error: {os.getpid()} {filename}')
    #     raise Exception(f'Process Error: {os.getpid()} {filename}: {e}')

step2_drain_one_file("/home/lizongyang/hdd-lizongyang/LOG_ARENA/labeled_log_reader/asset/Thunderbird_asset/two_letter_idx_files/anomaly_log_dir_inner/-##R.txt","/home/lizongyang/hdd-lizongyang/LOG_ARENA/labeled_log_reader/asset/Thunderbird_asset/drain_Thunderbird_simth08.ini")

