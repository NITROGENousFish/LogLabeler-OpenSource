# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 35678))
# ptvsd.wait_for_attach()

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



#! do not modify this function
def get_template_from_drain(drain_obj):
    """ 
    Dump template information from drain object
        Access to drain 
            _ = list(drain_obj.drain.id_to_cluster.values())[0]
            ### _ is drain3.drain.LogCluster object
            cluster_id = _.cluster_id
            template_string = _.get_template()
            size = _.size
    """
    
    
    
    return [
        {
            "cluster_id":_log_cluster.cluster_id,
            "template":_log_cluster.get_template(),
            "size":_log_cluster.size
        }
        for _log_cluster in drain_obj.drain.id_to_cluster.values()
    ]


SIMTH = 8
#* read thunderbird data
HADOOP_DATASET_FOLDER = config.HADOOP_DATASET_FOLDER
drain_config_path = os.path.join(config.HADOOP_ASSET_DIR,"drain_Hadoop_simth08.ini")



rregex  = re.compile(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2},\d{3}) (INFO|WARN|ERROR) (\[.*?\]) ([\w.]*?): ((?:.|\n)*?)(?=^(?:\d{4}-\d{2}-\d{2}))",re.M)
allres = []
for dirname in tqdm(os.listdir(HADOOP_DATASET_FOLDER)):
    if dirname == "README.md" or dirname == "abnormal_label.txt":
        continue
    for filename in os.listdir(os.path.join(HADOOP_DATASET_FOLDER,dirname)):
        absfilepath = os.path.join(os.path.join(HADOOP_DATASET_FOLDER,dirname),filename)
        
        with open(absfilepath,'r') as fp:
            content = fp.read()
            res_list = rregex.findall(content+"0000-00-00")
        _res = [
            {
                'dirname':dirname,
                'filename':filename,
                'date':ttuple[0],
                'time':ttuple[1],
                'level':ttuple[2],
                'domain':ttuple[3],
                'str_pkg':ttuple[4],
                'str_content':ttuple[5]
            }
            for ttuple in res_list
        ]
        allres.extend(_res)

infolog = [i['str_content'].split("\n")[0] for i in allres if i['level']=='INFO']
warnlog = [i['str_content'].split("\n")[0] for i in allres if i['level']=='WARN']
errorlog = [i['str_content'].split("\n")[0] for i in allres if i['level']=='ERROR']



def deal_loglist(loglist):
    config = flinkdrain3.template_miner_config.TemplateMinerConfig_ini_modified()
    config.load(drain_config_path)
    miner = flinkdrain3.TemplateMiner(None,config)
    for line in loglist:
        result = miner.add_log_message(line)
    rescontent = get_template_from_drain(miner)
    return [i['template'] for i in rescontent]

infodrain = deal_loglist(infolog)
warndrain = ['~ '+i for i in deal_loglist(warnlog)]
errordrain = ['- '+i for i in deal_loglist(errorlog)]

res = []
res.extend(infodrain)
res.extend(warndrain)
res.extend(errordrain)

with open(os.path.join(config.HADOOP_ASSET_DIR,"reduced-maskall-3label.txt"),'w+') as fp:
    for i in res:
        fp.write(i+"\n")