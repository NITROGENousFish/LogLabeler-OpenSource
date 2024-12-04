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





#* subprocess main
def process_lines(filename:str, start:int, end:int, log_masker, result:multiprocessing.Queue):
    lines = []
    print(f'Process: {os.getpid()} {start} {end}')
    total_lines = end - start
    progress_bar = tqdm(total=total_lines)
    with open(filename,"r", encoding='utf-8', errors='replace') as file:
        for i, line in enumerate(file):
            if i >= start and i < end:
                res = log_masker.mask(line)
                lines.append(res)
                progress_bar.update()
    result.put(lines)
    progress_bar.close()




def step2_drain_one_file_normal(filename:str,drain_config_path,respipe:multiprocessing.Queue):
    #* init miner
    config = flinkdrain3.template_miner_config.TemplateMinerConfig_ini_modified()
    config.load(drain_config_path)
    miner = flinkdrain3.TemplateMiner(None,config)
    
    # all_log_res = [] #* each with a dict {'log':log_line,'cluster_id':cluster_id}
    print(f'Process Start: {os.getpid()} {filename}')
    
    process_date_id = re.compile(r" \.\. \w*? [A-z]{3,4}  :: (?:(\w*?)\/\1|.*?@.*?) ")
    
    with open(filename,"r", encoding='utf-8', errors='replace') as file:
        for line in tqdm(file):
            if len(line)>=2:
                line = process_date_id.sub("",line)
                result = miner.add_log_message(line,flagSkipMask=True)
            # cluster_id = result["cluster_id"]
            # all_log_res.append({'log':line,'cluster_id':cluster_id})
    rescontent = get_template_from_drain(miner)
    respipe.put([i['template'] for i in rescontent])
    print(f'Process Finish: {os.getpid()} {filename}')
    
def step2_drain_one_file_anomaly(filename:str,drain_config_path,respipe:multiprocessing.Queue):
    #* init miner
    config = flinkdrain3.template_miner_config.TemplateMinerConfig_ini_modified()
    config.load(drain_config_path)
    miner = flinkdrain3.TemplateMiner(None,config)
    
    # all_log_res = [] #* each with a dict {'log':log_line,'cluster_id':cluster_id}
    print(f'Process Start: {os.getpid()} {filename}')
    
    process_date_id = re.compile(r" \.\. \w*? [A-z]{3,4}  :: (?:(\w*?)\/\1|.*?@.*?) ")
    
    with open(filename,"r", encoding='utf-8', errors='replace') as file:
        for line in tqdm(file):
            if len(line)>=2:
                line = line[2:]
                line = process_date_id.sub("",line)
                result = miner.add_log_message(line,flagSkipMask=True)
                # cluster_id = result["cluster_id"]
                # all_log_res.append({'log':line,'cluster_id':cluster_id})
    rescontent = get_template_from_drain(miner)
    respipe.put(["- "+i['template'] for i in rescontent])
    print(f'Process Finish: {os.getpid()} {filename}')

def step2_drain_one_file_special(filename:str,drain_config_path,respipe:multiprocessing.Queue):
    #* init miner
    config = flinkdrain3.template_miner_config.TemplateMinerConfig_ini_modified()
    config.load(drain_config_path)
    miner = flinkdrain3.TemplateMiner(None,config)
    
    # all_log_res = [] #* each with a dict {'log':log_line,'cluster_id':cluster_id}
    print(f'Process Start: {os.getpid()} {filename}')
    
    
    re1 = re.compile(r"(?<=\W)\w*?(?:[A-Za-z]\d|\d[A-Za-z])\w*?(?=\W)")
    re2 = re.compile(r"## ##\/##")
    with open(filename,"r", encoding='utf-8', errors='replace') as file:
        for line in tqdm(file):
            if len(line)>=2:
                line = line[2:]
                line = re1.sub("",line)
                line = re2.sub("",line)
                result = miner.add_log_message(line,flagSkipMask=True)
            # cluster_id = result["cluster_id"]
            # all_log_res.append({'log':line,'cluster_id':cluster_id})
    rescontent = get_template_from_drain(miner)
    respipe.put(["- "+i['template'] for i in rescontent])
    print(f'Process Finish: {os.getpid()} {filename}')


if __name__ == "__main__":
    SIMTH = 8
    #* read thunderbird data
    THUNDERBIRD_DATASET_FOLDER = config.THUNDERBIRD_DATASET_FOLDER
    thunderbird_log_data_path = os.path.join(THUNDERBIRD_DATASET_FOLDER,"Thunderbird.log")
    
    _asset_dir = config.THUNDERBIRD_ASSET_DIR
    drain_config_path = os.path.join(_asset_dir,f"drain_Thunderbird_simth0{SIMTH}.ini")
    thunderbird_template_path = os.path.join(_asset_dir,f"template_mined-simth0{SIMTH}.json")
    simplified = os.path.join(_asset_dir,f"simple-template_mined-simth0{SIMTH}.json")
    miner_path = os.path.join(_asset_dir,f"miner-simth0{SIMTH}.pkl")
    drain_id_to_simplified_hash_path = os.path.join(_asset_dir,f"drain_id_to_simplified_id_hash-simth0{SIMTH}.json")
    two_letter_idx_folder = os.path.join(_asset_dir,'two_letter_idx_files')
    
    
    os.makedirs(two_letter_idx_folder,exist_ok=True)
    anomaly_log_dir_inner = os.path.join(two_letter_idx_folder,"anomaly_log_dir_inner");os.makedirs(anomaly_log_dir_inner,exist_ok=True)
    normal_log_dir_inner = os.path.join(two_letter_idx_folder,"normal_log_dir_inner");os.makedirs(normal_log_dir_inner,exist_ok=True)

    

    #**** begin of 1 *****************************************************************************************************
    def load_drain3_config(path):
        config = flinkdrain3.template_miner_config.TemplateMinerConfig_ini_modified()
        config.load(path)
        return config
    _conf = load_drain3_config(drain_config_path)

    from flinkdrain3.masking import LogMasker
    
    #* all matched regex will replace with blank
    log_masker = LogMasker(_conf.masking_instructions, _conf.mask_prefix, _conf.mask_suffix)
    
    
    
    #* main
    NUM_PROCESSES = 150
    
    #* read logs
    # def count_lines(path):
    #     total_lines = 0
    #     with open(path,"r", encoding='utf-8', errors='replace')  as fp:
    #         for count, line in tqdm(enumerate(fp)):
    #             total_lines = count
    #     total_lines += 1
    #     return total_lines
    # total_lines = count_lines(thunderbird_log_data_path)
    total_lines = 211212192
    
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        # 计算每个进程要处理的行数
        chunk_size = total_lines // NUM_PROCESSES
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()
    
        
        WORKS = []
        for i in range(NUM_PROCESSES):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < NUM_PROCESSES - 1 else total_lines
            WORKS.append((thunderbird_log_data_path, start, end, log_masker, result_queue))
        
        results = [pool.apply_async(process_lines,w) for w in WORKS]

        print('pool running')

        while True:
            time.sleep(1)
            # catch exception if results are not ready yet
            try:
                ready = [result.ready() for result in results]
                successful = [result.successful() for result in results]
                # print('123')
            except Exception as e:
                # print(e)
                continue
            # exit loop if all tasks returned success
            if all(successful):
                break
            # raise exception reporting exceptions received from workers
            if all(ready) and not all(successful):
                raise Exception(f'Workers raised following exceptions {[result._value for result in results if not result.successful()]}')

    print('pool closed')

    # 收集所有进程的结果
    all_lines = []
    while not result_queue.empty():
        lines = result_queue.get()
        all_lines.extend(lines)
        
    with open(os.path.join(_asset_dir,'maskall.txt'),'w') as fp:
        fp.writelines(all_lines)
        

    # 处理你的文件行数据
    fp_hash_normal = {}
    fp_hash_anomaly = {}
    def get_first_wd(line):
        _ = line.split()
        
        return _[0] != '-',re.sub(r'[\\/*?:"<>|]', '_', "".join(i[0] for i in _)[:4])
    for line in tqdm(all_lines):
        isnormal,_word_hash = get_first_wd(line)
        if isnormal:
            if _word_hash not in fp_hash_normal.keys():
                fp_hash_normal[_word_hash] = open(os.path.join(normal_log_dir_inner,f'{_word_hash}.txt'),'w')
            fp_hash_normal[_word_hash].write(line)
            fp_hash_normal[_word_hash].write("\n")
      
        else: #anmomaly
            if _word_hash not in fp_hash_anomaly.keys():
                fp_hash_anomaly[_word_hash] = open(os.path.join(anomaly_log_dir_inner,f'{_word_hash}.txt'),'w')
            fp_hash_anomaly[_word_hash].write(line)
            fp_hash_anomaly[_word_hash].write("\n")
    for k,v in fp_hash_normal.items():
        v.close()
    for k,v in fp_hash_anomaly.items():
        v.close()    
    #**** end of 1 *****************************************************************************************************
    
    
    
    # #**** begin of 2 *****************************************************************************************************    

    #* main
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    WORKS_NORMAL = [(os.path.join(normal_log_dir_inner,filename),drain_config_path,result_queue) for filename in os.listdir(normal_log_dir_inner)]
    WORKS_ANOMALY = [(os.path.join(anomaly_log_dir_inner,filename),drain_config_path,result_queue) for filename in os.listdir(anomaly_log_dir_inner) if "-##p.txt" not in filename]
    #* only one
    WORKS_SPECIAL = [(os.path.join(anomaly_log_dir_inner,filename),drain_config_path,result_queue) for filename in os.listdir(anomaly_log_dir_inner) if "-##p.txt" in filename]
    
    def generate_drain_works(dirpath):
        return 

    NUM_PROCESSES = 245 # len of total works
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        
        results = []
        results.extend([pool.apply_async(step2_drain_one_file_normal,w) for w in WORKS_NORMAL])
        results.extend([pool.apply_async(step2_drain_one_file_anomaly,w) for w in WORKS_ANOMALY])
        results.extend([pool.apply_async(step2_drain_one_file_special,w) for w in WORKS_SPECIAL])

        print('pool running')

        while True:
            time.sleep(1)
            # catch exception if results are not ready yet
            try:
                ready = [result.ready() for result in results]
                successful = [result.successful() for result in results]
                # print('123')
            except Exception as e:
                # print(e)
                continue
            # exit loop if all tasks returned success
            if all(successful):
                break
            # raise exception reporting exceptions received from workers
            if all(ready) and not all(successful):
                raise Exception(f'Workers raised following exceptions {[result._value for result in results if not result.successful()]}')

    print('pool closed')

    # 收集所有进程的结果
    all_lines = []
    while not result_queue.empty():
        lines = result_queue.get()
        all_lines.extend(lines)

    with open(os.path.join(_asset_dir,'simple-template_mined-simth08.txt'),'w') as fp:
        for i in all_lines:
            fp.write(i)
            fp.write("\n")
    # #**** end of 2 *****************************************************************************************************
    
    #**** start of 2 *****************************************************************************************************


    r = []
    with open(os.path.join(_asset_dir,'simple-template_mined-simth08.txt'),'r') as fp:
        r = fp.readlines()

    res = []
    for i in tqdm(r):
        _i = i.replace("<~*~>","")
        _i = _i.replace(" ## ##/## ","")

        #* remove patterns like r".. ibr2-northern Jan :: ibr2-northern/ibr2-northern "
        _i = re.sub(r"(?<=\s)\.\.\s*?[\w-]*? [A-z]{3,4}\s*?::\s*?[\w-]*?(?:@|\/)[\w-]*?(?=\s)", '', _i)
        #* remove username like jAIHfWTE026352@aaa(\.sdasdsf)?
        _i = re.sub(r"(?<=\W)\w{1,}?(?:[A-z][0-9]|[0-9][A-z])\w{1,}?@\w{1,}?(\.\w{1,}?)?(?=\W)",'',_i)
        #* remove Hashes like jAIHfWTE026352
        _i = re.sub(r"(?<=\W)\w{1,}?(?:[A-z][0-9]|[0-9][A-z])\w{1,}?(?=\W)",'',_i)
        _i = re.sub(r"(?<=\W)[A-z]\d{2,}?(?=\W)",'',_i)
        _i = re.sub(r"(?<=\W)\d{2,}?[A-z](?=\W)",'',_i)
        _i = re.sub(r"(?<=\W)[A-z]\d{2,}?[A-z](?=\W)",'',_i)
        #* remove sizenum like 21312kB
        _i = re.sub(r"(?<=\W)\d{3,}?[kK][bB](?=\W)",'',_i)
        #* remove patterns like r'..  Feb ::  '
        _i = re.sub(r"\.\.  [A-z]{3,4} ::  ",'',_i)
        
        res.append(_i)

    res = list(set(res))

    p = os.path.join(_asset_dir,'reduced-maskall.txt')
    # p = "/Users/nitrogenousfish/Desktop/Code/tmp_postprocess/sendmail.txt"

    with open(p,'w+') as fp:
        for i in res:
            fp.write(i)
    # #**** end of 2 *****************************************************************************************************