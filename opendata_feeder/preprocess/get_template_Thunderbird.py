import os,sys,time,json,pickle,re
from tqdm import tqdm

import multiprocessing, logging
import queue
import shutil
import itertools

from typing import Callable
# raletive packages
from opendata_feeder.preprocess import drain_wrapper

# root packages
import flinkdrain3 as drain3
from tools import file_tools,csv_tools

NUM_PROCESS = 200
RUN_DATASET = sys.argv[1]
FLAG_ENABLE_FAST_TEST = False
FLAG_REMOVE_TEMP_DIR = False
PARAM_DICT = {
    "Thunderbird":{
        "LINE_NUMS":211212192,# wc -l LOG_DATA_PATH | | awk '{print $1}'
        "LOG_DATA_PATH":PROJ_ABS_FILEPATH+"./opendata_feeder/original_datasets/Thunderbird/Thunderbird.log",
        "DRIAN_CONFIG_PATH":PROJ_ABS_FILEPATH+"./opendata_feeder/data/drain_configs/drain_Thunderbird_simth08.ini",
        "RESULT_DIR":PROJ_ABS_FILEPATH+"./opendata_feeder/data/output_Thunderbird/"
    },
    "Spirit":{
        "LINE_NUMS":272298969,# wc -l LOG_DATA_PATH | | awk '{print $1}'
        "LOG_DATA_PATH":PROJ_ABS_FILEPATH+"./opendata_feeder/original_datasets/spirit2",
        "DRIAN_CONFIG_PATH":PROJ_ABS_FILEPATH+"./opendata_feeder/data/drain_configs/drain_Thunderbird_simth08.ini",
        "RESULT_DIR":PROJ_ABS_FILEPATH+"./opendata_feeder/data/output_Spirit/"
    },
    "Liberty":{
        "LINE_NUMS":265569231,# wc -l LOG_DATA_PATH | | awk '{print $1}'
        "LOG_DATA_PATH":PROJ_ABS_FILEPATH+"./opendata_feeder/original_datasets/liberty2",
        "DRIAN_CONFIG_PATH":PROJ_ABS_FILEPATH+"./opendata_feeder/data/drain_configs/drain_Thunderbird_simth08.ini",
        "RESULT_DIR":PROJ_ABS_FILEPATH+"./opendata_feeder/data/output_Liberty/"
    }
}

LINE_NUMS = PARAM_DICT[RUN_DATASET]["LINE_NUMS"]
LOG_DATA_PATH = PARAM_DICT[RUN_DATASET]["LOG_DATA_PATH"]
DRIAN_CONFIG_PATH = PARAM_DICT[RUN_DATASET]["DRIAN_CONFIG_PATH"]
RESULT_DIR = PARAM_DICT[RUN_DATASET]["RESULT_DIR"]

os.makedirs(RESULT_DIR,exist_ok=True)


def thunderbird_preprorcess(rawlog:str):
    # awk '!/^-/ {print NR, $0}' your_file.txt
    # 带有行号的输出不以 - 开头的行
    r = rawlog.strip()
    splited = r.split(" ",8)
    if len(splited) == 8:
        splited.append("ORGINAL LOG IS EMPTY")
    res_is_anomaly = False if splited[0] == '-' else True
    actual_log = splited[-1]
    machine_iid = splited[-2]
    return str(res_is_anomaly),str(machine_iid),str(actual_log)


def _get_a_multiprocess_queue():
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    return result_queue


def _parallel_framework(multiprocessing_function:Callable,work_pool_param_list:list,NUM_PROCESSES=150):
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        results = [pool.apply_async(multiprocessing_function,w) for w in work_pool_param_list]
        print('multi process pool running')

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
    print('multi process pool closed')


# * output: all_miner_info.json, template_id_hash.txt (aligned with the orginal dataset)

if __name__ == "__main__":
    # * inner public variables
    _INNER_NUM_PROCESSES = NUM_PROCESS
    # * at & slash: in machine_iid, no
    _RESULT_QUEUE_AT_ANOMALY = _get_a_multiprocess_queue()
    _RESULT_QUEUE_SLASH_ANOMALY = _get_a_multiprocess_queue()
    _RESULT_QUEUE_OTHER_ANOMALY = _get_a_multiprocess_queue()
    _RESULT_QUEUE_AT_NORMAL = _get_a_multiprocess_queue()
    _RESULT_QUEUE_SLASH_NORMAL = _get_a_multiprocess_queue()
    _RESULT_QUEUE_OTHER_NORMAL = _get_a_multiprocess_queue()

    _CONFIG = drain3.TemplateMinerConfig_ini_modified()
    _CONFIG.load(DRIAN_CONFIG_PATH)

    _LOG_MASKER = drain3.LogMasker(_CONFIG.masking_instructions, _CONFIG.mask_prefix, _CONFIG.mask_suffix) #* all matched regex will replace with blank

    stage1dir = os.path.join(RESULT_DIR,"tmp_stage1_bined")
    stage2dir = os.path.join(RESULT_DIR,"tmp_stage2_bined")
    os.makedirs(stage1dir,exist_ok=True)
    os.makedirs(stage2dir,exist_ok=True)

    def __each_multiprocessing_lines(filename:str, start:int, end:int, _RESULT_QUEUE_AT_ANOMALY,_RESULT_QUEUE_SLASH_ANOMALY,_RESULT_QUEUE_OTHER_ANOMALY,_RESULT_QUEUE_AT_NORMAL,_RESULT_QUEUE_SLASH_NORMAL,_RESULT_QUEUE_OTHER_NORMAL):
        print(f'Process: {os.getpid()} {start} {end}')
        total_lines = end - start

        atanomaly = []
        slashanomaly = []
        otheranomaly = []
        atnormal = []
        slashnormal = []
        othernormal = []

        progress_bar = tqdm(total=total_lines)
        with open(filename,"r", encoding='utf-8', errors='replace') as file:
            for i, line in enumerate(file):
                if i >= start and i < end:
                    # *** line logic begin
                    str_is_anomaly,machine_iid,log_line = thunderbird_preprorcess(line)
                    glo_iid = i
                    maskedlog = _LOG_MASKER.mask(log_line)

                    res_str = csv_tools.iterable_to_csv_single_line_with_escape([glo_iid,"template_id",str_is_anomaly,machine_iid,maskedlog])+"\n"
                    # bin process
                    if str_is_anomaly=="True":
                        if "@" in machine_iid:
                            atanomaly.append(res_str)
                        elif "/" in machine_iid:
                            slashanomaly.append(res_str)
                        else:
                            otheranomaly.append(res_str)
                    else:
                        if "@" in machine_iid:
                            atnormal.append(res_str)
                        elif "/" in machine_iid:
                            slashnormal.append(res_str)
                        else:
                            othernormal.append(res_str)
                    # *** line logic end
                    progress_bar.update()

        progress_bar.close()
        _RESULT_QUEUE_AT_ANOMALY.put(atanomaly)
        _RESULT_QUEUE_SLASH_ANOMALY.put(slashanomaly)
        _RESULT_QUEUE_OTHER_ANOMALY.put(otheranomaly)
        _RESULT_QUEUE_AT_NORMAL.put(atnormal)
        _RESULT_QUEUE_SLASH_NORMAL.put(slashnormal)
        _RESULT_QUEUE_OTHER_NORMAL.put(othernormal)

    def __generate_work_paramlist()->list:
        # 计算每个进程要处理的行数
        chunk_size = LINE_NUMS // _INNER_NUM_PROCESSES    

        _works = []
        for i in range(_INNER_NUM_PROCESSES):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < _INNER_NUM_PROCESSES - 1 else LINE_NUMS
            _works.append((LOG_DATA_PATH, start, end, _RESULT_QUEUE_AT_ANOMALY,_RESULT_QUEUE_SLASH_ANOMALY,_RESULT_QUEUE_OTHER_ANOMALY,_RESULT_QUEUE_AT_NORMAL,_RESULT_QUEUE_SLASH_NORMAL,_RESULT_QUEUE_OTHER_NORMAL))
        return _works

    # * Run multiprocessing
    with multiprocessing.Pool(processes=_INNER_NUM_PROCESSES) as pool:
        results = [pool.apply_async(__each_multiprocessing_lines,w) for w in __generate_work_paramlist()]
        print('multi process pool running')        
        while True:

            # * check states, # catch exception if results are not ready yet
            try:
                ready = [result.ready() for result in results]
                successful = [result.successful() for result in results]
            except Exception as e:
                # print(e) # continue report <multiprocessing.pool.ApplyResult object at 0x7f731fa03290> not ready, only if subprocess finished
                # * still some subprocess not finished, keep going...
                continue

            # raise exception reporting exceptions received from workers
            if all(ready) and not all(successful):
                raise Exception(f'Workers raised following exceptions {[result._value for result in results if not result.successful()]}')

            # exit loop if all tasks returned success
            if all(successful):
                print("All subprocess finish, Queue is empty, waiting...")
                time.sleep(0.5)
                break

    print('multi process pool closed')

    # * stage1 post process
    stage1dir = os.path.join(RESULT_DIR,"tmp_stage1_bined")
    os.makedirs(stage1dir,exist_ok=True)

    for name,q in zip(["AT_ANOMALY","SLASH_ANOMALY","OTHER_ANOMALY","AT_NORMAL","SLASH_NORMAL","OTHER_NORMAL"],[_RESULT_QUEUE_AT_ANOMALY,_RESULT_QUEUE_SLASH_ANOMALY,_RESULT_QUEUE_OTHER_ANOMALY,_RESULT_QUEUE_AT_NORMAL,_RESULT_QUEUE_SLASH_NORMAL,_RESULT_QUEUE_OTHER_NORMAL]):
        with open(os.path.join(stage1dir,f'{name}_noheader.csv'),'w') as fp:
            all_content = []
            while not q.empty():
                lines = q.get()
                all_content.extend([[int(l.split("|",1)[0]),l] for l in lines])
            sorted_content = sorted(all_content,key=lambda x: x[0])
            for i in sorted_content:
                fp.write(i[1])

    # * stage2 begin
    def drain_one(filepath,drain_hash:str):
        # * drain3 init
        config = drain3.TemplateMinerConfig_ini_modified()
        config.load(DRIAN_CONFIG_PATH)
        miner = drain3.TemplateMiner(None,config)

        # * read logs
        raw_loglist = []
        with open(filepath,"r") as fp:
            raw_loglist = fp.readlines()

        parsed_loglist = [csv_tools.csv_to_innerlist_single_line_with_escape(i) for i in raw_loglist]
        del raw_loglist

        print("=======begin mining=======")
        res_str_list = [] #* each with a dict {'log':log_line,'cluster_id':cluster_id}        
        counter = 0

        for logtuple in tqdm(parsed_loglist):
            # logtuple consists of [glo_iid,"template_id",str_is_anomaly,machine_iid,maskedlog]
            result = miner.add_log_message(logtuple[-1],flagSkipMask=True)
            cluster_id = drain_hash+":"+str(result["cluster_id"])
            logtuple[1] = cluster_id
            res_str_list.append(csv_tools.iterable_to_csv_single_line_with_escape(logtuple)+"\n")
            if FLAG_ENABLE_FAST_TEST:
                counter+=1
                if counter >=100000:
                    break
        return miner,res_str_list

    miner_list = []
    for name,q in zip(["AT_ANOMALY","SLASH_ANOMALY","OTHER_ANOMALY","AT_NORMAL","SLASH_NORMAL","OTHER_NORMAL"],[_RESULT_QUEUE_AT_ANOMALY,_RESULT_QUEUE_SLASH_ANOMALY,_RESULT_QUEUE_OTHER_ANOMALY,_RESULT_QUEUE_AT_NORMAL,_RESULT_QUEUE_SLASH_NORMAL,_RESULT_QUEUE_OTHER_NORMAL]):
        miner,res_str_list = drain_one(os.path.join(stage1dir,f'{name}_noheader.csv'),name)
        minerinfo = drain_wrapper.get_template_from_drain(miner)
        for item in minerinfo:
            item['cluster_id'] = name +":"+str(item['cluster_id'])
            miner_list.append(item)
        with open(os.path.join(stage2dir,f'{name}_noheader.csv'),'w') as fp:       
            fp.writelines(res_str_list)

    file_tools.json_save(miner_list,os.path.join(stage2dir,"all_miner_info.json"))

    print("All done")        

    # #* get machine iid begin
    # def get_machine_iid():
    #     def deal(filepath):
    #         #* read logs
    #         raw_loglist = []
    #         with open(filepath,"r") as fp:
    #             raw_loglist = fp.readlines()

    #         parsed_loglist = list(set([csv_tools.csv_to_innerlist_single_line_with_escape(i[3:-1])[3] for i in raw_loglist]))
    #         return parsed_loglist

    #     res_d = {}
    #     for name,q in zip(["AT_ANOMALY","SLASH_ANOMALY","OTHER_ANOMALY","AT_NORMAL","SLASH_NORMAL","OTHER_NORMAL"],[_RESULT_QUEUE_AT_ANOMALY,_RESULT_QUEUE_SLASH_ANOMALY,_RESULT_QUEUE_OTHER_ANOMALY,_RESULT_QUEUE_AT_NORMAL,_RESULT_QUEUE_SLASH_NORMAL,_RESULT_QUEUE_OTHER_NORMAL]):
    #         _ = deal(os.path.join(stage1dir,f'{name}_noheader.csv'))
    #         res_d[name] = _

    #     file_tools.json_save(res_d,os.path.join(stage2dir,"machine_iid_set.json"))

    # * step3 get raw line num to template id

    def _read_iid_and_template_tuple(filepath):
        raw_loglist = []
        with open(filepath,"r") as fp:
            raw_loglist = fp.readlines()
        def __inner(i):
            ret = i[:2]
            ret[0] = int(ret[0])
            return ret
        parsed_loglist = [__inner(csv_tools.csv_to_innerlist_single_line_with_escape(i)) for i in raw_loglist]
        del raw_loglist
        return parsed_loglist

    _all_list = [_read_iid_and_template_tuple(os.path.join(stage2dir,f'{name}_noheader.csv')) for name in ["AT_ANOMALY","SLASH_ANOMALY","OTHER_ANOMALY","AT_NORMAL","SLASH_NORMAL","OTHER_NORMAL"]]
    print("begin_sorted")
    sorted_hash = sorted(itertools.chain.from_iterable(_all_list),key=lambda x: x[0])
    with open(os.path.join(RESULT_DIR,"template_id_hash.txt"),'w') as fp:
        for i in sorted_hash:
            fp.write(i[1])
            fp.write("\n")


    #* clean all_miner_info.json to desired format

    # shutil.copy(os.path.join(stage2dir,"all_miner_info.json"),os.path.join(RESULT_DIR,"all_miner_info.json"))
    _all_miner_info = file_tools.json_load(os.path.join(stage2dir,"all_miner_info.json"))
    for item in _all_miner_info:
        item['is_anomaly'] = True if "ANOMALY" in item['cluster_id'] else False
        item['template_id'] = item['cluster_id']
        del item['cluster_id']
    
    file_tools.json_save(_all_miner_info,os.path.join(RESULT_DIR, "all_miner_info.json"))

    #* clean up
    if FLAG_REMOVE_TEMP_DIR:
        shutil.rmtree(stage1dir)
        shutil.rmtree(stage2dir)
