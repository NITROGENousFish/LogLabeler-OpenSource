# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 35678))
# ptvsd.wait_for_attach()

import os,sys
from labeled_log_reader.tools import file_tools
from labeled_log_reader.parsing import drain_wrapper

from labeled_log_reader import config
import drain3
import json
import pickle
from tqdm import tqdm

ENABLE_FAST_TEST = False


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


def is_anomaly(s:str):
    return not s.startswith("-")



# SIMTH:int = sys.argv[1]

SIMTH = 8

#* read BGL data
BGL_DATASET_FOLDER = config.BGL_DATASET_FOLDER
bgl_log_data = os.path.join(BGL_DATASET_FOLDER,"BGL.log")

_asset_dir = config.BGL_ASSET_DIR
drain_config_dir = os.path.join(_asset_dir,"BGL_drain_configs")
drain_config_path = os.path.join(drain_config_dir,f"drain_BGL_simth0{SIMTH}.ini")
bgl_template_path = os.path.join(_asset_dir,f"template_mined-simth0{SIMTH}.json")
simplified = os.path.join(_asset_dir,f"simple-template_mined-simth0{SIMTH}.json")
miner_path = os.path.join(_asset_dir,f"miner-simth0{SIMTH}.pkl")
maskall_txt = os.path.join(_asset_dir,f"reduced-maskall.txt")
drain_id_to_simplified_hash_path = os.path.join(_asset_dir,f"drain_id_to_simplified_id_hash-simth0{SIMTH}.json")


# #** Miner process ***************************************************************************
# #* read logs
# all_log = []
# with open(bgl_log_data,"r") as fp:
#     all_log = fp.readlines()

# #* init miner
# config = drain3.template_miner_config.TemplateMinerConfig()
# config.load(drain_config_path)
# miner = drain3.TemplateMiner(None,config)

# all_log_res = [] #* each with a dict {'log':log_line,'cluster_id':cluster_id}
# print("begin mining")
# counter = 0
# for log_line in tqdm(all_log):
#     result = miner.add_log_message(log_line)    
#     cluster_id = result["cluster_id"]
#     all_log_res.append({'log':log_line,'cluster_id':cluster_id})
#     if ENABLE_FAST_TEST:
#         counter+=1
#         if counter >=100000:
#             break
    
# rescontent = get_template_from_drain(miner)
# file_tools.pickle_save(miner,miner_path)
# file_tools.json_save(rescontent,bgl_template_path)
# file_tools.pickle_save(all_log_res,os.path.join(_asset_dir,f'BGLsim0{SIMTH}-all_log_res.pkl'))
# #*****************************************************************************

# d = file_tools.json_load(bgl_template_path)

# import re
# r = []


# def further_reduce(template_before:str) -> str:
#     """reduce
#     """
#     _ = re.sub(r":>.<:","",template_before)   #* remove connected placeholders combine placeholders as much as possible with 1 token split
#     _ = re.sub(r" ?:[a-f0-9]{8}","",_)
#     _ = re.sub(r"<:.*?:>", "", _)             #* remove connected placeholders
#     return _



# reduced_template_to_drain_hash = {}

# reduced_template_to_reduced_id = {}

# r = []
# for k in d:
#     l = k['template']
#     ccc = k['cluster_id']
#     sss = k['size']

#     res = further_reduce(k['template'])
#     k['reduced_template'] = res
#     r.append(res)


# reduced_template_to_reduced_id = {r:iid for  iid,r in enumerate(list(set(r)))}

# for k in d:
#     k['reduced_cluster_id'] = reduced_template_to_reduced_id[k['reduced_template']]


# file_tools.json_save({iid:r for iid,r in enumerate(list(set(r)))},simplified)

# file_tools.json_save({k['cluster_id']:k['reduced_cluster_id'] for k in d},drain_id_to_simplified_hash_path)



# #************************


# drain_id_to_simplified_hash = file_tools.json_load(drain_id_to_simplified_hash_path)

# miner = file_tools.pickle_load(miner_path)
# all_log_res = file_tools.pickle_load(os.path.join(_asset_dir,f'BGLsim0{SIMTH}-all_log_res.pkl'))

# header = ['template_id','log','is_anomaly']


# direct_num = 0
# matchres_none_num = 0
# matchres_success_num = 0




# with open(os.path.join(_asset_dir,f'BGLsim0{SIMTH}.jsonl'), 'w',encoding='utf-8', newline='') as f:
#     for dic in tqdm(all_log_res):
#         log = dic['log']
#         cluster_id = dic['cluster_id']

#         if str(cluster_id) in drain_id_to_simplified_hash.keys():
#             reduced_cluster_id = drain_id_to_simplified_hash[str(cluster_id)];direct_num+=1
#         else: #* should not run the following code, unless max_cluster is limited (lead to LogClusterCache is on)
#             matchres = miner.match(log,full_search_strategy="fallback")
#             if matchres == None:
#                 matchres_none_num += 1
#             else:
#                 reduced_cluster_id = drain_id_to_simplified_hash[str(matchres.cluster_id)]
#                 matchres_success_num += 1
#         json_record = json.dumps([reduced_cluster_id,log,is_anomaly(log)], ensure_ascii=False)
#         f.write(json_record + '\n')

# print(direct_num,matchres_none_num,matchres_success_num)





with open(maskall_txt,'w+') as fp:
    for k,v in file_tools.json_load(simplified).items():
        fp.write(v+"\n")
    

