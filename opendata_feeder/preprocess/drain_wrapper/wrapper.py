import flinkdrain3
import json
import sys
import os
import pickle
from tools import path_tools
from tools import file_tools


# semantic annotation


def fast_drain(l: list, fp=None, config_path=None):
    config = flinkdrain3.template_miner_config.TemplateMinerConfig()
    config.load(
        os.path.join(os.path.dirname(__file__), "drain.ini")
        if config_path == None
        else config_path
    )
    miner = flinkdrain3.TemplateMiner(None, config)
    print("begin mining")
    for log_line in l:
        result = miner.add_log_message(log_line)
        result_json = json.dumps(result)
        # print(result_json)
        template = result["template_mined"]
        # print("Parameters: " + str(params))
    for cluster in miner.drain.clusters:
        print(cluster, file=sys.stdout if not fp else fp)

    # #todo 这里的代码有问题
    # result= {"not found":[]}
    # for log_line in l:
    #     cluster = miner.match(log_line)
    #     if cluster is None:
    #         # print(f"No match found")
    #         result['not found'].append(log_line)
    #     else:
    #         template = cluster.get_template()
    #         if cluster.cluster_id not in result.keys():
    #             result[cluster.cluster_id] = [log_line]
    #         else:
    #             result[cluster.cluster_id].append(log_line)
    #         # print(f"Matched template #{cluster.cluster_id}: {template}")
    # res = {k:len(v) for k,v in result.items()}
    # print(f"{res}",file=sys.stdout if not fp else fp)
    return miner


def get_drain_miner():
    config = flinkdrain3.template_miner_config.TemplateMinerConfig()
    config.load(os.path.join(os.path.dirname(__file__), "drain.ini"))
    miner = flinkdrain3.TemplateMiner(None, config)
    return miner


""" HOW TO USE THE MINER
result = miner.add_log_message(log_line)
template = result["template_mined"]
params = miner.extract_parameters(template, log_line)
"""


def get_template_from_drain(drain_obj):
    """Dump template information from drain object
    Access to drain
        _ = list(drain_obj.drain.id_to_cluster.values())[0]
        ### _ is flinkdrain3.drain.LogCluster object
        cluster_id = _.cluster_id
        template_string = _.get_template()
        size = _.size
    """
    return [
        {
            "cluster_id": _log_cluster.cluster_id,
            "template": _log_cluster.get_template(),
            "size": _log_cluster.size,
        }
        for _log_cluster in drain_obj.drain.id_to_cluster.values()
    ]
