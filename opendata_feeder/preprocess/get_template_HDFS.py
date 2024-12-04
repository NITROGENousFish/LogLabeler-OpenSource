# library packages
import os, sys
from tqdm import tqdm

# raletive packages
from opendata_feeder.preprocess import drain_wrapper

# root packages
import flinkdrain3 as drain3
from tools import file_tools

ENABLE_FAST_TEST = False


def hdfs_preprorcess(rawlog: str):
    rawlog = rawlog.strip()
    res_log = rawlog.split(" ", 5)
    res_is_anomaly = res_log[3]
    return res_is_anomaly, res_log[-1]


def main_miner(hdfs_log_data, drain_config_path):
    # * drain3 init
    config = drain3.TemplateMinerConfig_ini_modified()
    config.load(drain_config_path)
    miner = drain3.TemplateMiner(None, config)

    # * read logs
    raw_loglist = []
    with open(hdfs_log_data, "r") as fp:
        raw_loglist = fp.readlines()

    print("=======begin mining=======")
    res_template_id_hash = []  # list of cluster_id
    drain_cluster_id_to_is_anomaly = {}
    counter = 0
    for rawlog in tqdm(raw_loglist):
        is_anomaly, log_line = hdfs_preprorcess(rawlog)  # * preprocess BGL
        result = miner.add_log_message(log_line)
        cluster_id = int(result["cluster_id"])
        res_template_id_hash.append(cluster_id)
        drain_cluster_id_to_is_anomaly[cluster_id] = is_anomaly

        if ENABLE_FAST_TEST:
            counter += 1
            if counter >= 10000:
                break

    return miner, res_template_id_hash, drain_cluster_id_to_is_anomaly


if __name__ == "__main__":
    hdfs_log_data = PROJ_ABS_FILEPATH+"./opendata_feeder/original_datasets/HDFS_v1/HDFS.log"
    drain_config_path = PROJ_ABS_FILEPATH+"./opendata_feeder/data/drain_configs/drain_BGL_simth08.ini"
    BGL_WORK_FOLDER = (
        PROJ_ABS_FILEPATH+"./opendata_feeder/data/output_HDFS"
    )
    os.makedirs(BGL_WORK_FOLDER, exist_ok=True)
    miner, res_template_id_hash, _drain_cluster_id_to_is_anomaly = main_miner(
        hdfs_log_data, drain_config_path
    )

    file_tools.pickle_save(miner, os.path.join(BGL_WORK_FOLDER, "miner.pkl"))

    file_tools.txtline_save(
        res_template_id_hash, os.path.join(BGL_WORK_FOLDER, "template_id_hash.txt")
    )

    # * save all_miner_info
    all_miner_info = drain_wrapper.get_template_from_drain(miner)
    for i in all_miner_info:
        # * add is_anomaly to all_miner_info
        i["is_anomaly"] = _drain_cluster_id_to_is_anomaly[i["cluster_id"]]
        i["template_id"] = i["cluster_id"]
        del i["cluster_id"]
    file_tools.json_save(
        all_miner_info, os.path.join(BGL_WORK_FOLDER, "all_miner_info.json")
    )
