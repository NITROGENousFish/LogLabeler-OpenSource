import re
from tqdm import tqdm

# raletive packages
from opendata_feeder.preprocess import drain_wrapper

# root packages
...

ENABLE_FAST_TEST = False

# awk '!/^-/ {print NR, $0;exit 0}' opendata_feeder/original_datasets/Thunderbird/Thunderbird.log

THUNDERBIRD_LINE_NUMS = 211212192
THUNDERBIRD_LOG_DATA_PATH = PROJ_ABS_FILEPATH+"./opendata_feeder/original_datasets/Thunderbird/Thunderbird.log"
THUNDERBIRD_DRIAN_CONFIG_PATH = PROJ_ABS_FILEPATH+"./opendata_feeder/data/drain_configs/drain_Thunderbird_simth08.ini"
RESULT_DIR = PROJ_ABS_FILEPATH+"./opendata_feeder/data/output_Thunderbird/"



def thunderbird_preprorcess(rawlog:str):
    # awk '!/^-/ {print NR, $0}' your_file.txt
    r = rawlog.strip()
    splited = r.split(" ",8)
    res_is_anomaly = False if splited[0] == '-' else True
    actual_log = splited[-1]
    machine_iid = splited[-2]
    return str(res_is_anomaly),str(machine_iid),str(actual_log)


reg = re.compile(r'\d{2}:\d{2}:\d{2}')
with open(THUNDERBIRD_LOG_DATA_PATH,"r", encoding='utf-8', errors='replace') as file:
    for i, line in tqdm(enumerate(file)):
        _1,_2,_3 = thunderbird_preprorcess(line)
        if reg.match(_2):
            print(i+1)