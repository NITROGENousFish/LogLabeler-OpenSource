import os

# * place your dataset folder here.
_opendata_feeder_dir = os.path.dirname(__file__)
ORGINAL_DATA_FOLDER = os.path.join(_opendata_feeder_dir, "original_datasets")
os.makedirs(ORGINAL_DATA_FOLDER, exist_ok=True)
PROCESSED_DATA_FOLDER = os.path.join(_opendata_feeder_dir, "data")
os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

SUPPORT_DATASET_NAME_LIST = ['BGL',"Thunderbird","Spirit","Liberty"]


def get_processed_dataset_folder(name):
    if name not in SUPPORT_DATASET_NAME_LIST:
        raise ValueError(f"Dataset {name} is not supported. Current supported datasets are {SUPPORT_DATASET_NAME_LIST}")
    _processed_dir = os.path.join(PROCESSED_DATA_FOLDER, f"output_{name}")
    if not os.path.exists(_processed_dir):
        raise RuntimeError(
            f"Dataset {_processed_dir} is not processed yet. Please run the processing script first."
        )
    return _processed_dir