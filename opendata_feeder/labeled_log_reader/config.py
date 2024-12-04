import os

# * place your dataset folder here.

DATASET_FOLDER = PROJ_ABS_FILEPATH+"./opendata_feeder" 

BGL_DATASET_FOLDER = os.path.join(DATASET_FOLDER, "BGL")
HADOOP_DATASET_FOLDER = os.path.join(DATASET_FOLDER, "Hadoop")
HDFS_V1_DATASET_FOLDER = os.path.join(DATASET_FOLDER, "HDFS_v1")
HDFS_V3_TRACEBENCH_DATASET_FOLDER = os.path.join(DATASET_FOLDER, "HDFS_v3_TraceBench")
OPENSTACK_DATASET_FOLDER = os.path.join(DATASET_FOLDER, "OpenStack")
THUNDERBIRD_DATASET_FOLDER = os.path.join(DATASET_FOLDER, "Thunderbird")

_TOOL_DIR = os.path.dirname(__file__)
ASSET_DIR = os.path.join(_TOOL_DIR, "asset")
os.makedirs(ASSET_DIR, exist_ok=True)
BGL_ASSET_DIR = os.path.join(ASSET_DIR, "BGL_asset")
os.makedirs(BGL_ASSET_DIR, exist_ok=True)
HADOOP_ASSET_DIR = os.path.join(ASSET_DIR, "Hadoop_asset")
os.makedirs(HADOOP_ASSET_DIR, exist_ok=True)
HDFS_V1_ASSET_DIR = os.path.join(ASSET_DIR, "HDFS_v1_asset")
os.makedirs(HDFS_V1_ASSET_DIR, exist_ok=True)
HDFS_V3_TRACEBENCH_ASSET_DIR = os.path.join(ASSET_DIR, "HDFS_v3_TraceBench_asset")
os.makedirs(HDFS_V3_TRACEBENCH_ASSET_DIR, exist_ok=True)
OPENSTACK_ASSET_DIR = os.path.join(ASSET_DIR, "OpenStack_asset")
os.makedirs(OPENSTACK_ASSET_DIR, exist_ok=True)
THUNDERBIRD_ASSET_DIR = os.path.join(ASSET_DIR, "Thunderbird_asset")
os.makedirs(THUNDERBIRD_ASSET_DIR, exist_ok=True)


# if __name__ == "__main__":
#     print(CURRENT_DATASET_FOLDER)
#     breakpoint()
