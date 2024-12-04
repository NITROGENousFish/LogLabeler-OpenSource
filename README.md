# Project overview

- `baselines` is for evaluation
- `flinkdrain3` is a self-modified drain tool that served as an infrastructure for other modules
- `opendata_feeder` is for preprocessing datasets and feed data into evaluations
- `opensource_experiment` if for evaluating RQ1&2 in Opensource dataset.
    - Since we cannot opensource core processing logic due to commercial restrictions. In this folder, we only provide label generation logic **without** detailed functional implementation. This part of the code cannot be run.
- `tools` is for usefull tools.
- 

# About data preprocessing

We construct opendata_feeder for preprocessing all open source data.

- The results of each dataset are written into the `opendata_feeder/data` folder. Since the processing results of the log files for all files would be extremely large, we only keep the original parser configuration files and the miner info obtained from parser preprocessing to prove that our preprocessing steps have been executed (see **After Prorprocess Format**). We will upload the processed data to a separate link (TBD) later. 
    - You can also re-execute the preprocessing steps according to the following steps to reproduce our work:
        1. Place the zip files of all datasets in the `opendata_feeder/original_datasets`, and directly unzip them
        2. Run `get_template_{datasetname}.py` for preprocess. 
- :warning: in the subsequent processing, `opendata_feeder/data_feeder/glove.6B.300d.txt` will be used for embedding word vectors. Please pay attention to the configuration in **Usage For Running Evaluation**. 

## After Preprocess Format

For each dataset, we have the following structure:

```shell
opendata_feeder
└── data
    └── output_<dataset_name>
        ├── all_miner_info.json
        └── template_id_hash.txt
    └── ...
```

In `all_miner_info.json`:

```json
[
    {
        "template": str,
        "size": int,
        "is_anomaly": bool,
        "template_id": str
    }
]
```

In `template_id_hash.txt`: eachline is the template_id of the original dataset.


# Usage For Running Evaluation

Please  link `glove.6B.300d.txt` inside the following folder:

1. `baselines/logdeep_base/logdeep/data` for logdeep；used in `baselines/logdeep_base/logdeep/tools/embedding.py`
2. `opendata_feeder/data_feeder` for opendata_feeder

You can download  `glove.6B.300d.txt` from [Stanford NLP word embeddings](https://nlp.stanford.edu/projects/glove/)



Please build necessary runtime for three baselines, we provide a Dockerfile in `baselines/Dockerfile` for easy build ups, see the following commands:

```shell
docker build -t baseline:main 

docker run --gpus all --name container_name -v $(pwd):/workcenter -p 49869:5678 -dit baseline:main # 5678 for inners python remote debug port, 49896 could be any other port
docker exec -it container_name python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())" # check can torch run successfully

# attach to exection env
docker exec -it container_name /bin/bash
# or run directly
docker exec -it container_name python --wait-for-client {files like `approaches/PLELog.py`}
# or run directly with optional python debug exec env
docker exec -it container_name python -m debugpy --listen 0.0.0.0:5678 --wait-for-client {files like `approaches/PLELog.py`}
```

For more commands,  please see `CMD_FOR_BASELINES.txt.sh`



