# After Preprocess Format

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