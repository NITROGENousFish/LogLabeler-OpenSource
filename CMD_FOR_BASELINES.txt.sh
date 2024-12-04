python -Xfrozen_modules=off -m debugpy  --listen 0.0.0.0:49979  --wait-for-client opendata_feeder/data_feeder/main_feeder.py

docker run --gpus all --name logarena -v $(pwd):/workcenter -p 50970:5678 -dit plelog:main
docker run --gpus all --name logarena_v2 -v $(pwd):/workcenter -p 50971:5678 -dit loglabeler-pytorch1.10-cuda-11.3 


docker exec -it logarena_v2 /bin/bash
docker exec -it logarena /bin/bash

docker exec -it logarena python -Xfrozen_modules=off -m debugpy  --listen 0.0.0.0:5678  --wait-for-client baselines/plelog_wrapper.py
docker exec -it logarena python -Xfrozen_modules=off -m debugpy  --listen 0.0.0.0:5678  --wait-for-client  baselines/logdeep_wrapper.py
docker exec -it logarena python -Xfrozen_modules=off -m debugpy  --listen 0.0.0.0:5678  --wait-for-client  baselines/logdeep_wrapper.py
docker exec -it logarena python baselines/logdeep_wrapper.py 1
docker exec -it logarena python baselines/plelog_wrapper.py
