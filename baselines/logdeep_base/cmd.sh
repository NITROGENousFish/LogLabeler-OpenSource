docker run --gpus all --name deeplog_env -v $(pwd):/workcenter -p 49870:5678 -dit plelog:main

docker exec -it deeplog_env /bin/bash
docker exec -it deeplog_env python -m debugpy --listen 0.0.0.0:5678 --wait-for-client demo/deeplog.py train

# docker exec -it CONTAINER python -m debugpy --listen 0.0.0.0:5678 --wait-for-client PYFILE

docker exec -it deeplog_env python -Xfrozen_modules=off -m debugpy  --listen 0.0.0.0:5678  --wait-for-client demo/deeplog.py train
docker exec -it deeplog_env python -Xfrozen_modules=off demo/deeplog.py train



docker exec -it deeplog_env /bin/bash





#* exec step for BGL
#* 1. structure_bgl -> generate data/sampling_example/bgl/BGL_100k_structured.csv
docker exec -it deeplog_env python -Xfrozen_modules=off data/sampling_example/structure_bgl.py
#* 2. sample bgl
docker exec -it deeplog_env python -Xfrozen_modules=off -m debugpy  --listen 0.0.0.0:5678  --wait-for-client data/sampling_example/sample_bgl.py
docker exec -it deeplog_env python -Xfrozen_modules=off data/sampling_example/sample_bgl.py
#* 3. train
docker exec -it deeplog_env python -Xfrozen_modules=off -m debugpy  --listen 0.0.0.0:5678  --wait-for-client demo/deeplog.py train
docker exec -it deeplog_env python -Xfrozen_modules=off -m debugpy  --listen 0.0.0.0:5678  --wait-for-client demo/deeplog.py train_modify

docker exec -it deeplog_env python -Xfrozen_modules=off demo/deeplog.py train






docker exec -it deeplog_env python -Xfrozen_modules=off -m debugpy  --listen 0.0.0.0:5678  --wait-for-client logdeep/dataset/sample.py