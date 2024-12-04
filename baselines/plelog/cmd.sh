mkdir datasets && cd datasets
cd datasets
axel -n 8 https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip
cd ..

docker build -t plelog:main .
docker build -t plelog:py311 .
# docker run --gpus all --name pytorchtest -v $(pwd):/workspace -dit pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
# docker exec -it pytorchtest python
docker run --gpus all --name plelog_test1 -v $(pwd):/workcenter -p 49869:5678 -dit plelog:py311
docker run --gpus all --name plelog_test1 -v $(pwd):/workcenter -p 49869:5678 -dit plelog:main
docker exec -it plelog_test1 python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
docker exec -it plelog_test1 /bin/bash
docker exec -it plelog_test1 python -m debugpy --listen 0.0.0.0:5678 --wait-for-client approaches/PLELog.py

# docker exec -it CONTAINER python -m debugpy --listen 0.0.0.0:5678 --wait-for-client PYFILE

docker exec -it plelog_test1 python -Xfrozen_modules=off -m debugpy  --listen 0.0.0.0:5678  --wait-for-client approaches/PLELog.py
docker exec -it plelog_test1 python -Xfrozen_modules=off -m debugpy  --listen 0.0.0.0:5678  --wait-for-client approaches/PLELog-customed-dataset.py
docker exec -it plelog_test1 python -Xfrozen_modules=off approaches/PLELog.py

docker exec -it plelog_test1 /bin/bash


rm -rf datasets/BGL
rm -rf my_cache/ & rm -rf outputs/
rm -rf logs/
mkdir logs
mkdir outputs
