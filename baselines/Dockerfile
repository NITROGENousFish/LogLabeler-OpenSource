# FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN mkdir /workcenter && \ 
    cd /workcenter && \
    apt update && apt install gcc build-essential -y && \
    pip install hdbscan==0.8.33 overrides==6.1.0 drain3 tqdm regex pandas debugpy && \
    export PYTHONPATH=/workcenter


WORKDIR /workcenter
ENV PYTHONPATH="/workcenter"
EXPOSE 5678