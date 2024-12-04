#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

from baselines.logdeep_base.logdeep.models.lstm import deeplog, loganomaly
from baselines.logdeep_base.logdeep.tools.predict_modify_opendatafeeder import Predicter
from baselines.logdeep_base.logdeep.tools.train_modify_opendatafeeder import Trainer
from baselines.logdeep_base.logdeep.tools.utils import *

from baselines.logdeep_base.logdeep.dataset.log import log_dataset
from tools.file_tools import print_to_file

# Config Parameters

import opendata_feeder

import multiprocessing
from functools import wraps


def run_in_process(queue_param=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 如果外部没有传入 Queue，则内部创建一个新的 Queue
            queue = queue_param

            # 定义一个包装器，将结果放入队列
            def process_func(queue, *args, **kwargs):
                result = func(*args, **kwargs)
                queue.put(result)

            # 创建并启动一个新进程
            process = multiprocessing.Process(
                target=process_func, args=(queue, *args), kwargs=kwargs
            )
            process.start()
            # 返回进程对象和队列
            return process

        return wrapper

    return decorator


if __name__ == "__main__":
    res_queue = multiprocessing.Queue()
    global model_baseline_main

    @run_in_process(res_queue)
    def model_baseline_main(dataset_test_name, MODLE_NAME, device, train, val, test):
        options = dict()
        options["data_dir"] = "data/"
        options["window_size"] = 10
        print()
        options["device"] = device

        # Smaple
        options["sample"] = "sliding_window"
        options["window_size"] = 10  # if fix_window

        # Features
        options["sequentials"] = False
        options["quantitatives"] = False
        options["semantics"] = True
        options["feature_num"] = sum(
            [options["sequentials"], options["quantitatives"], options["semantics"]]
        )

        # Models
        options["input_size"] = 300
        options["hidden_size"] = 128
        options["num_layers"] = 2
        options["num_classes"] = 2

        # Train
        options["batch_size"] = 1024
        options["accumulation_step"] = 5

        options["optimizer"] = "adam"
        options["lr"] = 0.001
        options["max_epoch"] = 300 #if 'bird' in dataset_test_name else 300
        options["lr_step"] = (300, 350)
        options["lr_decay_ratio"] = 0.1

        options["resume_path"] = None
        options["model_name"] = f"{MODLE_NAME}"
        options["save_dir"] = (
            f"baselines/logdeep_base/result/{dataset_test_name}{MODLE_NAME}/"
        )

        # Predict
        options["model_path"] = (
            f"baselines/logdeep_base/result/{dataset_test_name}{MODLE_NAME}/{MODLE_NAME}_last.pth"
        )
        options["num_candidates"] = 9

        def adapter(data):
            res_logs = []
            res_labels = []
            for logs, label in data:
                res_logs.append(logs)
                res_labels.append(int(label))

            return log_dataset(
                logs={"Semantics": res_logs},
                labels=res_labels,
                seq=False,
                quan=False,
                sem=True,
            )

        train_dataset = adapter(train)
        valid_dataset = adapter(val)
        test_dataset = adapter(test)

        seed_everything(seed=1234)

        _selected_model = None
        if MODLE_NAME == "deeplog":
            _selected_model = deeplog
        elif MODLE_NAME == "loganomaly":
            _selected_model = loganomaly
        else:
            print(f"Model {MODLE_NAME} not found")
            sys.exit(1)

        def train():
            Model = _selected_model(
                input_size=options["input_size"],
                hidden_size=options["hidden_size"],
                num_layers=options["num_layers"],
                num_keys=options["num_classes"],
            )
            trainer = Trainer(Model, train_dataset, valid_dataset, options)
            # os._exit(1)
            trainer.start_train()

        def predict():
            Model = _selected_model(
                input_size=options["input_size"],
                hidden_size=options["hidden_size"],
                num_layers=options["num_layers"],
                num_keys=options["num_classes"],
            )
            predicter = Predicter(Model, test_dataset, options)
            return predicter.predict()

        train()
        P, R, F1 = predict()
        print(f"Precision: {P:.3f}%, Recall: {R:.3f}%, F1-measure: {F1:.3f}%")
        return P, R, F1

    def main_with_dataset_process(
        dataset_test_name: str, MODLE_NAME: str, evalmode: str, window_size: int
    ):
        print(dataset_test_name, MODLE_NAME, evalmode, window_size)
        assert evalmode in ["online", "offline"]
        assert MODLE_NAME in ["loganomaly", "deeplog"]
        process_lists = []

        if "Thunderbird" in dataset_test_name and evalmode == "online":
            device_list = [
                "cuda:0",
                "cuda:1",
                "cuda:2",
                "cuda:3",
                "cuda:0",
                "cuda:1",
                "cuda:2",
                "cuda:3",
            ]

            for idx, prf1 in enumerate(
                opendata_feeder.main_feeder.get_thunderbird_online(window_size)
            ):
                process_lists.append(
                    model_baseline_main(
                        f"{dataset_test_name}{idx}",
                        MODLE_NAME,
                        device_list[idx],
                        prf1[0],
                        prf1[1],
                        prf1[2],
                    )
                )
        if "Thunderbird" in dataset_test_name and evalmode == "offline":
            train, val, test = opendata_feeder.main_feeder.get_thunderbird_offline(
                window_size
            )
            process_lists.append(
                model_baseline_main(
                    dataset_test_name,
                    MODLE_NAME,
                    "cuda:0",
                    train,
                    val,
                    test,
                )
            )
        if "BGL" in dataset_test_name and evalmode == "online":
            device_list = ["cuda:1", "cuda:2", "cuda:3"]
            for idx, prf1 in enumerate(
                opendata_feeder.main_feeder.get_bgl_online(window_size)
            ):
                process_lists.append(
                    model_baseline_main(
                        f"{dataset_test_name}{idx}",
                        MODLE_NAME,
                        device_list[idx],
                        prf1[0],
                        prf1[1],
                        prf1[2],
                    )
                )
        if "BGL" in  dataset_test_name and evalmode == "offline":
            train, val, test = opendata_feeder.main_feeder.get_bgl_offline(window_size)
            process_lists.append(
                model_baseline_main(
                    dataset_test_name,
                    MODLE_NAME,
                    "cuda:2",
                    train,
                    val,
                    test,
                )
            )

        # * wait all
        for p in process_lists:
            p.join()

        P = []
        R = []
        F1 = []
        while not res_queue.empty():
            _ = res_queue.get()
            P.append(_[0])
            R.append(_[1])
            F1.append(_[2])

        pstring = f"{dataset_test_name},{MODLE_NAME}: P:{sum(P) / len(P)},R:{sum(R) / len(R)},F1:{sum(F1) / len(F1)}"

        @print_to_file(f"./result-{dataset_test_name}-{MODLE_NAME}-{evalmode}.txt")
        def _():
            print(pstring)

        _()

    # main_with_dataset_process("BGL", "deeplog", "online",100)
    # main_with_dataset_process("BGL", "loganomaly", "online",100)
    # main_with_dataset_process("BGL", "deeplog","offline",100)
    # main_with_dataset_process("BGL", "loganomaly", "offline",100)
    # main_with_dataset_process("Thunderbird", "deeplog", "online",100)
    # main_with_dataset_process("Thunderbird", "loganomaly", "online",100)
    # main_with_dataset_process("Thunderbird", "deeplog","offline",100)
    # main_with_dataset_process("Thunderbird", "loganomaly", "offline",100)
    
    import sys
    if sys.argv[1] == '0':
        main_with_dataset_process("BGL555", "deeplog", "offline",5)
    if sys.argv[1] == '1':
        main_with_dataset_process("BGL555", "loganomaly", "offline",5)
    if sys.argv[1] == '2':
        main_with_dataset_process("Thunderbird555", "deeplog", "offline",5)
    if sys.argv[1] == '3':
        main_with_dataset_process("Thunderbird555", "loganomaly", "offline",5)