import opendata_feeder
import numpy as np
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score

import lightgbm as lgb


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys


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
    global main_GBM
    @run_in_process(res_queue)
    def main_GBM(dataset_test_name, MODLE_NAME, device, train, val, test):

        def adapter(data):
            res_logs = []
            res_labels = []
            for logs, label in data:
                res_logs.append(logs)
                res_labels.append(int(label))

            return np.mean(np.array(res_logs), axis=1), np.array(res_labels)

        X_train, y_train = adapter(train)
        X_val, y_val = adapter(val)
        X_test, y_test = adapter(test)

        # 创建SVM模型
        model = lgb.LGBMClassifier(
            boosting_type="dart", num_leaves=20, n_estimators=100, n_jobs=64
        )

        # 训练模型
        model.fit(X_train, y_train)

        # 在验证集上进行预测
        y_test_pred = model.predict(X_test)
        print(
            (
                precision_score(y_test, y_test_pred),
                recall_score(y_test, y_test_pred),
                f1_score(y_test, y_test_pred),
            )
        )
        return (
            precision_score(y_test, y_test_pred),
            recall_score(y_test, y_test_pred),
            f1_score(y_test, y_test_pred),
        )

    def main_with_dataset_process(
        dataset_test_name: str, MODLE_NAME: str, evalmode: str, window_size: int
    ):
        print(dataset_test_name, MODLE_NAME, evalmode, window_size)
        assert evalmode in ["online", "offline"]
        assert dataset_test_name in ["BGL", "Thunderbird"]
        process_lists = []

        if dataset_test_name == "Thunderbird" and evalmode == "online":
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
                    main_GBM(
                        f"{dataset_test_name}{idx}",
                        MODLE_NAME,
                        device_list[idx],
                        prf1[0],
                        prf1[1],
                        prf1[2],
                    )
                )
        if dataset_test_name == "Thunderbird" and evalmode == "offline":
            train, val, test = opendata_feeder.main_feeder.get_thunderbird_offline(
                window_size
            )
            process_lists.append(
                main_GBM(
                    dataset_test_name,
                    MODLE_NAME,
                    "cuda:0",
                    train,
                    val,
                    test,
                )
            )
        if dataset_test_name == "BGL" and evalmode == "online":
            device_list = ["cuda:1", "cuda:2", "cuda:3"]
            for idx, prf1 in enumerate(
                opendata_feeder.main_feeder.get_bgl_online(window_size)
            ):
                process_lists.append(
                    main_GBM(
                        f"{dataset_test_name}{idx}",
                        MODLE_NAME,
                        device_list[idx],
                        prf1[0],
                        prf1[1],
                        prf1[2],
                    )
                )
        if dataset_test_name == "BGL" and evalmode == "offline":
            train, val, test = opendata_feeder.main_feeder.get_bgl_offline(window_size)
            process_lists.append(
                main_GBM(
                    dataset_test_name,
                    MODLE_NAME,
                    "cuda:0",
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

    # main_with_dataset_process("BGL", "GBM", "online",100)
    # main_with_dataset_process("BGL", "GBM","offline",100)
    main_with_dataset_process("Thunderbird", "GBM", "online",100)
    # main_with_dataset_process("Thunderbird", "GBM", "offline",100)
