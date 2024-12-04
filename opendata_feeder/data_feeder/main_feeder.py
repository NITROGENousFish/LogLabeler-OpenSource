import os, sys, re, gc
from .abstraction import EmbeddingInterface
from typing import Callable, List, Iterable, Tuple
from .embedding import Simple_template_TF_IDF
import random
import numpy as np
import opendata_feeder.config as ofconfig

from tools import file_tools


def _load_dataset(dataset_name: str,select_start:int=-1, selected_end:int=-1):
    _dataset_folder = ofconfig.get_processed_dataset_folder(dataset_name)

    all_miner_info_path = os.path.join(_dataset_folder, "all_miner_info.json")
    template_id_hash_path = os.path.join(_dataset_folder, "template_id_hash.txt")
    if not os.path.exists(all_miner_info_path) or not os.path.exists(
        template_id_hash_path
    ):
        raise RuntimeError(
            "dataset is not processed yet. Please run the processing script first."
        )
    all_miner_info = file_tools.json_load(all_miner_info_path)
    if select_start == -1 and selected_end == -1:
        template_id_hash = file_tools.txtline_load(template_id_hash_path)
    else:
        assert select_start >= 0 and selected_end >= 0
        template_id_hash = file_tools.txtline_load_with_bound(template_id_hash_path,select_start,selected_end)
    return all_miner_info, template_id_hash


class DataFeederHelper:
    def __init__(self): ...
    @staticmethod
    def fixed_window_helper(
        obj, window_size: int, pad_with_last: bool = True
    ) -> List[List]:
        if not window_size > 0:
            raise ValueError(f"Window size not less or equal then 0, got {window_size}")
        l = len(obj)
        times, remain = divmod(l, window_size)
        if remain > 0:
            if pad_with_last:
                obj += [obj[-1]] * (window_size - remain)
            times += 1
        assert len(obj) % window_size == 0
        return [obj[t * window_size : (t + 1) * window_size] for t in range(times)]

    @staticmethod
    def cut_by_712(instances: List) -> Tuple[List, List, List]:
        val_split = int(0.1 * len(instances))
        train_split = int(0.7 * len(instances))
        train = instances[: (train_split + val_split)]
        random.shuffle(train)
        val = train[train_split:]
        train = train[:train_split]
        test = instances[(train_split + val_split) :]
        return train, val, test

    @staticmethod
    def cut_online_in_to_five_groups(
        instances: List,
    ) -> Tuple[List, List, List, List, List]:
        _l = len(instances)
        #* split _l into 5 groups
        g1 = instances[: int(_l * 0.2)]
        g2 = instances[int(_l * 0.2) : int(_l * 0.4)]
        g3 = instances[int(_l * 0.4) : int(_l * 0.6)]
        g4 = instances[int(_l * 0.6) : int(_l * 0.8)]
        g5 = instances[int(_l * 0.8) :]
        return g1, g2, g3, g4, g5
    def cut_online_in_to_ten_groups(
        instances: List,
    ) -> Tuple[List, List, List, List, List, List, List, List, List, List]:
        _l = len(instances)
        #* split _l into 10 groups
        g1 = instances[: int(_l * 0.1)]
        g2 = instances[int(_l * 0.1) : int(_l * 0.2)]
        g3 = instances[int(_l * 0.2) : int(_l * 0.3)]
        g4 = instances[int(_l * 0.3) : int(_l * 0.4)]
        g5 = instances[int(_l * 0.4) : int(_l * 0.5)]
        g6 = instances[int(_l * 0.5) : int(_l * 0.6)]
        g7 = instances[int(_l * 0.6) : int(_l * 0.7)]
        g8 = instances[int(_l * 0.7) : int(_l * 0.8)]
        g9 = instances[int(_l * 0.8) : int(_l * 0.9)]
        g10 = instances[int(_l * 0.9) :]
        return g1, g2, g3, g4, g5, g6, g7, g8, g9, g10


class DataFeeder:
    def __init__(self, dataset_name):
        if dataset_name == "Thunderbird_offline":
            all_miner_info,self.template_id_hash = _load_dataset("Thunderbird",1,10000000)
        elif dataset_name == "Thunderbird_online":
            all_miner_info,self.template_id_hash = _load_dataset("Thunderbird")
        else:
            all_miner_info, self.template_id_hash = _load_dataset(dataset_name)

        # tid for template id
        self.tid_to_all_miner_info: dict = {
            str(i["template_id"]): i for i in all_miner_info
        }
        self.flag_calc_template_embedding = (
            False  # * embedding can be accessed in self.tid_to_all_miner_info
        )

    def _calc_template_embedding(
        self, embedding_func: Callable[[List[str]], List[np.ndarray]]
    ):
        """calculate embedding for each key in id_has"""
        template_list = [i["template"] for i in self.tid_to_all_miner_info.values()]

        template_embedding = embedding_func(template_list)
        for v, e in zip(self.tid_to_all_miner_info.values(), template_embedding):
            v["embedding"] = e
        self.flag_calc_template_embedding = True
        gc.collect()

    def get_fixed_window(
        self,
        window_size,
        embedding_func: Callable[[List[str]], List[np.ndarray]],
        cutting_func: Callable[[List], Tuple[List, List, List]],
    ):
        self._calc_template_embedding(embedding_func)
        windows = DataFeederHelper.fixed_window_helper(
            self.template_id_hash, window_size, pad_with_last=True
        )

        embeddings = [
            np.array([self.tid_to_all_miner_info[tid]["embedding"] for tid in ww])
            for ww in windows
        ]
        labels = [
            any([self.tid_to_all_miner_info[tid]["is_anomaly"] for tid in ww])
            for ww in windows
        ]

        return cutting_func(list(zip(embeddings, labels)))

    # todo
    def get_sliding_window(self): ...

    # todo
    def get_time_window(self): ...

    # todo
    def get_session_fixed_window(self, window_size):
        # * first groupby session(machine_id, etc.), then apply fixed window
        # todo assert session identifider in id_hash
        ...


def _get_bgl_offline_with_DF(window_size):
    DF = DataFeeder("BGL")
    embedding_tfidf = Simple_template_TF_IDF()
    train, val, test = DF.get_fixed_window(
        window_size, embedding_tfidf.embedding_list, DataFeederHelper.cut_by_712
    )
    return [train, val, test], DF

def get_bgl_offline(window_size):
    return _get_bgl_offline_with_DF(window_size)[0]

def get_bgl_online(window_size):
    DF = DataFeeder("BGL")
    embedding_tfidf = Simple_template_TF_IDF()
    g1, g2, g3, g4, g5 = DF.get_fixed_window(
        window_size,
        embedding_tfidf.embedding_list,
        DataFeederHelper.cut_online_in_to_five_groups,
    )
    return [
        [g1+g2, g3, g3],
        [g2+g3, g4, g4],
        [g3+g4, g5, g5]
    ]


def _get_thunderbird_offline_with_DF(window_size):
    DF = DataFeeder("Thunderbird_offline")
    embedding_tfidf = Simple_template_TF_IDF()
    train, val, test = DF.get_fixed_window(
        window_size, embedding_tfidf.embedding_list, DataFeederHelper.cut_by_712
    )
    return [train, val, test], DF

def get_thunderbird_offline(window_size):
    return _get_thunderbird_offline_with_DF(window_size)[0]

def get_thunderbird_online(window_size):
    DF = DataFeeder("Thunderbird_online")
    embedding_tfidf = Simple_template_TF_IDF()
    g1, g2, g3, g4, g5, g6, g7, g8, g9, g10 = DF.get_fixed_window(
        window_size,
        embedding_tfidf.embedding_list,
        DataFeederHelper.cut_online_in_to_ten_groups,
    )
    return [
        [g1+g2, g3, g3],
        [g2+g3, g4, g4],
        [g3+g4, g5, g5],
        [g4+g5, g6, g6],
        [g5+g6, g7, g7],
        [g6+g7, g8, g8],
        [g7+g8, g9, g9],
        [g8+g9, g10, g10]
    ]
