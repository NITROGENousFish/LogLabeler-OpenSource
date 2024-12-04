import opendata_feeder
from baselines.plelog.CONSTANTS import *

from baselines.plelog.approaches.PLELog import PLELog
from baselines.plelog.entities.instances import Instance
from baselines.plelog.representations.sequences.statistics import Sequential_TF

from baselines.plelog.preprocessing.AutoLabeling import Probabilistic_Labeling
from baselines.plelog.utils.Vocab import Vocab
from baselines.plelog.utils import addedtool
from baselines.plelog.module.Optimizer import Optimizer

from baselines.plelog.module.Common import (
    data_iter,
    generate_tinsts_binary_label,
    batch_variable_inst,
)

from collections import Counter
import numpy as np
import os
import time
from sklearn.decomposition import FastICA
from tools.file_tools import print_to_file

# * overwrite in the main function

flag_prob_labeling = True

parser = "Drain"
WINDOW_SIZE = 5
lstm_hiddens = 300
num_layer = 2
batch_size = 64
epochs = 20
min_cluster_size = 100
min_samples = 1
reduce_dimension = 50
# reduce_dimension = -1
threshold = 0.5


def main(dataset):
    print(device)
    save_dir = os.path.join(PROJECT_ROOT, "outputs")

    base = os.path.join(PROJECT_ROOT, "datasets/" + dataset)
    output_model_dir = os.path.join(
        save_dir, "models/PLELog/" + dataset + "_" + parser + "/model"
    )
    output_res_dir = os.path.join(
        save_dir, "results/PLELog/" + dataset + "_" + parser + "/detect_res"
    )
    prob_label_res_file = os.path.join(
        save_dir,
        "results/PLELog/"
        + dataset
        + "_"
        + parser
        + "/prob_label_res/mcs-"
        + str(min_cluster_size)
        + "_ms-"
        + str(min_samples),
    )
    rand_state = os.path.join(
        save_dir,
        "results/PLELog/" + dataset + "_" + parser + "/prob_label_res/random_state",
    )

    # * load dataset
    all_miner_info, template_id_hash = None,None
    if dataset == "Thunderbird":
        all_miner_info, template_id_hash = (
            opendata_feeder.data_feeder.main_feeder._load_dataset(
                "Thunderbird", 1, 10000000
            )
        )
    if dataset == "BGL":
        all_miner_info, template_id_hash = opendata_feeder.data_feeder.main_feeder._load_dataset('BGL')

    # * calculate template embedding
    tid_to_all_miner_info: dict = {
        str(i["template_id"]): i for i in all_miner_info
    }
    template_list = [i["template"] for i in tid_to_all_miner_info.values()]
    embedding_tfidf = opendata_feeder.Simple_template_TF_IDF()
    template_embedding = embedding_tfidf.embedding_list(template_list)
    for v, e in zip(tid_to_all_miner_info.values(), template_embedding):
        v["embedding"] = e

    id2emb = {iid: v["embedding"] for iid, v in tid_to_all_miner_info.items()}

    # * window generator
    windows = (
        opendata_feeder.data_feeder.main_feeder.DataFeederHelper.fixed_window_helper(
            template_id_hash, WINDOW_SIZE, pad_with_last=True
        )
    )

    embeddings = [
        np.array([tid_to_all_miner_info[tid]["embedding"] for tid in ww])
        for ww in windows
    ]
    labels = [
        any([tid_to_all_miner_info[tid]["is_anomaly"] for tid in ww])
        for ww in windows
    ]

    Instance_list = []
    for idx, w,e,l in zip(list(range(len(windows))),windows,embeddings,labels):
        _i = Instance(int(f"1{idx}"), w, "Anomalous" if l else "Normal")
        _i.repr = e.flatten()
        Instance_list.append(_i)

    train, val, test = opendata_feeder.data_feeder.main_feeder.DataFeederHelper.cut_by_712(Instance_list)
    train_reprs = [inst.repr for inst in train]

    labeled_train = train

    # Counter([i.label for i in train])
    # sum([len(list(set(i.sequence))) for i in train])/ len(train)

    _transformer = None
    if reduce_dimension != -1:  
        start_time = time.time()
        print("Start FastICA, target dimension: %d" % reduce_dimension)
        _transformer = FastICA(n_components=reduce_dimension)
        train_reprs = _transformer.fit_transform(train_reprs)
        for idx, inst in enumerate(train):
            inst.repr = train_reprs[idx]
        print("Finished at %.2f" % (time.time() - start_time))

    # Probabilistic labeling.
    # Sample normal instances.

    if flag_prob_labeling:
        # * original labal method in PLElog random drop 50% normal
        train_normal = [x for x, inst in enumerate(train) if inst.label == "Normal"]
        normal_ids = train_normal[: int(0.5 * len(train_normal))]
        label_generator = Probabilistic_Labeling(
            min_samples=min_samples,
            min_clust_size=min_cluster_size,
            res_file=prob_label_res_file,
            rand_state_file=rand_state,
        )

        labeled_train = label_generator.auto_label(train, normal_ids)

        # Below is used to test if the loaded result match the original clustering result.
        TP, TN, FP, FN = 0, 0, 0, 0

        for inst in labeled_train:
            if inst.predicted == "Normal":
                if inst.label == "Normal":
                    TN += 1
                else:
                    FN += 1
            else:
                if inst.label == "Anomalous":
                    TP += 1
                else:
                    FP += 1

        from baselines.plelog.utils.common import get_precision_recall

        # print(len(normal_ids))
        @print_to_file(f"./wrapper-inner_{dataset}.txt")
        def __():
            print("TP %d TN %d FP %d FN %d" % (TP, TN, FP, FN))
            p, r, f = get_precision_recall(TP, TN, FP, FN)
            print("%.4f, %.4f, %.4f" % (p, r, f))
        __()
    # exit(1)

    vocab = Vocab()
    vocab.load_from_dict(id2emb)
    plelog = PLELog(vocab, num_layer, lstm_hiddens, {"Normal": 0, "Anomalous": 1})

    log = "layer={}_hidden={}_epoch={}".format(num_layer, lstm_hiddens, epochs)
    best_model_file = os.path.join(output_model_dir, log + "_best.pt")
    last_model_file = os.path.join(output_model_dir, log + "_last.pt")
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    # Train

    optimizer = Optimizer(filter(lambda p: p.requires_grad, plelog.model.parameters()))
    bestClassifier = None
    global_step = 0
    bestF = 0
    batch_num = int(np.ceil(len(labeled_train) / float(batch_size)))

    for epoch in range(epochs):
        plelog.model.train()
        start = time.strftime("%H:%M:%S")
        plelog.logger.info(
            "Starting epoch: %d | phase: train | start time: %s | learning rate: %s"
            % (epoch + 1, start, optimizer.lr)
        )
        batch_iter = 0
        correct_num, total_num = 0, 0
        # start batch
        for onebatch in data_iter(labeled_train, batch_size, True):
            plelog.model.train()
            tinst = generate_tinsts_binary_label(onebatch, vocab)
            tinst.to_cuda(device)
            loss = plelog.forward(tinst.inputs, tinst.targets)
            loss_value = loss.data.cpu().numpy()
            loss.backward()
            if batch_iter % 100 == 0:
                plelog.logger.info(
                    "Step:%d, Iter:%d, batch:%d, loss:%.2f"
                    % (global_step, epoch, batch_iter, loss_value)
                )
            batch_iter += 1
            if batch_iter % 1 == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, plelog.model.parameters()),
                    max_norm=1,
                )
                optimizer.step()
                plelog.model.zero_grad()
                global_step += 1
            if val:
                if batch_iter % 500 == 0 or batch_iter == batch_num:
                    plelog.logger.info("Testing on val set.")
                    _, _, f = plelog.evaluate(val)
                    if f > bestF:
                        plelog.logger.info(
                            "Exceed best f: history = %.2f, current = %.2f" % (bestF, f)
                        )
                        torch.save(plelog.model.state_dict(), best_model_file)
                        bestF = f
        plelog.logger.info("Training epoch %d finished." % epoch)
        torch.save(plelog.model.state_dict(), last_model_file)

    final_p, final_r, final_f = 0, 0, 0
    best_p, best_r, best_f = 0,0,0

    if os.path.exists(
        "baselines/plelog/outputs/models/PLELog/BGL_Drain/model/layer=1_hidden=300_epoch=20_last.pt"
    ):
        plelog.logger.info(f"=== Final Model ==={last_model_file}")
        plelog.model.load_state_dict(
            torch.load(last_model_file,map_location='cpu')
        )
        plelog.model.to(device)
        final_p, final_r, final_f = plelog.evaluate(test, threshold)
        print(final_p, final_r, final_f)
    if os.path.exists(
        "baselines/plelog/outputs/models/PLELog/BGL_Drain/model/layer=1_hidden=300_epoch=20_best.pt"
    ):
        plelog.logger.info(f"=== Best Model ==={best_model_file}")
        plelog.model.load_state_dict(torch.load(best_model_file, map_location="cpu"))
        plelog.model.to(device)
        best_p, best_r, best_f = plelog.evaluate(test, threshold)
        print(best_p, best_r, best_f)
    plelog.logger.info("All Finished")
    return final_p, final_r, final_f


if __name__ == "__main__":
    
    # dataset = 'BGL'
    # @print_to_file(f"./result-{dataset}-PLElog-555.txt")
    # def _():
    #     print(main(dataset))
    # _()
    
    # del _
    
    dataset = 'Thunderbird'
    @print_to_file(f"./result-{dataset}-PLElog-555.txt")
    def _():
        print(main(dataset))
    _()
    # dataset 
