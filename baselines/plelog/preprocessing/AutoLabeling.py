from CONSTANTS import *
from entities.instances import Instance
from models.clustering import Solitary_HDBSCAN
from utils.common import get_precision_recall
from utils import addedtool
from collections import defaultdict
from tqdm import tqdm

NUMPROCESS = 200

class Probabilistic_Labeling():
    def __init__(self, min_samples, min_clust_size, res_file=None, rand_state_file=None):
        self.model = Solitary_HDBSCAN(min_cluster_size=min_clust_size, min_samples=min_samples)
        self.random_state_file = rand_state_file
        # self.random_state_file = None
        self.res_file = res_file
        if res_file:
            folder, file = os.path.split(res_file)
            if not os.path.exists(folder):
                os.makedirs(folder)
        ProbLabelLogger = logging.getLogger('Prob_Label')
        ProbLabelLogger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'Prob_Label.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        ProbLabelLogger.addHandler(console_handler)
        ProbLabelLogger.addHandler(file_handler)
        ProbLabelLogger.info(
            'Construct logger for Probabilistic labeling succeeded, current working directory: %s, logs will be written in %s' %
            (os.getcwd(), LOG_ROOT))

        self.logger = ProbLabelLogger

    def auto_label(self, instances, normal_ids):
        if os.path.exists(self.res_file) and os.path.exists(self.random_state_file) and os.path.getsize(
                self.res_file) != 0:
            ...
            # self.logger.info('Found previous labeled file, will load and continue to accelerate the process.')

            # with open(self.random_state_file, 'rb') as reader:
            #     state = pickle.load(reader)
            # if state == random.getstate():
            #     self.load_label_res(instances)
            # else:
            #     self.logger.error('Random state does not match, please check or re-train.')
            #     exit(-1)
            # labeled_inst = instances
        # else:
        # * always relabel
        inputs = [inst.repr for inst in instances]
        inputs = np.asarray(inputs, dtype=np.float64)
        ground_truth = [inst.label for inst in instances]

        labels = self.model.fit_predict(inputs)
        outliers = self.model.outliers
        predicts = self.model.predict(inputs, normal_ids)

        # * subprocess function

        assert len(instances)==len(labels)==len(predicts)==len(outliers)

        def _process_each_chunk(chunk_start,chunk_len):
            # * sharing READONLY obj across process: instances, labels, predicts, outliers
            nonlocal instances, labels, predicts, outliers, normal_ids, ground_truth
            local_TP, local_TN, local_FP, local_FN = 0, 0, 0, 0
            def _inner_process_one(inst, label, predict, outlier,idx):
                nonlocal local_TP, local_TN, local_FP, local_FN, normal_ids
                if idx in normal_ids:
                    inst.predicted = 'Normal'
                    return inst
                else:
                    new_instance = Instance(inst.id, inst.sequence,inst.label)
                    new_instance.repr = inst.repr 
                    if label == -1:
                        # -1 cluster, all instances should have confidence 0
                        confidence = 0
                        new_instance.confidence = confidence
                        new_instance.predicted = predict
                    else:
                        # other clusters, instances should have confidence according to the outlier score.
                        confidence = 1 if np.isnan(outlier) else outlier
                        new_instance.confidence = confidence
                        new_instance.predicted = predict

                    if new_instance.predicted == 'Normal':
                        if ground_truth[idx] == 'Normal':
                            local_TN += 1
                        else:
                            local_FN += 1
                    else:
                        if ground_truth[idx] == 'Anomalous':
                            local_TP += 1
                        else:
                            local_FP += 1
                    return new_instance

            idx_list = [i for i in range(chunk_start, chunk_start + chunk_len)]
            ret_labeled_inst_with_id = []
            for idx in tqdm(idx_list):
                ret_labeled_inst_with_id.append(
                    _inner_process_one(
                        instances[idx],
                        labels[idx],
                        predicts[idx],
                        outliers[idx],
                        idx,
                    )
                )
            return local_TP, local_TN, local_FP, local_FN, ret_labeled_inst_with_id

        _num_process = NUMPROCESS
        totallen = len(instances)

        start_and_chunklen_list = addedtool.split_length_to_start_and_chunklen(totallen, _num_process)
        import multiprocessing
        self.logger.info("Zongyang: Start multiprocessing with %d process" % _num_process)
        _start_time = time.time()
        with addedtool.globalized(_process_each_chunk), multiprocessing.Pool(_num_process) as pool:
            results = pool.starmap(_process_each_chunk, start_and_chunklen_list)

        TP = sum(r[0] for r in results)
        TN = sum(r[1] for r in results)
        FP = sum(r[2] for r in results)
        FN = sum(r[3] for r in results)
        labeled_inst = [pred for result in results for pred in result[4]]
        self.logger.info("Finished at %.2f" % (time.time() - _start_time))

        # *================================================================================================
        self.logger.info('TP: %d. TN: %d, FP: %d. FN: %d' % (TP, TN, FP, FN))
        precision, recall, f = get_precision_recall(TP, TN, FP, FN)
        self.logger.info('Training set precision: %.4f, recallL: %.4f, F1: %.4f.' % (precision, recall, f))
        self.record_label_res(labeled_inst)
        with open(self.random_state_file, 'wb') as writer:
            state = random.getstate()
            pickle.dump(state, writer)
        return labeled_inst

    def record_label_res(self, instances):
        with open(self.res_file, 'w', encoding='utf-8') as writer:
            for inst in instances:
                writer.write(str(inst.id) + ' ' + str(inst.predicted) + ' ' + str(inst.confidence) + '\n')

    def load_label_res(self, instances):
        self.logger.info('Start load previous clustered results from %s' % self.res_file)
        self.logger.warning('Please NOTE that this may cause some problem due to incomplete cluster settings.')
        block2conf = {}
        block2label = {}
        with open(self.res_file, 'r', encoding='utf-8') as reader:
            for line in reader.readlines():
                block_id, label, confidence = line.strip().split()
                block2conf[block_id] = np.float64(confidence)
                block2label[block_id] = label
        for inst in instances:
            if inst.id in block2label.keys():
                inst.predicted = block2label[inst.id]
                inst.confidence = block2conf[inst.id]
            else:
                self.logger.error('Found mismatch block %s, please check and re-cluster if necessary' % inst.id)
                exit(-1)
