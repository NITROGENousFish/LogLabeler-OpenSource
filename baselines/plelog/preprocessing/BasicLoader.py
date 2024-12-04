import sys

from CONSTANTS import *
from parsers.Drain_IBM import Drain3Parser
import abc


def _async_parsing(parser, lines, log2temp):
    for id, line in lines:
        cluster = parser.match(line)
        if cluster == None:
            log2temp[id] = -1
        else:
            log2temp[id] = cluster.cluster_id


class BasicDataLoader:
    def __init__(self):
        self.in_file = None
        self.logger = None
        self.block2emb = {} #* useless
        self.blocks = []
        self.templates = {}
        self.log2temp = {}  # * {line_num: drain_cluster_id}
        self.rex = []
        self.remove_cols = []
        self.id2label = {0: "Normal", 1: "Anomalous"}
        self.label2id = {"Normal": 0, "Anomalous": 1}
        self.block_set = set()  #* useless
        self.block2seqs = (
            {}
        )  # * {block_id: list(line_num)}, block size (or windows size) is 120
        self.block2label = {}  # * read from label file datasets/BGL/label.txt, if not exists, splited from the orginal log
        self.block2eventseq = {}  # * {block_id: list(drain_cluster_id)}
        self.id2embed = {}
        self.semantic_repr_func = None

    @abc.abstractmethod
    def _load_raw_log_seqs(self):
        return

    @abc.abstractmethod
    def logger(self):
        return

    @abc.abstractmethod
    def _pre_process(self, line):
        return

    def parse_by_IBM(self, config_file, persistence_folder, core_jobs=16):
        """
        #* core jobs: 多线程
        Load parsing results by IDM Drain
        :param config_file: IDM Drain configuration file.
        :param persistence_folder: IDM Drain persistence file.
        :return: Update templates, log2temp attributes in self.
        """
        # * init self.block2emb self.templates self.log2temp to {}
        self.block2emb = {}
        self.templates = {}
        self.log2temp = {}
        if not os.path.exists(config_file):
            self.logger.error("IBM Drain config file %s not found." % config_file)
            exit(1)
        parser = Drain3Parser(
            config_file=config_file, persistence_folder=persistence_folder
        )
        persistence_folder = parser.persistence_folder  # * 设置持久化路径

        # Specify persistence files.
        log_event_seq_file = os.path.join(persistence_folder, "log_sequences.txt")
        log_template_mapping_file = os.path.join(
            persistence_folder, "log_event_mapping.dict"
        )
        templates_embedding_file = os.path.join(
            parser.persistence_folder, "templates.vec"
        )
        start_time = time.time()
        if parser.to_update:
            self.logger.info("No trained parser found, start training.")
            parser.parse_file(self.in_file, remove_cols=self.remove_cols)
            self.logger.info(
                "Get total %d templates." % len(parser.parser.drain.clusters)
            )

        # Load templates from trained parser.
        for cluster_inst in parser.parser.drain.clusters:
            self.templates[int(cluster_inst.cluster_id)] = cluster_inst.get_template()

        # check parsing resutls such as log2event dict and template embeddings.
        # * log_template_mapping_file -> self.log2temp: {line_num: drain_cluster_id}
        # * log_event_seq_file -> self.block2eventseq
        if self._check_parsing_persistences(
            log_template_mapping_file, log_event_seq_file
        ):
            self._load_parsing_results(log_template_mapping_file, log_event_seq_file)
            pass
        else:
            # parsing results not found, or somehow missing.
            self.logger.info(
                "Missing persistence file(s), start with a full parsing process."
            )
            self.logger.warning(
                "If you don't want this to happen, please copy persistence files from somewhere else and put it in %s"
                % persistence_folder
            )
            ori_lines = []
            with open(self.in_file, "r", encoding="utf-8") as reader:
                log_id = 0
                for line in tqdm(reader.readlines()):
                    processed_line = self._pre_process(
                        line
                    )  # * ABC method, preprocess the line
                    ori_lines.append((log_id, processed_line))
                    log_id += 1
            self.logger.info("Parsing raw log....")
            if core_jobs:
                m = Manager()
                log2temp = m.dict()
                pool = Pool(core_jobs)

                def _split(X, copies=5):
                    quota = int(len(X) / copies) + 1
                    res = []
                    for i in range(copies):
                        res.append(X[i * quota : (i + 1) * quota])
                    return res

                splitted_lines = _split(
                    ori_lines, core_jobs
                )  # * each: list of splitted lines, each splitted line is a list like: [(id, line),(),...]
                inputs = zip(
                    [parser] * core_jobs, splitted_lines, [log2temp] * core_jobs
                )
                pool.starmap(_async_parsing, inputs)
                pool.close()
                pool.join()
                self.log2temp = dict(log2temp)
                # * self.log2temp: {line_num: drain_cluster_id}
                pass
            else:
                for id, message in ori_lines:
                    cluster = parser.match(message)
                    self.log2temp[id] = cluster.cluster_id

            self.logger.info("Finished parsing in %.2f" % (time.time() - start_time))

            # Transform original log sequences with log ids(line number) to log event sequence.
            for block, seq in self.block2seqs.items():
                self.block2eventseq[block] = []
                for log_id in seq:
                    self.block2eventseq[block].append(self.log2temp[log_id])

            # Record block id and log event sequences.

            self._record_parsing_results(log_template_mapping_file, log_event_seq_file)
        # Prepare semantic embeddings.
        self._prepare_semantic_embed(templates_embedding_file)
        self.logger.info(
            "All data preparation finished in %.2f" % (time.time() - start_time)
        )

    def _load_parsing_results(self, log_template_mapping_file, event_seq_file):
        start = time.time()
        self.logger.info("Start loading previous parsing results.")

        with open(
            log_template_mapping_file, "r", encoding="utf-8"
        ) as log_template_mapping_reader:
            for line in log_template_mapping_reader.readlines():
                logid, tempid = line.strip().split(",")
                self.log2temp[int(logid)] = int(tempid)
            self.logger.info(
                "Loaded %d log sequences and their mappings." % len(self.log2temp)
            )

        with open(event_seq_file, "r", encoding="utf-8") as event_seq_reader:
            for line in event_seq_reader.readlines():
                tokens = line.strip().split(":")
                block = tokens[0]
                seq = tokens[1].split()
                self.block2eventseq[block] = [int(x) for x in seq]
            self.logger.info("Loaded %d blocks" % len(self.block2eventseq))

        self.logger.info("Finished in %.2f" % (time.time() - start))

    def _record_parsing_results(self, log_template_mapping_file, evet_seq_file):
        # Recording IBM parsing result.
        start_time = time.time()
        log_template_mapping_writer = open(
            log_template_mapping_file, "w", encoding="utf-8"
        )
        event_seq_writer = open(evet_seq_file, "w", encoding="utf-8")

        def _save_log2temp(writer):
            for log_id, temp_id in self.log2temp.items():
                writer.write(str(log_id) + "," + str(temp_id) + "\n")
            self.logger.info("Log2Temp saved.")

        _save_log2temp(log_template_mapping_writer)

        def _save_log_event_seqs(writer):
            self.logger.info("Start saving log event sequences.")
            for block, event_seq in self.block2eventseq.items():
                event_seq = map(lambda x: str(x), event_seq)
                seq_str = " ".join(event_seq)
                writer.write(str(block) + ":" + seq_str + "\n")
            self.logger.info("Log event sequences saved.")

        _save_log_event_seqs(event_seq_writer)

        log_template_mapping_writer.close()
        event_seq_writer.close()
        self.logger.info("Done in %.2f" % (time.time() - start_time))

    def _prepare_semantic_embed(self, semantic_emb_file):
        # * convert all drain template to semantic representations
        if self.semantic_repr_func:
            # * function semantic_repr_func: Simple_template_TF_IDF.present
            self.id2embed = self.semantic_repr_func(self.templates)
            with open(semantic_emb_file, "w", encoding="utf-8") as writer:
                for id, embed in self.id2embed.items():
                    # * id
                    writer.write(str(id) + " ")
                    # * np.array vector, vector length = 300, same with datasets/glove.6B.300d.txt
                    writer.write(" ".join([str(x) for x in embed.tolist()]) + "\n")
            self.logger.info(
                "Finish calculating semantic representations, please found the vector file at %s"
                % semantic_emb_file
            )
        else:
            self.logger.warning(
                "No template encoder. Please be NOTED that this may lead to duplicate full parsing process."
            )

        pass

    def _check_parsing_persistences(self, log_template_mapping_file, event_seq_file):
        def _check_file_existence_and_contents(file):
            flag = os.path.exists(file) and os.path.getsize(file) != 0
            self.logger.info("checking file %s ... %s" % (file, str(flag)))
            return flag

        return _check_file_existence_and_contents(
            log_template_mapping_file
        ) and _check_file_existence_and_contents(event_seq_file)

    # def _save_templates(self, writer):
    #     for id, template in self.templates.items():
    #         writer.write(','.join([str(id), template]) + '\n')
    #     self.logger.info('Templates saved.')

    # def _load_templates(self, reader):
    #     for line in reader.readlines():
    #         tokens = line.strip().split(',')
    #         id = tokens[0]
    #         template = ','.join(tokens[1:])
    #         self.templates[int(id)] = template
    #     self.logger.info('Loaded %d templates' % len(self.templates))

    # def _load_semantic_embed(self, reader):
    #     for line in reader.readlines():
    #         token = line.split()
    #         template_id = int(token[0])
    #         embed = np.asarray(token[1:], dtype=np.float64)
    #         self.id2embed[template_id] = embed
    #     self.logger.info('Load %d templates with embedding size %d' % (len(self.id2embed), self.id2embed[1].shape[0]))
