import re
import os
from typing import List
from collections import Counter

from tqdm import tqdm
import numpy as np


VOCAB_PATH = "data/glove.6B.300d.txt"


def _not_empty(s):
    return s and s.strip()


def _like_camel_to_tokens(camel_format):
    simple_format = []
    temp = ""
    flag = False

    if isinstance(camel_format, str):
        for i in range(len(camel_format)):
            if camel_format[i] == "-" or camel_format[i] == "_":
                simple_format.append(temp)
                temp = ""
                flag = False
            elif camel_format[i].isdigit():
                simple_format.append(temp)
                simple_format.append(camel_format[i])
                temp = ""
                flag = False
            elif camel_format[i].islower():
                if flag:
                    w = temp[-1]
                    temp = temp[:-1]
                    simple_format.append(temp)
                    temp = w + camel_format[i].lower()
                else:
                    temp += camel_format[i]
                flag = False
            else:
                if not flag:
                    simple_format.append(temp)
                    temp = ""
                temp += camel_format[i].lower()
                flag = True  # 需要回退
            if i == len(camel_format) - 1:
                simple_format.append(temp)
        simple_format = list(filter(_not_empty, simple_format))
    return simple_format


class Simple_template_TF_IDF:
    total_words = 0
    num_oov = 0

    def __init__(self, embed_file=VOCAB_PATH):
        self._word2vec = {}
        self.vocab_size = 0
        # * load word2vec
        if os.path.exists(embed_file):
            with open(embed_file, "r", encoding="utf-8") as reader:
                print("Reading embedding glove")
                for line in tqdm(reader.readlines()):
                    tokens = line.strip().split()
                    word = tokens[0]
                    embed = np.asarray(tokens[1:], dtype=np.float64)
                    self._word2vec[word] = embed
                    self.vocab_size = len(tokens) - 1
                    if len(tokens) != 301:
                        print(f"Word of total vocabulary: Not 301 but {line}")
        else:
            print(
                f"Init Simple_template_TF_IDF failed, no pre-trained embedding file found. Please check path: {embed_file}"
            )
            exit(2)

    def _transform(self, words):
        if isinstance(words, list):
            return_list = []
            for word in words:
                Simple_template_TF_IDF.total_words += 1
                word = word.lower()
                # word = re.sub('[·’!"$%&\'()＃！（）*+,-./:;<=>?，：￥★、…．＞【】［］《》？“”‘\[\\]^_`{|}~]+', '', word)
                if word in self._word2vec.keys():
                    return_list.append(self._word2vec[word])
                else:
                    print(word, end=" ")
                    Simple_template_TF_IDF.num_oov += 1
            return np.asarray(return_list, dtype=np.float64).sum(axis=0) / len(words)
        else:
            Simple_template_TF_IDF.total_words += 1
            word = words.lower()
            # word = re.sub('[·’!"$%&\'()＃！（）*+,-./:;<=>?，：￥★、…．＞【】［］《》？“”‘\[\\]^_`{|}~]+', '', word)
            if word in self._word2vec.keys():
                return self._word2vec[word]
            else:
                Simple_template_TF_IDF.num_oov += 1
                return np.zeros(self.vocab_size)

    def present(self, id2templates):
        """
        :param id2templates: dict, key: id, value: template
        return: dict{key id, value len 300 vector}
        """
        processed_id2templates = {}
        all_tokens = set()
        tokens_template_counter = Counter()

        # Preprocessing templates and calculate token-in-template apperance.
        id2embed = {}
        for id, template in id2templates.items():
            # Preprocess: split by spaces and special characters.
            template_tokens = re.split(r"[,\!:=\[\]\(\)\$\s\.\/\#\|\\ ]", template)
            filtered_tokens = []
            for simplified_token in template_tokens:
                if re.match("[\_]+", simplified_token) is not None:
                    filtered_tokens.append("")
                elif re.match("[\-]+", simplified_token) is not None:
                    filtered_tokens.append("")
                else:
                    filtered_tokens.append(simplified_token)
            template_tokens = list(filter(_not_empty, filtered_tokens))

            # Update token-in-template counter for idf calculation.
            for token in template_tokens:
                tokens_template_counter[token] += 1
                all_tokens = all_tokens.union(template_tokens)

            # Update new processed templates
            processed_id2templates[id] = " ".join(template_tokens)

        print(
            "Found %d tokens in %d log templates"
            % (len(all_tokens), len(processed_id2templates))
        )

        # Calculate IDF score.
        total_templates = len(processed_id2templates)
        token2idf = {}
        for token, count in tokens_template_counter.most_common():
            token2idf[token] = np.log(total_templates / count)

        # Calculate TF score and summarize template embedding.
        for id, template in processed_id2templates.items():
            template_tokens = template.split()
            N = len(template_tokens)
            token_counter = Counter(template_tokens)
            template_emb = np.zeros(self.vocab_size)
            if N == 0:
                id2embed[id] = template_emb
                continue
            for token in template_tokens:
                simple_words = _like_camel_to_tokens(token)
                tf = token_counter[token] / N
                if token in token2idf.keys():
                    idf = token2idf[token]
                else:
                    idf = 1
                embed = self._transform(simple_words)
                template_emb += tf * idf * embed
            id2embed[id] = template_emb
        print(
            "OOV Rate: %d/%d = %.4f"
            % (
                Simple_template_TF_IDF.num_oov,
                Simple_template_TF_IDF.total_words,
                (Simple_template_TF_IDF.num_oov / Simple_template_TF_IDF.total_words),
            )
        )
        return id2embed


def fast_tfidf_embedding(template:List[str])->List[np.ndarray]:
    tfidf = Simple_template_TF_IDF()
    res = tfidf.present({k: v for k, v in enumerate(template)})
    return list(res.values())

def test_Simple_template_TF_IDF():
    tfidf = Simple_template_TF_IDF()
    loglist = [
        "i-cache parity error..............0",
        "icache prefetch .*",
        "Ido chip status changed: .* .* .* .* .* .* .* .* .* .* .*",
        "Ido packet timeout",
        "idoproxy communication failure: socket closed",
        "idoproxydb has been started: \$Name: .* \$ Input parameters: -enableflush -loguserinfo db.properties BlueGene1",
        "idoproxydb hit ASSERT condition: ASSERT expression=.* Source file=.* Source line=.* Function=.* IdoTransportMgr::SendPacket\(IdoUdpMgr.*, BglCtlPavTrace.*\)",
        "idoproxydb hit ASSERT condition: ASSERT expression=!\(nMsgLen > 0x10000\) Source file=idomarshalerio.cpp Source line=1929 Function=int IdoMarshalerRecvBuffer::ReadBlock\(IdoMsg::IdoMsgHdr.*&\)",
        "idoproxydb hit ASSERT condition: ASSERT expression=pTargetMgr Source file=idoclientmgr.cpp Source line=353 Function=int IdoClientMgr::TargetClose\(const char.*\)",
        "idoproxydb hit ASSERT condition: ASSERT expression=!\(RecvMsgHdr.ulLen > 0x10000\) Source file=idomarshalerio.cpp Source line=387 Function=virtual int IdoMarshalerIo::RunRecv\(\)",
        "imprecise machine .*",
        "inexact .*",
        "instance of a correctable ddr. RAS KERNEL INFO .* microseconds spent in the rbs signal handler during .* calls. .* microseconds was the maximum time for a single instance of a correctable ddr.",
        "instruction address: .*",
        "instruction address space.........0",
    ]
    resid2embed = tfidf.present({k: v for k, v in enumerate(loglist)})
    assert isinstance(resid2embed, dict)
    assert isinstance(resid2embed[0], np.ndarray)
    assert resid2embed[0].shape == (300,)


if __name__ == "__main__":
    test_Simple_template_TF_IDF()
