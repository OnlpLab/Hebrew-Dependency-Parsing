import re
import json
import torch
import string

from typing import List
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizerFast

from embeddings_utils import get_embeddings
from data_utils import PAD_TOKEN, PADDING_FOR_PREDICTION, DataProcessor, get_tokens_data, NON_ROOT_DEP


class Vocab:
    UNKNOWN_TOKEN = "[UNK]"

    def __init__(self, sentences: List[List[str]], poses: List[List[str]], deps: List[List[str]],
                 heads: List[List[str]], genders: List[List[str]], numbers: List[List[str]], persons: List[List[str]],
                 embeddings_vocab):

        self.words = self.get_tokens_set(sentences)
        self.pos = self.get_tokens_set(poses, padding=True)
        self.pos.add(NON_ROOT_DEP)
        self.deps = self.get_tokens_set(deps)
        self.deps.add(NON_ROOT_DEP)
        self.heads = self.get_tokens_set(heads)
        self.genders = self.get_tokens_set(genders, padding=True)
        self.numbers = self.get_tokens_set(numbers, padding=True)
        self.persons = self.get_tokens_set(persons, padding=True)

        self.pos_size = len(self.pos)
        self.deps_size = len(self.deps)
        self.gender_size = len(self.genders)
        self.number_size = len(self.numbers)
        self.person_size = len(self.persons)

        if embeddings_vocab is not None:
            self.word2i = embeddings_vocab
        else:
            self.word2i = {w: i for i, w in enumerate(self.words)}

        self.vocab_size = len(self.word2i)
        self.i2word = {i: w for w, i in self.word2i.items()}

        self.i2pos = {i: l for i, l in enumerate(self.pos)}
        self.i2pos[PADDING_FOR_PREDICTION] = PAD_TOKEN
        self.pos2i = {l: i for i, l in self.i2pos.items()}
        self.i2gender = {i: l for i, l in enumerate(self.genders)}
        self.i2gender[PADDING_FOR_PREDICTION] = PAD_TOKEN
        self.gender2i = {l: i for i, l in self.i2gender.items()}
        self.i2number = {i: l for i, l in enumerate(self.numbers)}
        self.i2number[PADDING_FOR_PREDICTION] = PAD_TOKEN
        self.number2i = {l: i for i, l in self.i2number.items()}
        self.i2person = {i: l for i, l in enumerate(self.persons)}
        self.i2person[PADDING_FOR_PREDICTION] = PAD_TOKEN
        self.person2i = {l: i for i, l in self.i2person.items()}

        # predict
        self.i2dep = {i: l for i, l in enumerate(self.deps)}
        self.i2dep[PADDING_FOR_PREDICTION] = PAD_TOKEN
        self.dep2i = {l: i for i, l in self.i2dep.items()}

    def get_pos_index(self, pos):
        if pos in self.pos2i:
            return self.pos2i[pos]
        return self.pos2i[self.UNKNOWN_TOKEN]

    def get_dep_index(self, dep):
        if dep in self.dep2i:
            return self.dep2i[dep]
        return self.dep2i[self.UNKNOWN_TOKEN]

    def get_gender_index(self, gender):
        if gender in self.gender2i:
            return self.gender2i[gender]
        return self.gender2i[self.UNKNOWN_TOKEN]

    def get_number_index(self, number):
        if number in self.number2i:
            return self.number2i[number]
        return self.number2i[self.UNKNOWN_TOKEN]

    def get_person_index(self, person):
        if person in self.person2i:
            return self.person2i[person]
        return self.person2i[self.UNKNOWN_TOKEN]

    def get_tokens_set(self, sentences, padding=False):
        tokens_set = set()
        for sent in sentences:
            tokens_set.update(set(sent))
        try:
            if not padding:
                tokens_set.remove(PAD_TOKEN)
                tokens_set.remove(PADDING_FOR_PREDICTION)
        except Exception as error:
            pass
        return tokens_set


class TrainData(Dataset):
    PAD_TOKEN = "[PAD]"

    def __init__(self, device, data_path: str, embeddings_vocab, mtl_task: str, tokenizer: BertTokenizerFast,
                 bert_model: BertModel, embeddings_path_load: str):
        self.device = device
        self.data_path = data_path
        self.mtl_task = mtl_task
        self.bert_model = bert_model
        self.tokenizer = tokenizer

        self.ma_generator = MAGenerator(self.data_path, add_gold_seg=True)
        self.data_processor = DataProcessor(self.ma_generator, train=True)
        _, _, self.sentence_analyses, self.sentences_segments, _, self.poses, self.deps, self.heads, self.genders, self.numbers, self.persons = self.data_processor.get_data(self.data_path)

        if not embeddings_path_load:
            self.create_embedding()
        else:
            try:
                self.embeddings = torch.load(embeddings_path_load, map_location=self.device)
            except Exception as error:
                print("Can't load embedding file for train!")
                self.create_embedding()

        # create vocab after data preparation
        self.vocab = Vocab(self.sentences_segments, self.poses, self.deps, self.heads, self.genders, self.numbers, self.persons, embeddings_vocab)

    def create_embedding(self):
        print("start to create embedding for training!")
        self.embeddings = get_embeddings(self.sentence_analyses, self.tokenizer, self.bert_model, self.device)
        model_name = self.bert_model.name_or_path.replace("/", "-")
        torch.save(self.embeddings, f"{model_name}_train_embeddings.bin")
        print("finish to create embedding for training!")

    def __len__(self):
        return len(self.sentences_segments)

    def __getitem__(self, index):
        curr_dep = self.deps[index]
        curr_head = self.heads[index]
        curr_pos = self.poses[index]

        # new context
        all_analyses_embedding = self.embeddings[index]

        pos_tensor = torch.tensor([self.vocab.get_pos_index(w) for w in curr_pos]).to(torch.int64)
        head_tensor = torch.tensor(curr_head).to(torch.int64)
        dep_tensor = torch.tensor([self.vocab.get_dep_index(w) for w in curr_dep]).to(torch.int64)

        # Features
        gender_tensor = None
        number_tensor = None
        person_tensor = None

        if self.mtl_task:
            if "gender" in self.mtl_task:
                curr_gender = self.genders[index]
                gender_tensor = torch.tensor([self.vocab.get_gender_index(w) for w in curr_gender]).to(torch.int64)

            if "number" in self.mtl_task:
                curr_number = self.numbers[index]
                number_tensor = torch.tensor([self.vocab.get_number_index(w) for w in curr_number]).to(torch.int64)

            if "person" in self.mtl_task:
                curr_person = self.persons[index]
                person_tensor = torch.tensor([self.vocab.get_person_index(w) for w in curr_person]).to(torch.int64)

        return all_analyses_embedding, pos_tensor, dep_tensor, head_tensor, gender_tensor, person_tensor, number_tensor


class TestData(Dataset):
    def __init__(self, device: str, gold_test_path: str, test_tokenization_path: str, test_pos_path: str, vocab: Vocab,
                 mtl_task: str, tokenizer: BertTokenizerFast, bert_model: BertModel, add_gold_seg: bool,
                 embeddings_path_load: str):
        self.vocab = vocab
        self.device = device
        self.gold_test_path = gold_test_path
        self.test_tokenization_path = test_tokenization_path
        self.test_pos_path = test_pos_path
        self.mtl_task = mtl_task
        self.bert_model = bert_model
        self.tokenizer = tokenizer

        self.ma_generator = MAGenerator(self.gold_test_path, add_gold_seg=add_gold_seg)
        self.data_processor = DataProcessor(self.ma_generator, train=False)
        _, _, self.sentence_analyses, self.sentences_segments, self.analyses_mask, self.poses, self.deps, self.heads, self.genders, self.numbers, self.persons = self.data_processor.get_data(self.gold_test_path)
        self.sentences_id, self.raw_sentences, _, self.sentences_segments_gold, _, self.poses_gold, self.deps_gold, self.heads_gold, self.genders_gold, self.numbers_gold, self.persons_gold = self.data_processor.get_data(self.gold_test_path, gold=True)

        if not embeddings_path_load:
            self.create_embedding()
        else:
            try:
                self.embeddings = torch.load(embeddings_path_load, map_location=self.device)
            except Exception as error:
                print("Can't load embedding file for test!")
                self.create_embedding()

        self.test_sentences_tokens = get_tokens_data(self.test_tokenization_path)
        self.test_poses = get_tokens_data(self.test_pos_path)

    def create_embedding(self):
        print("start to create embedding for test!")
        self.embeddings = get_embeddings(self.sentence_analyses, self.tokenizer, self.bert_model, self.device)
        model_name = self.bert_model.name_or_path.replace("/", "-")
        torch.save(self.embeddings, f"{model_name}_test_embeddings.bin")
        print("finish to create embedding for test!")

    def __len__(self):
        return len(self.sentences_segments)

    def __getitem__(self, index):
        # test inputs
        test_sentence = self.sentences_segments[index]
        test_pos = self.poses[index]
        analysis_mask = self.analyses_mask[index]
        embedding = self.embeddings[index]

        # gold sequences
        gold_sentence = self.sentences_segments_gold[index]
        gold_pos = self.poses_gold[index]
        gold_dep = self.deps_gold[index]
        gold_head = self.heads_gold[index]
        gold_gender = self.genders_gold[index]
        gold_number = self.numbers_gold[index]
        gold_person = self.persons_gold[index]

        pos_tensor = torch.tensor([self.vocab.get_pos_index(w) for w in test_pos]).to(torch.int64)

        return embedding, analysis_mask, pos_tensor, test_sentence, test_pos, gold_sentence, gold_pos, gold_dep, gold_head, gold_gender, gold_number, gold_person


class MAGenerator:
    RE_NUM = r"[\d]"
    MA_FILE = "full_ma_ud_format.json"
    PREFIXES_FILE = "hebrew_prefixes.txt"

    def __init__(self, data_path: str = "", add_gold_seg: bool = False):
        self.ud_ma = self.get_ud_ma()

        if add_gold_seg:
            self.update_ma_with_gold(data_path=data_path)

        self.prefixes = self.load_hebrew_prefixes()

    def load_hebrew_prefixes(self):
        prefixes = {}
        with open(self.PREFIXES_FILE, encoding="utf8") as f:
            lines = f.readlines()
        for line in lines:
            splitted = line.split()
            prefixes[splitted[0]] = (splitted[1].split("^"), splitted[2].split("+"))
        return prefixes

    def get_ud_ma(self):
        with open(self.MA_FILE, encoding="utf8") as json_file:
            data = json.load(json_file)
            return data

    def is_equal_analysis(self, analysis, other_analysis):
        if len(analysis) != len(other_analysis):
            return False

        for seg in analysis.keys():
            if seg not in other_analysis:
                return False

        return True

    def any_equal_analysis(self, other_analyses, analysis):
        for op in other_analyses:
            if self.is_equal_analysis(analysis, op):
                return True
        return False

    def get_gold_ma(self, data_path: str):
        d = defaultdict(list)

        with open(data_path, encoding="utf-8") as f:
            lines = f.readlines()
            curr_segments = {}
            token_counter = 0

            for line in lines:
                if not line[0].isdigit():
                    continue

                splitted_line = line.split("\t")
                curr_token = splitted_line[1].replace('”', '"')
                if curr_token != "__":
                    curr_token = curr_token.replace("_", "")
                lemma = splitted_line[2]  # lemma form
                pos = splitted_line[3]
                features = self.get_all_features(splitted_line[5])

                if "-" in splitted_line[0]:
                    full_curr_token = curr_token
                    x = splitted_line[0].split("-")
                    token_counter = int(x[1]) - int(x[0]) + 1
                    continue
                elif token_counter != 0:
                    curr_segments[curr_token] = self.get_seg_analysis(pos, lemma, features)
                    token_counter -= 1

                    if token_counter == 0:
                        if full_curr_token not in d or not self.any_equal_analysis(d[full_curr_token], curr_segments):
                            d[full_curr_token].append(curr_segments)
                        curr_segments = {}
                elif curr_token not in d or not self.any_equal_analysis(d[curr_token], {curr_token: self.get_seg_analysis(pos, lemma, features)}):
                    d[curr_token].append({curr_token: self.get_seg_analysis(pos, lemma, features)})

        return d

    def get_seg_analysis(self, pos: str, lemma: str, features):
        return {"POS": [pos], "lemma": [lemma], "gender": [features["gender"]], "number": [features["number"]], "person": [features["person"]]}

    def get_all_features(self, features: str):
        f_lst = features.split("|")
        gender = "_"
        if "gen=F" in f_lst and "gen=M" in f_lst:
            gender = "Fem,Masc"
        elif "gen=F" in f_lst:
            gender = "Fem"
        elif "gen=M" in f_lst:
            gender = "Masc"

        number = "_"
        if "num=D" in f_lst and "num=P" in f_lst:
            number = "Dual,Plur"
        elif "num=S" in f_lst and "num=P" in f_lst:
            number = "Plur,Sing"
        elif "num=S" in f_lst:
            number = "Sing"
        elif "num=P" in f_lst:
            number = "Plur"
        elif "num=D" in f_lst:
            number = "Dual"

        person = "_"
        if "per=A" in f_lst:
            person = "1,2,3"
        elif "per=1" in f_lst:
            person = "1"
        elif "per=2" in f_lst:
            person = "2"
        elif "per=3" in f_lst:
            person = "3"

        all_suf_features = ""
        for f in f_lst:
            if f.startswith("suf_"):
                all_suf_features += f[-1]

        return {"gender": gender, "number": number, "person": person, "suffix": all_suf_features}

    def get_acronym_analysis(self, token: str):
        all_analyses = [{token: self.get_segment_analysis(token, "NOUN")}]
        # the acronym should be at least 2 characters + "
        max_pre_length = len(token) - 3
        for pre, (split_pre, pos_pre) in self.prefixes.items():
            if len(pre) <= max_pre_length:
                if token.startswith(pre):
                    analysis = {}
                    for letter, pos in zip(split_pre, pos_pre):
                        analysis[letter] = self.get_segment_analysis(letter, pos)
                    token_without_pre = token[len(pre):]
                    analysis[token_without_pre] = self.get_segment_analysis(token_without_pre, "NOUN")
                    all_analyses.append(analysis)
        return all_analyses

    def get_apostrophes_analysis(self, token: str):
        analysis = {}
        before_apos, after_apos = token.split('"')
        for pre, (split_pre, pos_pre) in self.prefixes.items():
            if pre == before_apos:
                for letter, pos in zip(split_pre, pos_pre):
                    analysis[letter] = self.get_segment_analysis(letter, pos)
                break
        analysis['"'] = self.get_segment_analysis('"', "PUNCT")
        if after_apos in self.ud_ma:
            for ana in self.ud_ma[after_apos]:
                if after_apos in ana:
                    analysis[after_apos] = ana[after_apos]
        else:
            analysis[after_apos] = self.get_segment_analysis(after_apos, "NOUN")

        return [analysis]

    def get_segment_analysis(self, segment: str, pos: str):
        return {"POS": [pos], "lemma": [segment], "gender": ["_"], "number": ["_"], "person": ["_"]}

    def update_ma_with_gold(self, data_path: str):
        gold_ma = self.get_gold_ma(data_path=data_path)
        for token, analyses in gold_ma.items():
            if token not in self.ud_ma:
                self.ud_ma[token] = analyses
            else:
                for a in analyses:
                    if not self.any_equal_analysis(self.ud_ma[token], a):
                        self.ud_ma[token].append(a)

    def get_token_analysis(self, token):
        if token in self.ud_ma:
            return self.ud_ma[token]

        num_match = re.search(r"\d+[\d.,_-]*", token)
        if num_match:
            start_offset = num_match.regs[0][0]
            end_offset = num_match.regs[0][1]
            if start_offset == 0 and len(token) == end_offset:
                return [{token: self.get_segment_analysis(token, "NUM")}]
            else:
                analysis = {}
                for letter in token[:start_offset]:
                    analysis[letter] = self.get_segment_analysis(letter, "ADP")

                analysis[token[start_offset:]] = self.get_segment_analysis(token[start_offset:], "NUM")
                return [analysis]

        if token == "%":
            return [{token: self.get_segment_analysis(token, "NOUN")}]

        if all(j in string.punctuation for j in token):
            return [{token: self.get_segment_analysis(token, "PUNCT")}]

        if '"' in token:
            if token[-2] == '"':
                return self.get_acronym_analysis(token)
            else:
                return self.get_apostrophes_analysis(token)

        # think about the pos for unknown tokens
        return [{token: self.get_segment_analysis(token, "NOUN")}]
