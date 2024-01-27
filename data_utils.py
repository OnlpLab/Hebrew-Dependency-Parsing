import re
import os
import torch

from typing import List
from collections import Counter
from dataclasses import dataclass
from torch import Tensor, block_diag

PAD_TOKEN = "[PAD]"
PADDING_FOR_PREDICTION = -100
NON_ROOT_DEP = "non root"
GENDER_RE = r"Gender=([A-Za-z,]*)"
NUMBER_RE = r"Number=([A-Za-z,]*)"
PERSON_RE = r"Person=([1-3,]*)"
VERB_FORM_RE = r"VerbForm=([A-Za-z]*)"
not_hebrew = '!#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))


def to_cut(i, c, t):
    if c not in not_hebrew and c not in "0123456789" and i != len(t) - 1 and t[i + 1] in "0123456789":
        return True
    if c in not_hebrew:
        return True
    if c == '"' and (i == 0 or i == len(t) - 1 or not (t[i - 1] not in not_hebrew and t[i + 1] not in not_hebrew)):
        return True

    return False


def get_hebrew_tokens(sentence):
    sentence = sentence.strip()
    splitted_sent = sentence.split()
    tokens = []
    for t in splitted_sent:
        last_p = 0
        for i, c in enumerate(t):
            if to_cut(i, c, t):
                if last_p == i:
                    tokens.append(t[last_p:i + 1])
                    last_p = i + 1
                else:
                    tokens.append(t[last_p:i])
                    last_p = i
                    if i != len(t) - 1 and t[i + 1] not in not_hebrew:
                        tokens.append(t[i:i + 1])
                        last_p = i + 1
        tokens.append(t[last_p:])
    return tokens


def get_tokens_data(data_file):
    with open(data_file, encoding="utf8") as f:
        lines = f.readlines()
    sentences = []
    curr_seq = [PAD_TOKEN]
    for line in lines:
        if line == "\n":
            sentences.append(clean_data(curr_seq))
            curr_seq = [PAD_TOKEN]
            continue
        curr_seq.append(line.strip())
    return sentences


def extract_features(features):
    gender = "_"
    number = "_"
    person = "_"
    verb_form = "_"
    if features != "_":
        re_gender = re.search(GENDER_RE, features)
        if re_gender:
            gender = re_gender.group(1)

        re_number = re.search(NUMBER_RE, features)
        if re_number:
            number = re_number.group(1)

        re_person = re.search(PERSON_RE, features)
        if re_person:
            person = re_person.group(1)

        re_verb_form = re.search(VERB_FORM_RE, features)
        if re_verb_form:
            verb_form = re_verb_form.group(1)
    return gender, number, person, verb_form


def clean_data(sentence):
    for i, w in enumerate(sentence):
        if "_" in w and len(w) > 1:
            clean_word = w.replace("_", "")
            if len(clean_word) > 0:
                sentence[i] = clean_word
    return sentence


@dataclass
class Analysis:
    analysis: List[str]
    start: int
    end: int


@dataclass
class Token:
    token: str
    start: int
    end: int
    gold_segmentation: List
    analyses: List[Analysis]


@dataclass
class Sentence:
    sent_id: str  # In data
    tokens: List[Token]


class DataProcessor:

    def __init__(self, ma_generator, train=False):
        self.ma_generator = ma_generator
        self.train = train

    def initialize_curr_parameters(self, gold):
        if gold:
            # add padding for demo token (for root word)
            self.curr_sentence_segments = []
            self.curr_tokens = []
            self.curr_tokens_raw = []
            self.curr_mask = []
            self.curr_pos = []
            self.curr_deps = []
            self.curr_head = []
            self.curr_genders = []
            self.curr_numbers = []
            self.curr_persons = []

            # helpers for updating the heads after optional segments addition
            self.original_heads_helper = []
            self.new_heads_helper = []
        else:
            # add padding for aux token (for root word)
            self.curr_sentence_segments = [PAD_TOKEN, PAD_TOKEN]
            self.curr_tokens = []
            self.curr_tokens_raw = [PAD_TOKEN, PAD_TOKEN]
            self.curr_mask = [Tensor([[0]]), Tensor([[0]])]
            self.curr_pos = [PAD_TOKEN, PAD_TOKEN]
            self.curr_deps = [PAD_TOKEN, PAD_TOKEN]
            self.curr_head = [PADDING_FOR_PREDICTION, PADDING_FOR_PREDICTION]
            self.curr_genders = [PAD_TOKEN, PAD_TOKEN]
            self.curr_numbers = [PAD_TOKEN, PAD_TOKEN]
            self.curr_persons = [PAD_TOKEN, PAD_TOKEN]

            # helpers for updating the heads after optional segments addition
            self.original_heads_helper = []
            self.new_heads_helper = [PADDING_FOR_PREDICTION, PADDING_FOR_PREDICTION]

    def get_most_common_feature(self, feature_options):
        f = Counter(feature_options).most_common(1)[0][0]
        return f

    def add_analysis(self, analysis, gold_analysis):
        curr_gold = self.ma_generator.is_equal_analysis(analysis, gold_analysis)

        for segment, segment_analysis in analysis.items():
            if not curr_gold:
                self.curr_deps.append(NON_ROOT_DEP)

                self.curr_pos.append(self.get_most_common_feature(segment_analysis["POS"]))
                self.curr_genders.append(self.get_most_common_feature(segment_analysis["gender"]))
                self.curr_numbers.append(self.get_most_common_feature(segment_analysis["number"]))
                self.curr_persons.append(self.get_most_common_feature(segment_analysis["person"]))
                self.curr_sentence_segments.append(segment)
                self.new_heads_helper.append(segment)

            else:
                if self.train:
                    self.curr_pos.append(gold_analysis[segment]["pos"])
                else:
                    # use predicted POS for gold segmentation in test time
                    self.curr_pos.append(self.get_most_common_feature(segment_analysis["POS"]))
                self.curr_deps.append(gold_analysis[segment]["dep"])
                self.curr_genders.append(gold_analysis[segment]["gender"])
                self.curr_numbers.append(gold_analysis[segment]["number"])
                self.curr_persons.append(gold_analysis[segment]["person"])
                self.curr_sentence_segments.append(segment)
                self.new_heads_helper.append(f"{segment}_{gold_analysis[segment]['id_token']}")

    def add_token_analyses(self, token, true_segments):
        analyses = self.ma_generator.get_token_analysis(token)
        analyses.sort(key=lambda x: len(x), reverse=True)

        token_analyses_segments = []
        token_analyses = []
        start_token_ind = len(self.curr_sentence_segments)
        added_segments = 0

        for a in analyses:
            # if (not self.only_gold_analysis) or (list(a.keys()) == list(true_segments.keys())):
            # if list(a.keys()) == list(true_segments.keys()):
            # for token object
            start_ana_ind = len(self.curr_sentence_segments)
            curr_analysis = list(a.keys())
            added_segments += len(curr_analysis)
            token_analyses.append(Analysis(analysis=curr_analysis, start=start_ana_ind, end=start_ana_ind + len(curr_analysis) - 1))

            # got token segments (text)
            token_analyses_segments.append(list(a.keys()))
            self.add_analysis(a, true_segments)

        token_mask = self.get_token_mask(token_analyses_segments)
        self.curr_mask.append(token_mask)
        self.curr_tokens.append(Token(token=token, start=start_token_ind, end=start_token_ind + added_segments - 1, gold_segmentation=list(true_segments.keys()), analyses=token_analyses))

    def get_sentence_mask(self):
        mask = block_diag(*self.curr_mask)
        mask = mask[2:, :]
        mask = mask.to(torch.bool)
        return mask

    def get_token_mask(self, token_analyses):
        """
        :return: mask with True for segments in same analysis, otherwise False
        """
        mask = []
        for analysis1 in token_analyses:
            for seg1 in analysis1:
                seg1_mask = []
                for analysis2 in token_analyses:
                    if analysis1 == analysis2:
                        for seg2 in analysis2:
                            if seg1 == seg2:
                                seg1_mask.append(True)
                            else:
                                seg1_mask.append(False)
                    else:
                        seg1_mask.extend([True]*len(analysis2))

                mask.append(seg1_mask)

        return Tensor(mask)

    def get_curr_heads(self):
        segment2index = {}
        # update indexes to non-location based format
        for (segment, head_index) in self.original_heads_helper:
            if head_index == 0:
                segment2index[segment] = head_index
            else:
                head_segment, _ = self.original_heads_helper[head_index-1]
                segment2index[segment] = head_segment

        curr_indexes = []
        for token in self.new_heads_helper:
            if token in segment2index:

                head_token = segment2index[token]
                if head_token == 0:
                    head = 0
                else:
                    if head_token in self.new_heads_helper:
                        head = self.new_heads_helper.index(head_token)
                    else:
                        # when gold segmentation is missing
                        head = -1

                curr_indexes.append(head)

            elif token == PADDING_FOR_PREDICTION:
                curr_indexes.append(PADDING_FOR_PREDICTION)

            else:
                curr_indexes.append(1)

        return curr_indexes

    def get_info(self, splitted_line):
        id_token = splitted_line[0]
        form = splitted_line[1]
        if form != "__":
            form = form.replace("_", "")
        lemma = splitted_line[2]  # lemma form
        pos = splitted_line[3]
        head_id = int(splitted_line[6])
        deprel = splitted_line[7]
        features = splitted_line[5]
        gender, number, person, _ = extract_features(features)
        return id_token, form, lemma, pos, deprel, head_id, features, gender, number, person

    def get_data(self, file_path, gold=False):
        sentences_id = []
        sentences_raw = []
        masks = []
        sentences = []
        sentences_segments = []
        poses = []
        deps = []
        heads = []
        genders = []
        numbers = []
        persons = []
        curr_segments = {}
        token_counter = 0

        self.initialize_curr_parameters(gold=gold)

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
            not_tokens_indexes = set()

            for line in lines:

                if line == "\n":
                    not_tokens_indexes = set()
                    sentences.append(Sentence(sent_id=sentences_id[-1], tokens=self.curr_tokens))
                    sentences_segments.append(clean_data(self.curr_sentence_segments))
                    masks.append(self.get_sentence_mask())
                    poses.append(self.curr_pos)
                    deps.append(self.curr_deps)
                    genders.append(self.curr_genders)
                    numbers.append(self.curr_numbers)
                    persons.append(self.curr_persons)

                    if not gold:
                        self.curr_head = self.get_curr_heads()
                    heads.append(self.curr_head)

                    self.initialize_curr_parameters(gold)
                    continue

                if line.startswith("# sent_id"):
                    sentences_id.append(line[12:].strip())

                if line.startswith("# text = "):
                    sentences_raw.append(line[9:].strip())

                if line.startswith("#"):
                    continue

                splitted_line = line.split("\t")
                curr_token = splitted_line[1].replace('”', '"')

                # create list of tokens (in dataset the tokenization is sometimes bad so we need to do it like it)
                if "-" in splitted_line[0]:
                    x = splitted_line[0].split("-")
                    for k in range(int(x[0]), int(x[1]) + 1):
                        not_tokens_indexes.add(k)
                    self.curr_tokens_raw.append(curr_token)
                elif int(splitted_line[0]) not in not_tokens_indexes:
                    self.curr_tokens_raw.append(curr_token)

                if curr_token != "__":
                    curr_token = curr_token.replace("_", "")

                # if there is - so the token contains multiple tokens (will be added separately)
                if "-" not in splitted_line[0] and "." not in splitted_line[0]:
                    id_token, form, lemma, pos, deprel, head_id, feature, gender, number, person = self.get_info(splitted_line)
                    if gold:
                        self.curr_sentence_segments.append(form)
                        self.curr_pos.append(pos)
                        self.curr_deps.append(deprel)
                        self.curr_genders.append(gender)
                        self.curr_numbers.append(number)
                        self.curr_persons.append(person)
                    else:
                        self.original_heads_helper.append((f"{form}_{id_token}", head_id))

                    # for adding non root
                    if head_id != 0:
                        head_id += 1
                    self.curr_head.append(head_id)

                # code for inserting optional segments
                # line which describes token that will be divided in next lines
                if not gold:
                    if "-" in splitted_line[0]:
                        full_curr_token = curr_token
                        x = splitted_line[0].split("-")
                        token_counter = int(x[1]) - int(x[0]) + 1

                    # add tokens that are part of another token
                    elif token_counter != 0:
                        curr_segments[curr_token] = {"dep": deprel, "head": head_id, "pos": pos, "id_token": id_token, "gender": gender, "number": number, "person": person}
                        token_counter -= 1

                        if token_counter == 0:
                            self.add_token_analyses(full_curr_token, curr_segments)
                            curr_segments = {}

                    # add tokens that are not divided to segments
                    else:
                        self.add_token_analyses(curr_token, {curr_token: {"dep": deprel, "head": head_id, "pos": pos, "id_token": id_token, "gender": gender, "number": number, "person": person}})

        return sentences_id, sentences_raw, sentences, sentences_segments, masks, poses, deps, heads, genders, numbers, persons
