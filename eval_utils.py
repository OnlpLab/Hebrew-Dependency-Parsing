from collections import Counter
from data_utils import PAD_TOKEN


def remove_for_evaluation(sentences_label, gold_pos):
    labels = []
    for sent_label, sent_pos in zip(sentences_label, gold_pos):
        curr_sent_label = []
        for word_label, word_pos in zip(sent_label, sent_pos):
            if word_pos not in ["PUNCT"]:
                curr_sent_label.append(word_label)
        labels.append(curr_sent_label)

    return labels


def remove_padding(data_seq):
    data_seq = [seq[2:] for seq in data_seq]
    return data_seq


def head_ids_to_head_forms(sentences, head_ids):
    head_form = []
    for j, (head_sent, sentence) in enumerate(zip(head_ids, sentences)):
        curr_sent = []
        for i, head in enumerate(head_sent):
            if head == 0:
                curr_sent.append("root")
            elif head == 1 or (head - 2 >= len(sentence)):
                curr_sent.append("non root")
            else:
                curr_sent.append(sentence[head - 2])
        head_form.append(curr_sent)
    return head_form


def remove_non_root(head, tag):
    new_tag = []

    for sent_h, sent_t in zip(head, tag):
        curr_t = []
        for h, t in zip(sent_h, sent_t):
            if h != "non root" and h != 1:
                curr_t.append(t)
        new_tag.append(curr_t)

    return new_tag


def calculate_scores(gold_counts, pred_counts, intersection_counts):
    precision = intersection_counts / pred_counts if pred_counts else 0.0
    recall = intersection_counts / gold_counts if gold_counts else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
    print(f"precision: {round(precision*100, 2)}")
    print(f"recall: {round(recall*100, 2)}")
    print(f"f1: {round(f1*100, 2)}")
    print()
    return round(f1 * 100, 2)


def eval_segmentation(test_sentences, gold_sentences):
    gold_counts, pred_counts, intersection_counts = 0, 0, 0

    for i, (test_sent, gold_sent) in enumerate(zip(test_sentences, gold_sentences)):

        gold_count, pred_count = Counter(gold_sent), Counter(test_sent)
        del gold_count[PAD_TOKEN]
        del pred_count[PAD_TOKEN]

        intersection_count = gold_count & pred_count

        gold_counts += sum(gold_count.values())
        pred_counts += sum(pred_count.values())
        intersection_counts += sum(intersection_count.values())

    return gold_counts, pred_counts, intersection_counts


def las_evaluation(dep_predict, gold_dep, head_predict, gold_head, test_sentence, gold_sentence):
    gold_counts, pred_counts, intersection_counts = 0, 0, 0
    for i, (p_dep, g_dep, p_head, g_head, p_sentence, g_sentence) in enumerate(zip(dep_predict, gold_dep, head_predict, gold_head, test_sentence, gold_sentence)):
        pred_pair = [(word, head, dep) for word, head, dep in zip(p_sentence, p_head, p_dep)]
        gold_pair = [(word, head, dep) for word, head, dep in zip(g_sentence, g_head, g_dep)]

        gold_count, pred_count = Counter(gold_pair), Counter(pred_pair)
        intersection_count = gold_count & pred_count

        gold_counts += sum(gold_count.values())
        pred_counts += sum(pred_count.values())
        intersection_counts += sum(intersection_count.values())

    return gold_counts, pred_counts, intersection_counts


def eval_prediction(pred_tag, gold_tag, test_sentence, gold_sentence):
    gold_counts, pred_counts, intersection_counts = 0, 0, 0
    for p_tag, g_tag, p_sent, g_sent in zip(pred_tag, gold_tag, test_sentence, gold_sentence):
        pred_pair = [(word, tag) for word, tag in zip(p_sent, p_tag)]
        gold_pair = [(word, pos) for word, pos in zip(g_sent, g_tag)]

        gold_count, pred_count = Counter(gold_pair), Counter(pred_pair)
        intersection_count = gold_count & pred_count

        gold_counts += sum(gold_count.values())
        pred_counts += sum(pred_count.values())
        intersection_counts += sum(intersection_count.values())

    return gold_counts, pred_counts, intersection_counts


def predictions_to_sentences_labels(data_lens, head_pred, dep_pred, i2dep):
    sentences_head = []
    sentences_dep = []
    for i, sent_len in enumerate(data_lens):
        start_offset = sum(data_lens[:i])
        sent_head = head_pred[start_offset: start_offset + sent_len].tolist()
        sent_dep_ind = dep_pred[start_offset: start_offset + sent_len].tolist()
        sent_dep = [i2dep[dep] for dep in sent_dep_ind]
        sentences_head.append(sent_head)
        sentences_dep.append(sent_dep)

    return sentences_head, sentences_dep


def mtl_to_labels(data_lens, pred, i2label):
    sent_tags = []

    for i, sent_len in enumerate(data_lens):
        start_offset = sum(data_lens[:i])
        sent_tag_ind = pred[start_offset: start_offset + sent_len].tolist()
        sent_tag = [i2label[ner] for ner in sent_tag_ind]
        sent_tags.append(sent_tag)

    return sent_tags
