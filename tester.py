import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tree_parser import mst
from data import Vocab, TestData
from data_utils import PAD_TOKEN
from eval_utils import calculate_scores, eval_segmentation, eval_prediction, las_evaluation, mtl_to_labels, \
    predictions_to_sentences_labels, remove_for_evaluation, head_ids_to_head_forms, remove_non_root, remove_padding


class Tester:
    BATCH_TEST_SIZE = 128

    def __init__(self, test_data: TestData, vocab: Vocab, device: str):
        self.test_dl = DataLoader(test_data, batch_size=self.BATCH_TEST_SIZE, collate_fn=self.pad_collate_test)
        self.ma_generator = test_data.ma_generator
        self.sentences = test_data.sentence_analyses
        self.sentences_id = test_data.sentences_id
        self.raw_sentences = test_data.raw_sentences
        self.vocab = vocab
        self.device = device

    def test(self, model):
        model.eval()

        # model's accuracy
        pred_pos_gold_counts, pred_pos_pred_counts, pred_pos_intersection_counts = 0, 0, 0
        dep_gold_counts, dep_pred_counts, dep_intersection_counts = 0, 0, 0
        head_gold_counts, head_pred_counts, head_intersection_counts = 0, 0, 0
        gender_gold_counts, gender_pred_counts, gender_intersection_counts = 0, 0, 0
        number_gold_counts, number_pred_counts, number_intersection_counts = 0, 0, 0
        person_gold_counts, person_pred_counts, person_intersection_counts = 0, 0, 0

        # model's input
        seg_gold_counts, seg_pred_counts, seg_intersection_counts = 0, 0, 0
        las_gold_counts, las_pred_counts, las_intersection_counts = 0, 0, 0

        # model's output
        all_predicted_seg = []
        all_predicted_head = []
        all_predicted_dep = []
        all_predicted_pos = []

        with torch.no_grad():
            for step, (embeddings, sentences_lens, analysis_mask, test_pos_input, test_sentence, test_pos,
                       gold_sentence, gold_pos, gold_dep, gold_head, gold_gender, gold_number, gold_person) in \
                    tqdm(enumerate(self.test_dl), total=len(self.test_dl)):

                batch_sentences = self.sentences[step*self.BATCH_TEST_SIZE: (step + 1)*self.BATCH_TEST_SIZE]
                head_predict, dep_predict, pos_predict, gender_predict, number_predict, person_predict = self.get_epoch_predictions(model, batch_sentences, sentences_lens, embeddings, analysis_mask, test_pos_input)

                test_sentence = remove_padding(test_sentence)
                test_pos = remove_padding(test_pos)

                all_predicted_seg.extend(remove_non_root(head_predict, test_sentence))
                all_predicted_head.extend(self.get_relevant_head_ids(head_predict, remove_non_root(head_predict, head_predict)))
                all_predicted_dep.extend(remove_non_root(head_predict, dep_predict))
                if model.mtl_task and "pos" in model.mtl_task:
                    all_predicted_pos.extend(remove_non_root(head_predict, pos_predict))
                else:
                    all_predicted_pos.extend(["_"]*len(all_predicted_seg[-1]))

                head_predict = head_ids_to_head_forms(test_sentence, head_predict)
                dep_predict = remove_for_evaluation(dep_predict, test_pos)
                head_predict_with_non_root = remove_for_evaluation(head_predict, test_pos)
                test_sentence = remove_for_evaluation(test_sentence, test_pos)

                head_predict = remove_non_root(head_predict_with_non_root, head_predict_with_non_root)
                dep_predict = remove_non_root(head_predict_with_non_root, dep_predict)
                test_sentence = remove_non_root(head_predict_with_non_root, test_sentence)

                gold_head = head_ids_to_head_forms(gold_sentence, gold_head)
                gold_sentence = remove_for_evaluation(gold_sentence, gold_pos)
                gold_dep = remove_for_evaluation(gold_dep, gold_pos)
                gold_head = remove_for_evaluation(gold_head, gold_pos)
                gold_gender = remove_for_evaluation(gold_gender, gold_pos)
                gold_number = remove_for_evaluation(gold_number, gold_pos)
                gold_person = remove_for_evaluation(gold_person, gold_pos)

                seg_gold_count, seg_pred_count, seg_intersection_count = eval_segmentation(test_sentence, gold_sentence)
                seg_gold_counts += seg_gold_count
                seg_pred_counts += seg_pred_count
                seg_intersection_counts += seg_intersection_count

                gold_counts, pred_counts, intersection_counts = eval_prediction(dep_predict, gold_dep,
                                                                                test_sentence, gold_sentence)
                dep_gold_counts += gold_counts
                dep_pred_counts += pred_counts
                dep_intersection_counts += intersection_counts

                gold_counts, pred_counts, intersection_counts = eval_prediction(head_predict, gold_head,
                                                                                test_sentence, gold_sentence)
                head_gold_counts += gold_counts
                head_pred_counts += pred_counts
                head_intersection_counts += intersection_counts

                if model.mtl_task:
                    if "pos" in model.mtl_task:
                        gold_pos_without_punct = remove_for_evaluation(gold_pos, gold_pos)
                        pos_predict = remove_for_evaluation(pos_predict, test_pos)
                        pos_predict = remove_non_root(head_predict_with_non_root, pos_predict)
                        gold_counts, pred_counts, intersection_counts = eval_prediction(pos_predict,
                                                                                        gold_pos_without_punct,
                                                                                        test_sentence, gold_sentence,)
                        pred_pos_gold_counts += gold_counts
                        pred_pos_pred_counts += pred_counts
                        pred_pos_intersection_counts += intersection_counts
                    if "gender" in model.mtl_task:
                        gender_predict = remove_for_evaluation(gender_predict, test_pos)
                        gender_predict = remove_non_root(head_predict_with_non_root, gender_predict)
                        gold_counts, pred_counts, intersection_counts = eval_prediction(gender_predict,
                                                                                        gold_gender, test_sentence,
                                                                                        gold_sentence)
                        gender_gold_counts += gold_counts
                        gender_pred_counts += pred_counts
                        gender_intersection_counts += intersection_counts
                    if "number" in model.mtl_task:
                        number_predict = remove_for_evaluation(number_predict, test_pos)
                        number_predict = remove_non_root(head_predict_with_non_root, number_predict)
                        gold_counts, pred_counts, intersection_counts = eval_prediction(number_predict,
                                                                                        gold_number, test_sentence,
                                                                                        gold_sentence)
                        number_gold_counts += gold_counts
                        number_pred_counts += pred_counts
                        number_intersection_counts += intersection_counts
                    if "person" in model.mtl_task:
                        person_predict = remove_non_root(head_predict_with_non_root, person_predict)
                        gold_counts, pred_counts, intersection_counts = eval_prediction(person_predict,
                                                                                        gold_person, test_sentence,
                                                                                        gold_sentence)
                        person_gold_counts += gold_counts
                        person_pred_counts += pred_counts
                        person_intersection_counts += intersection_counts

                gold_counts, pred_counts, intersection_counts = las_evaluation(dep_predict, gold_dep, head_predict,
                                                                               gold_head,
                                                                               test_sentence, gold_sentence)
                las_gold_counts += gold_counts
                las_pred_counts += pred_counts
                las_intersection_counts += intersection_counts

            print(f"segmentation input:")
            seg_score = calculate_scores(seg_gold_counts, seg_pred_counts, seg_intersection_counts)

            print(f"dependency label:")
            dep_score = calculate_scores(dep_gold_counts, dep_pred_counts, dep_intersection_counts)

            print(f"head:")
            head_score = calculate_scores(head_gold_counts, head_pred_counts, head_intersection_counts)

            print(f"head+dependency:")
            las_score = calculate_scores(las_gold_counts, las_pred_counts, las_intersection_counts)

            results = {"seg": seg_score, "head": head_score, "dep": dep_score, "las": las_score}

            if model.mtl_task:
                if "pos" in model.mtl_task or "udify" in model.mtl_task:
                    print("predicted pos:")
                    pos_score = calculate_scores(pred_pos_gold_counts, pred_pos_pred_counts, pred_pos_intersection_counts)
                    results["pos"] = pos_score
                if "gender" in model.mtl_task:
                    print("predicted gender:")
                    gender_score = calculate_scores(gender_gold_counts, gender_pred_counts, gender_intersection_counts)
                    results["gender"] = gender_score

                if "number" in model.mtl_task:
                    print("predicted number:")
                    number_score = calculate_scores(number_gold_counts, number_pred_counts, number_intersection_counts)
                    results["number"] = number_score

                if "person" in model.mtl_task:
                    print("predicted person:")
                    person_score = calculate_scores(person_gold_counts, person_pred_counts, person_intersection_counts)
                    results["person"] = person_score

            model.train()

            self.create_conll_file("prediction", all_predicted_seg, all_predicted_head, all_predicted_dep, all_predicted_pos)
            return results

    def pad_collate_test(self, batch):
        (embeddings, analysis_mask, pos_tensor, test_sentence, test_pos, gold_sentence, gold_pos, gold_dep, gold_head,
         gold_gender, gold_number, gold_person) = zip(*batch)
        sent_lens = [len(s) for s in test_sentence]

        p_pad = pad_sequence(pos_tensor, batch_first=True, padding_value=self.vocab.pos2i[PAD_TOKEN]).to(self.device)

        embed_pad = None
        if embeddings[0] is not None:
            embed_pad = pad_sequence(embeddings, batch_first=True, padding_value=self.vocab.word2i[PAD_TOKEN]).to(self.device)

        return embed_pad, sent_lens, analysis_mask, p_pad, test_sentence, test_pos, gold_sentence, gold_pos, gold_dep, gold_head, gold_gender, gold_number, gold_person

    def remove_impossible_analysis(self, predicted_head, analysis_masks):
        for i in range(predicted_head.shape[0]):
            curr_analysis = analysis_masks[i].to(self.device)
            curr_head = predicted_head[i].to(self.device)
            rows_addition1 = torch.full((2, curr_analysis.shape[1]), False).to(self.device)
            new_mask = torch.cat((rows_addition1, curr_analysis), dim=0).to(self.device)
            rows_addition2 = torch.full((curr_head.shape[0] - curr_analysis.shape[1], curr_analysis.shape[1]), False).to(self.device)
            new_mask = torch.cat((new_mask, rows_addition2), dim=0).to(self.device)
            cols_addition = torch.full((curr_head.shape[0], predicted_head[i].shape[0] - curr_analysis.shape[1]), False).to(self.device)
            padded_mask = torch.cat((new_mask, cols_addition), dim=1).to(self.device)
            predicted_head[i] = curr_head.masked_fill(padded_mask == True, float('-inf')).to(self.device)

        return predicted_head

    def add_missing_token_analysis(self, token, sentence_stats):
        sec_top = torch.topk(sentence_stats, 2, dim=1).values[:, 1]
        max_score = float('-inf')
        max_ana_ind = -1
        for i, ana in enumerate(token.analyses):
            second_analysis_stat = sec_top[ana.start: ana.end + 1]
            # ana_score = torch.mean(second_analysis_stat)
            ana_score = torch.max(second_analysis_stat)
            if max_score < ana_score:
                max_score = ana_score
                max_ana_ind = i

        for i, ana in enumerate(token.analyses):
            if i == max_ana_ind:
                sentence_stats[ana.start: ana.end + 1][:, 1:2] = float('-inf')
            else:
                sentence_stats[ana.start: ana.end + 1][:, 1:2] = float('inf')

        return sentence_stats

    def well_formed_segmentation(self, batch_sentences, head_output_improved):
        # The code below add segments of partial analyses
        for i, sent in enumerate(batch_sentences):
            curr_head_pred = torch.argmax(head_output_improved[i], dim=1)
            for token in sent.tokens:
                tokens_analysis_exists = 0
                for ana in token.analyses:
                    curr_analysis_pred = curr_head_pred[ana.start: ana.end + 1]

                    # check if any analysis exists for missing tokens
                    if torch.any(curr_analysis_pred != 1):
                        tokens_analysis_exists += 1

                        # update partial analysis to full analysis
                        if torch.any(curr_analysis_pred == 1):
                            head_output_improved[i][ana.start: ana.end + 1][:, 1:2] = float('-inf')

                if tokens_analysis_exists != 1:
                    head_output_improved[i] = self.add_missing_token_analysis(token, head_output_improved[i])

        return head_output_improved

    def get_relevant_head_ids(self, ids_with_non_root, ids_without_non_root):
        relevant = []
        for k, (curr_sent_with_root, curr_sent_without_root) in enumerate(zip(ids_with_non_root, ids_without_non_root)):
            ind_helper = list(range(2, len(curr_sent_with_root)+2))
            ind_helper_without_non_root = []
            for ind_with_non_root, j in zip(ind_helper, curr_sent_with_root):
                if j != 1:
                    ind_helper_without_non_root.append(ind_with_non_root)

            old_ind2new_ind = {old_ind: i+1 for i, old_ind in enumerate(ind_helper_without_non_root)}
            relevant_curr_sent = []
            for old_ind in curr_sent_without_root:
                if old_ind == 0:
                    relevant_curr_sent.append(0)
                else:
                    relevant_curr_sent.append(old_ind2new_ind[old_ind])

            relevant.append(relevant_curr_sent)
        return relevant

    def get_sentence_conll_format(self, sent_id, sent_raw, sent_segs, sent_heads, sent_labels, sent_pos):
        sent_format = f"# sent_id = {sent_id}\n# text = {sent_raw}\n"
        for i, (seg, head, label, pos) in enumerate(zip(sent_segs, sent_heads, sent_labels, sent_pos)):
            lemma = self.get_lemma(seg, pos)
            seg_line = f"{i+1}\t{seg}\t{lemma}\t{pos}\t{pos}\t_\t{head}\t{label}\t_\t_\n"
            sent_format += seg_line
        return sent_format

    def create_conll_file(self, file_name, segs, heads, labels, pos):
        file_content = ""

        for sent_id, sent_raw, sent_segs, sent_heads, sent_labels, sent_pos in zip(self.sentences_id, self.raw_sentences, segs, heads, labels, pos):
            sent_str = self.get_sentence_conll_format(sent_id, sent_raw, sent_segs, sent_heads, sent_labels, sent_pos)
            file_content += f"{sent_str}\n"

        with open(f'{file_name}.conllu', 'w', encoding="utf8") as f:
            f.write(file_content)

    def get_lemma(self, segment, pos):
        if len(segment) <= 2:
            return segment

        for analysis in self.ma_generator.get_token_analysis(segment):
            if len(analysis) == 1 and pos in analysis[segment]["POS"]:
                return analysis[segment]["lemma"][0]

        return segment

    def well_formed_heads(self, head_output_improved):
        non_root_mask = (torch.argmax(head_output_improved, dim=2) == 1)
        for i in range(non_root_mask.shape[0]):
            curr_sent_col_mask = torch.stack([non_root_mask[i]] * non_root_mask.shape[1])
            # prevent the chosen segments to move to the non root (could happen in MST)
            curr_sent_col_mask[:, 1:2] = curr_sent_col_mask[:, 1:2].squeeze(dim=1).masked_fill(non_root_mask[i] == False, float('-inf')).unsqueeze(dim=1)
            head_output_improved[i] = head_output_improved[i].masked_fill(curr_sent_col_mask, float('-inf'))
        return head_output_improved

    def get_epoch_predictions(self, model, batch_sentences, data_lens, embeddings, analysis_mask, pos):
        dep_output, head_output, pos_output, gender_output, number_output, person_output = model(embeddings, data_lens)
        vocab = model.vocab
        mask = pos.view(-1) != vocab.pos2i[PAD_TOKEN]
        # -inf prob for padding rows
        head_output[:, :2, 1:] = float('-inf')

        # Constraints on output
        head_output_improved = self.remove_impossible_analysis(head_output, analysis_mask)
        head_output_improved = self.well_formed_segmentation(batch_sentences, head_output_improved)
        # remove head which are not in the final output (their head is the non root)
        head_output_improved = self.well_formed_heads(head_output_improved)

        predicted_head = mst(head_output_improved, pos != vocab.pos2i[PAD_TOKEN], self.device)

        # mask labels prediction
        masked_predictions_dep = dep_output[mask]
        masked_predictions_dep = masked_predictions_dep[torch.arange(len(predicted_head)), predicted_head]
        _, predicted_dep = torch.max(masked_predictions_dep, 1)

        # remove 2 from all length for root and non root deletion
        data_lens = [sent_len - 2 for sent_len in data_lens]
        head_predict, dep_predict = predictions_to_sentences_labels(data_lens, predicted_head, predicted_dep, vocab.i2dep)

        pos_predict = None
        gender_predict = None
        number_predict = None
        person_predict = None
        if model.mtl_task:
            if "pos" in model.mtl_task:
                masked_predictions_pos = pos_output[mask]
                _, predicted_pos = torch.max(masked_predictions_pos, 1)
                pos_predict = mtl_to_labels(data_lens, predicted_pos, vocab.i2pos)
            if "gender" in model.mtl_task:
                masked_predictions_gender = gender_output[mask]
                _, predicted_gender = torch.max(masked_predictions_gender, 1)
                gender_predict = mtl_to_labels(data_lens, predicted_gender, vocab.i2gender)
            if "number" in model.mtl_task:
                masked_predictions_number = number_output[mask]
                _, predicted_number = torch.max(masked_predictions_number, 1)
                number_predict = mtl_to_labels(data_lens, predicted_number, vocab.i2number)
            if "person" in model.mtl_task:
                masked_predictions_person = person_output[mask]
                _, predicted_person = torch.max(masked_predictions_person, 1)
                person_predict = mtl_to_labels(data_lens, predicted_person, vocab.i2person)

        return head_predict, dep_predict, pos_predict, gender_predict, number_predict, person_predict
