import torch
import torch.nn as nn

from tqdm import tqdm
from torch import optim
from tester import Tester
from torch.utils.data import DataLoader
from data import Vocab, TestData, TrainData
from torch.nn.utils.rnn import pad_sequence
from data_utils import PAD_TOKEN, PADDING_FOR_PREDICTION
import time


class Trainer:
    def __init__(self, model: nn.Module, output_name: str, train_data: TrainData, test_data: TestData, vocab: Vocab,
                 mtl_task: str, n_ep: int, lr: float, batch: int, steps_to_eval: int, device: str, debug=False):
        self.device = device
        print(f"Device: {self.device}")
        self.model = model.to(device)
        self.patience = 15
        self.vocab = vocab
        self.debug = debug
        self.n_epochs = n_ep
        self.mtl_task = mtl_task
        self.train_batch_size = batch
        self.steps_to_eval = steps_to_eval

        self.train_data = DataLoader(train_data, batch_size=self.train_batch_size, collate_fn=self.pad_collate)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=PADDING_FOR_PREDICTION)
        self.tester = Tester(test_data=test_data, vocab=self.vocab, device=self.device)

        self.output_name = f"{output_name}.bin"
        self.best_score = -1
        self.train_time_sum = 0
        self.train_time_max = 0

    def train(self):
        num_samples = 0
        for epoch in range(self.n_epochs):
            print(f"start epoch: {epoch + 1}")
            pos_train_loss = 0.0
            dep_train_loss = 0.0
            head_train_loss = 0.0
            self.model.train()

            for step, (word_embeddings, sentences_lens, pos, dep, head, gender, person, number) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
                pos_train_loss = 0.0
                gender_train_loss = 0.0
                number_train_loss = 0.0
                person_train_loss = 0.0
                num_samples += 1
                start = time.time()
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                self.model.zero_grad()

                dep_output, head_output, pos_output, gender_output, number_output, person_output = self.model(word_embeddings, sentences_lens)

                head_output = head_output.flatten(0, 1)
                mask = dep.view(-1) != self.vocab.dep2i[PAD_TOKEN]
                head_train_loss, dep_train_loss = self.loss_model_output(head_output, head, dep_output, dep, mask)

                if self.mtl_task:

                    if "pos" in self.mtl_task or "udify" in self.mtl_task:
                        pos_train_loss = self.loss_mtl(pos_output, pos, mask)
                    if "gender" in self.mtl_task:
                        gender_train_loss = self.loss_mtl(gender_output, gender, mask)
                    if "number" in self.mtl_task:
                        number_train_loss = self.loss_mtl(number_output, number, mask)
                    if "person" in self.mtl_task:
                        person_train_loss = self.loss_mtl(person_output, person, mask)

                loss = head_train_loss + pos_train_loss + dep_train_loss + gender_train_loss + number_train_loss + person_train_loss

                loss.backward()
                loss = 0
                self.optimizer.step()
                end = time.time()
                train_time = end - start
                self.train_time_sum += train_time
                if train_time > self.train_time_max:
                    self.train_time_max = train_time

                if self.debug and num_samples >= self.steps_to_eval:
                    num_samples = 0
                    print(f"in step: {step+1} pos train loss: {pos_train_loss}")
                    print(f"in step: {step+1} dep train loss: {dep_train_loss}")
                    print(f"in step: {step+1} head train loss: {head_train_loss}")
                    pos_train_loss = 0.0
                    dep_train_loss = 0.0
                    head_train_loss = 0.0
                    self.tester.test(self.model)

            print(f"in epoch: {epoch + 1} pos train loss: {pos_train_loss}")
            print(f"in epoch: {epoch + 1} head train loss: {head_train_loss}")
            print(f"in epoch: {epoch + 1} dep train loss: {dep_train_loss}")

            print(f"Dev data evaluation in epoch {epoch + 1}:")
            results = self.tester.test(self.model)
            las_score = results["las"]
            if las_score > self.best_score:
                self.best_score = las_score
                torch.save(self.model, self.output_name)
            print("\n")

        print("The best results are:")
        best_model = torch.load(self.output_name, map_location=torch.device(self.device))
        best_results = self.tester.test(best_model)
        print(f"maximum epoch: {self.train_time_max}")
        print(f"sum time: {self.train_time_sum}")
        print(f"AVG time: {(self.train_time_sum/self.n_epochs)}")
        return best_results

    def loss_mtl(self, pred, gold, mask):
        masked_gold = torch.masked_select(gold.view(-1), mask)
        masked_predictions = pred[mask]
        loss_score = self.loss_func(masked_predictions, masked_gold)
        return loss_score

    def loss_model_output(self, pred_head, gold_head, pred_dep, gold_dep, mask):
        masked_gold_head = torch.masked_select(gold_head.view(-1), mask)
        masked_gold_dep = torch.masked_select(gold_dep.view(-1), mask)
        masked_predictions_head = pred_head[mask]
        masked_predictions_dep = pred_dep[mask]
        masked_predictions_dep = masked_predictions_dep[torch.arange(len(masked_gold_head)), masked_gold_head]
        loss_head = self.loss_func(masked_predictions_head, masked_gold_head)
        loss_dep = self.loss_func(masked_predictions_dep, masked_gold_dep)

        return loss_head, loss_dep

    def pad_collate(self, batch):
        (embeddings, pos, dep, head, gender, person, number) = zip(*batch)
        sent_lens = [len(s) for s in dep]

        embed_pad = None
        if embeddings[0] is not None:
            embed_pad = pad_sequence(embeddings, batch_first=True, padding_value=self.vocab.word2i[PAD_TOKEN]).to(self.device)

        p_pad = pad_sequence(pos, batch_first=True, padding_value=self.vocab.pos2i[PAD_TOKEN]).to(self.device)
        d_pad = pad_sequence(dep, batch_first=True, padding_value=self.vocab.dep2i[PAD_TOKEN]).to(self.device)
        h_pad = pad_sequence(head, batch_first=True, padding_value=PADDING_FOR_PREDICTION).to(self.device)

        g_pad = None
        if gender[0] is not None:
            g_pad = pad_sequence(gender, batch_first=True, padding_value=self.vocab.gender2i[PAD_TOKEN]).to(self.device)

        n_pad = None
        if number[0] is not None:
            n_pad = pad_sequence(number, batch_first=True, padding_value=self.vocab.number2i[PAD_TOKEN]).to(self.device)

        per_pad = None
        if person[0] is not None:
            per_pad = pad_sequence(person, batch_first=True, padding_value=self.vocab.person2i[PAD_TOKEN]).to(self.device)

        return embed_pad, sent_lens, p_pad, d_pad, h_pad, g_pad, per_pad, n_pad
