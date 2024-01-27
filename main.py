import os
import torch
import random
import argparse
import numpy as np

from loguru import logger
from transformers import BertModel, BertTokenizerFast

from tester import Tester
from trainer import Trainer
from models import DeepBiaffine
from data import TrainData, TestData


def set_seed(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() == 'cuda':
        torch.cuda.manual_seed_all(seed)


def main(mode: str, bert_model_name: str, lr: int, epochs: int, batch: int, train_path: str, eval_path: str,
         embeddings_path_load_eval: str, embeddings_path_load_train: str, model_path: str, test_tokenization_file: str,
         test_pos_file: str, add_gold_seg_test: bool, mtl_task: str, mtl_dim: int, head_mlp_dim: int, dep_mlp_dim: int,
         bilstm_dim: int, bilstm_layers: int, output_name: str, debug: bool, seed: int, steps_to_eval: int, gpu: int):
    set_seed(seed)
    device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else "cpu"
    logger.add(f"{output_name}.log")

    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name, output_hidden_states=True).to(device)

    for param in list(bert_model.parameters()):
        param.requires_grad = False

    pre_trained_embeddings = bert_model.embeddings.word_embeddings.weight.data
    embedding_dim = pre_trained_embeddings.size()[1]

    if mode == "test":
        model = torch.load(model_path, map_location=torch.device(device))

        test_data = TestData(device=device, gold_test_path=eval_path, test_tokenization_path=test_tokenization_file,
                             test_pos_path=test_pos_file, vocab=model.vocab, mtl_task=model.mtl_task,
                             tokenizer=bert_tokenizer, add_gold_seg=add_gold_seg_test, bert_model=bert_model,
                             embeddings_path_load=embeddings_path_load_eval)
        test_model = Tester(test_data=test_data, device=device, vocab=model.vocab)
        test_model.test(model)

    else:
        embeddings_vocab = bert_tokenizer.vocab

        train_data = TrainData(device=device, data_path=train_path, embeddings_vocab=embeddings_vocab,
                               mtl_task=mtl_task, tokenizer=bert_tokenizer, bert_model=bert_model,
                               embeddings_path_load=embeddings_path_load_train)

        test_data = TestData(device=device, gold_test_path=eval_path, test_tokenization_path=test_tokenization_file,
                             test_pos_path=test_pos_file, vocab=train_data.vocab, mtl_task=mtl_task,
                             tokenizer=bert_tokenizer, add_gold_seg=add_gold_seg_test, bert_model=bert_model,
                             embeddings_path_load=embeddings_path_load_eval)

        model = DeepBiaffine(embedding_dim=embedding_dim,
                             bilstm_layers=bilstm_layers,
                             dep_mlp_dim=dep_mlp_dim,
                             head_mlp_dim=head_mlp_dim,
                             bilstm_dim=bilstm_dim,
                             vocab=train_data.vocab,
                             mtl_task=mtl_task,
                             mtl_dim=mtl_dim)

        trainer = Trainer(model=model,
                          output_name=f"{output_name}_{seed}",
                          train_data=train_data,
                          test_data=test_data,
                          lr=lr,
                          vocab=train_data.vocab,
                          mtl_task=mtl_task,
                          steps_to_eval=steps_to_eval,
                          batch=batch,
                          n_ep=epochs,
                          device=device,
                          debug=debug)

        results = trainer.train()
        logger.info(f"Results of seed {seed}: {results}")
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aligner model')

    parser.add_argument('mode', help='choose test or train', type=str)
    parser.add_argument('-tokf', '--tokenization_file', help='tokenization file in data folder', default="data/gold_tokenization_test.txt", type=str)
    parser.add_argument('-posf', '--pos_file', help='pos file in data folder', default="data/gold_pos_test.txt", type=str)
    parser.add_argument('-mtl', '--mtl_task', help='concatenate one or more: pos, gender, number, person', type=str, default="pos")
    parser.add_argument('-ep', '--eval_path', help='path to the evaluation UD dataset', type=str, default="data/he_htb-ud-test.conllu")
    parser.add_argument('-tp', '--train_path', help='path to the train UD dataset', type=str, default="data/he_htb-ud-train.conllu")
    parser.add_argument('-mp', '--model_path', help='path to load model to test', type=str)
    parser.add_argument('-embeddings_path_load_eval', '--embeddings_path_load_eval', help='path to load from the embedding file for eval dataset', type=str)
    parser.add_argument('-embeddings_path_load_train', '--embeddings_path_load_train', help='path to load from the embedding file for train dataset', type=str)
    parser.add_argument('-seed', '--seed', help='seed', default=1, type=int)
    parser.add_argument('-s', '--steps_to_eval', help='number of steps to evaluate the model', default=500, type=int)
    parser.add_argument('--gpu', type=int, default=0, required=False)
    parser.add_argument('-dmlp', '--dep_mlp', help='dep mlp dimension', default=100, type=int)
    parser.add_argument('-hmlp', '--head_mlp', help='head mlp dimension', default=500, type=int)
    parser.add_argument('-bidim', '--bilstm_dim', help='out dimension of bilstm', default=600, type=int)
    parser.add_argument('-mtl_dim', '--mtl_dim', help='out dimension of mtl linear layer', default=600, type=int)
    parser.add_argument('-debug', '--debug', help='debug test time after number of steps', action='store_true')
    parser.add_argument('-o', '--output_name', help='output model name', type=str)
    parser.add_argument('-bl', '--bilstm_layers', help='number of layers of the bilstm', default=1, type=int)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=0.001, type=float)
    parser.add_argument('-e', '--epochs', help='number of epochs', default=30, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=32, type=int)
    parser.add_argument('-add_gold_seg_test', '--add_gold_seg_test', help='always contains gold segmentation in test input', action='store_true')
    parser.add_argument('-bm', '--bert_model_name', help='Bert model for contextualized embeddings and tokenizer', default="onlplab/alephbert-base", type=str)

    args = parser.parse_args()

    main(mode=args.mode,
         eval_path=args.eval_path,
         train_path=args.train_path,
         embeddings_path_load_eval=args.embeddings_path_load_eval,
         embeddings_path_load_train=args.embeddings_path_load_train,
         model_path=args.model_path,
         test_tokenization_file=args.tokenization_file,
         test_pos_file=args.pos_file,
         mtl_task=args.mtl_task,
         mtl_dim=args.mtl_dim,
         output_name=args.output_name,
         bilstm_layers=args.bilstm_layers,
         dep_mlp_dim=args.dep_mlp,
         head_mlp_dim=args.head_mlp,
         bilstm_dim=args.bilstm_dim,
         steps_to_eval=args.steps_to_eval,
         lr=args.learning_rate,
         epochs=args.epochs,
         batch=args.batch,
         gpu=args.gpu,
         seed=args.seed,
         debug=args.debug,
         add_gold_seg_test=args.add_gold_seg_test,
         bert_model_name=args.bert_model_name)
