import os
import argparse
import sys
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, Optional, Sequence, Union
from datetime import datetime

from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import gluonnlp as nlp
from tqdm import tqdm, tqdm_notebook

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


from arg_parser import build_default_arg_parser
from kobertt import BERTDataset,BERTClassifier,BERTSentenceTransform


def main() -> None:
    args = build_default_arg_parser().parse_args()
    run(args)

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def get_tag(name: str) -> str:
    now = datetime.now()
    return f"{name}_run-{now}"

def preprocess(path):
  df = pd.read_csv(path)

  df = df.replace('부정',int(0))
  df = df.replace('긍정',int(1))
  df = df.replace('중립',int(2))
  
  data_list = []
  for q, label in zip(df['review'], df['polarity']):
      data = []
      data.append(q)
      data.append(str(label))

      data_list.append(data)
  return data_list


def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if directory is not None and tag is not None:
        os.makedirs(name=directory, exist_ok=True)

        main_log_path = os.path.join(directory, f"{tag}.log")
        fh_main = logging.FileHandler(main_log_path)
        fh_main.setLevel(level)
        fh_main.setFormatter(formatter)
        logger.addHandler(fh_main)

def run(args: argparse.Namespace) -> None:
    data_list = preprocess(args.train_path)

    dataset_train, dataset_test = train_test_split(data_list, test_size=0.25, random_state=0)
    tag = get_tag('capstone')
    
    
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')


    data_train = BERTDataset(dataset_train, 0, 1, tokenizer, vocab, args.max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tokenizer, vocab, args.max_len, True, False)
    
    
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, num_workers=5)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, num_workers=5)
    
    
    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(args.device)



    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    t_total = len(train_dataloader) * args.num_epochs
    warmup_step = int(t_total * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
    
    setup_logger(tag=tag, directory=os.getcwd())
    
    for e in range(args.num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(args.device)
            segment_ids = segment_ids.long().to(args.device)
            valid_length= valid_length
            label = label.long().to(args.device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step() 
            train_acc += calc_accuracy(out, label)
            if batch_id % args.log_interval == 0:
                logging.info("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
        logging.info("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(args.device)
            segment_ids = segment_ids.long().to(args.device)
            valid_length= valid_length
            label = label.long().to(args.device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        logging.info("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))


        torch.save(model,'capstone.model')
        
if __name__ == "__main__":
    main()
        