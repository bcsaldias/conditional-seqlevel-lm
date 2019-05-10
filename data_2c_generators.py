import pandas as pd
import numpy as np
import re
import spacy
NLP = spacy.load('en')
MAX_CHARS = 20000

def tokenizer(comment):
    comment = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", 
        str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]
    return [
        x.text for x in NLP.tokenizer(comment) if x.text != " "]



import logging
import torch
from torchtext import data
LOGGER = logging.getLogger("reviews_dataset")

    

def get_dataset(fix_length=100, lower=False, vectors=None, prepare_data=False):
    
    if vectors is not None:
        # pretrain vectors only supports all lower cases
        lower = True
    
    LOGGER.debug("Preparing CSV files...")
    
    if prepare_data:
        prepare_csv()
    
    review_text = data.Field(
        sequential = True,
        #fix_length=fix_length,
        tokenize = tokenizer,
        pad_first = False,
        dtype = torch.int64,
        lower = lower,
        init_token = '<sos>',
        eos_token = '<eos>',
    )
    
    theme = data.Field(
                use_vocab=True, 
                sequential=False, 
                dtype=torch.int64)
    
    meta_id = data.Field(
                use_vocab=True, 
                sequential=False, 
                dtype=torch.int64)
    
    perspective = data.Field(
                use_vocab=True, 
                sequential=False, 
                dtype=torch.int64)

    fields=[
            ('meta_id', meta_id),
            ('review_text', review_text),
            ('theme', theme),
            ('perspective', perspective)]
    
    LOGGER.debug("Reading train csv file...")
    train, val = data.TabularDataset.splits(
        path='cache/', format='csv', skip_header=True,
        train='dataset_2c_train.csv', validation='dataset_2c_val.csv',
        fields = fields
        )
    
    LOGGER.debug("Reading test csv file...")
    test = data.TabularDataset(
        path='cache/dataset_2c_test.csv', format='csv', 
        skip_header=True, fields=fields)
    
    LOGGER.debug("Building vocabulary...")
    
    review_text.build_vocab(
        train, val, test,
        max_size=20000,
        min_freq=50,
        vectors=vectors
    )

    meta_id.build_vocab(
        train, val, test,
        max_size=float('inf'),
        min_freq=0,
    )
    
    theme.build_vocab(
        train, val, test,
        max_size=10,
        min_freq=0,
    )
    
    perspective.build_vocab(
        train, val, test,
        max_size=10,
        min_freq=0,
    )
    
    LOGGER.debug("Done preparing the datasets")
    return train, val, test, review_text, theme, perspective



def get_iterator(dataset, batch_size, train=True, shuffle=True, repeat=False):
    
    dataset_iter = data.Iterator(
        dataset, batch_size=batch_size, device='cuda',
        train=train, shuffle=shuffle, repeat=repeat,
        
        sort_key = lambda x: len(x.review_text),
        sort_within_batch=False,
        sort=True
    )
    
    return dataset_iter

















