# from fastNLP.embeddings import StaticEmbedding
# from fastNLP.io import CSVLoader
# from fastNLP import Const
# import numpy as np
# import fitlog
# import pickle

import os

from fastNLP_module import StaticEmbedding
from fastNLP import cache_results
from fastNLP import Vocabulary


@cache_results(_cache_fp='cache/databaker_tn', _refresh=False)
def load_databaker_tn(path, char_embedding_path=None, bigram_embedding_path=None, index_token=True,
                    char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0):

    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    train_path = os.path.join(path, 'train.char.bmes')
    dev_path = os.path.join(path, 'dev.char.bmes')
    test_path = os.path.join(path, 'test.char.bmes')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)


    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']


    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()

    print(datasets.keys())
    print('CEHCK dev dataset LENGTH', len(datasets['dev']))
    print('CEHCK test dataset LENGTH', len(datasets['test']))
    print('CEHCK train dataset LENGTH', len(datasets['train']))

    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']] )
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])

    # label vocab do not take test/dev datasets as additional input
    # label_vocab.from_dataset(datasets['train'], field_name='target')
    label_vocab.from_dataset(datasets['train'],field_name='target',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    print('CHECK label_vocab IN load_databaker_tn:', len(label_vocab))
    print('CHECK label_vocab IN load_databaker_tn:', label_vocab._word2idx)

    if index_token:
        char_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                 field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                 field_name='target', new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
                                         min_freq=char_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings
