#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 22:46:10 2017

@author: nymph332088
@class: 
  BatchLoaderPairsMD: using mongodb to store training and testing pairs, tensors
"""

from os import path
import numpy as np
import pandas as pd
from ReprsReader import word_rep, word_rep_bin
from scipy.spatial import distance
import scipy.sparse as ss
import time
from collections import Counter
from pymongo import MongoClient
client = MongoClient()
UNK = '_'
def unpack(fn):
  data = pd.read_pickle(fn)
  return data['train_phrases'], data['test_phrases'], \
        data['idx2chars'], data['chars2idx']

        
class BatchLoaderPairs(object):
  def __init__(self, opt, train_phrases, test_phrases):
    self.data_dir = opt.data_dir
    
    ## mongo database to store: pairs, tensors.
    
    
    vocab_file = path.join(opt.data_dir, 'vocab.pkl')
    
    if (path.exists(vocab_file)):
      print('Unpacking vocabulary....')
      train_phrases, test_phrases, self.idx2chars, self.chars2idx = unpack(vocab_file)
    else:
      print('Creating vocabulary....')
      np.random.shuffle(train_phrases)
      np.random.shuffle(test_phrases)
  #    if (opt.use_default_vocab):
  #      self.idx2chars = list(string.printable)
  #    else:
      self.data_dir = opt.data_dir
      self.idx2chars = [UNK] # Unknown characters
      self.chars2idx = {UNK:1}
      charcounts = Counter()
      for phrase in train_phrases:
        phrase = phrase.replace(UNK, ' ')
        for c in phrase: charcounts.update(c)
      
      for ii, cc in enumerate(charcounts.most_common(len(charcounts))):
        char = cc[0]
        self.idx2chars.append(char)
        self.chars2idx[char] = ii + 2 # 0: zeropading, 1: unknown
      print('Saving vocabulary...')
      pd.to_pickle({'idx2chars':self.idx2chars, 'chars2idx':self.chars2idx, \
                'train_phrases':train_phrases, \
                'test_phrases':test_phrases},\
                path.join(opt.data_dir, 'vocab.pkl'))    
    
    opt.char_vocab_size = len(self.idx2chars)

    pairdb_name = opt.data_dir.split('/')[-1]
    self.db = client[pairdb_name]
    pair_collection = 'data_0_pairs'

    if not pair_collection in self.db.collection_names():
      print('One time pair file processing....')
      self.create_pairs(opt.w2v_fn, train_phrases, test_phrases, \
                        threshold=opt.threshold)
    else:
      print('Pair file exists....Updating database according to new threshold')
#      self.update_db(opt.threshold)

    max_sent_len = 0
    max_word_len = 0
    if opt.archi == 'hi':
      print('Building tensor for hieararchical LSTM....')
      for phrase in train_phrases:
        words = phrase.split(opt.phrase_sep)
        if (max_sent_len < len(words)): max_sent_len = len(words)
        for word in words:
          if (max_word_len) < len(word): max_word_len = len(word)
      
      self.tensor_coll = 'tensors_hi'
      if not self.tensor_coll in self.db.collection_names():
        print('One time creating phrase tensors...')
        self.create_tensor_hi(train_phrases, test_phrases, max_sent_len, 
                         max_word_len, sep=opt.phrase_sep) 
      else:
        print('Phrase tensor database exist ...%s' %(self.tensor_coll))
              
    elif opt.archi == 'bi':
      print('Building tensor for bidirectional LSTM....')
      for phrase in train_phrases:
        if (max_sent_len < len(phrase)): max_sent_len = len(phrase)

      self.tensor_coll = 'tensors_bi'
      
      if not self.tensor_coll in self.db.collection_names():        
        self.create_tensor_bi(train_phrases, test_phrases, max_sent_len, 
                            sep=opt.phrase_sep)
      else:
        print('Phrase tensor database exist...%s' %(self.tensor_coll))
    

    opt.max_sent_len = max_sent_len
    opt.max_word_len = max_word_len


    
    self.all_tensors = [[], [], []]
    self.batch_idx = [-1, -1, -1]
    self.num_batches = [0, 0, 0]
    self.sample_size = [0, 0, 0]
    
    for split in xrange(3):
      self.sample_size[split] = self.db['data_%s_pairs' %(split)] \
                                    .count({'mute':0})
      self.num_batches[split] = int(np.ceil(self.sample_size[split] / \
                                   float(opt.batch_size)))
    
    self.num_batches[1] = 500
    self.num_batches[2] = 500
    """ TODO: create batch index for non-muted entries. Done
        TODO: create batch index with faster speed. Current permutation too bad.
    """
#    for split in xrange(3):
#      print("Creating batch index for split %s" %(split))
#      batch_idx = np.repeat(range(self.num_batches[split]), opt.batch_size)
#      batch_idx = batch_idx[:self.sample_size[split]]
#      batch_idx = np.random.permutation(batch_idx)
#      for i, doc in enumerate(self.db['data_%s_pairs' %(split)].find({'mute':0})):
#        self.db['data_%s_pairs' %(split)].update_one({'_id':doc['_id']}, \
#                                {'$set': {'batch_idx': batch_idx[i]}}, \
#                                upsert=False)
    

    print('#Pairs in each split: %s, %s, %s...' %(self.sample_size[0], 
                                                  self.sample_size[1],
                                                  self.sample_size[2]))

    print('#Batches in each split: %s, %s, %s...' %(self.num_batches[0], 
                                                  self.num_batches[1],
                                                  self.num_batches[2]))
    
    
  def next_batch(self, split_idx):
    while True:
      self.batch_idx[split_idx] += 1
      if self.batch_idx[split_idx] >= self.num_batches[split_idx]:
        self.batch_idx[split_idx] = 0 # cycle around to beginning
      # pull out the correct next batch
      idx = self.batch_idx[split_idx]
      inp = list(self.db['data_%s_pairs' %(split_idx)].find({'batch_idx':idx}, \
                          {'label':1, 'x1':1, 'x2':1}))
      inp_t1 = np.array([self.db[self.tensor_coll].find_one({'_id': d['x1']})['tensor'] \
                for d in inp])
      
      inp_t2 = np.array([self.db[self.tensor_coll].find_one({'_id': d['x2']})['tensor'] \
                for d in inp])
      
      y = np.array([d['label'] for d in inp]).astype(np.float32).reshape((-1, 1))
      yield ({'charinp1':inp_t1, 'charinp2':inp_t2}, y)
      
  
  def create_tensor_bi(self, train_phrases, test_phrases, max_sent_len, 
                       max_word_len=None, sep='_'):
    def get_tensor(phrase):
      phrase = phrase.replace(sep, ' ')
      phrase = phrase.replace(UNK, ' ')
      char_tensor = []
      for c in phrase:
        try:
          char_tensor.append(self.chars2idx[c])
        except KeyError: 
          char_tensor.append(self.chars2idx[UNK])
      if (len(char_tensor) < max_sent_len):
        while (len(char_tensor) < max_sent_len):
          char_tensor.append(0)
      else:
        char_tensor = char_tensor[:max_sent_len]
        
      return char_tensor
      
    for phrase in train_phrases:
      tensor = {'_id': phrase, 'tensor': get_tensor(phrase)}
      self.db[self.tensor_coll].insert_one(tensor)
      
    for phrase in test_phrases:
      tensor = {'_id': phrase, 'tensor': get_tensor(phrase)}
      self.db[self.tensor_coll].insert_one(tensor)
    
    print('Tensor database created')    
        
  def create_tensor_hi(self, train_phrases, test_phrases, max_sent_len,
                    max_word_len, sep='_'):
    
    print('Creating training and testing tensors.....')

    def get_tensor(phrase):
      char_tensor = []
      for word in phrase.split(sep):
        word = word.replace(UNK, ' ')
        t = []
        for c in word:
          if c in self.idx2chars: t.append(self.chars2idx[c])
          else: t.append(self.chars2idx[UNK])
        char_tensor.append(t)
      for i, tensor in enumerate(char_tensor):
        if (len(tensor) < max_word_len):
          while (len(tensor) < max_word_len):
            char_tensor[i].append(0)
        else:
          char_tensor[i] = char_tensor[i][:max_word_len]
        
      if (len(char_tensor) < max_sent_len):
        while (len(char_tensor) < max_sent_len):
          char_tensor.append([0] * max_word_len)
      else:
        char_tensor = char_tensor[:max_sent_len]
      return char_tensor
      
    for phrase in train_phrases:
      tensor = {'_id': phrase, 'tensor': get_tensor(phrase)}
      self.db[self.tensor_coll].insert_one(tensor)
      
    for phrase in test_phrases:
      tensor = {'_id': phrase, 'tensor': get_tensor(phrase)}
      self.db[self.tensor_coll].insert_one(tensor)
    
    print('Tensor database created')

  def create_pairs(self, w2v_fn, train_phrases, test_phrases, val_rate=0.3,
                   threshold=0.8):
    """
    w2v: phrase embedding model to generate labels.
    train_phrases: very frequent phrases
    test_phrases: very rare phrases
    """
#    train_size = np.ceil(sample_size * (1 - val_rate))
#    val_size = sample_size - train_size
#    
#    train_val_pairs = np.random.choice(train_phrases, size=(sample_size+val_size, 2))
#    train_val_pairs[train_val_pairs[:, 0] != train_val_pairs[:, 1]]
#    train_val_pairs = [tuple(row) for row in train_val_pairs]
#    train_val_pairs = np.unique(train_val_pairs)
#    
#    train_pairs = train_val_pairs[:train_size, :]
#    val_pairs = train_val_pairs[train_size:sample_size, :]    
#    test_pairs = np.random.choice(test_phrases, size=(test_sample_size, 2))
  
    print('Creating training and testing pairs.....')    
    if (w2v_fn[-3:] == 'bin'):
      train_vecs, _ = word_rep_bin(train_phrases, w2v_fn)
      test_vecs, _ = word_rep_bin(test_phrases, w2v_fn)
    else:
      train_vecs, _ = word_rep(train_phrases, w2v_fn)
      test_vecs, _ = word_rep(test_phrases, w2v_fn)

    print('Calculating similarities for train and validation.....')
    num_pairs = 0
    for i, phrase in enumerate(train_phrases):
      sims = 1 - distance.cdist(train_vecs[i:(i+1), :], train_vecs[i+1:, :], \
                                metric='cosine')
      pairs_tr = []
      pairs_val = []
      for j, sim in enumerate(sims[0]):
        pair = {'x1': phrase, \
                'x2': train_phrases[i + 1 + j], \
                'sim': sim
                }        
        pair['label'] = 1 if sim >= threshold else 0
        pair['mute'] = 1 
        pair['batch_idx'] = -1
        split = np.random.rand()
        if split > val_rate: 
          pairs_tr.append(pair)
        else:
          pairs_val.append(pair)
      
      num_pairs = num_pairs + len(pairs_tr) + len(pairs_val)
      
      if len(pairs_tr):
        self.db['data_0_pairs'].insert_many(pairs_tr)
      if len(pairs_val):
        self.db['data_1_pairs'].insert_many(pairs_val)
      
      if (num_pairs % 100 == 0): 
        print("Inserted %s pairs" %(num_pairs))
      
    for i, phrase in enumerate(test_phrases):
      vec = test_vecs[i:(i+1), :]
      sims = 1 - distance.cdist(vec, train_vecs, metric='cosine')
      pairs = []
      for j, sim in enumerate(sims[0]):
        pair = {'x1': phrase, \
                'x2': train_phrases[j], \
                'sim': sim}
        pair['label'] = 1 if sim >= threshold else 0
        pair['mute'] = 1
        pair['batch_idx'] = -1
        pairs.append(pair)
      
      if len(pairs):
        self.db['data_2_pairs'].insert_many(pairs)
      
      
      
    #%% Create indexing
    print("Creating indexing")
    for i in range(3):
      self.db['data_%s_pairs' %(i)].create_index('label')
      self.db['data_%s_pairs' %(i)].create_index('mute')
      self.db['data_%s_pairs' %(i)].create_index('batch_idx')
      self.db['data_%s_pairs' %(i)].create_index('sim')        

  
  #%% Muting some negative pairs to balance the positive and negative training
  def update_db(self, threshold):
    for i in range(3):
      self.db['data_%s_pairs' %(i)].update_many({'sim': {'$gte': threshold}}, \
                                                {'$set': {'label': 1}})
      self.db['data_%s_pairs' %(i)].update_many({'sim': {'$lt': threshold}}, \
                                                {'$set': {'label': 0}})
    pos_size = []
    neg_size = []
    for i in range(3):
      pos_size.append(self.db['data_%s_pairs' %(i)].count({'label': 1}))
      neg_size.append(self.db['data_%s_pairs' %(i)].count({'label': 0}))
    
    sample_neg = [4 * s for s in pos_size]
    sample_neg_ratio = np.array(sample_neg, dtype=float) / np.array(neg_size)
    
    for i in range(3):
      for doc in self.db['data_%s_pairs' %(i)].find({'label':0}):
        sample = np.random.rand()
        if sample <= sample_neg_ratio[i]:
          self.db['data_%s_pairs' %(i)].update_one({'_id':doc['_id']}, \
                                          {'$set': {'mute': 0}}, \
                                          upsert=False)
    
    
    