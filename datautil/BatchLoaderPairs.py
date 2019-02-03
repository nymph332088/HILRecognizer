# -*- coding: utf-8 -*-
"""
Created on Thu May  4 22:47:00 2017
@author: nymph
"""
from os import path
import numpy as np
import pandas as pd
from ReprsReader import word_rep, word_rep_bin
from scipy.spatial import distance
import scipy.sparse as ss
import time
from collections import Counter
UNK = '_'
def unpack(fn):
  data = pd.read_pickle(fn)
  return data['train_phrases'], data['test_phrases'], \
        data['idx2chars'], data['chars2idx']

        
class BatchLoaderPairs(object):
  def __init__(self, opt, train_phrases, test_phrases):
    self.data_dir = opt.data_dir
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
    pair_file = path.join(opt.data_dir, 'data_0_pairs.pkl')
    if not path.exists(pair_file):
      print('One time pair file processing....')
      self.create_pairs(opt.w2v_fn, train_phrases, test_phrases)
    else:
      print('Pair file exists....')
      
    max_sent_len = 0
    max_word_len = 0
    if opt.archi == 'hi':
      print('Building tensor for hieararchical LSTM....')
      for phrase in train_phrases:
        words = phrase.split(opt.phrase_sep)
        if (max_sent_len < len(words)): max_sent_len = len(words)
        for word in words:
          if (max_word_len) < len(word): max_word_len = len(word)
      
      phrase_tensor_file = path.join(opt.data_dir, 'phrase_tensor_hi.pkl')
      if not path.exists(phrase_tensor_file):
        print('One time creating phrase tensors...')
        self.create_tensor_hi(train_phrases, test_phrases, max_sent_len, 
                         max_word_len, sep=opt.phrase_sep) 
        print('Saving phrase tensor for hi...')
        pd.to_pickle(self.phrase_tensor,\
                  path.join(opt.data_dir, 'phrase_tensor_hi.pkl')) 
      else:
        print('Unpacking phrase tensor file...%s' %(phrase_tensor_file))
        self.phrase_tensor = pd.read_pickle(phrase_tensor_file)      
        
    elif opt.archi == 'bi':
      print('Building tensor for bidirectional LSTM....')
      for phrase in train_phrases:
        if (max_sent_len < len(phrase)): max_sent_len = len(phrase)

      phrase_tensor_file = path.join(opt.data_dir, 'phrase_tensor_bi.pkl')
      if not path.exists(phrase_tensor_file):        
        self.create_tensor_bi(train_phrases, test_phrases, max_sent_len, 
                            sep=opt.phrase_sep)
        print('Saving phrase tensor for bi...')
        pd.to_pickle(self.phrase_tensor,\
                  path.join(opt.data_dir, 'phrase_tensor_bi.pkl'))      
      else:
        print('Unpacking phrase tensor file....%s' %(phrase_tensor_file))
        self.phrase_tensor = pd.read_pickle(phrase_tensor_file)
      

    opt.max_sent_len = max_sent_len
    opt.max_word_len = max_word_len

    self.all_tensors = [[], [], []]
    self.batch_idx = [0, 0, 0]
    self.batch_size = [0, 0, 0]
    self.sample_size = [0, 0, 0]
    for split in xrange(3):
      pairs = pd.read_pickle(path.join(opt.data_dir, 'data_%s_pairs.pkl' %(split)))
#      pairs = pairs[:20]
      inp1 = [pair[0] for pair in pairs]
      inp2 = [pair[1] for pair in pairs]
      y = [pair[2] for pair in pairs]
      
      sections = np.arange(len(pairs), step=opt.batch_size)[1:]
      ## reorder each batches:
      self.all_tensors[split].append(np.split(inp1, sections, axis=0))
      self.all_tensors[split].append(np.split(inp2, sections, axis=0))
      self.all_tensors[split].append(np.split(y, sections, axis=0))

      self.sample_size[split] = len(pairs)      
      self.batch_size[split] = len(self.all_tensors[split][0])

    print('#Pairs in each split: %s, %s, %s...' %(self.sample_size[0], 
                                                  self.sample_size[1],
                                                  self.sample_size[2]))

    print('#Batches in each split: %s, %s, %s...' %(self.batch_size[0], 
                                                  self.batch_size[1],
                                                  self.batch_size[2]))
    
    
  def next_batch(self, split_idx):
    while True:
      self.batch_idx[split_idx] += 1
      if self.batch_idx[split_idx] >= self.batch_size[split_idx]:
        self.batch_idx[split_idx] = 0 # cycle around to beginning
      # pull out the correct next batch
      idx = self.batch_idx[split_idx]
      inp1 = np.array([self.phrase_tensor[x] for x in self.all_tensors[split_idx][0][idx]])
      inp2 = np.array([self.phrase_tensor[x] for x in self.all_tensors[split_idx][1][idx]])
      y = self.all_tensors[split_idx][2][idx].astype(np.float32).reshape((-1, 1))
      yield ({'charinp1':inp1, 'charinp2':inp2}, y)

      
  
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
      
    self.phrase_tensor = {}
    for phrase in train_phrases:
      self.phrase_tensor[phrase] = get_tensor(phrase)
    for phrase in test_phrases:
      self.phrase_tensor[phrase] = get_tensor(phrase)
    
        
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
          else: char_tensor.append(self.chars2idx[UNK])
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
      
    self.phrase_tensor = {}
    for phrase in train_phrases:
      self.phrase_tensor[phrase] = get_tensor(phrase)
    
    for phrase in test_phrases:
      self.phrase_tensor[phrase] = get_tensor(phrase)
      
#    for split in xrange(3):
##      tuples = np.load(path.join(self.data_dir, 'data_%s_pairs.npy' %(split)))
#      tuples = pd.read_pickle(path.join(self.data_dir, 'data_%s_pairs.pkl' %(split)))
#      split_tensors = []
#      print('Creating training and testing tensors..#pairs %s' %(len(tuples)))
#      
#      for tup in tuples:
#        tup_tensor = []
#        phrase2 = train_phrases[tup[1]]
#        if (split != 2): phrase1 = train_phrases[tup[0]]
#        else: phrase1 = test_phrases[tup[0]]
#        tup_tensor.append(get_tensor(phrase1))
#        tup_tensor.append(get_tensor(phrase2))
#        tup_tensor.append(tup[2])
#        split_tensors.append(tup_tensor)
#      pd.to_pickle(split_tensors,
#                   path.join(self.data_dir, 'data_%s_tensors.pkl' %(split)))
#      np.save(path.join(self.data_dir, 'data_%s_tensors.npy' %(split)), split_tensors)

  def create_pairs(self, w2v_fn, train_phrases, test_phrases, val_rate=0.3):
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

    print('Calculating similarities .....')    
    train_sims = 1 - distance.cdist(train_vecs, train_vecs, metric='cosine')
#    train_sims = ss.coo_matrix(train_sims >= 0.5)
    
    test_sims = 1 - distance.cdist(test_vecs, train_vecs, metric='cosine')
#    test_sims = ss.coo_matrix(test_sims >= 0.5)
    
    start_time = time.time()
    train_pairs = []
    for ii in range(train_sims.shape[0]):
      for jj in range(train_sims.shape[1]):
        if (ii == jj): continue
        if (train_sims[ii, jj] >= 0.8):
          train_pairs.append([train_phrases[ii], train_phrases[jj], 1])
        elif (train_sims[ii, jj] < 0.0):
          train_pairs.append([train_phrases[ii], train_phrases[jj], 0])
    
    
    print("Time spent %s" %(time.time() - start_time))
    start_time = time.time()
    test_pairs = []
    for ii in range(test_sims.shape[0]):
      for jj in range(ii + 1, test_sims.shape[1]):
        if (test_sims[ii, jj] >= 0.8):
          test_pairs.append([test_phrases[ii], train_phrases[jj], 1])
        elif (test_sims[ii, jj] < 0.0):
          test_pairs.append([test_phrases[ii], train_phrases[jj], 0])
    
    print("Time spent %s" %(time.time() - start_time))     
#    start_time = time.time()
#    ii, jj, kk = ss.find(ss.coo_matrix(train_sims >= 0.5))
#    train_pairs = zip(ii, jj, kk.astype(float))
#    
#    ii, jj, kk = ss.find(ss.coo_matrix(train_sims < 0.5))
#    train_pairs = train_pairs + zip(ii, jj, kk.astype(float) - 1)
#    print("Time spent %s" %(time.time() - start_time))
    
    train_size = int(len(train_pairs) * (1 - val_rate))

#    ii, jj, kk = ss.find(test_sims)
#    test_pairs = zip(ii, jj, kk.astype(float))
#    
#    ii, jj, kk = ss.find(test_sims != True)
#    test_pairs = test_pairs + zip(ii, jj, kk.astype(float) - 1)
    
#    print('Shuffling similarities .....')
#    np.random.shuffle(train_pairs)
#    np.random.shuffle(test_pairs)
    
#    data_tuples = [train_pairs[:train_size], train_pairs[train_size:], 
#                        test_pairs]

    start_time = time.time()
    print('Shuffling pairs....')
    x = np.random.permutation(len(train_pairs))
    train_pairs = np.array(train_pairs)[x].tolist()
    for i in range(len(train_pairs)): train_pairs[i][2] = int(train_pairs[i][2])
    print('Time spent %s' %(time.time() - start_time))

    
    print('Saving data .....')
    for split in xrange(3):
      start_time = time.time()
      if split == 0:
        pd.to_pickle(train_pairs[:train_size],
                     path.join(self.data_dir, 'data_%s_pairs.pkl' %(split))) 
        print('Saving train .....%s' %(time.time() - start_time))
      
#        np.save(path.join(self.data_dir, 'data_%s_pairs.npy' %(split)), 
#                train_pairs[:train_size])
#        print('Saving train .....%s' %(time.time() - start_time))

      if split == 1:
        pd.to_pickle(train_pairs[train_size:], 
                     path.join(self.data_dir, 'data_%s_pairs.pkl' %(split)))
        print('Saving val .....%s' %(time.time() - start_time))
#        np.save(path.join(self.data_dir, 'data_%s_pairs.npy' %(split)), 
#                train_pairs[train_size:])
      if split == 2:
        pd.to_pickle(test_pairs,
                     path.join(self.data_dir, 'data_%s_pairs.pkl' %(split)))
        print('Saving test .....%s' %(time.time() - start_time))
#        np.save(path.join(self.data_dir, 'data_%s_pairs.npy' %(split)), 
#                test_pairs)
        
        