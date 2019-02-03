#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:15:37 2018

@author: tuf14438
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datautil.util import  *
import numpy as np
import pandas as pd
import time
import os
from model.SBLClassifier import BiLSTMTagger
#%%

def train_test_tagger(x_train, y_train, x_test, y_test, args, model=None):
  if model==None:
    model = load_pretrain(args)



  loss_function = nn.NLLLoss()

  # print(args)
  if args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
  elif args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
  else:
    raise Exception('For other optimizers, please add it yourself. '
                      'supported ones are: SGD and Adam.')


  for epoch in range(args.epochs):
    total_loss = 0
    start_time = time.time()
    model.train()
    for batch, i in enumerate(range(0, len(x_train), args.batch_size)):

      start_inds, end_inds = i, min(len(x_train), i+args.batch_size)

      # Step 1. Remember that Pytorch accumulates gradients.
      # We need to clear them out before each instance
      model.zero_grad()
      # Step 2. Get our inputs ready for the network, that is, turn them into
      # Variables of word indices.
      sentence_in, targets = package(x_train[start_inds:end_inds], y_train[start_inds:end_inds], volatile=False)
      # print(sentence_in, targets)
      if args.cuda:
        sentence_in = sentence_in.cuda()
        targets = targets.cuda()
      # Also, we need to clear out the hidden state of the LSTM,
      # detaching it from its history on the last instance.
      hidden = model.init_hidden(sentence_in.size(1))

      # Step 3. Run our forward pass.
      tag_scores, _ = model.forward(sentence_in, hidden)

      # Step 4. Compute the loss, gradients, and update the parameters by
      #  calling optimizer.step()
      size = tag_scores.size()

      loss = loss_function(tag_scores.view(-1, size[2]), targets.view(targets.nelement()))
      total_loss += loss.data
      loss.backward()
      optimizer.step()
      if i % (args.batch_size) == 0:
        elapsed = time.time() - start_time
        # print("Finished %s batches out of %s, took %s seconds, loss = %s" \
        #       %(batch, len(x_train) // args.batch_size, elapsed, total_loss[0] / (i+1)))
        start_time = time.time()
    
  
    tags_pred, tags_probs = predict(model, x_test, y_test)
    # tags_true = np.array([[tags.word2idx[tag] for tag in y] for y in y_test])
    _, tags_true = package(x_test, y_test, volatile=True)
    tags_true = tags_true.data.numpy() 
    # print(tags_true)
    perf = perfs(tags_true, tags_pred)
    # print(perf)
  return model, tags_pred, tags_probs, perf

             


def predict(model, x_test, y_test):
  model.eval()
  total_loss = 0
  start_time = time.time()
  tags_pred = []
  tags_probs = []
  loss_function = nn.NLLLoss()
  for batch, i in enumerate(range(0, len(x_test), args.batch_size)):
    start_inds, end_inds = i, min(len(x_test), i+args.batch_size)
     # Step 2. Get our inputs ready for the network, that is, turn them into
    # Variables of word indices.
    sentence_in, targets = package(x_test[start_inds:end_inds], y_test[start_inds:end_inds], volatile=True)
    if args.cuda:
      sentence_in = sentence_in.cuda()
      targets = targets.cuda()
   
    # Step 1. Clear out the hidden state.
    hidden = model.init_hidden(sentence_in.size(1))

    # Step 3. Run our forward pass.
    # tag scores: [bsz, len, tag size]
    tag_scores, _ = model.forward(sentence_in, hidden)
    tags_probs.append(tag_scores.cpu().data.numpy())
    # preds : [bsz, len]
    _, preds = torch.max(tag_scores, dim=2)
    tags_pred.append(preds.cpu().data.numpy())
    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    size = tag_scores.size()
    loss = loss_function(tag_scores.view(-1, size[2]), targets.view(targets.nelement()))
    total_loss += loss.data

  elapsed = time.time() - start_time

  # print("Took %s seconds, loss = %s" \
  #       %(elapsed, total_loss[0] / len(x_test)))
  
  tags_pred = np.vstack(tags_pred)
  tags_probs = np.vstack(tags_probs)
  return tags_pred, tags_probs



def load_pretrain(args):
  model = BiLSTMTagger({
      'dropout': args.dropout,
      'ntoken': n_token,
      'nlayers': args.nlayers,
      'nhid': args.nhid,
      'ninp': args.emsize,
      'pooling': args.pooling,
      'attention-unit': args.attention_unit,
      'attention-hops': args.attention_hops,
      'filters': args.filters,
      'filter_nums': args.filter_nums,
      'nfc': args.nfc,
      'dictionary': dictionary,
      'word-vector': args.word_vector,
      'class-number': args.class_number,
      'max_len': args.max_len, 
      'target_size': len(args.tags)
  })
  
  
  if args.pretrain != '':
    print("Loading pre-trained models.....")
    pretrained_dict = torch.load(args.pretrain)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    # pretrained_dict = {k[8:]: v for k, v in pretrained_dict.items() if k[8:] in model_dict}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

  if args.cuda:
    model = model.cuda()
  
  return model

def pretrain_predict(x_test, y_test, args):
  model = BiLSTMTagger({
      'dropout': args.dropout,
      'ntoken': n_token,
      'nlayers': args.nlayers,
      'nhid': args.nhid,
      'ninp': args.emsize,
      'pooling': args.pooling,
      'attention-unit': args.attention_unit,
      'attention-hops': args.attention_hops,
      'filters': args.filters,
      'filter_nums': args.filter_nums,
      'nfc': args.nfc,
      'dictionary': dictionary,
      'word-vector': args.word_vector,
      'class-number': args.class_number,
      'max_len': args.max_len, 
      'target_size': len(args.tags)
  })
  
  

  if args.cuda:
    model = model.cuda()

  print("Loading pre-trained models.....")
  pretrained_dict = torch.load(args.pretrain)
  model_dict = model.state_dict()
  # 1. filter out unnecessary keys
  # pretrained_dict = {k[8:]: v for k, v in pretrained_dict.items() if k[8:] in model_dict}
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  # 2. overwrite entries in the existing state dict
  model_dict.update(pretrained_dict)
  # 3. load the new state dict
  model.load_state_dict(model_dict)

  model.eval()

  sentence_in, targets = package(x_test, y_test, volatile=True)
  if args.cuda:
    sentence_in = sentence_in.cuda()
    targets = targets.cuda()
 
  # Step 1. Clear out the hidden state.
  hidden = model.init_hidden(sentence_in.size(1))

  # Step 3. Run our forward pass.
  # tag scores: [bsz, len, tag size]
  tag_scores, _ = model.forward(sentence_in, hidden)

  # preds : [bsz, len]
  _, preds = torch.max(tag_scores, dim=2)
  
  return preds.cpu().data.numpy()



