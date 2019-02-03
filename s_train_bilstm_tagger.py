# -*- coding: utf-8 -*-

#CUDA_VISIBLE_DEVICES=0 python s_train_alstm.py --pooling conv --epochs 50 --batch-size 300 --cuda

from datautil.util import Dictionary, get_args, random_part, loo_part
import pandas as pd
import numpy as np
import os
import torch
import torch._utils
import random
from sklearn import metrics
from sklearn.cross_validation import KFold
import time
import matplotlib.pylab as plt
import seaborn as sns
from torch.autograd import Variable
import copy
try:
  torch._utils._rebuild_tensor_v2
except AttributeError:
  def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    tensor._backward_hooks = backward_hooks
    return tensor
  torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
execfile('text_character_bilstm_tagger.py')
execfile('active_learning.py')
# exec(open('./text_character_bilstm_tagger.py').read())
# kaggle_seq_label = pd.read_excel('data/junkdata/testKaggle2_seq_tolabel_all.xlsx', encoding='utf-8')
# for i in range(len(kaggle_seq_label)):
#     if i % 3 == 0 :
#         kaggle_seq_label.iloc[i, :] = kaggle_seq_label.iloc[i, :].fillna(' ')
#     else:
#         kaggle_seq_label.iloc[i, :] = kaggle_seq_label.iloc[i, :].fillna('o')
#        
# seq_label = kaggle_seq_label.iloc[[i for i in range(len(kaggle_seq_label)) if i % 3 == 1], :]
# seq_label = np.array([u''.join(row) for _, row in seq_label.iterrows()])
# strings = kaggle_seq_label.iloc[[i for i in range(len(kaggle_seq_label)) if i % 3 == 0], :]
# strings = np.array([u''.join(row) for _, row in strings.iterrows()])
# str_len = np.array([len(s) for s in strings])
# tag_len = np.array([len(s) for s in seq_label])
# np.all(str_len == tag_len)
# bound_label = kaggle_seq_label.iloc[[i for i in range(len(kaggle_seq_label)) if i % 3 == 2], :]
# bound_label = np.array([u''.join(row) for _, row in bound_label.iterrows()])
# bound_label = np.array([s.replace('s', 'o').replace('O', 'o') for s in bound_label])
#
# seq_label = np.array([s.replace('d', 'D').replace('A', 'o').replace('P', 'o').replace('O', 'o') \
#             for s in seq_label])

# kaggle_data = pd.read_csv('data/junkdata/testKaggle2_25regex.csv', encoding='utf-8')
# kaggle_data.columns = ['Media', 'PublishTime', 'Title', 'Outlet', 'String', 'Label', \
#        'R.E.tag', 'R.E.datetime', 'R.E.1', 'R.E.2', 'R.E.3', 'R.E.4', \
#        'R.E.5', 'R.E.6', 'R.E.7', 'R.E.8', 'R.E.9', 'R.E.10', 'R.E.11', \
#        'R.E.12', 'R.E.13', 'R.E.14', 'R.E.15', 'R.E.16', 'R.E.17', \
#        'R.E.18', 'R.E.19', 'R.E.20', 'R.E.21', 'R.E.22', 'R.E.23', \
#        'R.E.24', 'R.E.25', 'R.E.label']
# kaggle_data.Media.fillna(method='ffill', inplace=True)
# kaggle_data.Title.fillna(method='ffill', inplace=True)
# kaggle_data.Outlet.fillna(method='ffill', inplace=True)
# kaggle_data.String[kaggle_data.Label == 1] = strings
# kaggle_data['DatetimeLabel'] = [u'o'*len(r['String']) for _, r in kaggle_data.iterrows()]
# kaggle_data.DatetimeLabel[kaggle_data.Label == 1] = seq_label
# kaggle_data['TagLabel'] = [u'o'*len(r['String']) for _, r in kaggle_data.iterrows()]
# kaggle_data.TagLabel[kaggle_data.Label == 1] = bound_label

# from GenerateCharLabels import getBoundTag
# re_bound, re_tag = getBoundTag(kaggle_data.String.values)
# kaggle_data['R.E.tag'] = re_tag
# kaggle_data['R.E.bound'] = re_bound
# kaggle_data.to_csv('data/position/testKaggle2.csv', index=False, encoding='utf-8')
def package(x_train, y_train, volatile=False):
  """Package data for training / evaluation."""
  dat = map(lambda x: map(lambda y: dictionary.word2idx[y], x.encode('ascii', errors='replace')), x_train)
  y_train = map(lambda x: map(lambda y: tags.word2idx[y], x), y_train)
  # maxlen = 0
  # for item in dat:
  #     maxlen = max(maxlen, len(item))
  maxlen = args.max_len
  # maxlen = min(maxlen, 500)
  for i in range(len(x_train)):
    if maxlen < len(dat[i]):
      dat[i] = dat[i][:maxlen]
      y_train[i] = y_train[i][:maxlen]
    else:
      for j in range(maxlen - len(dat[i])):
        dat[i].append(dictionary.word2idx['<pad>'])
        y_train[i].append(0)
  # print(np.array(dat).shape)
  # print(np.unique([len(y) for y in y_train]))
  dat = Variable(torch.LongTensor(dat), volatile=volatile)
  targets = Variable(torch.LongTensor(y_train), volatile=volatile)
  return dat.t(), targets


# def perfs(tags_true, tags_pred):
#   micro_acc = [metrics.f1_score(tags_true[i], tags_pred[i]) for i in range(len(tags_true))]
#   macro_acc = 
#   micro_f1 = 
#   macro_f1 = 


def position_eval(tags_true, tags_pred):
  pos_baselines = [np.mean(true == 0) for true in tags_true]
  pos_accs = [np.mean(tags_true[i] == tags_pred[i]) for i in range(len(tags_pred))]
  
  whole_accs = [np.all(tags_true[i] == tags_pred[i]) for i in range(len(tags_pred))]
  

  perf = {'ACC': np.mean(pos_accs), \
          'All': np.mean(whole_accs)}

  tags_true = tags_true.flatten()
  tags_pred = tags_pred.flatten()
#  macro_f1 = metrics.f1_score(tags_true, tags_pred, \
#                              labels=range(1, max(tags_true)+1), \
#                              average='macro')
  micro_f1 = metrics.f1_score(tags_true, tags_pred, \
                              labels=range(1, max(tags_true)+1), \
                              average='micro')
  micro_prec = metrics.precision_score(tags_true, tags_pred, \
                              labels=range(1, max(tags_true)+1), \
                              average='micro')
  micro_recall = metrics.recall_score(tags_true, tags_pred, \
                              labels=range(1, max(tags_true)+1), \
                              average='micro')

#  perf['Precision'] = micro_prec
#  perf['Recall'] = micro_recall
#  perf['F1'] = micro_f1
#  return perf
  return micro_prec, micro_recall, micro_f1
  
#%% Entity Level evaluation
def conll_entity_eval(tags_true, tags_pred, tag2idx, original_str):
  import re
  regex = re.compile('y+')
  tags_pred_str = np.array([''.join([tag2idx.idx2word[int(i)] for i in t]) for t in tags_pred])
  tags_true_str = np.array([''.join([tag2idx.idx2word[int(i)] for i in t]) for t in tags_true])
                   
  # original_str = kaggle2.String.ix[test_idx].values
  
  true_entities = [[[i, original_str[i][m.span()[0]:m.span()[1]], m.span()[0], m.span()[1], -1, ''] \
                    for j, m in enumerate(regex.finditer(s))] \
                   for i, s in enumerate(tags_true_str)]
                   
  pred_entities = [[[i, original_str[i][m.span()[0]:m.span()[1]], m.span()[0], m.span()[1], -1, ''] \
                   for m in regex.finditer(s)] \
                   for i, s in enumerate(tags_pred_str)]
  
  # calculate the overlapped entities
  
  for i, eps in enumerate(pred_entities):
    ets = true_entities[i]
    for ep_i, ep in enumerate(eps):
      for et_j, et in enumerate(ets):
        if et[4] != -1 or ep[4] != -1: continue
        # exact match
        if ep == et:
          ep[4] = et_j
          et[4] = ep_i
          ep[5], et[5] = 'Exact', 'Exact'
        # overlap
        elif ep[2] <= et[3] and ep[3] >= et[2]:
          ep[4] = et_j
          et[4] = ep_i
          ep[5], et[5] = 'Part', 'Part'
                   
  
  
  
  from itertools import chain
  true_entities_df = pd.DataFrame(list(chain(*true_entities)))
  pred_entities_df = pd.DataFrame(list(chain(*pred_entities)))
  
  print(true_entities_df.shape, pred_entities_df.shape)
  if len(pred_entities_df) == 0: return 0, 0, 0
  true_pos = float(sum(pred_entities_df.iloc[:, 5] == 'Exact'))
  prec = true_pos / len(pred_entities_df)
  recall = true_pos / len(true_entities_df)
  if prec == 0 and recall == 0: return 0, 0, 0
  f1 = 2 * (prec*recall) / (prec + recall)
  return prec, recall, f1
#%%
def perfs(tags_true, tags_pred, tag2idx=None, original_str=[]):
  # print(tags_pred[0], tags_true[0], np.mean(tags_true[0] == tags_pred[0]))
  # print(tags_true[0] == tags_pred[0])
  
  pos_baselines = [np.mean(true == 0) for true in tags_true]
  pos_accs = [np.mean(tags_true[i] == tags_pred[i]) for i in range(len(tags_pred))]
  
  whole_accs = [np.all(tags_true[i] == tags_pred[i]) for i in range(len(tags_pred))]
  
          
  # perf = {'Baseline': np.mean(pos_baselines), \
  #         'ACC': np.mean(pos_accs), \
  #         'All': np.mean(whole_accs)}

  perf = {'ACC': np.mean(pos_accs), \
          'All': np.mean(whole_accs)}

  if (tag2idx != None) and (original_str != []):
    perf['PosPrec'], perf['PosRecall'], perf['PosF1'] \
      = position_eval(tags_true, tags_pred)
    perf['EntPrec'], perf['EntRecall'], perf['EntF1'] \
      = conll_entity_eval(tags_true, tags_pred, \
                      tag2idx, original_str)
  return perf

def model_selection(data, kf_func):
  kf = kf_func(data)
  model_dir = '%s/%s/' %(args.save, kf_func.__name__)
  for name in ['lr', 'nlayers', 'nhid', 'emsize']:
    model_dir += '%s-%s_' %(name, vars(args)[name])

  
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    

  table = []
  fold = 0
  xs = []
  ys = []
  preds = []
  
  for train_idx, test_idx in kf:
    x_train = data.iloc[train_idx, :]['String'].values
    y_train = data.iloc[train_idx, :]['Tag'].values
    x_test = data.iloc[test_idx, :]['String'].values
    y_test = data.iloc[test_idx, :]['Tag'].values
    
    _, tags_true = package(x_test, y_test, volatile=True)
    tags_true = tags_true.data.numpy()
    print('#train = %s, #test = %s' %(len(x_train), len(x_test)))
    
    model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args)
    perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
    xs += x_test.tolist()
    ys += tags_true.tolist()
    preds += tags_pred.tolist()
    table.append(perf)
    fold = fold + 1
    with open(os.path.join(model_dir, 'model_fold_%s.pt' %(fold)), 'wb') as f:
      torch.save(model, f)
    f.close()  

  pd.to_pickle(args, os.path.join(model_dir, 'args.pkl'))
  pd.to_pickle({'tag2idx':tags, 'preds':preds, 'labels':ys, 'strings':xs}, os.path.join(model_dir, \
             'meta-%s.pkl' %(int(time.time()))))
  table = pd.DataFrame(table, index = ['Fold %s' %(i) for i in range(5)])
  ave = table.mean(axis=0)
  ave.name = 'Average'
  table = table.append(ave)
  
  overall = pd.Series(perfs(np.array(ys), np.array(preds), tag2idx=tags, original_str=np.array(xs)))
  overall.name='Overall'
  table = table.append(overall)
  print(table)
  return table  
 

def get_results(kaggle_data):
  flname = args.label
#  kaggle_records = pd.read_pickle('%s/%s_%s_pred.pkl' %(args.save, args.partition, flname))
  kaggle_records = []
  for i in range(10):
    lr = 1.0 / (2** np.random.random_integers(6, 13))
    emsize = np.random.choice([20, 30, 40, 50, 60, 70, 80])
    nhid =  np.random.choice([50, 75, 100, 125, 150, 200])
    nlayers = np.random.choice([1, 2, 5])
    args.lr = lr
    args.emsize = emsize
    args.nhid = nhid
    args.nlayers = nlayers
    
    record = {'args':copy.deepcopy(args)}  

    print("Training with %s" %(str(record)))
    if args.partition == 'loo':
      table1 = model_selection(kaggle_data, loo_part)
      record['loo'] = table1
      record['loo_acc'] = table1.ix['Overall', 'ACC']

    if args.partition == 'random':
      table2 = model_selection(kaggle_data, random_part)    
      record['random'] = table2
      record['random_acc'] = table2.ix['Overall', 'ACC']

    kaggle_records.append(record)
    if not os.path.exists('%s/'%(args.save)):
      os.makedirs('%s'%(args.save))
    
    
    pd.to_pickle(kaggle_records, '%s/%s_%s_pred.pkl' %(args.save, args.partition, flname)) 

  accs = np.array([r['%s_acc' %(args.partition)] for r in kaggle_records])
  best_perf = np.where(accs == np.max(accs))[0][0]
#  print(kaggle_records[best_perf]['loo'])
  print(kaggle_records[best_perf][args.partition])

  print('Start training the best model.....')
  best_args = kaggle_records[best_perf]['args']
  x_train = kaggle_data['String'].values
  y_train = kaggle_data['Tag'].values
  model, _, _, _ = train_test_tagger(x_train, y_train, x_train[:10], y_train[:10], best_args)

  with open(os.path.join(args.save, '%s_%s_best.pt'%(args.partition, flname)), 'wb') as f:
    torch.save(model.state_dict(), f)
  f.close()
  pd.to_pickle(best_args, os.path.join(args.save, '%s_%s_best_args.pkl'%(args.partition, flname)))




#%%
def tagger():
  kaggle_data = pd.read_csv(args.data, encoding='utf-8')
  kaggle_data['Tag'] = kaggle_data[args.label]
  # kaggle_data.columns = ['Outlet', 'kaggle_time', 'title', 'url', 'String', 'Prediction',
  #        u'boundary labels', u'Tag']
  # kaggle_data.Outlet.fillna(method='ffill', inplace=True)      
  # kaggle_data.title.fillna(method='ffill', inplace=True)      
  # kaggle_data.kaggle_time.fillna(method='ffill', inplace=True)
  # kaggle_data.url.fillna(method='ffill', inplace=True)
  
  # kaggle_data = kaggle_data[~ pd.isnull(kaggle_data.Tag)]
  
  str_len = np.array([len(s) for s in kaggle_data.String])
  tag_len = np.array([len(s) for s in kaggle_data.Tag])
  assert(np.all(str_len == tag_len))
  seq_data = kaggle_data[str_len == tag_len] 
  get_results(seq_data)
  

  
def tagger_trsize():
  kaggle_data = pd.read_csv(args.data, encoding='utf-8')
  kaggle_data['Tag'] = kaggle_data[args.label]
  # kaggle_data.columns = ['Outlet', 'kaggle_time', 'title', 'url', 'String', 'Prediction',
  #        u'boundary labels', u'Tag']
  # kaggle_data.Outlet.fillna(method='ffill', inplace=True)      
  # kaggle_data.title.fillna(method='ffill', inplace=True)      
  # kaggle_data.kaggle_time.fillna(method='ffill', inplace=True)
  # kaggle_data.url.fillna(method='ffill', inplace=True)  
  # kaggle_data = kaggle_data[~ pd.isnull(kaggle_data.Tag)]

  str_len = np.array([len(s) for s in kaggle_data.String])
  tag_len = np.array([len(s) for s in kaggle_data.Tag])
  assert(np.all(str_len == tag_len))
  seq_data = kaggle_data[str_len == tag_len]
  if not os.path.exists(args.save):
    os.makedirs(args.save)  

  flname = args.label
  # ts_record = pd.read_pickle('%s/%s_%s_trsize_%s.pkl' %(args.save, \
  #                 args.partition, flname, args.fold))
  # train_idx, test_idx = ts_record['train_idx'], ts_record['test_idx']
  # x_test = seq_data.ix[train_idx, 'String'].values
  # y_test = seq_data.ix[test_idx, 'Tag'].values
  # tr_perfs = ts_record['perfs']
  # tr_preds = ts_record['preds']
  # tr_size = ts_record['tr_size']

  if args.partition == 'random':
    kf = list(random_part(seq_data))
  else:
    kf = loo_part(seq_data)
    
  train_idx, test_idx = kf[args.fold][0], kf[args.fold][1]
  print(len(train_idx), len(test_idx))
  x_test = seq_data.iloc[test_idx, :]['String'].values
  y_test = seq_data.iloc[test_idx, :]['Tag'].values
  tr_perfs = []
  tr_preds = []
  if (len(train_idx) < 10000):
    tr_size = [20, 50, 100, 200, 500, 1000, 2000, len(train_idx)]
   
  else:
    tr_size = [200, 500, 1000, 2000, 5000, 10000, len(train_idx)]
  # tr_size = [len(train_idx)]
  # ts_record = {'x_test':x_test, 'y_test':y_test, 'tag2idx':tags, 'tr_size':tr_size}
  ts_record = {'tag2idx':tags, 'tr_size':tr_size, 'train_idx':kaggle_data.index.values[train_idx],\
               'test_idx':kaggle_data.index.values[test_idx]}

  _, tags_true = package(x_test, y_test, volatile=True)
  tags_true = tags_true.data.numpy()
  ts_record['tags_true'] = tags_true

  if args.pretrain != '':
    # Baseline
    baseline = np.zeros(tags_true.shape)
    tr_perfs.append(perfs(tags_true, baseline, tag2idx=tags, original_str=x_test))
    tr_preds.append(baseline)

    # Regular Expression
    re_test = seq_data.ix[test_idx, 'R.E.tag'].values
    _, tags_re = package(x_test, re_test, volatile=True)
    tags_re = tags_re.data.numpy()
    tr_perfs.append(perfs(tags_true, tags_re, tag2idx=tags, original_str=x_test))
    tr_preds.append(tags_re)

    # Pretrained Regular Expression
    # pretrain_pred = pretrain_predict(x_test, y_test, args)
    mre_model = load_pretrain(args)
    pretrain_pred, _ = predict(mre_model, x_test, y_test)

    tr_perfs.append(perfs(tags_true, pretrain_pred, tag2idx=tags, original_str=x_test))
    tr_preds.append(pretrain_pred)

    # Trainining set Re and MRE for active samplling.
    print('Active Sampling....')
    x_train_all = seq_data.iloc[train_idx, ]['String'].values
    re_train_all = seq_data.ix[train_idx, 'R.E.tag'].values
    _, tags_re_trainall = package(x_train_all, re_train_all, volatile=True)
    tags_re_trainall = tags_re_trainall.data.numpy()
    # pretrain_pred_trainall = pretrain_predict(x_train_all, re_train_all, args)
    pretrain_pred_trainall, _ = predict(mre_model, x_train_all, re_train_all)
    disagree_scores = np.array([sum(tags_re_trainall[i] != pretrain_pred_trainall[i]) / len(tags_re_trainall[i]) \
                for i in range(len(tags_re_trainall))])
    disagree_idx = np.argsort(disagree_scores)

  random_sample_idxs = []
  active_sample_idxs = []
  for i, ts in enumerate(tr_size):
    np.random.seed(i)
    # if args.pretrain == '':
    #   train_idx_s = np.random.choice(train_idx, ts, replace=False)
    # else:
    #   print('Active Sampling....')
    #   random_sample_idxs.append(np.random.choice(train_idx, ts, replace=False))

    #   top_ts_disagree = disagree_idx[-ts:]
    #   train_idx_s = train_idx[top_ts_disagree]
    #   active_sample_idxs.append(train_idx_s)

    train_idx_s = np.random.choice(train_idx, ts, replace=False)
    x_train = seq_data.iloc[train_idx_s, ]['String'].values
    y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
    print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
    model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args)
    perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
    print(perf)
    tr_perfs.append(perf)
    tr_preds.append(tags_pred)
    
    ts_record['preds'] = tr_preds
    ts_record['perfs'] = tr_perfs
    ts_record['random_sample_idxs'] = random_sample_idxs
    ts_record['active_sample_idxs'] = active_sample_idxs
    pd.to_pickle(ts_record, '%s/%s_%s_trsize_%s.pkl' %(args.save, args.partition, flname, args.fold))


def evaluate_rerandom():
  random_data = pd.read_csv(args.data, encoding='utf-8', lineterminator='\n')
  str_len = np.array([len(s) for s in random_data.String])
  tag_len = np.array([len(s) for s in random_data['R.E.tag']])
  assert(np.all(str_len == tag_len))

  if 'kaggle' in random_data:
    test_data = pd.read_csv('data/position/testKaggle2.csv', encoding='utf-8', lineterminator='\n')
  else:
    test_data = pd.read_csv(args.data[:-10] + '.csv', encoding='utf-8', lineterminator='\n')

  model, tags_pred, _, perf = train_test_tagger(random_data.String.values, \
                                                random_data['R.E.tag'].values, \
                                                test_data.String.values, \
                                                test_data.TagLabel.values, args)


  pd.to_pickle(perf, os.path.join(args.save, 'evaluate_t_random.pkl'))

def evaluate_retoy():
  retoy = pd.read_csv(args.data + 'All.csv', encoding='utf-8')
  retoy = retoy[~ retoy['R.E.tag'].isnull()]
  str_len = np.array([len(s) for s in retoy.String])
  tag_len = np.array([len(s) for s in retoy['R.E.tag']])
  assert(np.all(str_len == tag_len))
  retoy = retoy[str_len == tag_len]

  random = pd.read_csv(args.data + 'Random_REtag.csv', encoding='utf-8')
  kaggle_data = pd.read_csv(args.data +'2.csv', encoding='utf-8')
  str_len = np.array([len(s) for s in kaggle_data.String])
  tag_len = np.array([len(s) for s in kaggle_data.TagLabel])
  assert(np.all(str_len == tag_len))
  seq_data = kaggle_data[str_len == tag_len]

  if not os.path.exists(args.save):
    os.makedirs(args.save)  

  if args.partition == 'random':
    kf = list(random_part(seq_data))
  else:
    kf = loo_part(seq_data)
    
  train_idx, test_idx = kf[args.fold][0], kf[args.fold][1]
  print(len(train_idx), len(test_idx))
  x_test = seq_data.iloc[test_idx, :].String.values
  y_test = seq_data.iloc[test_idx, :].TagLabel.values

  size = min([len(retoy), len(random)])
  percents = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
  
  # percents = []
  save = args.save

  results = {'retoy_preds':[], 'random_preds':[], 'retoy_perfs':[], 'random_perfs':[]}
  inds_retoy = np.arange(len(retoy))
  inds_random = np.arange(len(random))
  np.random.shuffle(inds_retoy)
  np.random.shuffle(inds_random)

  _, tags_true = package(x_test, y_test, volatile=True)
  tags_true = tags_true.data.numpy()
  for pert in percents:
    print('Training model with %s percent of REtoy.' %(pert * 100))
    tr_sz = int(size * pert)
    inds = inds_retoy[:tr_sz]
    model, tags_pred, _, perf = train_test_tagger(retoy.String.values[inds], \
                                               retoy['R.E.tag'].values[inds], \
                                               x_test, \
                                               y_test, args)
    perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=seq_data.TagLabel.values)
    print(perf)
    results['retoy_preds'].append(tags_pred)
    results['retoy_perfs'].append(perf)
    pd.to_pickle(results, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))
    with open(os.path.join(args.save, 'retoy_%i.pt'%(pert*100)), 'wb') as f:
      torch.save(model.state_dict(), f)

  for pert in percents:
    print('Training model with %s percent of RErandom.' %(pert * 100))
    tr_sz = int(size * pert)

    inds = inds_random[:tr_sz]
    model, tags_pred, _, perf = train_test_tagger(random.String.values[inds], \
                                               random['R.E.tag'].values[inds], \
                                               x_test, \
                                               y_test, args)
    perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=seq_data.TagLabel.values)
    print(perf)
    results['random_preds'].append(tags_pred)
    results['random_perfs'].append(perf)
    pd.to_pickle(results, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))
    with open(os.path.join(args.save, 'random_%i.pt'%(pert*100)), 'wb') as f:
      torch.save(model.state_dict(), f)


def pretrain_model():
  kaggle_data = pd.read_csv(args.data, encoding='utf-8', lineterminator='\n')
  kaggle_data['Tag'] = kaggle_data[args.label]
#  kaggle_data.columns = ['Outlet', 'kaggle_time', 'title', 'url', 'String', 'Prediction',
#         u'boundary labels', u'Tag']
#  kaggle_data.Outlet.fillna(method='ffill', inplace=True)      
#  kaggle_data.title.fillna(method='ffill', inplace=True)      
#  kaggle_data.kaggle_time.fillna(method='ffill', inplace=True)
#  kaggle_data.url.fillna(method='ffill', inplace=True)
  flname = args.label
  kaggle_data = kaggle_data[~ pd.isnull(kaggle_data.Tag)]

  str_len = np.array([len(s) for s in kaggle_data.String])
  tag_len = np.array([len(s) for s in kaggle_data.Tag])
  assert(np.all(str_len == tag_len))
  seq_data = kaggle_data[str_len == tag_len]

  if not os.path.exists(args.save):
    os.makedirs(args.save)  
  x_train = seq_data['String'].values
  y_train = seq_data['Tag'].values
  print(x_train.shape)
  size = int(0.1*len(x_train))
  test_inds = np.random.choice(np.arange(len(x_train)), size=size, replace=False)
  x_test = x_train[test_inds]
  y_test = y_train[test_inds]
  print(x_test.shape)
  model, tags_pred, _, _ = train_test_tagger(x_train, y_train, x_test, y_test, args)
  with open(os.path.join(args.save, '%s_%s_pretrain_%s.pt'%(flname, args.data.split('/')[-1][:-4], args.epochs)), 'wb') as f:
    torch.save(model.state_dict(), f)
  f.close()
  _, tags_true = package(x_test, y_test, volatile=True)
  tags_true = tags_true.data.numpy()
  args.train_perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_train)
  print(args.train_perf)
  pd.to_pickle(args, os.path.join(args.save, '%s_%s_pretrain_args.pkl'%(flname, args.data.split('/')[-1][:-4]))) 

if __name__ == '__main__':
 
 args = get_args()
 dictionary = Dictionary()
 dictionary.idx2word = ['<pad>'] + [chr(i) for i in range(256)]
 dictionary.word2idx = {w:i for i, w in enumerate(dictionary.idx2word)}
 n_token = len(dictionary)

 tags = Dictionary()
 tags.idx2word = [tag for tag in args.tags]
 tags.word2idx = {tag:i for i, tag in enumerate(tags.idx2word)}
 print(tags.idx2word, tags.word2idx)

 torch.manual_seed(args.seed)
 if torch.cuda.is_available():
   if not args.cuda:
     print("WARNING: You have a CUDA device, so you should probably run with --cuda")
   else:
     torch.cuda.manual_seed(args.seed)
 random.seed(args.seed)    

 if args.params != '':
   kaggle_records = pd.read_pickle(args.params)
   if type(kaggle_records) == list:
     accs = np.array([r['%s_acc' %(args.partition)] for r in kaggle_records])
     best_perf = np.where(accs == np.max(accs))[0][0]
     args.emsize = kaggle_records[best_perf]['args'].emsize
     args.lr = kaggle_records[best_perf]['args'].lr
     args.nhid = kaggle_records[best_perf]['args'].nhid  
     args.nlayers = kaggle_records[best_perf]['args'].nlayers
   else:
     args.emsize = kaggle_records.emsize
     args.nhid = kaggle_records.nhid
     args.nlayers = kaggle_records.nlayers
     # args.dropout = kaggle_records.dropout
     # args.max_len = kaggle_records.max_len
     args.lr = kaggle_records.lr
   
   if args.run == 'pretrain':
     pretrain_model()
   if args.run == 'train_sz':
     tagger_trsize()
   if args.run == 'eval_retoy':
     evaluate_retoy()
   if args.run == 'al':
     active_train()
 else:
   tagger()
