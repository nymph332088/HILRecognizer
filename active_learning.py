from sklearn.model_selection import ShuffleSplit
def active_train():
  # Random selection no pretraining.
  def strategy0():
    args.pretrain = ''
    ts_record['strategy0_sample'] = random_sample_idxs
    ts_record['strategy0_perfs'] = [tr_perfs[-1]]
    model = load_pretrain(args)
    for i in range(iterations):
      print('Random sampling without pretrain iteration %s' %(i))
      # sample_size = 2 ** (i+4)
      sample_size = bs * (i + 1)
      if sample_size > len(train_idx): break
      train_idx_s = random_sample_idxs[:sample_size]
      # train_idx_s = random_sample_idxs[i*bs:(i+1)*bs]
      assert(len(np.unique(train_idx_s)) == len(train_idx_s))
      np.random.shuffle(train_idx_s)

      x_train = seq_data.iloc[train_idx_s, ]['String'].values
      y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
      print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
      model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args, model)
      perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
      print(perf)
      args.pretrain = os.path.join(args.save, 'strategy0_model.pt')
      ts_record['strategy0_perfs'].append(perf)
      ts_record['strategy0_pred'] = tags_pred
      pd.to_pickle(ts_record, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))
    with open(os.path.join(args.save, 'strategy0_model_%s.pt'%(args.fold)), 'wb') as f:
      torch.save(model.state_dict(), f)
  
  # Random Selection with pretrain
  def strategy1():
    args.pretrain = mre
    ts_record['strategy1_sample'] = random_sample_idxs
    ts_record['strategy1_perfs'] = [tr_perfs[-1]]
    model = load_pretrain(args)

    for i in range(iterations):
      print('Random sampling iteration %s' %(i))
      # sample_size = 2 ** (i+4)
      sample_size = bs * (i + 1)
      if sample_size > len(train_idx): break
      train_idx_s = random_sample_idxs[:sample_size]
      # train_idx_s = random_sample_idxs[i*bs:(i+1)*bs]
      assert(len(np.unique(train_idx_s)) == len(train_idx_s))
      np.random.shuffle(train_idx_s)

      x_train = seq_data.iloc[train_idx_s, ]['String'].values
      y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
      print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
      model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args, model)
      perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
      print(perf)
      # with open(os.path.join(args.save, 'strategy1_model.pt'), 'wb') as f:
      #   torch.save(model.state_dict(), f)
      args.pretrain = os.path.join(args.save, 'strategy1_model.pt')
      ts_record['strategy1_perfs'].append(perf)
      ts_record['strategy1_pred'] = tags_pred
      pd.to_pickle(ts_record, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))

  # # Active selection 1
  # def strategy2():
  #   args.pretrain = mre
  #   # Trainining set Re and MRE for active samplling.
  #   print('Calculating disagree scores....')
  #   x_train_all = seq_data.iloc[train_idx, ]['String'].values
  #   re_train_all = seq_data.ix[train_idx, 'R.E.tag'].values
  #   _, tags_re_trainall = package(x_train_all, re_train_all, volatile=True)
  #   tags_re_trainall = tags_re_trainall.data.numpy()
  #   # pretrain_pred_trainall = pretrain_predict(x_train_all, re_train_all, args)

  #   pretrain_pred_trainall, _ = predict(mre_model, x_train_all, re_train_all)

  #   disagree_scores = np.array([sum(tags_re_trainall[i] != pretrain_pred_trainall[i]) / len(tags_re_trainall[i]) \
  #               for i in range(len(tags_re_trainall))])
  #   disagree_idx = np.argsort(-disagree_scores)

  #   ts_record['strategy2_sample'] = disagree_idx
  #   ts_record['strategy2_perfs'] = [tr_perfs[-1]]
    
  #   for i in range(iterations):
  #     print('Disagree sampling iteration %s' %(i))
  #     # sample_size = 2 ** (i+4) 
  #     sample_size = bs * (i + 1)
  #     if sample_size > len(train_idx): break
  #     top_ts_disagree = disagree_idx[:sample_size]
  #     # top_ts_disagree = disagree_idx[i*bs:(i+1)*bs]
  #     train_idx_s = train_idx[top_ts_disagree]
  #     assert(len(np.unique(train_idx_s)) == len(train_idx_s))
  #     np.random.shuffle(train_idx_s)

  #     x_train = seq_data.iloc[train_idx_s, ]['String'].values
  #     y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
  #     print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
  #     model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args)
  #     perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
  #     print(perf)
  #     with open(os.path.join(args.save, 'strategy2_model.pt'), 'wb') as f:
  #       torch.save(model.state_dict(), f)
  #     args.pretrain = os.path.join(args.save, 'strategy2_model.pt')
  #     ts_record['strategy2_perfs'].append(perf)
  #     pd.to_pickle(ts_record, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))

  #   Average average
  def strategy3():
    args.pretrain = mre
    ts_record['strategy3_sample'] = []
    ts_record['strategy3_perfs'] = [tr_perfs[-1]]
    model = load_pretrain(args)
    for i in range(iterations):
      print('Average average sampling iteration %s' %(i))
      # sample_size = 2 ** (i+4)
      # bs = sample_size / 2
      sample_size = bs * (i + 1)
      if sample_size > len(train_idx): break
      train_pred, train_prob = predict(model, x_train_all, re_train_all)
      entropy = entropies(train_prob, 'ave')
      rank_idx = np.argsort(-entropy)
      sorted_train_idx = train_idx[rank_idx]
      sorted_train_idx = sorted_train_idx[~ pd.Series(sorted_train_idx).isin(ts_record['strategy3_sample'])]
      train_idx_s = list(ts_record['strategy3_sample']) + list(sorted_train_idx[:bs])
      ts_record['strategy3_sample'] = train_idx_s
      # train_idx_s = list(sorted_train_idx[:bs])
      # ts_record['strategy3_sample'].extend(train_idx_s)
      assert(len(np.unique(train_idx_s)) == len(train_idx_s))
      np.random.shuffle(train_idx_s)

      x_train = seq_data.iloc[train_idx_s, ]['String'].values
      y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
      print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
      model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args, model)
      perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
      print(perf)
      # with open(os.path.join(args.save, 'strategy3_model.pt'), 'wb') as f:
      #   torch.save(model.state_dict(), f)
      args.pretrain = os.path.join(args.save, 'strategy3_model.pt')
      ts_record['strategy3_perfs'].append(perf)
      ts_record['strategy3_pred'] = tags_pred
      pd.to_pickle(ts_record, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))


  # Entropy max
  def strategy4():
    args.pretrain = mre
    ts_record['strategy4_sample'] = []
    ts_record['strategy4_perfs'] = [tr_perfs[-1]]
    model = load_pretrain(args)
    for i in range(iterations):
      print('Max sampling iteration %s' %(i))
      # sample_size = 2 ** (i+4)
      # bs = sample_size / 2
      sample_size = bs * (i + 1)
      if sample_size > len(train_idx): break
      train_pred, train_prob = predict(model, x_train_all, re_train_all)
      entropy = entropies(train_prob, 'max')
      rank_idx = np.argsort(-entropy)
      sorted_train_idx = train_idx[rank_idx]
      sorted_train_idx = sorted_train_idx[~ pd.Series(sorted_train_idx).isin(ts_record['strategy4_sample'])]
      train_idx_s = list(ts_record['strategy4_sample']) + list(sorted_train_idx[:bs])
      ts_record['strategy4_sample'] = train_idx_s
      # train_idx_s = list(sorted_train_idx[:bs])
      # ts_record['strategy4_sample'].extend(train_idx_s)
      assert(len(np.unique(train_idx_s)) == len(train_idx_s))
      np.random.shuffle(train_idx_s)
      x_train = seq_data.iloc[train_idx_s, ]['String'].values
      y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
      print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
      model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args, model)
      perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
      print(perf)
      # with open(os.path.join(args.save, 'strategy4_model.pt'), 'wb') as f:
      #   torch.save(model.state_dict(), f)
      args.pretrain = os.path.join(args.save, 'strategy4_model.pt')
      ts_record['strategy4_perfs'].append(perf)
      ts_record['strategy4_pred'] = tags_pred 
      pd.to_pickle(ts_record, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))

  # # Entropy sliding window max 
  # args.pretrain = mre
  # ts_record['strategy5_sample'] = []
  # ts_record['strategy5_perfs'] = [tr_perfs[-1]]

  # if 'Kaggle' in args.data:
  #   window = 20
  # elif 'CourseNumber' in args.data:
  #   # 5     773
  #   # 6     476
  #   # 7     209
  #   # 4     177
  #   window = 5
  # elif 'BillDate' in args.data:
  #   # 10    262
  #   # 17     31
  #   # 12     28
  #   window = 10
  # elif 'Email' in args.data:
  #   # 23    222
  #   # 20     91
  #   # 26     69
  #   window = 23
  # elif 'Phone' in args.data:
  #   # 14    590
  #   # 12    320
  #   # 13    112
  #   window = 14
        
  # model = load_pretrain(args)
  # for i in range(iterations):
  #   print('Sliding window iteration %s' %(i))
  #   # sample_size = 2 ** (i+4)
  #   # bs = sample_size / 2
  #   sample_size = bs * (i + 1)
  #   if sample_size > len(train_idx): break

  #   train_pred, train_prob = predict(model, x_train_all, re_train_all)
  #   entropy = entropies(train_prob, 'conv', window)
  #   rank_idx = np.argsort(-entropy)
  #   sorted_train_idx = train_idx[rank_idx]
  #   sorted_train_idx = sorted_train_idx[~ pd.Series(sorted_train_idx).isin(ts_record['strategy5_sample'])]
  #   train_idx_s = list(ts_record['strategy5_sample']) + list(sorted_train_idx[:bs])
  #   ts_record['strategy5_sample'] = train_idx_s
    
  #   # train_idx_s = list(sorted_train_idx[:bs])
  #   # ts_record['strategy5_sample'].extend(train_idx_s)
  #   assert(len(np.unique(train_idx_s)) == len(train_idx_s))
  #   np.random.shuffle(train_idx_s)
  #   x_train = seq_data.iloc[train_idx_s, ]['String'].values
  #   y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
  #   print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
  #   model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args)
  #   perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
  #   print(perf)
  #   with open(os.path.join(args.save, 'strategy5_model.pt'), 'wb') as f:
  #     torch.save(model.state_dict(), f)
  #   args.pretrain = os.path.join(args.save, 'strategy5_model.pt')
  #   ts_record['strategy5_perfs'].append(perf)

  #   pd.to_pickle(ts_record, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))

  # # Half average plus halp random 
  # args.pretrain = mre
  # ts_record['strategy6_sample'] = []
  # ts_record['strategy6_perfs'] = [tr_perfs[-1]]
  # model = load_pretrain(args)
  # for i in range(iterations):
  #   print('Half random and half uncertainty sampling iteration %s' %(i))
  #   # sample_size = 2 ** (i+4)
  #   # bs = sample_size / 2
  #   sample_size = bs * (i + 1)
  #   if sample_size > len(train_idx): break

  #   train_pred, train_prob = predict(model, x_train_all, re_train_all)
  #   entropy = entropies(train_prob, 'ave')
  #   rank_idx = np.argsort(-entropy)
  #   sorted_train_idx = train_idx[rank_idx]
  #   sorted_train_idx = sorted_train_idx[~ pd.Series(sorted_train_idx).isin(ts_record['strategy6_sample'])]
  #   half = int(bs/2)
  #   exist = list(ts_record['strategy6_sample'])
  #   top_half = list(sorted_train_idx[:half])
  #   random_half = list(np.random.choice(sorted_train_idx[half:], half, replace=False))
  #   train_idx_s = exist + top_half + random_half
  #   ts_record['strategy6_sample'] = train_idx_s

  #   # train_idx_s = top_half + random_half    
  #   # ts_record['strategy6_sample'].extend(train_idx_s)
  #   assert(len(np.unique(train_idx_s)) == len(train_idx_s))

  #   np.random.shuffle(train_idx_s)
  #   x_train = seq_data.iloc[train_idx_s, ]['String'].values
  #   y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
  #   print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
  #   model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args)
  #   perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
  #   print(perf)
  #   with open(os.path.join(args.save, 'strategy6_model.pt'), 'wb') as f:
  #     torch.save(model.state_dict(), f)
  #   args.pretrain = os.path.join(args.save, 'strategy6_model.pt')
  #   ts_record['strategy6_perfs'].append(perf)
  #   pd.to_pickle(ts_record, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))


  # First random then uncertainty
  def strategy7():
    args.pretrain = mre
    pool_size = int(len(train_idx) * 0.1)
    ts_record['strategy7_sample'] = []
    ts_record['strategy7_perfs'] = [tr_perfs[-1]]
    model = load_pretrain(args)
    for i in range(iterations):
      print('First random then uncertainty sampling iteration %s' %(i))
      # sample_size = 2 ** (i+4)
      # bs = sample_size / 2
      sample_size = bs * (i + 1)
      if sample_size > len(train_idx): break
      pool = train_idx[~ pd.Series(train_idx).isin(ts_record['strategy7_sample'])]
      subsample = np.random.choice(pool, pool_size, replace=False)
      subsample_x, subsample_y = seq_data.String.values[subsample], seq_data['R.E.tag'].values[subsample]
      train_pred, train_prob = predict(model, subsample_x, subsample_y)

      entropy = entropies(train_prob, 'ave')
      rank_idx = np.argsort(-entropy)
      sorted_train_idx = list(subsample[rank_idx][:bs])
      exist = list(ts_record['strategy7_sample'])
      train_idx_s = exist + sorted_train_idx
      ts_record['strategy7_sample'] = train_idx_s

      # train_idx_s = top_half + random_half    
      # ts_record['strategy6_sample'].extend(train_idx_s)
      assert(len(np.unique(train_idx_s)) == len(train_idx_s))

      np.random.shuffle(train_idx_s)
      x_train = seq_data.iloc[train_idx_s, ]['String'].values
      y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
      print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
      model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args, model)
      perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
      print(perf)
      # with open(os.path.join(args.save, 'strategy7_model.pt'), 'wb') as f:
      #   torch.save(model.state_dict(), f)
      args.pretrain = os.path.join(args.save, 'strategy7_model.pt')
      ts_record['strategy7_perfs'].append(perf)
      ts_record['strategy7_pred'] = tags_pred
      pd.to_pickle(ts_record, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))

  # # # We also need no pretrain but with active learning.
  def strategy8():
    args.pretrain = ''
    ts_record['strategy8_sample'] = []
    ts_record['strategy8_perfs'] = [tr_perfs[-1]]
    model = load_pretrain(args)
    for i in range(iterations):
      print('Max uncertainty without pretrain %s' %(i))
      # sample_size = 2 ** (i+4)
      # bs = sample_size / 2
      sample_size = bs * (i + 1)
      if sample_size > len(train_idx): break
      if i == 0:
        train_idx_s = random_sample_idxs[:sample_size]
      else:
        train_pred, train_prob = predict(model, x_train_all, re_train_all)
        entropy = entropies(train_prob, 'max')
        rank_idx = np.argsort(-entropy)
        sorted_train_idx = train_idx[rank_idx]
        sorted_train_idx = sorted_train_idx[~ pd.Series(sorted_train_idx).isin(ts_record['strategy8_sample'])]
        train_idx_s = list(ts_record['strategy8_sample']) + list(sorted_train_idx[:bs])
      ts_record['strategy8_sample'] = train_idx_s
      assert(len(np.unique(train_idx_s)) == len(train_idx_s))
      x_train = seq_data.iloc[train_idx_s, ]['String'].values
      y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
      print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
      model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args, model)
      perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
      print(perf)
      # with open(os.path.join(args.save, 'strategy8_model.pt'), 'wb') as f:
      #   torch.save(model.state_dict(), f)
      args.pretrain = os.path.join(args.save, 'strategy8_model.pt')
      ts_record['strategy8_perfs'].append(perf)
      ts_record['strategy8_pred'] = tags_pred
      pd.to_pickle(ts_record, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))

  kaggle_data = pd.read_csv(args.data, encoding='utf-8', lineterminator='\n')
  kaggle_data['Tag'] = kaggle_data[args.label]

  str_len = np.array([len(s) for s in kaggle_data.String])
  tag_len = np.array([len(s) for s in kaggle_data.Tag])
  assert(np.all(str_len == tag_len))
  seq_data = kaggle_data[str_len == tag_len]
  if not os.path.exists(args.save):
    os.makedirs(args.save)  

  flname = args.label
  if args.partition == 'random':
    kf = list(random_part(seq_data))
  elif args.partition == 'loo':
    kf = loo_part(seq_data)
  elif args.partition == 'outlet':
    kf = outlet_part(seq_data)
    
  # kf = ShuffleSplit(n_splits=10, test_size=1200, random_state=0).split(range(len(seq_data)))
  # kf = list(kf)
  
  
  
  train_idx, test_idx = kf[args.fold][0], kf[args.fold][1]
  print(len(train_idx), len(test_idx), args.strategies, args.al_bs, args.al_iter)
  L = min([10000, len(train_idx)])
  if len(train_idx) > L:
    np.random.seed(1234)
    train_idx = np.random.choice(train_idx, size=L, replace=False)
  print(len(train_idx), len(test_idx))

  np.random.seed(0)
  random_sample_idxs = np.random.choice(train_idx, len(train_idx), replace=False)
  x_test = seq_data.iloc[test_idx, :]['String'].values
  y_test = seq_data.iloc[test_idx, :]['Tag'].values
  x_train_all = seq_data.iloc[train_idx, ]['String'].values
  re_train_all = seq_data.ix[train_idx, 'R.E.tag'].values
  tr_perfs = []
  tr_preds = []

  # bs = 20
  # iterations = len(train_idx) / bs
  # iterations = 70

  bs = args.al_bs
  iterations = args.al_iter

  if os.path.exists(os.path.join(args.save, 'result_%s.pkl'%(args.fold))):
    print("######################Loading records.....")
    ts_record = pd.read_pickle(os.path.join(args.save, 'result_%s.pkl'%(args.fold)))
  else:
    ts_record = {'tag2idx':tags, 'train_idx':train_idx, 'test_idx':test_idx}
  

  _, tags_true = package(x_test, y_test, volatile=True)
  tags_true = tags_true.data.numpy()
  ts_record['tags_true'] = tags_true

  if args.pretrain != '':
    # Baseline
    baseline = np.zeros(tags_true.shape)
    tr_perfs.append(perfs(tags_true, baseline, tag2idx=tags, original_str=x_test))
    tr_preds.append(baseline)

    # Regular Expression Pretrain
    pre_re = args.pretrain.split('/')[-1].split('_')[0]
    re_test = seq_data.ix[test_idx, pre_re].values
    _, tags_re = package(x_test, re_test, volatile=True)
    tags_re = tags_re.data.numpy()
    tr_perfs.append(perfs(tags_true, tags_re, tag2idx=tags, original_str=x_test))
    tr_preds.append(tags_re)

    # Pretrained Regular Expression
    mre_model = load_pretrain(args)
    pretrain_pred, _ = predict(mre_model, x_test, y_test)
    # pretrain_pred = pretrain_predict(x_test, y_test, args)
    tr_perfs.append(perfs(tags_true, pretrain_pred, tag2idx=tags, original_str=x_test))
    tr_preds.append(pretrain_pred)

  
  for perf in tr_perfs: print(perf)
  if tr_perfs == []: 
    tr_perfs = [{'ACC': 0, 'All': 0, 'PosRecall': 0, 'EntPrec': 0,\
                 'PosF1': 0, 'PosPrec': 0, 'EntRecall': 0., 'EntF1': 0}] * 3

  ts_record['baselines'] = tr_preds
  ts_record['baseline_perfs'] = tr_perfs
  mre = args.pretrain
  print(mre)


  if 0 in args.strategies:
    # args.epochs = 10
    strategy0()
  if 1 in args.strategies:
    # args.epochs = 10
    strategy1()
  if 3 in args.strategies:
    strategy3()
  if 4 in args.strategies:
    strategy4()
  if 7 in args.strategies:
    strategy7()
  if 8 in args.strategies:
    strategy8()


def entropies(probs, mode='ave', window=10):
  from scipy.stats import entropy
  ens = np.array(map(lambda x: map(lambda y: entropy(y), x), probs))

  if mode == 'ave':
    ens = ens.mean(axis=-1)
  elif mode == 'max':
    ens = ens.max(axis=-1)
  elif mode == 'conv':
    ens_win = []
    for i in range(len(ens)):
      e = np.convolve(ens[i, :], np.ones((window,))/window, mode='valid')
      ens_win.append(max(e))
    ens = np.array(ens_win)
  return ens



# def active_train():
#   kaggle_data = pd.read_csv(args.data, encoding='utf-8')
#   kaggle_data['Tag'] = kaggle_data[args.label]

#   str_len = np.array([len(s) for s in kaggle_data.String])
#   tag_len = np.array([len(s) for s in kaggle_data.Tag])
#   assert(np.all(str_len == tag_len))
#   seq_data = kaggle_data[str_len == tag_len]
#   if not os.path.exists(args.save):
#     os.makedirs(args.save)  

#   flname = args.label


#   if args.partition == 'random':
#     kf = list(random_part(seq_data))
#   else:
#     kf = loo_part(seq_data)
    
#   train_idx, test_idx = kf[args.fold][0], kf[args.fold][1]
#   print(len(train_idx), len(test_idx))
#   x_test = seq_data.iloc[test_idx, :]['String'].values
#   y_test = seq_data.iloc[test_idx, :]['Tag'].values
#   tr_perfs = []
#   tr_preds = []

#   bs = 20
#   iterations = len(train_idx) / bs
#   iterations = 10

#   ts_record = {'tag2idx':tags, 'train_idx':kaggle_data.index.values[train_idx],\
#                'test_idx':kaggle_data.index.values[test_idx]}

#   _, tags_true = package(x_test, y_test, volatile=True)
#   tags_true = tags_true.data.numpy()
#   ts_record['tags_true'] = tags_true

#   if args.pretrain != '':
#     # Baseline
#     baseline = np.zeros(tags_true.shape)
#     tr_perfs.append(perfs(tags_true, baseline, tag2idx=tags, original_str=x_test))
#     tr_preds.append(baseline)

#     # Regular Expression
#     re_test = seq_data.ix[test_idx, 'R.E.tag'].values
#     _, tags_re = package(x_test, re_test, volatile=True)
#     tags_re = tags_re.data.numpy()
#     tr_perfs.append(perfs(tags_true, tags_re, tag2idx=tags, original_str=x_test))
#     tr_preds.append(tags_re)

#     # Pretrained Regular Expression
#     mre_model = load_pretrain(args)
#     pretrain_pred, _ = predict(mre_model, x_test, y_test)
#     # pretrain_pred = pretrain_predict(x_test, y_test, args)
#     tr_perfs.append(perfs(tags_true, pretrain_pred, tag2idx=tags, original_str=x_test))
#     tr_preds.append(pretrain_pred)



#   ts_record['baselines'] = tr_preds
#   ts_record['baseline_perfs'] = tr_perfs
#   mre = args.pretrain
#   # Random Selection
#   args.pretrain = mre
#   np.random.seed(0)
#   random_sample_idxs = np.random.choice(train_idx, len(train_idx), replace=False)
#   ts_record['random_sample_idxs'] = random_sample_idxs
#   ts_record['random_sample_preds'] = []
#   ts_record['random_sample_perfs'] = []
#   for i in range(iterations):
#     print('Random sampling iteration %s' %(i))
#     # sample_size = 2 ** (i+4)
#     sample_size = bs * (i + 1)
#     if sample_size > len(train_idx): break
#     train_idx_s = random_sample_idxs[:sample_size]
#     # train_idx_s = random_sample_idxs[:(i+1)*bs]
#     x_train = seq_data.iloc[train_idx_s, ]['String'].values
#     y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
#     print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
#     model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args)
#     perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
#     print(perf)
#     with open(os.path.join(args.save, 'random_model.pt'), 'wb') as f:
#       torch.save(model.state_dict(), f)
#     args.pretrain = os.path.join(args.save, 'random_model.pt')
#     ts_record['random_sample_preds'].append(tags_pred)
#     ts_record['random_sample_perfs'].append(perf)
#     pd.to_pickle(ts_record, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))

#   # Active selection 1
#   args.pretrain = mre
#   # Trainining set Re and MRE for active samplling.
#   print('Calculating disagree scores....')
#   x_train_all = seq_data.iloc[train_idx, ]['String'].values
#   re_train_all = seq_data.ix[train_idx, 'R.E.tag'].values
#   _, tags_re_trainall = package(x_train_all, re_train_all, volatile=True)
#   tags_re_trainall = tags_re_trainall.data.numpy()
#   # pretrain_pred_trainall = pretrain_predict(x_train_all, re_train_all, args)

#   pretrain_pred_trainall, _ = predict(mre_model, x_train_all, re_train_all)

#   disagree_scores = np.array([sum(tags_re_trainall[i] != pretrain_pred_trainall[i]) / len(tags_re_trainall[i]) \
#               for i in range(len(tags_re_trainall))])
#   disagree_idx = np.argsort(-disagree_scores)

#   ts_record['re_mre_sample_idxs'] = disagree_idx
#   ts_record['re_mre_sample_preds'] = []
#   ts_record['re_mre_sample_perfs'] = []
  
#   for i in range(iterations):
#     print('Disagree sampling iteration %s' %(i))
#     # sample_size = 2 ** (i+4) 
#     sample_size = bs * (i + 1)
#     if sample_size > len(train_idx): break
#     top_ts_disagree = disagree_idx[:sample_size]
#     # top_ts_disagree = disagree_idx[:(i+1)*bs]
#     train_idx_s = train_idx[top_ts_disagree]
#     x_train = seq_data.iloc[train_idx_s, ]['String'].values
#     y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
#     print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
#     model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args)
#     perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
#     print(perf)
#     with open(os.path.join(args.save, 're_mre_model.pt'), 'wb') as f:
#       torch.save(model.state_dict(), f)
#     args.pretrain = os.path.join(args.save, 're_mre_model.pt')
#     ts_record['re_mre_sample_preds'].append(tags_pred)
#     ts_record['re_mre_sample_perfs'].append(perf)
#     pd.to_pickle(ts_record, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))

#   # Hybrid
#   args.pretrain = mre
#   ts_record['hybrid_avg_sample_idxs'] = []
#   ts_record['hybrid_avg_sample_preds'] = []
#   ts_record['hybrid_avg_sample_perfs'] = []
#   for i in range(iterations):
#     print('Hybrid average sampling iteration %s' %(i))
#     # sample_size = 2 ** (i+4)
#     # bs = sample_size / 2
#     sample_size = bs * (i + 1)
#     if sample_size > len(train_idx): break
#     if i == 0:
#       top_ts_disagree = disagree_idx[:sample_size]
#       # top_ts_disagree = disagree_idx[:(i+1)*bs]
#       train_idx_s = train_idx[top_ts_disagree]
      
#     else:
#       train_pred, train_prob = predict(model, x_train_all, re_train_all)
#       entropy = entropies(train_prob, 'ave')
#       rank_idx = np.argsort(-entropy)
#       sorted_train_idx = train_idx[rank_idx]
#       sorted_train_idx = sorted_train_idx[~ pd.Series(sorted_train_idx).isin(ts_record['hybrid_avg_sample_idxs'])]
#       train_idx_s = list(ts_record['hybrid_avg_sample_idxs']) + list(sorted_train_idx[:bs])
#     ts_record['hybrid_avg_sample_idxs'] = train_idx_s
#     x_train = seq_data.iloc[train_idx_s, ]['String'].values
#     y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
#     print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
#     model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args)
#     perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
#     print(perf)
#     with open(os.path.join(args.save, 'hybrid_avg_model.pt'), 'wb') as f:
#       torch.save(model.state_dict(), f)
#     args.pretrain = os.path.join(args.save, 'hybrid_avg_model.pt')
#     ts_record['hybrid_avg_sample_preds'].append(tags_pred)
#     ts_record['hybrid_avg_sample_perfs'].append(perf)
#     pd.to_pickle(ts_record, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))


#   # Entropy max 
#   args.pretrain = mre
#   ts_record['hybrid_max_sample_idxs'] = []
#   ts_record['hybrid_max_sample_preds'] = []
#   ts_record['hybrid_max_sample_perfs'] = []
#   for i in range(iterations):
#     print('Hybrid max sampling iteration %s' %(i))
#     # sample_size = 2 ** (i+4)
#     # bs = sample_size / 2
#     sample_size = bs * (i + 1)
#     if sample_size > len(train_idx): break
#     if i == 0:
#       top_ts_disagree = disagree_idx[:sample_size]
#       # top_ts_disagree = disagree_idx[:(i+1)*bs]
#       train_idx_s = train_idx[top_ts_disagree]
      
#     else:
#       train_pred, train_prob = predict(model, x_train_all, re_train_all)
#       entropy = entropies(train_prob, 'max')
#       rank_idx = np.argsort(-entropy)
#       sorted_train_idx = train_idx[rank_idx]
#       sorted_train_idx = sorted_train_idx[~ pd.Series(sorted_train_idx).isin(ts_record['hybrid_max_sample_idxs'])]
#       train_idx_s = list(ts_record['hybrid_max_sample_idxs']) + list(sorted_train_idx[:bs])
#     ts_record['hybrid_max_sample_idxs'] = train_idx_s
#     x_train = seq_data.iloc[train_idx_s, ]['String'].values
#     y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
#     print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
#     model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args)
#     perf = perfs(tags_true, tags_pred, tag2idx=tags, original_str=x_test)
#     print(perf)
#     with open(os.path.join(args.save, 'hybrid_max_model.pt'), 'wb') as f:
#       torch.save(model.state_dict(), f)
#     args.pretrain = os.path.join(args.save, 'hybrid_max_model.pt')
#     ts_record['hybrid_max_sample_preds'].append(tags_pred)
#     ts_record['hybrid_max_sample_perfs'].append(perf)
#     pd.to_pickle(ts_record, os.path.join(args.save, 'result_%s.pkl'%(args.fold)))

