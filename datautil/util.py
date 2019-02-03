import json
import argparse

import numpy as np
from sklearn.cross_validation import KFold

class Dictionary(object):
  def __init__(self, path=''):
    self.word2idx = dict()
    self.idx2word = list()
    if path != '':  # load an external dictionary
      words = json.loads(open(path, 'r').readline())
      for item in words:
        self.add_word(item)

  def add_word(self, word):
    if word not in self.word2idx:
      self.idx2word.append(word)
      self.word2idx[word] = len(self.idx2word) - 1
    return self.word2idx[word]

  def __len__(self):
    return len(self.idx2word)


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
  parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
  parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers in BiLSTM')
  parser.add_argument('--attention-unit', type=int, default=350,
                        help='number of attention unit')
  parser.add_argument('--attention-hops', type=int, default=1,
                        help='number of attention hops, for multi-hop attention model')
  parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
  parser.add_argument('--clip', type=float, default=0.5,
                        help='clip to prevent the too large grad in LSTM')
  parser.add_argument('--nfc', type=int, default=512,
                        help='hidden (fully connected) layer size for classifier MLP')
  parser.add_argument('--lr', type=float, default=.001,
                        help='initial learning rate')
  parser.add_argument('--pooling', type=str, default='all',
                        help='embedding inp to classifier: max/mean, all, conv')
  parser.add_argument('--filters', type=int, nargs='+', default=[2, 3, 4, 5, 6],
                        help='kernel sizes for the cnn')
  parser.add_argument('--filter_nums', type=int, nargs='+', default=[50,25,25,25,50], 
                        help='number of kernels for each kernel size')
  parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
  parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
  parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
  parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
  parser.add_argument('--save', type=str, default='',
                        help='path to save the final model')
  parser.add_argument('--dictionary', type=str, default='',
                        help='path to save the dictionary, for faster corpus loading')
  parser.add_argument('--word-vector', type=str, default='',
                        help='path for pre-trained word vectors (e.g. GloVe), should be a PyTorch model.')
  parser.add_argument('--max_len', type=int, default=104, 
                        help='maximum length of sequence.')
  parser.add_argument('--train-data', type=str, default='',
                        help='location of the training data, should be a json file')
  parser.add_argument('--val-data', type=str, default='',
                        help='location of the development data, should be a json file')
  parser.add_argument('--test-data', type=str, default='',
                        help='location of the test data, should be a json file')
  parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training')
  parser.add_argument('--class-number', type=int, default=2,
                        help='number of classes')
  parser.add_argument('--optimizer', type=str, default='Adam',
                        help='type of optimizer')
  parser.add_argument('--penalization-coeff', type=float, default=1, 
                        help='the penalization coefficient')
  parser.add_argument('--partition', type=str, default='loo',
                      help='type of partition')
  parser.add_argument('--data', type=str, default='data/testKaggle2.xlsx')
  parser.add_argument('--params', type=str, default='', help='hyper parameters')
  parser.add_argument('--pretrain', type=str, default='', help='pretrained parameters')
  parser.add_argument('--tags', type=str, nargs='+', default=['y'], help='seq2seq tag space')
  parser.add_argument('--label', type=str, default='Label', help='column to build model on.')
  parser.add_argument('--fold', type=int, default=3, help='test idx fold')
  parser.add_argument('--run', type=str, default='pretrain', help='pretrain|train_sz|eval_retoy|al')
  parser.add_argument('--al_bs', type=int, default=20)
  parser.add_argument('--al_iter', type=int, default=70)
  parser.add_argument('--strategies', type=int, nargs='+', default=[0, 1, 3, 4, 7])
  return parser.parse_args()



def loo_part(data):
  counts = data.Media.value_counts()
  # print(counts)
  partitions = []
  per_part = counts.sum() / 5

  i, j = 0, len(counts)-1
  prev_i, prev_j = i, j
  while (i < j):
    sum = 0
    while (sum <= per_part) and (i < len(counts)) and (i < j):
      sum += counts.values[i]
      i += 1
    i -= 1
    sum -= counts.values[i]

    while (sum < per_part) and (i < len(counts)) and (i < j):
      sum += counts.values[j]
      j -= 1
    partitions.append(range(prev_i, i) + range(prev_j, j, -1))
    prev_i, prev_j = i, j

  partitions[-1].append(i)

  # print(partitions)
  print(len(partitions), partitions[-1])
  kf = []
  for k in range(5):
    train_idx, test_idx = np.where(~data.Media.isin(counts.index[partitions[k]]))[0],\
                          np.where(data.Media.isin(counts.index[partitions[k]]))[0]
    kf.append((train_idx, test_idx))

  return kf


def outlet_part(data):
  np.random.seed(2345)
  outlets = data.Outlet.unique()
  np.random.shuffle(outlets)
  kf_outlet = KFold(len(outlets), n_folds=5)
  
  kf = []
  for train, test in kf_outlet:
    train_idx = np.where(data.Outlet.isin(outlets[train]))[0]
    test_idx = np.where(data.Outlet.isin(outlets[test]))[0]
    kf.append((train_idx, test_idx))
    
  return kf
    
  
  
def random_part(data):
  # Prepare training and testing data
  kf = KFold(len(data), n_folds=5, random_state=3)
  return kf

