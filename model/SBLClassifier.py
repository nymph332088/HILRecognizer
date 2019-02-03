from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.functional as F
import torch.nn as nn
import os
import torch.autograd as autograd

def to_scalar(var):
  # returns a python float
  return var.view(-1).data.tolist()[0]

# return the argmax as a python int
def argmax(vec):
  _, idx = torch.max(vec, 1)
  # return idx.data.numpy()[0]
  return int(idx.data)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
  max_score = vec[0, argmax(vec)]
  max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
  return max_score + \
      torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTMCRFTagger(nn.Module):
  def __init__(self, config, tag_to_ix):
    super(BiLSTMCRFTagger, self).__init__()
    self.embedding_dim = config['ninp']
    self.hidden_dim = config['nhid']
    self.vocab_size = len(config['dictionary'].idx2word)
    self.nlayers = config['nlayers']
    self.tag_to_ix = tag_to_ix

    self.tagset_size = len(tag_to_ix)
    self.drop = nn.Dropout(config['dropout'])

    self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
    self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                        num_layers=config['nlayers'], bidirectional=True)
    # Maps the output of the LSTM into tag space.
    self.hidden2tag = nn.Linear(self.hidden_dim*2, self.tagset_size)

    # Matrix of transition parameters.  Entry i,j is the score of
    # transitioning *to* i *from* j.
    self.transitions = nn.Parameter(
        torch.randn(self.tagset_size, self.tagset_size))

    # These two statements enforce the constraint that we never transfer
    # to the start tag and we never transfer from the stop tag
    self.transitions.data[tag_to_ix['<START>'], :] = -10000
    self.transitions.data[:, tag_to_ix['<STOP>']] = -10000

    self.hidden = self.init_hidden()

  def init_hidden(self):
    weight = next(self.parameters()).data
    return (Variable(weight.new(self.nlayers * 2, 1, self.hidden_dim).zero_()),
            Variable(weight.new(self.nlayers * 2, 1, self.hidden_dim).zero_()))
  
  def init_weights(self, init_range=0.1):
    self.hidden2tag.weight.data.uniform_(-init_range, init_range)
    self.hidden2tag.bias.data.fill_(0)

  
  def _forward_alg(self, feats):
    # Do the forward algorithm to compute the partition function
    init_alphas = Variable(torch.zeros((1, self.tagset_size))).cuda() + -10000.
    # START_TAG has all of the score.
    init_alphas[0, self.tag_to_ix['<START>']] = 0.

    # Wrap in a variable so that we will get automatic backprop
    forward_var = init_alphas

    # Iterate through the sentence
    for feat in feats:
      alphas_t = []  # The forward tensors at this timestep
      for next_tag in range(self.tagset_size):
        # broadcast the emission score: it is the same regardless of
        # the previous tag
        emit_score = feat[next_tag].view(
            1, -1).expand(1, self.tagset_size)
        # the ith entry of trans_score is the score of transitioning to
        # next_tag from i
        trans_score = self.transitions[next_tag].view(1, -1)
        # The ith entry of next_tag_var is the value for the
        # edge (i -> next_tag) before we do log-sum-exp
        next_tag_var = forward_var + trans_score + emit_score
        # The forward variable for this tag is log-sum-exp of all the
        # scores.
        alphas_t.append(log_sum_exp(next_tag_var).view(1))
      forward_var = torch.cat(alphas_t).view(1, -1)
    terminal_var = forward_var + self.transitions[self.tag_to_ix['<STOP>']]
    alpha = log_sum_exp(terminal_var)
    return alpha

  def _get_lstm_features(self, sentence):
    self.hidden = self.init_hidden()
    embeds = self.drop(self.word_embeds(sentence)).view(len(sentence), 1, -1)
    lstm_out, self.hidden = self.lstm(embeds, self.hidden)
    lstm_out = lstm_out.view(len(sentence), self.hidden_dim * 2)
    lstm_feats = self.hidden2tag(lstm_out)
    return lstm_feats

  def _score_sentence(self, feats, tags):
    # Gives the score of a provided tag sequence
    score = Variable(torch.zeros(1)).cuda()
    tags = torch.cat([Variable(torch.LongTensor([self.tag_to_ix['<START>']])).cuda(), tags])
    for i, feat in enumerate(feats):
        score = score + \
            self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
    score = score + self.transitions[self.tag_to_ix['<STOP>'], tags.data[-1]]
    return score

  def _viterbi_decode(self, feats):
    backpointers = []

    # Initialize the viterbi variables in log space
    init_vvars = Variable(torch.zeros((1, self.tagset_size))).cuda() + -10000.
    init_vvars[0, self.tag_to_ix['<START>']] = 0

    # forward_var at step i holds the viterbi variables for step i-1
    forward_var = init_vvars
    for feat in feats:
      bptrs_t = []  # holds the backpointers for this step
      viterbivars_t = []  # holds the viterbi variables for this step

      for next_tag in range(self.tagset_size):
        # next_tag_var[i] holds the viterbi variable for tag i at the
        # previous step, plus the score of transitioning
        # from tag i to next_tag.
        # We don't include the emission scores here because the max
        # does not depend on them (we add them in below)
        next_tag_var = forward_var + self.transitions[next_tag]
        best_tag_id = argmax(next_tag_var)
        bptrs_t.append(best_tag_id)
        viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
      # Now add in the emission scores, and assign forward_var to the set
      # of viterbi variables we just computed
      forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
      backpointers.append(bptrs_t)

    # Transition to STOP_TAG
    terminal_var = forward_var + self.transitions[self.tag_to_ix['<STOP>']]
    best_tag_id = argmax(terminal_var)
    path_score = terminal_var[0][best_tag_id]

    # Follow the back pointers to decode the best path.
    best_path = [best_tag_id]
    for bptrs_t in reversed(backpointers):
      best_tag_id = bptrs_t[best_tag_id]
      best_path.append(best_tag_id)
    # Pop off the start tag (we dont want to return that to the caller)
    start = best_path.pop()
    assert start == self.tag_to_ix['<START>']  # Sanity check
    best_path.reverse()
    return path_score, best_path

  def neg_log_likelihood(self, sentence, tags):
    feats = self._get_lstm_features(sentence)
    forward_score = self._forward_alg(feats)
    gold_score = self._score_sentence(feats, tags)
    return forward_score - gold_score

  def forward(self, sentence):  # dont confuse this with _forward_alg above.
    # Get the emission scores from the BiLSTM
    lstm_feats = self._get_lstm_features(sentence)

    # Find the best path, given the features.
    score, tag_seq = self._viterbi_decode(lstm_feats)
    return score, tag_seq


class BiLSTMTagger(nn.Module):
  def __init__(self, config):
    super(BiLSTMTagger, self).__init__()
    self.bilstm = BiLSTM(config)

    # The linear layer that maps from hidden state space to tag space
    self.hidden2tag = nn.Linear(config['nhid'] * 2, config['target_size'])
    
  def init_weights(self, init_range=0.1):
    self.hidden2tag.weight.data.uniform_(-init_range, init_range)
    self.hidden2tag.bias.data.fill_(0)

  def init_hidden(self, bsz):
    return self.bilstm.init_hidden(bsz)
    
  def forward(self, inp, hidden):
    # inp size: [len, bsz] tokens
    # hidden size: [nlayers * 2, bsz, nhidden]
    # outp size: [bsz, len, nhid*2]
    
    outp, emb = self.bilstm.forward(inp, hidden)
    size = outp.size()  # [bsz, len, nhid*2]
    # print(size)
    new_outp = outp.view(-1, size[2])  # [bsz*len, nhid*2]
    
    tag_space = self.hidden2tag(new_outp) # [bsz*len, vocab size]
    # tag_scores = F.log_softmax(tag_space, dim=1).view(size[0], size[1], -1) # [bsz, len, vocab size]
    tag_scores = nn.LogSoftmax(dim=1)(tag_space).view(size[0], size[1], -1)
    return tag_scores, outp
    

  def encode(self, inp, hidden):
    return self.forward(inp, hidden)[0]

class LSTMTagger(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, cuda, n_layers, vocab_size, tagset_size):
    super(LSTMTagger, self).__init__()
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    self.iscuda = cuda
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

    # The LSTM takes word embeddings as inputs, and outputs hidden states
    # with dimensionality hidden_dim.
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, self.n_layers)

    # The linear layer that maps from hidden state space to tag space
    self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    self.hidden = self.init_hidden()

  def init_hidden(self):
    # Before we've done anything, we dont have any hidden state.
    # Refer to the Pytorch documentation to see exactly
    # why they have this dimensionality.
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    h0 = autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_dim))
    c0 = autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_dim))
    if self.iscuda:
      h0, c0 = h0.cuda(), c0.cuda()
    return (h0, c0)

  def forward(self, sentence):
    # sentence: [len(sent), emsize]
    embeds = self.word_embeddings(sentence)
    lstm_out, self.hidden = self.lstm(
        embeds.view(len(sentence), 1, -1), self.hidden)
    # tag space: [len(sent), tag size]
    tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

    tag_scores = F.log_softmax(tag_space, dim=1)
    return tag_scores
        

      
class CNN(nn.Module):
  def __init__(self, config):
    super(CNN, self).__init__()
    self.FILTERS = config["filters"]
    self.FILTER_NUM = config["filter_nums"]
    self.DROPOUT_PROB = config["dropout"]
    self.IN_CHANNEL = config['ninp']

    assert (len(self.FILTERS) == len(self.FILTER_NUM))

    # one for zero padding
    self.embedding = nn.Embedding(config['ntoken'], config['ninp'], padding_idx=0)
    for i in range(len(self.FILTERS)):
      conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.FILTERS[i])
      setattr(self, 'conv_%s'%(i), conv)

    self.relu = nn.ReLU()
  def get_conv(self, i):
    return getattr(self, 'conv_%s'%(i)) 

  def init_hidden(self, bsz):
    return None
    
  def forward(self, inp, hidden):
    # size of inp: [len, bsz] for the sake of LSTM needs [len, bsz, emb_size]
    #[len, bsz, emb_size] => [bsz, emb_size, len]
    emb = self.embedding(inp).permute(1, 2, 0)
      
    conv_results = []
    for i in range(len(self.FILTERS)):
      # [bsz, num_filter_i, len_out]
      conv_r = self.get_conv(i)(emb)
      conv_r = self.relu(conv_r)
      size = conv_r.size()
      # [bsz, num_filter_i, 1]
      pool = nn.MaxPool1d(size[2])(conv_r)
      # [bsz, num_filter_i]
      conv_results.append(pool.view(size[0], -1))
    # [ bsz, num_filter_total]
    outp = torch.cat(conv_results, 1)
    return outp, emb
    
class MLP(nn.Module):
  def __init__(self, config):
    super(MLP, self).__init__()
    self.embedding = nn.Embedding(config['ntoken'], config['ninp'], padding_idx=0)

  def init_hidden(self, bsz):
    return None
  
  def forward(self, inp, hidden):
    # [len, bsz] => [len, bsz, emb_dim] => [bsz, len, emb_dim]
    emb = self.embedding(inp).permute(1, 0, 2).contiguous()
    size = emb.size()
    outp = emb.view(size[0], -1)
    return outp, None
    
class BiLSTM(nn.Module):
  def __init__(self, config):
    super(BiLSTM, self).__init__()
    self.drop = nn.Dropout(config['dropout'])
    self.encoder = nn.Embedding(config['ntoken'], config['ninp'])
    self.bilstm = nn.LSTM(config['ninp'], config['nhid'], config['nlayers'], dropout=config['dropout'],
                          bidirectional=True)
    self.nlayers = config['nlayers']
    self.nhid = config['nhid']
    self.pooling = config['pooling']
    self.dictionary = config['dictionary']
#        self.init_weights()
    self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
    if os.path.exists(config['word-vector']):
      print('Loading word vectors from', config['word-vector'])
      vectors = torch.load(config['word-vector'])
      assert vectors[2] >= config['ninp']
      vocab = vectors[0]
      vectors = vectors[1]
      loaded_cnt = 0
      for word in self.dictionary.word2idx:
        if word not in vocab:
            continue
        real_id = self.dictionary.word2idx[word]
        loaded_id = vocab[word]
        self.encoder.weight.data[real_id] = vectors[loaded_id][:config['ninp']]
        loaded_cnt += 1
      print('%d words from external word vectors loaded.' % loaded_cnt)

  # note: init_range constraints the value of initial weights
  def init_weights(self, init_range=0.1):
    self.encoder.weight.data.uniform_(-init_range, init_range)

  def forward(self, inp, hidden):
    # emb size = [len, bsz, emb_size]
    emb = self.drop(self.encoder(inp))
    # print(emb)
    # print(hidden)
    # print(self.bilstm(emb, hidden))
    # outp size = [len, bsz, emb_size]        
    outp = self.bilstm(emb, hidden)[0]
    
    if self.pooling == 'mean':
        outp = torch.mean(outp, 0).squeeze()
    elif self.pooling == 'max':
        outp = torch.max(outp, 0)[0].squeeze()
    elif self.pooling == 'all' or self.pooling == 'all-word':
        outp = torch.transpose(outp, 0, 1).contiguous()
    return outp, emb

  def init_hidden(self, bsz):
    weight = next(self.parameters()).data
    return (Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()),
            Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()))


class SelfAttentiveEncoder(nn.Module):

  def __init__(self, config):
    super(SelfAttentiveEncoder, self).__init__()
    self.bilstm = BiLSTM(config)
    self.drop = nn.Dropout(config['dropout'])
    self.ws1 = nn.Linear(config['nhid'] * 2, config['attention-unit'], bias=False)
    self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax()
    self.dictionary = config['dictionary']
#        self.init_weights()
    self.attention_hops = config['attention-hops']

  def init_weights(self, init_range=0.1):
    self.ws1.weight.data.uniform_(-init_range, init_range)
    self.ws2.weight.data.uniform_(-init_range, init_range)

  def forward(self, inp, hidden):
    outp = self.bilstm.forward(inp, hidden)[0]
    size = outp.size()  # [bsz, len, nhid*2]
    compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
    transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
    transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
    concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
    concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

    hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
    alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
    alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
    penalized_alphas = alphas + (
        -10000 * (concatenated_inp == self.dictionary.word2idx['<pad>']).float())
        # [bsz, hop, len] + [bsz, hop, len]
    alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
    alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
    return torch.bmm(alphas, outp), alphas

  def init_hidden(self, bsz):
    return self.bilstm.init_hidden(bsz)


class Classifier(nn.Module):

  def __init__(self, config):
    super(Classifier, self).__init__()
    if config['pooling'] == 'mean' or config['pooling'] == 'max':
      self.encoder = BiLSTM(config)
      self.fc = nn.Linear(config['nhid'] * 2, config['nfc'])
      self.mode = 'lstm'
    elif config['pooling'] == 'all':
      self.encoder = SelfAttentiveEncoder(config)
      self.fc = nn.Linear(config['nhid'] * 2 * config['attention-hops'], config['nfc'])
      self.mode = 'alstm'
    elif config['pooling'] == 'conv':
      self.encoder = CNN(config)
      self.fc = nn.Linear(sum(config['filter_nums']), config['nfc'])
      self.mode = 'cnn'
    elif config['pooling'] == 'mlp':
      self.encoder = MLP(config)
      self.fc = nn.Linear(config['max_len']*config['ninp'], config['nfc'])
      self.mode = 'mlp'
    else:
      raise Exception('Error when initializing Classifier')
    self.drop = nn.Dropout(config['dropout'])
    self.tanh = nn.Tanh()
    self.pred = nn.Linear(config['nfc'], config['class-number'])
    self.dictionary = config['dictionary']
      
#        self.init_weights()

  def init_weights(self, init_range=0.1):
    self.fc.weight.data.uniform_(-init_range, init_range)
    self.fc.bias.data.fill_(0)
    self.pred.weight.data.uniform_(-init_range, init_range)
    self.pred.bias.data.fill_(0)

  def forward(self, inp, hidden):
    outp, attention = self.encoder.forward(inp, hidden)
    outp = outp.view(outp.size(0), -1)
    fc = self.tanh(self.fc(self.drop(outp)))
    pred = self.pred(self.drop(fc))
    if type(self.encoder) != SelfAttentiveEncoder:
      attention = None
    return pred, attention

  def init_hidden(self, bsz):
    return self.encoder.init_hidden(bsz)

  def encode(self, inp, hidden):
    return self.encoder.forward(inp, hidden)[0]

        
class MultiTaskClassifier(nn.Module):
  def __init__(self, config):
    super(Classifier, self).__init__()
    if config['pooling'] == 'mean' or config['pooling'] == 'max':
      self.encoder = BiLSTM(config)
      self.fc = nn.Linear(config['nhid'] * 2, config['nfc'])
      self.mode = 'lstm'
    elif config['pooling'] == 'all':
      self.encoder = SelfAttentiveEncoder(config)
      self.fc = nn.Linear(config['nhid'] * 2 * config['attention-hops'], config['nfc'])
      self.mode = 'alstm'
    elif config['pooling'] == 'conv':
      self.encoder = CNN(config)
      self.fc = nn.Linear(sum(config['filter_nums']), config['nfc'])
      self.mode = 'cnn'
    elif config['pooling'] == 'mlp':
      self.encoder = MLP(config)
      self.fc = nn.Linear(config['max_len']*config['ninp'], config['nfc'])
      self.mode = 'mlp'
    else:
      raise Exception('Error when initializing Classifier')
    self.drop = nn.Dropout(config['dropout'])
    self.tanh = nn.Tanh()
    self.preds = [nn.Linear(config['nfc'], class_num) \
                  for class_num in config['class-number']]
    self.dictionary = config['dictionary']
      
    self.init_weights()

  def init_weights(self, init_range=0.1):
    self.fc.weight.data.uniform_(-init_range, init_range)
    self.fc.bias.data.fill_(0)
    for pred in self.preds:
      pred.weight.data.uniform_(-init_range, init_range)
      pred.bias.data.fill_(0)

  def forward(self, inp, hidden):
    outp, attention = self.encoder.forward(inp, hidden)
    outp = outp.view(outp.size(0), -1)
    fc = self.tanh(self.fc(self.drop(outp)))
    pred = self.pred(self.drop(fc))
    if type(self.encoder) != SelfAttentiveEncoder:
      attention = None
    return pred, attention

  def init_hidden(self, bsz):
    return self.encoder.init_hidden(bsz)

  def encode(self, inp, hidden):
    return self.encoder.forward(inp, hidden)[0]
  