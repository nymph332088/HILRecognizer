# -*- coding: utf-8 -*-
import numpy as np
import cPickle as pickle
from os import path
from gensim.models import word2vec

def load_bin_vec(fname, vocab):
    try:
        model = word2vec.Word2Vec.load_word2vec_format(fname, binary=True)
    except UnicodeDecodeError:
        model = word2vec.Word2Vec.load_word2vec_format(fname, binary=True, encoding='ISO-8859-1')
#    _w2v_words = [w for w in model.vocab if w in vocab]
#    vocab = [w for w in vocab if w in _w2v_words]
    word_vecs = {}
    for i, word in enumerate(vocab):
        if (i % 10000 == 0) : print('Finished %s' %(i))        
        try:
            word_vecs[word] = model[word]
        except:
            pass
    return word_vecs, model.syn0.shape[1]
        
def load_bin_vec2(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            try:
                if word in vocab:
                   word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
                else:
                    f.read(binary_len)
            except:
                continue
    return word_vecs, layer1_size


    
def word_rep_bin(words, model):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    word_vecs, em_size = load_bin_vec2(model, words)
    embeddings = np.zeros((len(words), em_size))
    for i, word in enumerate(words):
        if word not in word_vecs:
            embeddings[i, :] = np.random.uniform(-0.25,0.25, em_size)  ### different w2v for OOV
        else:
            embeddings[i, :] = word_vecs[word]
    embeddings = embeddings / np.sqrt((embeddings ** 2).sum(-1))[..., np.newaxis]
    return embeddings, em_size


def word_rep(words, model=None, k=20):
    if(model is None):
        return np.random.uniform(-0.25, 0.25, (len(words), k))
    model = word2vec.Word2Vec.load(model)
    em_size = model.syn0.shape[1]
    embeddings = []
    for word in words:
        try:
            embeddings.append(model[word])
        except KeyError:
            embeddings.append(list(np.random.uniform(-0.25, 0.25, em_size)))
    embeddings = np.array(embeddings)
    embeddings = embeddings / np.sqrt((embeddings ** 2).sum(-1))[..., np.newaxis]
    return embeddings, em_size


