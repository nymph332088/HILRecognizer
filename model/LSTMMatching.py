 """
Created on Thu May  4 19:54:51 2017

@author: nymph
"""

from keras.models import Model, Sequential, model_from_json
from keras.layers import Layer, Input, Embedding, TimeDistributed, Dense, Reshape
from keras.layers import Dropout, Reshape, Merge, LSTM, Lambda, Bidirectional, AveragePooling1D
from keras import backend as K
from keras.optimizers import RMSprop
from keras.optimizers import Adam

#class MaxPooling(Layer):
#    def __init__(self, **kwargs):
#        self.supports_masking = True
#        super(MaxPooling, self).__init__(**kwargs)
#
#    def compute_mask(self, input, input_mask=None):
#        # do not pass the mask to the next layers
#        return None    
#
#    def call(self, x, mask=None):        
#        if mask is not None:
#          mask = tf.equal(mask, 0)
#          mask = tf.expand_dims(mask, -1)
#          mask = tf.tile(mask, [1, 1, [name]])
#            return masked_data.max(axis=1)
#        else:
#            return super().call(x)            
#    
#    def get_output_shape_for(self, input_shape):
#        return (input_shape[0], input_shape[2])
  
def cosine_distance(vects):
  import tensorflow as tf
  a, b = vects
  c=tf.sqrt(tf.reduce_sum(tf.multiply(a,a),axis=1))  
  d=tf.sqrt(tf.reduce_sum(tf.multiply(b,b),axis=1)) 
  e=tf.reduce_sum(tf.multiply(a, b),axis=1)
  f=tf.multiply(c,d)
  r=tf.div(e,f)
  return r

  
def cos_out_shape(shapes):
  import tensorflow as tf
  shape1, shape2 = shapes
  return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
  import tensorflow as tf
  margin = 0.4
  sim_loss = 0.25 * y_true * K.square(1 - K.square(y_pred))
  mask = tf.greater(y_pred, margin)
  default = tf.zeros_like(y_pred)
  error = tf.where(mask, x=y_pred, y=default)  
  dis_loss = (1 - y_true) * K.square(error)
  return K.mean(sim_loss + dis_loss)
     
def acc(y_true, y_pred):
  from keras import backend as K
  import tensorflow as tf
  thr = 0.5
  mask = tf.greater(y_pred, thr)
  pos = tf.ones_like(y_pred)
  neg = tf.zeros_like(y_pred)
  pred = tf.where(mask, x=pos, y=neg)
  return K.mean(tf.equal(pred, y_true))

def HierarchicalLSTM(opt):
  chars1 = Input(shape=(opt.max_sent_len, opt.max_word_len), dtype='int32', name='charinp1')
  chars2 = Input(shape=(opt.max_sent_len, opt.max_word_len), dtype='int32', name='charinp2')
 
  base_network = Sequential()
  base_network.add(TimeDistributed(Embedding(opt.char_vocab_size + 1,
                                             opt.char_vec_dim,
                                             mask_zero=True,
                                             input_length=opt.max_word_len,
                                             name='chars_embedding'), 
                                  input_shape=(opt.max_sent_len, opt.max_word_len)))
  base_network.add(TimeDistributed(LSTM(opt.char_rnn_size, activation='tanh', 
                                        inner_activation='sigmoid',
                                        return_sequences=False, stateful=False,
                                        name='char_lstm', 
                                        input_shape=(opt.max_word_len, opt.char_vec_dim),
                                        unroll=False),
                                  input_shape=(opt.max_sent_len, opt.max_word_len, opt.char_vec_dim)))

  base_network.add(LSTM(opt.word_rnn_size, activation='tanh', inner_activation='sigmoid',
                   return_sequences=False, stateful=False,
                   input_shape=(opt.max_sent_len, opt.char_rnn_size),
                   unroll=False,
                   name='word_lstm'))

  return chars1, chars2, base_network

def BiLSTM(opt):
  chars1 = Input(shape=(opt.max_sent_len, ), dtype='int32', name='charinp1')
  chars2 = Input(shape=(opt.max_sent_len, ), dtype='int32', name='charinp2')
   
  base_network = Sequential()
  base_network.add(Embedding(opt.char_vocab_size + 1,
                             opt.char_vec_dim,
                             mask_zero=True,
                             input_length=opt.max_sent_len,
                             name='chars_embedding'))
  layers = opt.num_layers
  while layers > 1:
    base_network.add(Bidirectional(LSTM(opt.char_rnn_size, activation='tanh',
                                        inner_activation='sigmoid',
                                        return_sequences=True),
                                  merge_mode='concat'))
    base_network.add(Dropout(opt.dropout))
    layers = layers - 1
    
  base_network.add(Bidirectional(LSTM(opt.char_rnn_size, activation='tanh',
                                      inner_activation='sigmoid',
                                      return_sequences=False),
                                merge_mode='concat'))
  
  # base_network.add(AveragePooling1D(pool_size=opt.max_sent_len))
  base_network.add(Dropout(opt.dropout))
  base_network.add(Reshape((opt.char_rnn_size * 2,), input_shape=(1, opt.char_rnn_size * 2)))
  base_network.add(Dense(opt.char_rnn_size * 2))
  return chars1, chars2, base_network

def Matching(opt):
  if (opt.archi == 'hi'):
    chars1, chars2, base_network = HierarchicalLSTM(opt)
  elif (opt.archi == 'bi'):
    chars1, chars2, base_network = BiLSTM(opt)
  
  emb1 = base_network(chars1)
  emb2 = base_network(chars2)
  
  # cont = Merge(mode='concat')([emb1, emb2])
  # drop = Dropout(opt.drop_rate)(cont)
  # dense = Dense(1, activation='sigmoid')(drop)
  
  cos_sim = Lambda(cosine_distance, output_shape=cos_out_shape)([emb1, emb2])
  
  model = Model([chars1, chars2], cos_sim)
  # op = RMSprop()
  op = Adam()
  model.compile(loss=contrastive_loss, optimizer=op, metrics=[acc])
  # model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy','binary_crossentropy'])
  return model
  
  