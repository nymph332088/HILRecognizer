#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:01:22 2017

@author: nymph332088
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import random
from math import floor
from pymongo import MongoClient
client = MongoClient()
db = client['linkedin']

from datautil import  major_rewrites
class PersonDoc(object):
  def __init__(self):
    self.docnum = 0
  
  def procList(self, majorList, prefix):
    positList = [x.encode('ascii', 'ignore').lower() for x in majorList]
    positList = [x.translate(string.maketrans(string.punctuation," "*len(string.punctuation)))\
            for x in positList]
    positList = ['_'.join(str(x).split()) for x in positList]
    positList = [prefix + x for x in positList]
    return positList

  def __iter__(self):
    """
    1. train with 10 million people from 300 universities
    """
    f = open('data/seed_0_usrs', 'rb')        
    for aid in f:
      aid = aid.strip()
      doc = db.profiles.find_one({'_id': aid}, {'exp_size':1, 'edu_size':1,\
          'experiences': 1, 'educations':1})
      if doc == None: continue
      if ((doc['edu_size'] + doc['exp_size']) <= 1) : 
        continue
      majorList = []
      for edu in doc['educations']:
        _,_,rewrite = major_rewrites.major_rewrite(edu['Major'].lower(), major_rewrites.majorBigram)
        majorList.extend(rewrite)
      # majorList = self.procList(majorList, 'm_')
      majorList = ['m_' + '_'.join(major.split()) for major in majorList]

      posList = []
      for exp in doc['experiences']:
        posList.append(exp['Position'])
      posList = self.procList(posList, 'p_')
      self.docnum += 1
      if(self.docnum % 1000000 == 0):
        print(self.docnum)
      yield doc['_id'], majorList + posList