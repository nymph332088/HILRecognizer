# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:27:26 2015

@author: tuf14438
"""
import pandas as pd
import re
import string
majorBigram = pd.read_pickle('data/majorBigram.pkl')
def major_rewrite(major, majorBigram):
  punct = ''.join(string.punctuation.split('\''))
  punct = ''.join(punct.split('.'))
  # punct = ''.join(punct.split('-'))
  r = re.compile(r'[{}]+'.format(re.escape(punct)))    
  mjPhrases = r.split(major)
  # mjPhrases = re.split("[" + punct + "]+", major)
  mjPhrases = [mj.strip() for mj in mjPhrases if(not mj.strip() in ['general', ''])]
  mjPhrasesNew = joinToken(mjPhrases, ' ', majorBigram)
  
  refined = []
  r = re.compile(r'\b({0})\b'.format('and'), flags=re.IGNORECASE)    

  for rw in mjPhrasesNew:
    if(bool(r.search(rw))):
      temp = r.split(rw)
      temp = [t.strip() for t in temp]
      temp = [t for t in temp if (not t in ['','and'])]
      refined.extend(joinToken(temp, ' and ', majorBigram))
    else:
      refined.append(rw)

  final = []
  gabage = ['general','major', 'in','with', 'w','a',\
  'minor','specialization','specializing','concentration',\
  'concentrating','dual program','dual degree','degree','emphasis','on']
  remove = '|'.join(gabage)
  r = re.compile(r'\b({0})\b'.format(remove), flags=re.IGNORECASE)
  for rw in refined:
    final.extend(r.split(rw))

  final = [rw.strip() for rw in final]
  final = [rw for rw in final if((not rw in gabage) and (rw != ''))]
  
  return mjPhrasesNew, refined, final

def joinToken(mjPhrases, joint, majorBigram):
  joints = []
  for i in range(len(mjPhrases) - 1):
    lastPre = mjPhrases[i].split()[-1]
    firstLat = mjPhrases[i+1].split()[0]
    if((firstLat == 'and') or (lastPre == 'and')):
        joints.append(i)
        continue
    bg = ' '.join([lastPre, firstLat])
    if( bg in majorBigram.index):
        joints.append(i)
  
  groups = []
  if(len(joints) == 0):
      for i in range(len(mjPhrases)):
          groups.append([i])
  else:
    from operator import itemgetter
    from itertools import groupby
    import itertools
    for k, g in groupby(enumerate(joints), lambda (i,x):i-x):
      groups.append(map(itemgetter(1), g))
    
    [g.append(g[-1] + 1) for g in groups]
    temp = list(itertools.chain.from_iterable(groups))
  
    for i in range(len(mjPhrases)):
      if(not i in temp):
        pos = 0
        while((groups[pos][0]<i)):
          pos += 1
          if(pos == len(groups)):
            break
        if(pos == len(groups)):
          groups.append([i])
        else:
          groups.insert(pos, [i])
  mjPhrasesNew = []
  for g in groups:
    mjPhrasesNew.append(joint.join(mjPhrases[g[0]:(g[-1]+1)]))
  
  return mjPhrasesNew