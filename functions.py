# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 11:54:04 2017

@author: damien
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import seaborn as sns
from sklearn.metrics import precision_recall_curve as prc
from sklearn.metrics import fbeta_score as fb
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as prec
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
import re
p=re.compile(r'\<http.+?\>', re.DOTALL)
import unicodedata
from string import punctuation
punctuation=punctuation.replace("#", "")

#fonction pour choisir le threshold optimal au sens de la Fbeta
def optThr(y_true,y_pred,beta):
    precision,recall,thresholds=prc(y_true,y_pred)
    F=(+beta**2)*precision*recall/((beta**2)*precision+recall)
    tosup=np.isnan(F)
    F=F[~tosup]
    thresholds=thresholds[~tosup[1:]]
    return thresholds[np.argmax(F)-1]

#fonction pour predire des labels a partir de probabilites
def pred(thrs,preds):
    beta=0.5
    preds_ret=np.zeros(preds.shape)
    preds_ret[:,1]=np.array(map(int,preds[:,1]>thrs[0]))
    preds_ret[:,0]=map(int,0==np.array(map(int,1-preds[:,0]>thrs[1]))+preds_ret[:,1])
    preds_ret[:,2]=map(int,1-preds_ret[:,0]-preds_ret[:,1])
    return preds_ret

#fonction pour cacluler le score selon la metrique demandee
def fs(y_true,y_pred):
    beta=0.5
    thrs=[]
    thrs.append(optThr(y_true[:,1],y_pred[:,1],beta))
    thrs.append(optThr(1-y_true[:,0],1-y_pred[:,0],beta))
    score=scorer(thrs,y_true,y_pred,beta)
    return score,thrs

def scorer(thrs,y_true,y_pred,beta):
    y_preds=pred(thrs,y_pred)
    fbs=[]
    fbs.append(fb(y_true[:,1],y_preds[:,1],beta))
    fbs.append(fb(1-y_true[:,0],1-y_preds[:,0],beta))
    score=np.mean(fbs)
    return score
    

#fonction pour traiter les tweets, supprime la ponctuation, transforme les liens, nombres...
def clean(word):
    word=re.sub(r'https?:\/\/.*[\r\n]*', 'lien', word, flags=re.MULTILINE)
    word=re.sub(r'http?:\/\/.*[\r\n]*', 'lien', word, flags=re.MULTILINE)
    re.sub(r'([a-z])\1+', r'\1\1', word)
    word=re.sub(r'[0-9]+h[0-9]*', 'heure', word, flags=re.MULTILINE)
    word=re.sub(r'[0-9]+km/h', 'vitesse', word, flags=re.MULTILINE)
    word=re.sub(r'[0-9]+km', 'distance', word, flags=re.MULTILINE)
    word=re.sub(r'[0-9][0-9]+', 'nombre', word, flags=re.MULTILINE)
    word=re.sub(r'\n','', word,flags=re.MULTILINE)
    word=re.sub(r'\r','', word,flags=re.MULTILINE)
    for p in list(punctuation):
        word=word.replace(p,' ')
    return word.lower()











