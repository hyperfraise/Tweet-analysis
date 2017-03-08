# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:07:25 2017

@author: Damien Menigaux

Voici le code qui a servi a labeliser le dataset de 12 000 tweets fourni.

Ce code est écrit en python 2. Cela posera problème pour un utilisateur de
python 3, en raison de la gestion des strings, qui va faire échouer la ligne
definissant l'objet "normal" (ligne censee remplacer tous les accents et caracteres
speciaux problematiques en python 2, elle va uniquement les supprimer en python 3)

Le package keras est utilise avec le backend theano, et non tensorflow, qui
gere mal le merging de modeles en keras. Il vaut mieux runner le NN avec 
le GPU, et non le CPU.

Le dataset de tweets scrapes par nos soins contient 150 000 tweets distincts.
"""




import functions
from functions import *

#voici les tweets scrapes en 2017
#index = tweetId, deux colonnes : le texte brut, et les labels
tocon=pd.read_csv('tweets_labeled.csv',sep='~',index_col=0)

#nous allons analyser les textes legerement corriges
normal = [clean(unicodedata.normalize('NFKD',x.decode('utf-8')).encode('ASCII', 'ignore')) for x in list(tocon['text'])]
tocon['cleaned']=normal

#On prend tout le vocabulaire qui nous est donné
tolabel=pd.read_csv('set_non_labelised_12k.csv',index_col=0,sep='~')[['text']]
normaltolabel = [clean(unicodedata.normalize('NFKD',x.decode('utf-8')).encode('ASCII', 'ignore')) for x in list(tolabel['text'])]
normal+=normaltolabel


#On tokenize les phrases vues, puis on les encode
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(tocon['cleaned'])
sequences = tokenizer.texts_to_sequences(tocon['cleaned'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
max_lent=np.max(map(len,sequences))
data = pad_sequences(sequences)

#Avant de donner ces phrases au modele, on va preparer une matrice d'embedding
#pour les mots du corpus
from gensim.models import word2vec
import logging

s='\n'.join(tocon['cleaned'])
f = open('tweets.txt','w')
f.write(s)
f.close()
sentences = word2vec.LineSentence('tweets.txt')

#le modele suivant apprend un representation en faible dimension des mot du corpus
embedding_dim=100
#cette etape prend quelques dizaines de secondes, parallelisable avec l'attribut
#"workers"
wordembed = word2vec.Word2Vec(sentences,size=embedding_dim,workers=1,min_count=-1)

wordembed.most_similar(['gare'])

embedding_matrix = np.zeros((len(word_index)+1,embedding_dim))
und=0
for word, i in word_index.items():
    embedding_vector = wordembed[word]
    embedding_matrix[i] = embedding_vector



#on prepare les datasets croises
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = to_categorical(np.asarray(tocon['Label']))[indices]
texts=np.array(tocon['text'])[indices]
indexes=tocon.index[indices]
nb_validation_samples = int(0.1 * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


#Construisons le modele
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers.recurrent import LSTM
from keras.layers.noise import GaussianDropout
from keras.layers import Input, Embedding, LSTM, Dense, merge, GRU, SimpleRNN
from keras.regularizers import l1,l2
from keras.layers.convolutional import Cropping1D

main_input = Input(shape=(x_train.shape[1],), dtype='int32', name='main_input')
g = Embedding(len(word_index)+1,embedding_dim,weights=[embedding_matrix])(main_input)

#Plusieurs couches de convolution pour analyser les motifs de tailles differentes
#dans les phrases interessantes
ms=[]
for i in range(1,3,2):
    m=Convolution1D(30,i,border_mode='valid',activation='relu',init='normal')(g)
    m=LSTM(30,W_regularizer=l2(0.1),return_sequences=True)(m)
    m=Flatten()(m)
    ms.append(m)

m=Dense(50,W_regularizer=l2(0.1),activation='relu')(m)
m=Dropout(0.5)(m)
main_output = Dense(3, activation='softmax', name='main_output')(m)
model = Model(input=main_input,output=main_output)
model.compile(optimizer='adam',loss=["categorical_crossentropy"],metrics=['fmeasure'])
class_weight = {0 : 1.,
    1: 500.,
    2: 200.}

from keras.utils.visualize_util import plot
plot(model,show_shapes=True,to_file='images/single_model_shapes.png')
plot(model,to_file='images/single_model.png')

#On donne plus d'importance aux tweets de 2013
sw=np.ones(x_train.shape[0])
sw[indexes[:-nb_validation_samples]<15000]*=7
beta=0.5

#On fitte le modele. Sans carte graphique, il prendra certainement
#au moins 10h de training. Avec : 10m/20min
g=model.fit([x_train],[y_train],class_weight=class_weight,nb_epoch=20,batch_size=2000,sample_weight=sw)


#on choisit les thresholds sur le dataset de validation 
pred_val=model.predict([x_val])
fi=np.where(indexes[-nb_validation_samples:]<20000)[0]

predfi=pred_val[fi]
yfi=y_val[fi]
np.sum(yfi,axis=0)
score,thrs=fs(yfi,predfi)


#On labelise le dataset de test
normal = [clean(unicodedata.normalize('NFKD',x.decode('utf-8')).encode('ASCII', 'ignore')) for x in list(tolabel['text'])]
tolabel['cleaned']=normal
sequences_lab = tokenizer.texts_to_sequences(tolabel['cleaned'])
word_index_lab = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index_lab))
indices_lab=tolabel.index
data_lab = pad_sequences(sequences_lab, maxlen=max_lent)

pred_lab=model.predict([data_lab])
pred=pred(thrs,pred_lab)
tolabel['Label']=[x[0][0] for x in map(np.where,pred)]


tolabel[['text','Label']].to_csv('set_12000_labeled.csv')
