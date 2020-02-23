import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from matplotlib import pyplot as plt
import networkx as nx
import  collections
import itertools

import nltk.sentiment.util as ut

TH_cJac=0.4

#  --- read file---
path = './sample_text/sample.txt'
with open(path, 'r', encoding='UTF-8') as f:
    s = f.read()
# f = open(path)
# f.close()
# print(sent_tokenize(s))
stop_words=set(stopwords.words("english"))
lem = WordNetLemmatizer()
sent = sent_tokenize(s)


# TFのカウント
# def TF_count (list term_list):
    
#     return (coll_dict)
# temp=[]
# for i in bigrms_list:
    

def sent_preprocess( in_sentense ):
    words = word_tokenize(in_sentense)
    filtered_word=[]
    for w in words:
        if w not in stop_words:
            temp_word=lem.lemmatize(w, pos='v')
            filtered_word.append(temp_word)
        
    temp_words2=nltk.pos_tag(filtered_word)
    filtered_word.clear()
    word_list=[]
    for j in temp_words2:
        key=j[1]
        if ('NN' in key) or ('JJ' in key) or ('VB' in key):
            word_list.append(j[0])
    
    return word_list


# ---ngrm list generator---
word_list_all=[]
bigrms_list=[]
trigrms_list=[]
filtered_sent=[]
for i in sent:

    word_list = sent_preprocess(i)

    bigrm=nltk.bigrams(word_list)
    tribigrm=nltk.trigrams(word_list)
    bigrms_list.extend(list(bigrm))
    trigrms_list.extend(list(tribigrm))
    word_list_all.extend(word_list)
    filtered_sent.append(word_list)
    # word_list.clear()

collection1= collections.Counter(word_list_all)
word_keys= list(collection1.keys())
word_comb=list(itertools.combinations(word_keys,2))
counter = 0
dist_freq_2word={}
for i in word_comb:
    for s in filtered_sent:
        if (i[0] in s) and (i[1] in s):
            counter = counter+1
    if counter >=1:
        dist_freq_2word[i]=counter
    counter=0

def calc_cJac(comb_df, tf):
    cJac={}
    for i in comb_df.keys():
        # print(i[0])
        a=tf[i[0]]
        b=tf[i[1]]
        x=comb_df[i]
        cJac[i]=x/(a+b-x)
    return cJac

cJac=calc_cJac(dist_freq_2word, collection1)
plt.hist(cJac.values())
cJac_list=[]
for i in cJac.keys():
    if cJac[i] > TH_cJac:
        cJac_list.append([i[0],i[1],cJac[i]])


G=nx.Graph()
G.add_weighted_edges_from(cJac_list)
# G.add_edges_from(bigrms_list)
nx.draw_networkx(G)
plt.show()

nx.write_gexf(G, "test.gexf")

# testet

print('finish!')

#