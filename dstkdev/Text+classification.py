
# coding: utf-8

# In[22]:


import numpy as np
from cytoolz.curried import get, groupby, valmap
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegressionCV
from nltk.stem.snowball import GermanStemmer
from collections import ChainMap


# In[2]:


docs = ["abc def\nabc def",
        "def\nghi",
       ]*20

labels = [0, 1]*20


# In[8]:


class NoFit:
    def fit(self, X, y=None):
        return self

    
class Doc2List(BaseEstimator, TransformerMixin, NoFit):
    def transform(self, docs):
        return [[line.split() for line in doc.split("\n")] for doc in docs]

    
class List2doc(BaseEstimator, TransformerMixin, NoFit):   
    def transform(self, doclists):
        return ["\n".join(" ".join(line) for line in doclist) for doclist in doclists] 
    
    
class DocTrans(BaseEstimator, TransformerMixin, NoFit):
    def __init__(self, doc_func):
        self.doc_func = doc_func
        
    def transform(self, docs):
        return [self.doc_func(doc) for doc in docs]
   

class DocFunc(BaseEstimator, TransformerMixin, NoFit):
    def __init__(self, doc_func):
        self.doc_func = doc_func
        
    def transform(self, docs):
        return [[self.doc_func(doc)] for doc in docs]
  

class CleanDoc(BaseEstimator, TransformerMixin, NoFit):
    def __init__(self):
        self.stemmer = GermanStemmer()
        
    def transform(self, docs):
        res = []
        for doc in docs:
            lines = doc.split("\n")
            lines = [" ".join(self.stemmer.stem(word)
                              for word in re.findall("[a-zäöüß]{3,}", line.lower()))
                     for line in lines]
            res.append("\n".join(lines))
            
        return res


# In[3]:


def subwords(word):
    return [word[:2], word[2:]]


# In[27]:


stem = GermanStemmer().stem

cnt_vect_splits = [("short", lambda doc: [line for line in doc if len(line)<=1], {}),
                   ("long", lambda doc: [line for line in doc if len(line)>1], {}),
                   ("subwords", lambda doc: [list(map(stem, concat(subwords(word) for word in line))) for line in doc], {"ngram_range":(1,1)}),
                  ]

doc_funcs = [("num_char", lambda doc: len(re.findall("[A-Za-zäöüÄÖÜß]", doc))),
            ]

cnt_vect_default_params = {"min_df": 3,
                           "strip_accents": "unicode",
                           "ngram_range": (1,2),
                           #"stop_words": [...],
                          }


# In[28]:


tfidf_pipe = Pipeline([("doc2list", Doc2List()),
                       ("cnt_vects", FeatureUnion([("cnt_vect_pipe_{}".format(filter_name),
                                                    Pipeline([("filter_{}".format(filter_name), DocTrans(filter_func)),
                                                              ("list2doc", List2doc()),
                                                              ("cnt_vect_{}".format(filter_name), CountVectorizer(**ChainMap(filter_cnt_vect_param,
                                                                                                                             cnt_vect_default_params))),
                                                             ])
                                                   ) for filter_name, filter_func, filter_cnt_vect_param in cnt_vect_splits
                                                  ])),
                       ("tfidf", TfidfTransformer(sublinear_tf=True)),
                      ])

clf = Pipeline([("clean", CleanDoc()),
                ("tfidf_and_meta", FeatureUnion([("tfidf_pipe", tfidf_pipe)]+ # needs to be first, or below vocabulary_map miscounts
                                                [(name, DocFunc(func)) for name, func in doc_funcs])),
                ("clf", LogisticRegressionCV()),
                ])


# In[29]:


clf.fit(docs, labels)
clf.predict(docs)


# In[30]:


def get_step_by_name(pipe, name):
    return [trans for name_, trans in pipe.steps if name_.startswith(name)][0]


# In[31]:


cnt_vects_pipe = get_step_by_name(tfidf_pipe, "cnt_vects")

cnt_vects = [get_step_by_name(pipe, "cnt_vect_") for _name, pipe in cnt_vects_pipe.transformer_list]

vocabulary_map = pipe(enumerate(concat(cnt_vect.vocabulary_ for cnt_vect in cnt_vects)),
                      groupby(get(1)),
                      valmap(lambda vals:list(pluck(0, vals))),
                     )
vocabulary_map


# In[ ]:




