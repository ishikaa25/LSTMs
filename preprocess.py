import re
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from collections import Counter



def clean(data):
  '''
  Pre-processes text data
  1. Lowercasing
  2. Removal of HTML parts
  3. Removal of Non-ASCII and Digits characters
  4. Removal of white and multiple spacing
  5. Removal of stop-words

  Args:
  data (numpy.ndarray) : Array containing (a list of) textual data 
  '''

  stop = stopwords.words()
  #To lowercase
  text = data.lower()

  #Remove HTML stuff
  text= re.sub("(<.*?>)","",text) 
  #remove non-ascii and digits
  text= re.sub("(\\W|\\d)"," ",text) 
    
  #Remove whitespace
  text= text.strip()
  #Remove multiple spaces
  text = re.sub(' +', ' ',text)

  #Remove stop words
  text = [word for word in text if word not in stop]

  return text

def count_tokenize(X,min_occur=100):
    '''
    Tokenize text data by count. 

    Args:
    X (numpy.ndarray) : Array containing (a list of) textual data.
    min_occur (int)   : Min. occurences of a word to filter vocabulary / Threshold
    '''

    words = Counter() 
    for i, sentence in enumerate(X):
    # Initialize example with empty list
        X[i] = []
        for word in nltk.word_tokenize(sentence):  # Tokenizing the words
            words.update([word])
            X[i].append(word)
        if i%5000 == 0:
            print("{}".format(int(i/len(X)*100)) + "%"+" done")

    print("100%"+"done")

    #Creates dictionary for words based on count with a threshold of min_occur
    words = {k:v for k,v in words.items() if v>min_occur}
    words = sorted(words,key=words.get,reverse=True)

    #Adding unknown and padding words to vocab
    words = ['_PAD','_UNK'] + words

    word2idx = {o:i for i,o in enumerate(words)}    #Maps word --> number
    idx2word = {i:o for i,o in enumerate(words)}    #Maps number --> word

    return word2idx,idx2word



def vectorize(X,word2idx,vec_length=200):
    '''
    Vectorize text data using a vocabulary (word-->number dictionary). 

    Args:
    X (numpy.ndarray) : Array containing (a list of) textual data.
    word2idx (dict)   : Ref. vocabulary to map word --> number.
    vec_length (int)  : Final/output length of sequence of each example
    '''

    for idx, example in enumerate(X):
        X[idx] = [word2idx[word] if word in word2idx else 0 for word in example]

    
    X_new = np.zeros((len(X),vec_length))

    for idx,example in enumerate(X):
        if len(example)!=0:
            X_new[idx,-len(example):] = np.array(example)[:vec_length]

    return X_new


def vect(X,word2idx,vec_length=200):
    '''
    Vectorize text data using a vocabulary (word-->number dictionary). 

    Args:
    X (numpy.ndarray) : Array containing (a list of) textual data.
    word2idx (dict)   : Ref. vocabulary to map word --> number.
    vec_length (int)  : Final/output length of sequence of each example
    '''

    for idx, example in enumerate(X):
        X[idx] = [word2idx[word] if word in word2idx else 0 for word in example]

    
    X_new = np.zeros((len(X),vec_length))

    for idx,example in enumerate(X):
        if len(example)!=0:
            X_new[idx,-len(example):] = np.array(example)[:vec_length]

    return X_new
