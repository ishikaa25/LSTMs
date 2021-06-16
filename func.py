import numpy as np
import re


def clean(df,nlp,col='Story',get_features=True):
    '''
    Remove punctations and stopwords, and tokenize


    Args:
    df : Pandas dataframe of dataset
    nlp : Spacy model
    col : Column name that has the docs
    '''
    # spacy_stopwords = nlp.Defaults.stop_words  
    #Sentence
    df['Sentences']  = df[col].apply(lambda x: [sent.text for sent in nlp(x).sents])
    #Stopwords
    df['Cleaned']  = df[col].apply(lambda row: " ".join(token.text for token in nlp(row) if not token.is_stop))
    #Punctuations  
    df['Cleaned']  = df['Cleaned'].apply(lambda x: " ".join(re.sub(r'[^a-zA-z\s]', '', str(token.text)) for token in nlp(x)))
    #Whitespaces
    df['Cleaned']  = df['Cleaned'].apply(lambda row: re.sub(' +', ' ',row))
    #Lemma
    df['Cleaned'] = df["Cleaned"].apply(lambda row: " ".join([w.lemma_ for w in nlp(row)]))
    
    df['Tokenized'] = [nlp(text) for text in df.Cleaned]
    # df["Tokenized"] = [nlp(w).lemma_ for w in df['Tokenized']]

    if get_features:
        df['mean_sen_length']  = df['Sentences'].apply(lambda x: np.mean([len(i) for i in x]))
        df['num_tokens'] = [len(token) for token in df.Tokenized]

    # df.drop('Cleaned',axis=1,inplace=True)

    return None