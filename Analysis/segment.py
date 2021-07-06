from nltk.tokenize import sent_tokenize, RegexpTokenizer
import pandas as pd
from contractions import fix
from os.path import basename, splitext
import glob
from preprocess import clean

#Initialize tokenizer (for word segmentation only)
tk = RegexpTokenizer("[\w]+")

#Make dataframe containting short stories (Cols- Title, Story)
df = pd.DataFrame()
for file in glob.glob("Stories/*.txt"):
    with open(file) as f:
        title = f.readline()
        textf = " ".join(line.strip() for line in f)
    df = pd.concat([df,pd.DataFrame(data = {"Title" : title.strip(), "Story" : [textf]})])  

df.reset_index(drop=True, inplace=True)

#Fix contractions
df['Story'] = df['Story'].apply(lambda x: fix(x))
#Word-segmentation
df['Word'] = df['Story'].apply(lambda x: tk.tokenize(x))
#Sentence-segmentation
df['Sent'] = df['Story'].apply(lambda x: sent_tokenize(x))

#Saving as .csv and .json
df.to_csv('Segmented.csv',index = False)
df.to_json('Segmented.json')
