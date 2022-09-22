import csv
import os
import string
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

csv.field_size_limit(1000000000)

# read csv file from data folder
df = pd.read_csv(os.path.join('data', 'state-of-the-union.csv'), names=['year', 'speech'], skiprows=1)

df['speech'] = df['speech'].str.replace('\nState of the Union Address\n', '')
df['speech'] = df['speech'].str.replace('\nAddress to Joint Session of Congress \n', '')
df['speech'] = df['speech'].str.replace('\nAddress on Administration Goals (Budget Message)\n', '')
df['speech'] = df['speech'].str.replace('\nAddress on Administration Goals\n', '')
df['speech'] = df['speech'].str.replace('\nAddress to Congress \n', '')

df['president'] = df['speech']

df['president'] = df['president'].str.split('\n').str[0]
df['date'] = df['speech'].str.split('\n').str[1]

temp_date = df[df['date'] == 'Address on Administration Goals (Budget Message)']['speech'].str.split('\n').str[3]
df['date'][df['date'] == 'Address on Administration Goals (Budget Message)'] = temp_date.values[0]

# delete first 3 lines of speech
df['speech'] = df['speech'].str.split('\n').str[3:]
# make a string list
df['speech'] = df['speech'].str.join(' ')
# replace \ with ''
df['speech'] = df['speech'].str.replace('\\\'', '')

# perform lemmatization
lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]


df['speech'] = df['speech'].apply(lemmatize_text)
df['speech'] = df['speech'].apply(lambda x: [item for item in x if item not in stop_words])
df['speech'] = df['speech'].apply(lambda x: ' '.join(x))

# perform stemming
stemmer = PorterStemmer()


def stem_text(text):
    return [stemmer.stem(w) for w in word_tokenize(text)]


df['speech'] = df['speech'].apply(stem_text)
df['speech'] = df['speech'].apply(lambda x: [item for item in x if item not in stop_words])
df['speech'] = df['speech'].apply(lambda x: ' '.join(x))

# remove punctuation

df['speech'] = df['speech'].str.replace('[{}]'.format(string.punctuation), '')

print(df['speech'].head(10))
