import csv
import os
import string
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from gensim import models, corpora
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

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
# decapitalize
df['speech'] = df['speech'].str.lower()
# remove punctuation
df['speech'] = df['speech'].str.replace('[{}]'.format(string.punctuation), '')

# perform lemmatization
lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]


df['speech'] = df['speech'].apply(lemmatize_text)
df['speech'] = df['speech'].apply(lambda x: [item for item in x if item not in stop_words])
df['speech'] = df['speech'].apply(lambda x: ' '.join(x))

# # perform stemming
# stemmer = PorterStemmer()
#
#
# def stem_text(text):
#     return [stemmer.stem(w) for w in word_tokenize(text)]
#
#
# df['speech'] = df['speech'].apply(stem_text)
# df['speech'] = df['speech'].apply(lambda x: [item for item in x if item not in stop_words])
# df['speech'] = df['speech'].apply(lambda x: ' '.join(x))
#
print(df['speech'].values[0])
#
# from gensim import corpora
#
# # create a dictionary from a list of speeches
# dictionary = corpora.Dictionary(df['speech'])
#
# # convert the dictionary to a bag of words
# corpus = [dictionary.doc2bow(speech) for speech in df['speech']]
# print(corpus[0])

NUM_TOPICS = 5
STOPWORDS = stopwords.words('english')

wnl = WordNetLemmatizer()


def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN': 'n', 'JJ': 'a',
                  'VB': 'v', 'RB': 'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'


def lemmatize_sent(text):
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag))
            for word, tag in pos_tag(word_tokenize(text))]


def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return lemmatize_sent(' '.join(cleaned_text))


# tokenize speeches and save them in a list
tokenized_speeches = [word_tokenize(speech) for speech in df['speech']]
# save tokenized speech to new column
df['tokens'] = tokenized_speeches
print(df['tokens'].values[0])
print(df['tokens'].values[1])
print(df['tokens'].values[2])
print(df['tokens'].values[3])

# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(df.tokens)
# dictionary.filter_extremes(no_below=3, no_above=.03)

# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in df.tokens]

# Build the LDA model
lda_model = models.LdaModel(corpus=corpus,
                            num_topics=20,
                            id2word=dictionary)

print()
print("LDA Model:")

for idx in range(20):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, lda_model.print_topic(idx, 20))


# build the LSI model
lsi_model = models.LsiModel(corpus=corpus, num_topics=20, id2word=dictionary)

print()
print("LSI Model:")
for idx in range(20):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, lsi_model.print_topic(idx, 20))

print()
print("first 20")
print(lsi_model.print_topic(1,20))
