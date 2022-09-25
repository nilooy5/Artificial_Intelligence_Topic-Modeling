import csv
import os
import string
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import re
from gensim import models, corpora
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import numpy as np

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

csv.field_size_limit(1000000000)

# read csv file from data folder
df = pd.read_csv(os.path.join('data', 'state-of-the-union.csv'), names=['year', 'speech'], skiprows=1)


def perform_initial_cleanup(data_frame):
    data_frame['speech'] = data_frame['speech'].str.replace('\nState of the Union Address\n', '')
    data_frame['speech'] = data_frame['speech'].str.replace('\nAddress to Joint Session of Congress \n', '')
    data_frame['speech'] = data_frame['speech'].str.replace('\nAddress on Administration Goals (Budget Message)\n', '')
    data_frame['speech'] = data_frame['speech'].str.replace('\nAddress on Administration Goals\n', '')
    data_frame['speech'] = data_frame['speech'].str.replace('\nAddress to Congress \n', '')
    data_frame['president'] = data_frame['speech']
    data_frame['president'] = data_frame['president'].str.split('\n').str[0]
    data_frame['date'] = data_frame['speech'].str.split('\n').str[1]
    temp_date = \
    data_frame[data_frame['date'] == 'Address on Administration Goals (Budget Message)']['speech'].str.split('\n').str[
        3]
    data_frame['date'][data_frame['date'] == 'Address on Administration Goals (Budget Message)'] = temp_date.values[0]
    # delete first 3 lines of speech
    data_frame['speech'] = data_frame['speech'].str.split('\n').str[3:]
    # make a string list
    data_frame['speech'] = data_frame['speech'].str.join(' ')
    # replace \ with ''
    data_frame['speech'] = data_frame['speech'].str.replace('\\\'', '')
    # decapitalize
    data_frame['speech'] = data_frame['speech'].str.lower()
    # remove punctuation
    data_frame['speech'] = data_frame['speech'].str.replace('[{}]'.format(string.punctuation), '')

    return data_frame


df = perform_initial_cleanup(df)

# perform initial lemmatization
lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]


df['speech'] = df['speech'].apply(lemmatize_text)
df['speech'] = df['speech'].apply(lambda x: [item for item in x if item not in stop_words])
df['speech'] = df['speech'].apply(lambda x: ' '.join(x))

NUM_TOPICS = 5
STOPWORDS = stopwords.words('english')
STOPWORDS.extend(['America', 'Americas', 'American', 'Americans', 'america', 'americas', 'american', 'americans'])

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


def clean_lemma_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return lemmatize_sent(' '.join(cleaned_text))


# clean the speeches
tokenized_speeches = df['speech'].apply(clean_lemma_text)

df['tokens'] = tokenized_speeches

# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(df.tokens)
# dictionary.filter_extremes(no_below=3, no_above=.03)

# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in df.tokens]


# # Build the LDA model
# lda_model = models.LdaModel(corpus=corpus,
#                             num_topics=10,
#                             id2word=dictionary)
#
# print()
# print("LDA Model:")
#
# for idx in range(10):
#     # Print the first 10 most representative topics
#     print("Topic #%s:" % idx, lda_model.print_topic(idx, 10))
#
# # build the LSI model
# lsi_model = models.LsiModel(corpus=corpus,
#                             num_topics=10,
#                             id2word=dictionary)
#
# print()
# print("LSI Model:")
# for idx in range(10):
#     # Print the first 10 most representative topics
#     print("Topic #%s:" % idx, lsi_model.print_topic(idx, 10))
#
#
# df['lda_topic'] = df['tokens']
# # def get_most_popular_topic(index)
# for j in np.arange(0, 225):
#     df['lda_topic'][j] = [i[0] for i in
#                           lda_model.get_document_topics(dictionary.doc2bow(df.tokens[j]), minimum_probability=0.2)]
#
# print(df.head(20))


def sotu_topic_finder(year):
    """
    Find SOTU topics using LDA. The LDA model is only trained on the text of that year topic
    Input: index i of the speech
    Output: list 5 topics found by the model
    """
    # Clean the text
    sent_text = sent_tokenize(df.speech[year - 1790])
    token_list = []
    for sent in sent_text:
        cleaned_sent = clean_lemma_text(sent)
        token_list.append(cleaned_sent)

    # Prepare the dictionary and corpus
    dictionary = corpora.Dictionary(token_list)
    corpus = [dictionary.doc2bow(text) for text in token_list]

    # Build the LDA model
    lda_model = models.LdaModel(corpus=corpus,
                                num_topics=10,
                                id2word=dictionary)

    # Output model
    print("LDA Model of %i:" % year)
    for idx in range(5):
        # Print the first 10 most representative topics
        print("Topic #%s:" % idx, lda_model.print_topic(idx, 10))

    # build the LSI model
    lsi_model = models.LsiModel(corpus=corpus,
                                num_topics=10,
                                id2word=dictionary)

    print()
    print("LSI Model of %i:" % year)
    for idx in range(5):
        # Print the first 10 most representative topics
        print("Topic #%s:" % idx, lsi_model.print_topic(idx, 10))

    return lda_model, lsi_model


yearly_models = []
for year in range(1990, 2005):
    yearly_models.append(sotu_topic_finder(year))

topics_1990 = yearly_models[0][0]


# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors


def visualize_topics(yearly_model):
    global cols, i
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)
    topics = yearly_model.show_topics(formatted=False)
    fig, axes = plt.subplots(1, 4, figsize=(10, 10), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()


for ldaM in yearly_models:
    visualize_topics(ldaM[0])

