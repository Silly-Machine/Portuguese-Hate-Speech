from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from collections import Counter
import pandas as pd
import spacy


def ColumnsOrder(df):
    return df[sorted(df.columns, reverse=True)]


def TextStats(df, column='text'):
    Stats = pd.DataFrame()
    Stats[f'length_{column}'] = df[column].apply(lambda x: len(x))
    Stats[f'count_word_{column}'] = df[column].apply(lambda x: len(set(x)))
    return Stats


def WordDist(df, column='text'):
    bag_words = ' '.join([word for word in df[column]])
    freq_dist = FreqDist(word_tokenize(bag_words, language='portuguese'))
    words_dist = pd.DataFrame(freq_dist.items(), columns=['word', 'frequency'])
    words_dist['density'] = words_dist['frequency']/len(bag_words)
    words_dist.sort_values(by=['frequency'], ascending=False, inplace=True)
    words_dist.reset_index(drop=True, inplace=True)
    return words_dist


def NgramsCount(corpus, nitems=2, rank=-1):
    nrange = (nitems, nitems)
    vectorizer = CountVectorizer(ngram_range=nrange).fit(corpus)
    ngrmas_bag = vectorizer.transform(corpus)
    words_freq = [(word, ngrmas_bag.sum(axis=0)[0, idx])
                  for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(words_freq[:rank], columns=[f'{str(nitems)}_gram', 'frequency'])


def ItemsCounter(text):
    nlp = spacy.load('pt_core_news_sm')
    item_count = [token.pos_ for token in nlp(text)]
    return Counter(item_count)


def TextualItems(df, column='text_nonstop'):
    items = df[column].apply(ItemsCounter).apply(pd.Series)
    items.columns = map(str.lower, items.columns)
    items.drop(columns=['x'], inplace=True)
    items.fillna(0, inplace=True)
    return items.astype(int)
