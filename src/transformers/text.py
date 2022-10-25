# Unable warnings
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Requirements
from nltk.stem.snowball import PortugueseStemmer
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from cleantext import clean
import nltk
import spacy
import re


class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords=True, wordlist=[], stemmer=False, lemma=False):
        self.stopwords = stopwords
        self.wordlist = wordlist
        self.stemmer = stemmer
        self.lemma = lemma

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Clean data
        X = X.apply(str).apply(lambda text: TextNormalizer.cleaner(text=text))
        # Remove stop words
        if self.stopwords:
            X = X.apply(str).apply(
                lambda text: TextNormalizer.remover(text=text, wordslist=self.wordlist)
            )
        # Lemmatizer
        if self.lemma:
            nlp = spacy.load("pt_core_news_sm")
            X = X.apply(str).apply(
                lambda x: " ".join([w.lemma_.lower() for w in nlp(x)])
            )
        # Stemmer
        if self.stemmer:
            X = X.apply(str).apply(PortugueseStemmer().stem)
        # Return
        return X.values.astype("U")

    @staticmethod
    def cleaner(text):
        text = re.sub(r"@[^\s]+", "nome_usuario", text)
        text = clean(
            text,
            fix_unicode=True,
            to_ascii=True,
            lower=True,
            no_emoji=True,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=False,
            no_punct=True,
            replace_with_punct="",
            replace_with_url="pagina_web",
            replace_with_email="email_usario",
            replace_with_phone_number="numero_telefone",
            replace_with_currency_symbol="simbolo_monetario",
        )
        return text

    @staticmethod
    def remover(text, wordslist):
        stopwords = nltk.corpus.stopwords.words("portuguese")
        stopwords.extend(wordslist)
        return " ".join([word for word in text.split() if word not in (stopwords)])
