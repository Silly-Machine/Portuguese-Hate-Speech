import re
import nltk
from cleantext import clean

import warnings
warnings.simplefilter("ignore")


def StandardNormalizer(text):
    text = text.lower()
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"#[^\s]+", "", text)
    text = re.sub(r"https?://[^\s]+", "", text)
    text = " ".join(text.split())
    return text


def Normalizer(text):

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
    #text = ''.join([i for i in text if not i.isdigit()])
    return text


def WordRemover(text, additional_words=['nomeusuario', 'paginaweb', 'emailusario', 'numerotelefone', 'simbolomonetario', 'rt']):
    stopwords = nltk.corpus.stopwords.words("portuguese")
    stopwords.extend(additional_words)
    return ' '.join([word for word in text.split() if word not in (stopwords)])
