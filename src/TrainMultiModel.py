# ETL
import numpy as np
import pandas as pd

# ML preprocessing
from sklearn.model_selection import train_test_split

# Embedding
from TextVectorization import MeanEmbeddingVectorizer
from gensim import models


# Pipe
from sklearn.pipeline import Pipeline
# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Metrics
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('Hate Speech')


# Set and split train and test data
df = pd.read_csv("data/corpus/augmented_corpus_fortuna.csv")
# Set target and features
target = "label"
features = "text_nonstop"

# Break apart dataset
X = df[features].values.astype("U")
y = df[target]

# Split train abd test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Class weights
pos = len(df.query('label==1'))
neg = len(df.query('label==0'))
weight_for_0 = (1 / neg) * (len(df) / 2.0)
weight_for_1 = (1 / pos) * (len(df) / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

classifiers = {'GradientBoosting': GradientBoostingClassifier(),
               'KNeighbors': KNeighborsClassifier(),
               'Bernoulli': BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None),
               'SVC': LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='crammer_singer', fit_intercept=True, intercept_scaling=1, class_weight=class_weight, verbose=0, random_state=42, max_iter=1000),
               'LogisticRegression': LogisticRegression(penalty='l2', max_iter=200, C=1),
               'SGDC': SGDClassifier(loss='hinge', max_iter=200),
               'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight=class_weight),
               'RandomForest': RandomForestClassifier(random_state=42, class_weight=class_weight),
               'SVM': svm.SVC(kernel='rbf')}


embedding = {"skip_50": "data/pretrained-skipgram/skip_s50.txt",
             "skip_100": "data/pretrained-skipgram/skip_s100.txt",
             "skip_300": "data/pretrained-skipgram/skip_s300.txt",
             "skip_1000": "data/pretrained-skipgram/skip_s1000.txt"}

for embedding_name, embedding_path in embedding.items():

    # # Load a pre-trained model
    print(f"loading {embedding_name} ...")

    pretrained_model = models.KeyedVectors.load_word2vec_format(
        embedding_path, binary=False)

    pretrained_w2v = dict(
        zip(pretrained_model.index_to_key, pretrained_model.vectors))

    for model_name, classifier in classifiers.items():

        print(f"model:{model_name} | traning:{embedding_name}")

        with mlflow.start_run():
            ml_pipe = Pipeline([('vectorizer', MeanEmbeddingVectorizer(pretrained_w2v)),
                                ('classifier', classifier)])

            # Model fit
            ml_pipe.fit(X_train, y_train)
            y_predict = ml_pipe.predict(X_test)

            # Tracking
            mlflow.log_params(ml_pipe.get_params())
            mlflow.log_metric('precision', precision_score(y_test, y_predict))
            mlflow.log_metric('accuracy', accuracy_score(y_test, y_predict))
            mlflow.log_metric('recall', recall_score(y_test, y_predict))
            mlflow.log_metric('auc', roc_auc_score(y_test, y_predict))
            mlflow.log_metric('f1', f1_score(y_test, y_predict))
            mlflow.sklearn.log_model(
                ml_pipe, model_name + "_" + embedding_name)


print("finished")
