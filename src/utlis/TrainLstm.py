# %% [markdown]
# ## Requirements
#

# %%
# Unable warnings
from numba import cuda
import mlflow
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Embedding
from keras.models import Sequential
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split
import gensim
from gensim import models
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# %%
cuda.get_current_device().reset()
cuda.current_context().reset()
cuda.current_context().get_memory_info()


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        K.clear_session()
        K._get_available_gpus()
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)

# %% [markdown]
# #### Data Processing
#

# %%
# ETL

# %% [markdown]
# #### Natural language processing
#

# %%


# %% [markdown]
# #### Models
# %%
# ML preprocessing


# Deep learnig model


# %%
# Metrics

# Train  metrics
METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]


# %% [markdown]
# #### Tracking

# %%
mlflow.set_tracking_uri('http://127.0.0.1:7500')
mlflow.set_experiment('Hate Speech')


# %% [markdown]
# ## Split dataset
#

# %%
# Get data
df = pd.read_csv("data/corpus/augmented_corpus_fortuna.csv")

# Set target and features
target = "label"
features = "text_stop"
count = f"length_{features}"
pos = len(df.query('label==1'))
neg = len(df.query('label==0'))


# Break apart dataset
X = df[features].values.astype("U")
y = df[target]

# Split train abd test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Set k-fold criteria
k_fold = KFold(n_splits=10, shuffle=True, random_state=42)

# Classes balancing
longest_text = df[count].max()
initial_bias = np.log([pos/neg])

weight_for_0 = (1 / neg) * (len(df) / 2.0)
weight_for_1 = (1 / pos) * (len(df) / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}


# %% [markdown]
# ## LSTM with Word2vec

# %%
# Load embedding

print("loading embedding ... \n")

w2v = KeyedVectors.load_word2vec_format(
    "data/pretrained-glove/glove_s50.txt", binary=False
)

# Embedding props
vocab_size = len(w2v) + 1
vec_dim = w2v.vectors.shape[1]
embedding_weights = np.vstack([
    np.zeros(w2v.vectors.shape[1]),
    w2v.vectors
])


# %%
class TokenizerTransformer(BaseEstimator, TransformerMixin, Tokenizer):
    def __init__(self, **tokenizer_params):
        Tokenizer.__init__(self, **tokenizer_params)

    def fit(self, X, y=None):
        self.fit_on_texts(X)
        return self

    def transform(self, X, y=None):
        X_transformed = self.texts_to_sequences(X)
        return X_transformed


class PadSequencesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, maxlen):
        self.maxlen = maxlen

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_padded = pad_sequences(X, maxlen=self.maxlen)
        return X_padded


class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)


# %% [markdown]
# #### Basic pipeline
# %%
# Preset parameters
experiment_parameters = {"classifier": "LSTM",
                         "class_weight": class_weight,
                         "epochs": 100,
                         "units": 50,
                         "dropout": 0.4,
                         "recurrent_dropout": 0.2,
                         "kernel_initializer": 'glorot_uniform',
                         "loss": "binary_crossentropy",
                         "optimizer": "adamax",
                         "embedding_input_dim": vec_dim,
                         "batch_size": 128}


# %%
# LSTM model
def lstm_builder(embedding_input_dim, embedding_output_dim, embedding_weights):
    output_bias = tf.keras.initializers.Constant(initial_bias)
    lstm = Sequential()

    lstm.add(
        Embedding(
            input_dim=embedding_input_dim,
            output_dim=embedding_output_dim,
            weights=[embedding_weights],
            trainable=False,
            mask_zero=True,
        )
    )
    lstm.add(Bidirectional(LSTM(units=experiment_parameters['units'],
                                dropout=experiment_parameters['dropout'],
                                recurrent_dropout=0,
                                kernel_initializer=experiment_parameters['kernel_initializer'])))

    lstm.add(Dropout(0.20))

    lstm.add(Dense(units=1,
                   activation="sigmoid",
                   bias_initializer=output_bias))

    lstm.compile(loss=experiment_parameters['loss'],
                 optimizer=experiment_parameters['optimizer'],
                 metrics=METRICS)
    return lstm


# %%
# Model execution
lstm = KerasClassifier(
    model=lstm_builder,
    epochs=experiment_parameters['epochs'],
    embedding_input_dim=len(w2v) + 1,
    embedding_output_dim=vec_dim,
    embedding_weights=embedding_weights,
    batch_size=experiment_parameters['batch_size'],
    callbacks=[EarlyStopping(monitor="loss",
                             patience=10,
                             restore_best_weights=True)],
    class_weight=class_weight
)


# %%
print("model tracking ... \n")
mlflow.sklearn.autolog()
with mlflow.start_run():

    ml_pipe = Pipeline(
        [("tokenizer",  TokenizerTransformer()),
         ("padder", PadSequencesTransformer(maxlen=longest_text)),
         ("model", lstm)])

    # Model fit
    ml_pipe.fit(X_train, y_train)
    y_predict = ml_pipe.predict(X_test)

    # Tracking
    mlflow.log_metric('precision', precision_score(y_test, y_predict))
    mlflow.log_metric('accuracy', accuracy_score(y_test, y_predict))
    mlflow.log_metric('recall', recall_score(y_test, y_predict))
    mlflow.log_metric('auc', roc_auc_score(y_test, y_predict))
    mlflow.log_metric('f1', f1_score(y_test, y_predict))
    mlflow.pyfunc.log_model(
        python_model=ModelWrapper(ml_pipe),
        artifact_path="LSTM",
    )

    K.clear_session()
    cuda.get_current_device().reset()
    cuda.current_context().reset()

# %%
