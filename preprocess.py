import pandas as pd
import numpy as np
import pickle
import os
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from keras.utils import np_utils
from keras.preprocessing import text, sequence
from sklearn.preprocessing import LabelEncoder


def get_data(filepath, header=None):
    """
    read data from file
    """
    df = pd.read_csv(filepath, header=header, names=['label', 'words']).astype(str)

    # 4/5 of data as training set, remaining as test set
    labels = np.unique(df['label'])
    train = df.iloc[:int(len(df) * 0.8), ]
    test = df.iloc[int(len(df) * 0.8):, ]

    x_train = train['words']
    y_train = train['label']

    x_test = test['words']
    y_test = test['label']

    return x_train, y_train, x_test, y_test, labels


def split_data(original_data_filepath, split_data_filepath):
    """
    Preprocess the data, if not done already.
    Return preprocessed data.
    """

    if os.path.isfile(split_data_filepath):
        with open(split_data_filepath, 'rb') as f:
            x_train, y_train, x_test, y_test, labels = pickle.load(f)
    else:
        # get the data
        x_train, y_train, x_test, y_test, labels = get_data(
            original_data_filepath)

        # save the preprocessed data into a pickle file
        saved = False
        with open(split_data_filepath, 'wb') as f:
            try:
                pickle.dump([x_train, y_train, x_test, y_test, labels], f)
                saved = True
            except:
                pass

        # detele the file if data was not saved
        if not saved:
            os.remove(split_data_filepath)

    return x_train, y_train, x_test, y_test, labels


def word2vec(corpus, embed_dim, embed_path):
    """
    train word/char embedding
    """
    if os.path.isfile(embed_path):
        model = KeyedVectors.load_word2vec_format(embed_path, binary=False)
    else:
        sequence = [sentence.split() for sentence in corpus]
        model = Word2Vec(sequence, size=embed_dim, min_count=2,
                         sg=1, negative=10, iter=10, workers=50)
        model.wv.save_word2vec_format(embed_path, binary=False)

    return model


def read_embed(embed_path, skip_first_row=False):
    """
    read embedding from path
    """
    if skip_first_row:
        df = pd.read_csv(embed_path, sep=' ', skiprows=1,
                         header=None, encoding='utf-8')
    else:
        df = pd.read_csv(embed_path, sep=' ', header=None, encoding='utf-8')
    embed = {}
    for _, row in df.iterrows():
        embed[row[0]] = row[1:].values.astype(np.float64)
    return embed


def get_tokenizer(corpus, num_words, lower=False):
    tokenizer = text.Tokenizer(num_words, lower=lower)
    tokenizer.fit_on_texts(corpus)
    return tokenizer


def encode_labels(labels):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return np_utils.to_categorical(y)


def get_embed_matrix(embed, token_index, max_features, embed_dim):
    """
    create embedding matrix
    """
    nb_words = min(max_features, len(token_index)) + 1
    embed_matrix = np.zeros((nb_words, embed_dim))
    for word, index in token_index.items():
        if index >= nb_words:
            continue
        vec = embed.get(word)
        if vec is not None:
            embed_matrix[index] = vec

    return embed_matrix
