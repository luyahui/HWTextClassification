import pandas as pd
import numpy as np
import pickle
import os


def get_data(filepath, header=None):
    """
    read data from file
    """
    df = pd.read_csv(filepath, header=header)

    # 4/5 of data as training set, remaining as test set
    train_raw = df.iloc[:len(df) * 0.8, ]
    test_raw = df.iloc[len(df)*0.8:, ]
    labels = np.unique(df[0])

    return train_raw, test_raw, labels


def extract_data(train_raw, test_raw, labels):
    """
    Convert the data from raw data to encoded data
    """
    x_train = train_raw[1]
    x_test = test_raw[1]

    # encod the label from string to integer
    def cast(label):
        for i in range(len(labels)):
            if label == labels[i]:
                return i
        return -1

    y_train = train_raw[0].apply(cast)
    y_test = test_raw[0].apply(cast)

    return x_train, y_train, x_test, y_test


def preprocess_data(data_filepath, preprocessed_filepath):
    """
    Preprocess the data, if not done already.
    Return preprocessed data.
    """

    if os.path.isfile(preprocessed_filepath):
        with open(preprocessed_filepath, 'rb') as f:
            x_train, y_train, x_test, y_test, labels = pickle.load(f)
    else:
        # get the data
        train_raw, test_raw, labels = get_data(data_filepath)
        x_train, y_train, x_test, y_test = extract_data(
            train_raw, test_raw, labels)

        # save the preprocessed data into a pickle file
        saved = False
        with open(preprocessed_filepath, 'wb') as f:
            try:
                pickle.dump([x_train, y_train, x_test, y_test, labels], f)
                saved = True
            except:
                pass
        
        # detele the file if data was not saved
        if not saved:
            os.remove(preprocessed_filepath)

    return x_train, y_train, x_test, y_test, labels
