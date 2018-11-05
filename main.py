"""
This is the entry point.
"""
from preprocess import *
from classifier import *
import keras


tokenizer_filepath = 'model_files/tokenizer.pickle'
label_encoder_filepath = 'model_files/label_encoder.pickle'
model_filepath = 'model_files/hw_classifier.h5'


def main():
    x_data, y_data, labels = get_data('shuffled-full-set-hashed.csv')

    # convert the word to vector
    corpus = x_data
    embed_dim = 300
    embed_path = 'embed_{}d.txt'.format(embed_dim)
    word2vec(corpus, embed_dim, embed_path)
    embed = read_embed(embed_path, skip_first_row=True)

    # define features
    max_features = 20000
    max_seq_len = 1000

    # convert words to sequence
    tokenizer = get_tokenizer(corpus, max_features, tokenizer_filepath)
    X = tokenizer.texts_to_sequences(corpus)
    X = sequence.pad_sequences(X, maxlen=max_seq_len)

    # encode labels
    Y = encode_labels(y_data, label_encoder_filepath=label_encoder_filepath)

    # create embed matrix
    embed_matrix = get_embed_matrix(
        embed, tokenizer.word_index, max_features, embed_dim)

    # create model
    model = get_model(embed_matrix, max_seq_len, labels)

    print("Model Created")

    # training model
    model.fit(X, Y, batch_size=256, epochs=5, verbose=1, validation_split=0.2)
    model.save(model_filepath)


class Classifier:
    def __init__(self, *args, **kwargs):
        self.max_seq_len = 1000
        with open(tokenizer_filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.model = keras.models.load_model(model_filepath)
        with open(label_encoder_filepath, 'rb') as f:
            self.encoder = pickle.load(f)

    def predict_label(self, test):
        X_test = self.tokenizer.texts_to_sequences(test)
        X_test = sequence.pad_sequences(X_test, maxlen=self.max_seq_len)

        y_test = self.model.predict(X_test, verbose=1)
        y_test = np.argmax(y_test, axis=1)

        labels = list(self.encoder.inverse_transform(y_test))
        return labels


if __name__ == '__main__':
    main()
