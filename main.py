"""
This is the entry point.
"""
from preprocess import *
from classifier import *
import keras


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
    tokenizer_filepath = 'tokenizer.pickle'
    tokenizer = get_tokenizer(corpus, max_features, tokenizer_filepath)
    X = tokenizer.texts_to_sequences(corpus)
    X = sequence.pad_sequences(X, maxlen=max_seq_len)

    # encode labels
    Y = encode_labels(y_data)

    # create embed matrix
    embed_matrix = get_embed_matrix(
        embed, tokenizer.word_index, max_features, embed_dim)

    # create model
    model = get_model(embed_matrix, max_seq_len, labels)

    print("Model Created")

    # training model
    model.fit(X, Y, batch_size=256, epochs=5, verbose=1, validation_split=0.2)
    model.save('hw_classifier.h5')

def predict_label(test):
    tokenizer_filepath = 'tokenizer.pickle'
    max_seq_len = 1000
    with open(tokenizer_filepath, 'rb') as f:
        tokenizer = pickle.load(f)
    X_test = tokenizer.texts_to_sequences(test)
    X_test = sequence.pad_sequences(X_test, maxlen=max_seq_len)

    model_filepath = 'hw_classifier.h5'
    classifier = keras.models.load_model(model_filepath)
    y_test = classifier.predict(X_test, verbose=1)
    y_test = np.argmax(y_test, axis = 1)

    label_encoder_filepath = 'label_encoder.pickle'
    with open(label_encoder_filepath, 'rb') as f:
        encoder = pickle.load(f)
    label = list(encoder.inverse_transform(y_test))
    return label


if __name__ == '__main__':
    main()
