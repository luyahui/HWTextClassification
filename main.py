"""
This is the entry point.
"""
from preprocess import *
from classifier import *


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
    tokenizer = get_tokenizer(corpus, max_features)
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


if __name__ == '__main__':
    main()
