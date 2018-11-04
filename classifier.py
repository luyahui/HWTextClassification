from keras import Model
from keras.layers import *


def get_model(embed_matrix, max_seq_len, labels):
    inp = Input(shape=(max_seq_len,))
    embedding_layer = Embedding(embed_matrix.shape[0], embed_matrix.shape[1],
                                weights=[embed_matrix], input_length=max_seq_len, trainable=False)
    lstm_layer = Bidirectional(LSTM(200, return_sequences=True))
    encoded = lstm_layer(embedding_layer(inp))
    concated = concatenate(
        [GlobalAvgPool1D()(encoded), GlobalMaxPool1D()(encoded)])
    mlp = Dropout(0.3)(concated)
    mlp = BatchNormalization()(mlp)
    mlp = Dense(300, activation='relu')(mlp)
    mlp = Dropout(0.3)(mlp)
    mlp = BatchNormalization()(mlp)
    mlp = Dense(300, activation='relu')(mlp)
    mlp = Dropout(0.3)(mlp)
    mlp = BatchNormalization()(mlp)
    outp = Dense(len(labels), activation='softmax')(mlp)
    model = Model(inp, outp)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
