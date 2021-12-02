from keras.datasets import imdb
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from keras_preprocessing import sequence

max_features = 20000
maxlen = 80

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
