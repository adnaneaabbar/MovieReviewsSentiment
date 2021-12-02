from keras.datasets import imdb
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from keras_preprocessing import sequence

max_features = 20000
maxlen = 80

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
