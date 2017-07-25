import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [series[i:i + window_size] for i in range(len(series) - window_size)]
    y = [i for i in series[window_size:]]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.summary()
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):

    # only chars that belongs to punctuation set and in ['a' .... 'z'] set
    # ['a' .... 'z'] chars are selected using ascii decimal code: 'a' = 97, 'z' = 122
    punctuation = ['!', ',', '.', ':', ';', '?']
    for i in range(255):
        if (chr(i) not in punctuation) and ( (i < 97) or (i > 122) ):
            text = text.replace(chr(i), ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs

    inputs = [text[i:i + window_size] for i in range(0, len(text) - window_size, step_size)]
    outputs = [i for i in text[window_size::step_size]]

    return inputs, outputs


# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    model.summary()
    return model
