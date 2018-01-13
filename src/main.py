from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import History
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Dense
from sys import argv
import pandas as pd
import numpy as np

# random seed
np.random.seed(1)

# initialize model
model = Sequential()

# records events into a History object
history = History()

# parameters
filename = "../database/yeast_modified.csv"
var = 8
input_layer = 7
output_layer = 7
hidden_layer = int(argv[1])
hidden_extra = int(argv[2])
input_act_fun = None
output_act_fun = "softmax"
hidden_act_fun = "relu"
learning_rate = float(argv[3])
epochs_nb = 150
loss_fun = "categorical_crossentropy"
mini_batches = 80
train_split = 0.93
metric = "categorical_accuracy"
logname = "../logs/training.log"

# database specific information
size = 1429
train_start = 0
train_end = int(size * train_split)
test_start = train_end + 1
test_end = size


# load data
def Load(start, end):
    # load input
    data = np.loadtxt(filename, delimiter=";", dtype=np.str)
    # split input
    X = data[start:end, 0:var].astype(float)
    # split output
    Y = data[start:end, var]
    # string --> int
    aux = pd.get_dummies(Y)
    Y = aux.values.argmax(1)
    # return a tuple with the input and the categorized output
    return (X, np_utils.to_categorical(Y))


# build model
def Build():
    # input layer
    model.add(Dense(input_layer, activation=input_act_fun, input_dim=var))
    # hidden layers
    model.add(Dense(hidden_layer, activation=hidden_act_fun))
    for i in range(0, hidden_extra):
        model.add(Dense(hidden_layer, activation=hidden_act_fun))
    # output layers
    model.add(Dense(output_layer, activation=output_act_fun))


# compile model
def Compile():
    adam = Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss=loss_fun, metrics=[metric])


# fit model / train the nn
def Fit(X, Y):
    # define the early stopping criteria
    early_stopping = EarlyStopping(monitor="val_loss", patience=30, verbose=1)
    # streams epoch results to a csv file
    train_csv_logger = CSVLogger(logname)

    model.fit(X, Y, batch_size=mini_batches, epochs=epochs_nb, verbose=1, callbacks=[early_stopping, train_csv_logger, history], validation_split=0.2, shuffle=True)


# evaluate model / test the nn
def Evaluate(X, Y):
    scores = model.evaluate(X, Y, verbose=1)
    print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def main():
    # define the train part of the database
    (train_X, train_Y) = Load(train_start, train_end)
    # define the test part of the database
    (test_X, test_Y) = Load(test_start, test_end)
    Build()
    Compile()
    Fit(train_X, train_Y)
    Evaluate(test_X, test_Y)


if __name__ == "__main__":
    main()
