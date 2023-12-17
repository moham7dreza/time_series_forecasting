# univariate cnn example
import matplotlib.pyplot as plt
import yfinance as yf
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, LSTM, Bidirectional
from keras.models import Sequential
from numpy import array


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def download_data(start, end):
    sequences = {}
    for ticker in ["GOOGL", "AAPL", "MSFT", "TSLA"]:
        # Download data from Yahoo Finance
        data = yf.download(ticker, start=start, end=end)
        seq = data["Close"].values
        sequences[ticker] = seq.reshape(len(seq), 1)
    return sequences


def univariant_CNN(dataset, x_input):
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(dataset, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=1000, verbose=0)
    # demonstrate prediction

    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def univariant_LSTM(dataset, x_input):
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(dataset, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=100, verbose=0)
    # demonstrate prediction

    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def univariant_BLSTM(dataset, x_input):
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(dataset, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=100, verbose=0)
    # demonstrate prediction

    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def flatten_arr(result):
    items = array(result).flatten()
    return [round(item, 2) for item in items]


def plot_result(x, h, title):
    plt.bar(x, h)
    plt.ylabel("Price")
    plt.title("Average Stock Price Predictions for " + title)
    plt.show()


def run(ticker):
    cnn = univariant_CNN(dataset[ticker], x_input[ticker][-4:-1])
    lstm = univariant_LSTM(dataset[ticker], x_input[ticker][-4:-1])
    b_lstm = univariant_BLSTM(dataset[ticker], x_input[ticker][-4:-1])
    last = flatten_arr(x_input[ticker][-1])

    plot_result(['CNN', 'BLSTM', 'LSTM', 'REAL'], [cnn[0], b_lstm[0], lstm[0], last[0]],
                ticker)


if __name__ == '__main__':
    dataset = download_data("2022-11-23", "2023-10-23")
    x_input = download_data("2023-10-23", "2023-11-24")

    run('GOOGL')
    run('AAPL')
    run('MSFT')
    run('TSLA')
