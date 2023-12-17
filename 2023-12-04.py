# multivariate cnn example
import yfinance as yf
from keras.layers import Dense, Flatten, Input, Conv1D, MaxPooling1D, LSTM, Bidirectional
from keras.models import Sequential, Model
from numpy import array
from numpy import hstack
import matplotlib.pyplot as plt


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def multi_variant_parallel_series_CNN():
    dataset = download_data("2022-11-23", "2023-10-23")
    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)
    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=100, verbose=0)
    # demonstrate prediction
    x_input = download_data("2023-10-23", "2023-11-24")[-4:-1]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def multi_variant_parallel_series_Stacked_LSTM():
    dataset = download_data("2022-11-23", "2023-10-23")
    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)
    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=100, verbose=0)
    # demonstrate prediction
    x_input = download_data("2023-10-23", "2023-11-24")[-4:-1]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def multi_variant_parallel_series_BLSTM():
    dataset = download_data("2022-11-23", "2023-10-23")
    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)
    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    # define model
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=100, verbose=0)
    # demonstrate prediction
    x_input = download_data("2023-10-23", "2023-11-24")[-4:-1]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def download_data(start, end):
    sequences = []
    for ticker in ["GOOGL", "AAPL", "MSFT", "TSLA"]:
        # Download data from Yahoo Finance
        data = yf.download(ticker, start=start, end=end)
        seq = data["Close"].values
        sequences.append(seq)
    return hstack([seq.reshape(len(seq), 1) for seq in sequences])


def multi_variant_parallel_series_CNN_multiple_output():
    dataset = download_data("2022-11-23", "2023-10-23")
    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)
    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    # separate output
    y1 = y[:, 0].reshape((y.shape[0], 1))
    y2 = y[:, 1].reshape((y.shape[0], 1))
    y3 = y[:, 2].reshape((y.shape[0], 1))
    y4 = y[:, 3].reshape((y.shape[0], 1))
    # define model
    visible = Input(shape=(n_steps, n_features))
    cnn = Conv1D(filters=64, kernel_size=2, activation='relu')(visible)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(50, activation='relu')(cnn)
    # define output 1
    output1 = Dense(1)(cnn)
    # define output 2
    output2 = Dense(1)(cnn)
    # define output 3
    output3 = Dense(1)(cnn)
    # define output 4
    output4 = Dense(1)(cnn)
    # tie together
    model = Model(inputs=visible, outputs=[output1, output2, output3, output4])
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, [y1, y2, y3, y4], epochs=100, verbose=0)
    # demonstrate prediction
    x_input = download_data("2023-10-23", "2023-11-24")[-4:-1]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def get_last_day():
    x_input = download_data("2023-10-23", "2023-11-24")
    return flatten_arr(x_input[-1])


def flatten_arr(result):
    items = array(result).flatten()
    return [round(item, 2) for item in items]


def plot_result(x, h, title):
    plt.bar(x, h)
    plt.ylabel("Price")
    plt.title("Average Stock Price Predictions for " + title)
    plt.show()


if __name__ == '__main__':
    cnn = multi_variant_parallel_series_CNN()
    b_lstm = multi_variant_parallel_series_BLSTM()
    s_lstm = multi_variant_parallel_series_Stacked_LSTM()
    mo_cnn = multi_variant_parallel_series_CNN_multiple_output()
    last = get_last_day()
    plot_result(['CNN', 'BLSTM', 'LSTM', 'MO CNN', 'REAL'], [cnn[0], b_lstm[0], s_lstm[0], mo_cnn[0], last[0]],
                'GOOGLE')
    plot_result(['CNN', 'BLSTM', 'LSTM', 'MO CNN', 'REAL'], [cnn[1], b_lstm[1], s_lstm[1], mo_cnn[1], last[1]], 'APPLE')
    plot_result(['CNN', 'BLSTM', 'LSTM', 'MO CNN', 'REAL'], [cnn[2], b_lstm[2], s_lstm[2], mo_cnn[2], last[2]],
                'MICROSOFT')
    plot_result(['CNN', 'BLSTM', 'LSTM', 'MO CNN', 'REAL'], [cnn[3], b_lstm[3], s_lstm[3], mo_cnn[3], last[3]], 'TESLA')
