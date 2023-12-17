# multivariate cnn example
import yfinance as yf
from keras.layers import Dense, Flatten, Input, Conv1D, MaxPooling1D, LSTM, Bidirectional
from keras.models import Sequential, Model
from numpy import array
from numpy import hstack
import matplotlib.pyplot as plt


# split a multivariate sequence into samples
def split_sequences_for_multivariate_parallel_series(sequences, n_steps):
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


def multivariate_parallel_series_CNN():
    dataset = download_data_for_multivariate_parallel_series("2022-11-23", "2023-10-23")
    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences_for_multivariate_parallel_series(dataset, n_steps)
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
    x_input = download_data_for_multivariate_parallel_series("2023-10-23", "2023-11-24")[-4:-1]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def multivariate_parallel_series_Stacked_LSTM():
    dataset = download_data_for_multivariate_parallel_series("2022-11-23", "2023-10-23")
    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences_for_multivariate_parallel_series(dataset, n_steps)
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
    x_input = download_data_for_multivariate_parallel_series("2023-10-23", "2023-11-24")[-4:-1]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def multivariate_parallel_series_BLSTM():
    dataset = download_data_for_multivariate_parallel_series("2022-11-23", "2023-10-23")
    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences_for_multivariate_parallel_series(dataset, n_steps)
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
    x_input = download_data_for_multivariate_parallel_series("2023-10-23", "2023-11-24")[-4:-1]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def download_data_for_multivariate_parallel_series(start, end):
    sequences = []
    for ticker in ["GOOGL", "AAPL", "MSFT", "TSLA"]:
        # Download data from Yahoo Finance
        data = yf.download(ticker, start=start, end=end)
        seq = data["Close"].values
        sequences.append(seq)
    return hstack([seq.reshape(len(seq), 1) for seq in sequences])


def multivariate_parallel_series_CNN_multiple_output():
    dataset = download_data_for_multivariate_parallel_series("2022-11-23", "2023-10-23")
    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences_for_multivariate_parallel_series(dataset, n_steps)
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
    x_input = download_data_for_multivariate_parallel_series("2023-10-23", "2023-11-24")[-4:-1]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def get_last_day_for_multivariate_parallel_series():
    x_input = download_data_for_multivariate_parallel_series("2023-10-23", "2023-11-24")
    return flatten_arr(x_input[-1])


def flatten_arr(result):
    items = array(result).flatten()
    return [round(item, 2) for item in items]


def plot_result(x, h, title):
    plt.bar(x, h)
    plt.ylabel("Price")
    plt.title("Average Stock Price Predictions for " + title)
    plt.show()


# split a univariate sequence into samples
def split_sequence_for_univariate_series(sequence, n_steps):
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


def download_data_for_univariate_series(start, end):
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
    X, y = split_sequence_for_univariate_series(dataset, n_steps)
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
    X, y = split_sequence_for_univariate_series(dataset, n_steps)
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
    X, y = split_sequence_for_univariate_series(dataset, n_steps)
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


def run_for_univariate_series(ticker):
    u_cnn = univariant_CNN(dataset[ticker], x_input[ticker][-4:-1])
    u_lstm = univariant_LSTM(dataset[ticker], x_input[ticker][-4:-1])
    u_b_lstm = univariant_BLSTM(dataset[ticker], x_input[ticker][-4:-1])
    u_last = flatten_arr(x_input[ticker][-1])

    return u_cnn, u_lstm, u_b_lstm


if __name__ == '__main__':
    dataset = download_data_for_univariate_series("2022-11-23", "2023-10-23")
    x_input = download_data_for_univariate_series("2023-10-23", "2023-11-24")

    cnn = multivariate_parallel_series_CNN()
    b_lstm = multivariate_parallel_series_BLSTM()
    s_lstm = multivariate_parallel_series_Stacked_LSTM()
    mo_cnn = multivariate_parallel_series_CNN_multiple_output()
    last = get_last_day_for_multivariate_parallel_series()

    u_cnn, u_lstm, u_b_lstm = run_for_univariate_series('GOOGL')

    plot_result(['CNN', 'U-CNN', 'BLSTM', 'U-BLSTM', 'LSTM', 'U-LSTM', 'MO CNN', 'REAL'],
                [cnn[0], u_cnn[0], b_lstm[0], u_b_lstm[0], s_lstm[0], u_lstm[0], mo_cnn[0], last[0]],
                'GOOGLE')

    u_cnn, u_lstm, u_b_lstm = run_for_univariate_series('AAPL')

    plot_result(['CNN', 'U-CNN', 'BLSTM', 'U-BLSTM', 'LSTM', 'U-LSTM', 'MO CNN', 'REAL'],
                [cnn[1], u_cnn[0], b_lstm[1], u_b_lstm[0], s_lstm[1], u_lstm[0], mo_cnn[1], last[1]],
                'AAPL')

    u_cnn, u_lstm, u_b_lstm = run_for_univariate_series('MSFT')

    plot_result(['CNN', 'U-CNN', 'BLSTM', 'U-BLSTM', 'LSTM', 'U-LSTM', 'MO CNN', 'REAL'],
                [cnn[2], u_cnn[0], b_lstm[2], u_b_lstm[0], s_lstm[2], u_lstm[0], mo_cnn[2], last[2]],
                'MSFT')

    u_cnn, u_lstm, u_b_lstm = run_for_univariate_series('TSLA')

    plot_result(['CNN', 'U-CNN', 'BLSTM', 'U-BLSTM', 'LSTM', 'U-LSTM', 'MO CNN', 'REAL'],
                [cnn[3], u_cnn[0], b_lstm[3], u_b_lstm[0], s_lstm[3], u_lstm[0], mo_cnn[3], last[3]],
                'TSLA')
