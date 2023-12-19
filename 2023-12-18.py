# 2023-12-17
# class based

import io
import os

import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files, drive
from keras.layers import Dense, Flatten, Input, Conv1D, MaxPooling1D, LSTM, Bidirectional
from keras.models import Sequential, Model
from numpy import array, hstack


def upload_and_read_csv_in_colab(name):
    print('\nSelect file for ' + name + ' ...\n')
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    if name.lower() not in file_name.lower():
        raise ValueError("uploaded dataset is incorrect")
    file_content = uploaded[file_name]
    print('\nSelected file ' + file_name + ' is begin to read ...\n')
    # load dataset
    series = pd.read_csv(io.BytesIO(file_content))
    return series


def read_csv_files_from_drive_in_colab():
    drive.mount('/content/drive')

    # Navigate to the folder containing your CSV files
    folder_path = '/content/drive/My Drive/iran_stock'
    %cd $folder_path

    # Get a list of all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Read each CSV file into a DataFrame and store them in a dictionary
    dataframes = {}
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        dataframes[file] = pd.read_csv(file_path)
    return dataframes


def stack_datasets(datasets, col):
    sequences = []
    for dataset in datasets:
        seq = dataset[col].values
        sequences.append(seq)
        # print(len(seq))
    return hstack([seq.reshape(len(seq), 1) for seq in sequences])


def data_preprocessing(dataset, date_col, start_date, end_date, format=True):
    # sort by date
    dataset = dataset.sort_values(by=date_col)

    if (format):
        # Assuming 'date' is the column containing date in integer format
        dataset[date_col] = pd.to_datetime(dataset[date_col], format='%Y%m%d')
    else:
        dataset[date_col] = pd.to_datetime(dataset[date_col])

    # Convert object columns to strings
    object_columns = dataset.select_dtypes(include='object').columns
    dataset[object_columns] = dataset[object_columns].astype(str)

    # Identify and exclude object columns
    non_object_columns = dataset.select_dtypes(exclude='object').columns

    # Create a new DataFrame without object columns
    dataset = dataset[non_object_columns]
    dataset = dataset.set_index(date_col)
    dataset = dataset.resample('W-Sat').mean().ffill()
    dataset = dataset.loc[start_date:end_date]

    return dataset


def train_test_split(dataset, test_size):
    index = -test_size - 1
    train = dataset[:index]
    test = dataset[index:-1]
    u_last = flatten_arr(dataset[-1])

    return train, test, u_last


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


def flatten_arr(result):
    items = array(result).flatten()
    return [round(item, 2) for item in items]


def get_CNN_model(n_steps, n_features):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')
    return model


def get_stacked_LSTM_model(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')
    return model


def get_multi_output_CNN_model(n_steps, n_features):
    # define model
    visible = Input(shape=(n_steps, n_features))
    cnn = Conv1D(filters=64, kernel_size=2, activation='relu')(visible)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(50, activation='relu')(cnn)
    # define outputs dynamically based on n_features
    outputs = [Dense(1)(cnn) for _ in range(n_features)]
    # tie together
    model = Model(inputs=visible, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def get_bidirectional_LSTM_model(n_steps, n_features):
    # define model
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')
    return model


def multivariate_parallel_series_CNN(dataset, x_input, n_steps, epochs):
    # convert into input/output
    X, y = split_sequences_for_multivariate_parallel_series(dataset, n_steps)
    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    # define model
    model = get_CNN_model(n_steps, n_features)
    # fit model
    model.fit(X, y, epochs, verbose=0)
    # demonstrate prediction
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def multivariate_parallel_series_Stacked_LSTM(dataset, x_input, n_steps, epochs):
    # convert into input/output
    X, y = split_sequences_for_multivariate_parallel_series(dataset, n_steps)
    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    # define model
    model = get_stacked_LSTM_model(n_steps, n_features)
    # fit model
    model.fit(X, y, epochs, verbose=0)
    # demonstrate prediction
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def multivariate_parallel_series_BLSTM(dataset, x_input, n_steps, epochs):
    # convert into input/output
    X, y = split_sequences_for_multivariate_parallel_series(dataset, n_steps)
    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    # define model
    model = get_bidirectional_LSTM_model(n_steps, n_features)
    # fit model
    model.fit(X, y, epochs, verbose=0)
    # demonstrate prediction
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def multivariate_parallel_series_CNN_multiple_output(dataset, x_input, n_steps, epochs):
    # convert into input/output
    X, y = split_sequences_for_multivariate_parallel_series(dataset, n_steps)
    # the dataset knows the number of features, e.g., 2
    n_features = X.shape[2]

    # separate output dynamically based on n_features
    y_outputs = [y[:, i].reshape((y.shape[0], 1)) for i in range(n_features)]

    # define model
    model = get_multi_output_CNN_model(n_steps, n_features)
    # fit model
    model.fit(X, y_outputs, epochs=epochs, verbose=0)

    # demonstrate prediction
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def plotting(dataset, name):
    # Plotting
    plt.figure(figsize=(12, 8))

    # Plotting open, high, low, close prices
    plt.plot(dataset.index, dataset['<OPEN>'], label='Open', linestyle='-', color='blue')
    plt.plot(dataset.index, dataset['<HIGH>'], label='High', linestyle='-', color='green')
    plt.plot(dataset.index, dataset['<LOW>'], label='Low', linestyle='-', color='red')
    plt.plot(dataset.index, dataset['<CLOSE>'], label='Close', linestyle='-', color='purple')

    # Adding grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adding labels and title
    plt.xlabel('Date(EN)')
    plt.ylabel('Price(Rial)')
    plt.title('<' + name + '> Prices Over Time')

    # Formatting x-axis with a readable date format
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    # Rotating x-axis labels for better readability
    plt.gcf().autofmt_xdate()

    # Adding legend with a shadow
    plt.legend(shadow=True)

    # Adding a background color to the plot
    plt.gca().set_facecolor('#f4f4f4')

    # Adding horizontal lines for reference
    plt.axhline(y=100, color='gray', linestyle='--', linewidth=1, label='Reference Line')

    # Display the plot
    plt.show()


def plot_result(x, h, title):
    plt.bar(x, h)
    plt.ylabel("Price")
    plt.title("Average Stock Price Predictions for " + title)
    plt.show()


def run_for_univariate_series_ir(dataset, col, n_steps, epochs):
    dataset = dataset[col].values

    train, test, u_last = train_test_split(dataset, n_steps)

    u_cnn = univariant_CNN(train, test, n_steps, epochs)
    u_lstm = univariant_LSTM(train, test, n_steps, epochs)
    u_b_lstm = univariant_BLSTM(train, test, n_steps, epochs)

    return u_cnn[0], u_lstm[0], u_b_lstm[0], u_last


def run_for_multivariate_series_ir(datasets, col, n_steps, epochs):
    dataset = stack_datasets(datasets, col)

    train, test, last = train_test_split(dataset, n_steps)

    cnn = multivariate_parallel_series_CNN(train, test, n_steps, epochs)
    b_lstm = multivariate_parallel_series_BLSTM(train, test, n_steps, epochs)
    s_lstm = multivariate_parallel_series_Stacked_LSTM(train, test, n_steps, epochs)
    mo_cnn = multivariate_parallel_series_CNN_multiple_output(train, test, n_steps, epochs)

    return cnn, b_lstm, s_lstm, mo_cnn, last


def univariant_CNN(dataset, x_input, n_steps, epochs):
    # split into samples
    X, y = split_sequence_for_univariate_series(dataset, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = get_CNN_model(n_steps, n_features)
    # fit model
    model.fit(X, y, epochs, verbose=0)
    # demonstrate prediction

    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def univariant_LSTM(dataset, x_input, n_steps, epochs):
    # split into samples
    X, y = split_sequence_for_univariate_series(dataset, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = get_stacked_LSTM_model(n_steps, n_features)
    # fit model
    model.fit(X, y, epochs, verbose=0)
    # demonstrate prediction

    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def univariant_BLSTM(dataset, x_input, n_steps, epochs):
    # split into samples
    X, y = split_sequence_for_univariate_series(dataset, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = get_bidirectional_LSTM_model(n_steps, n_features)
    # fit model
    model.fit(X, y, epochs, verbose=0)
    # demonstrate prediction

    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    yhat = flatten_arr(yhat)
    print(yhat)
    return yhat


def run_and_plot(dataset, title, cnn, b_lstm, s_lstm, mo_cnn):
    labels = ['CNN', 'U-CNN', 'BLSTM', 'U-BLSTM', 'LSTM', 'U-LSTM', 'MO CNN', 'REAL']

    # 10. run algorithms on univariate dollar
    u_cnn, u_lstm, u_b_lstm, u_last = run_for_univariate_series_ir(dataset, prediction_col,
                                                                   n_steps,
                                                                   epochs)
    data = [cnn, u_cnn, b_lstm, u_b_lstm, s_lstm, u_lstm, mo_cnn, u_last]
    # 13. plot all results of algorithms on car
    plot_result(labels, data, title)


def attack():
    # 3. get each dataset according to env
    if colab:
        dfs = read_csv_files_from_drive_in_colab()

        ir_dollar = dfs[dollar_file_name]
        ir_home = dfs[home_file_name]
        ir_oil = dfs[oil_file_name]
        ir_car = dfs[car_file_name]
        ir_gold = dfs[gold_file_name]
    else:
        ir_dollar = pd.read_csv(folder_path + dollar_file_name)
        ir_home = pd.read_csv(folder_path + home_file_name)
        ir_oil = pd.read_csv(folder_path + oil_file_name)
        ir_car = pd.read_csv(folder_path + car_file_name)
        ir_gold = pd.read_csv(folder_path + gold_file_name)

    # 5. preprocess each dataset (sort, date col check, set index, resample)
    ir_dollar = data_preprocessing(ir_dollar, date_col, start_date, end_date,
                                   format=False)
    ir_home = data_preprocessing(ir_home, date_col, start_date, end_date)
    ir_oil = data_preprocessing(ir_oil, date_col, start_date, end_date)
    ir_car = data_preprocessing(ir_car, date_col, start_date, end_date)
    ir_gold = data_preprocessing(ir_gold, date_col, start_date, end_date)

    # 6. plot each dataset
    if plotting:
        plotting(ir_dollar, titles[0])
        plotting(ir_home, titles[1])
        plotting(ir_oil, titles[2])
        plotting(ir_car, titles[3])
        plotting(ir_gold, titles[4])

    # 7. array datasets for horizontally stack in multivariate prediction
    datasets = [ir_dollar, ir_home, ir_oil, ir_car, ir_gold]

    # 9. run algorithms on multivariate series
    cnn, b_lstm, s_lstm, mo_cnn, last = run_for_multivariate_series_ir(datasets, prediction_col,
                                                                       n_steps,
                                                                       epochs)

    for i in range(len(datasets)):
        run_and_plot(datasets[i], titles[i], cnn[i], b_lstm[i], s_lstm[i], mo_cnn[i])


def single_run():
    # 3. get each dataset according to env
    if colab:
        dfs = read_csv_files_from_drive_in_colab()

        ir_dollar = dfs[dollar_file_name]
        ir_home = dfs[home_file_name]
        ir_oil = dfs[oil_file_name]
        ir_car = dfs[car_file_name]
        ir_gold = dfs[gold_file_name]
    else:
        ir_dollar = pd.read_csv(folder_path + dollar_file_name)
        ir_home = pd.read_csv(folder_path + home_file_name)
        ir_oil = pd.read_csv(folder_path + oil_file_name)
        ir_car = pd.read_csv(folder_path + car_file_name)
        ir_gold = pd.read_csv(folder_path + gold_file_name)
    # print(ir_dollar.head(5))
    # print(ir_dollar.tail(5))
    # 5. preprocess each dataset (sort, date col check, set index, resample)
    ir_dollar = data_preprocessing(ir_dollar, date_col, start_date, end_date,
                                   format=False)
    ir_home = data_preprocessing(ir_home, date_col, start_date, end_date)
    ir_oil = data_preprocessing(ir_oil, date_col, start_date, end_date)
    ir_car = data_preprocessing(ir_car, date_col, start_date, end_date)
    ir_gold = data_preprocessing(ir_gold, date_col, start_date, end_date)

    # 6. plot each dataset
    if plotting:
        plotting(ir_dollar, titles[0])
        plotting(ir_home, titles[1])
        plotting(ir_oil, titles[2])
        plotting(ir_car, titles[3])
        plotting(ir_gold, titles[4])

    # 7. array datasets for horizontally stack in multivariate prediction
    datasets = [ir_dollar, ir_home, ir_oil, ir_car, ir_gold]

    dataset = stack_datasets(datasets, prediction_col)
    # print('sample dataset after stack ', dataset)
    train, test, last = train_test_split(dataset, n_steps)
    # print('train ', train)
    cnn = multivariate_parallel_series_CNN(train, test, n_steps, epochs_for_multivariate_series)


if __name__ == '__main__':
    colab = True
    plotting = False
    date_col = "<DTYYYYMMDD>"
    start_date = '2017-06-10'
    end_date = '2022-12-03'
    epochs_for_multivariate_series = 3000
    epochs_for_univariate_series = 2000
    n_steps = 3
    prediction_col = '<CLOSE>'
    folder_path = '../iran_stock/'
    # 1. set csv dataset file names
    dollar_file_name = 'dollar_tjgu_from_2012.csv'
    car_file_name = 'Iran.Khodro_from_2001.csv'
    oil_file_name = 'S_Parsian.Oil&Gas_from_2012.csv'
    home_file_name = 'Maskan.Invest_from_2014.csv'
    gold_file_name = 'Lotus.Gold.Com.ETF_from_2017.csv'

    titles = ['Dollar', 'Home', 'Oil', 'Car', 'Gold']

    # 3. get each dataset according to env
    if colab:
        dfs = read_csv_files_from_drive_in_colab()

        ir_dollar = dfs[dollar_file_name]
        ir_home = dfs[home_file_name]
        ir_oil = dfs[oil_file_name]
        ir_car = dfs[car_file_name]
        ir_gold = dfs[gold_file_name]
    else:
        ir_dollar = pd.read_csv(folder_path + dollar_file_name)
        ir_home = pd.read_csv(folder_path + home_file_name)
        ir_oil = pd.read_csv(folder_path + oil_file_name)
        ir_car = pd.read_csv(folder_path + car_file_name)
        ir_gold = pd.read_csv(folder_path + gold_file_name)
    # print(ir_dollar.head(5))
    # print(ir_dollar.tail(5))
    # 5. preprocess each dataset (sort, date col check, set index, resample)
    ir_dollar = data_preprocessing(ir_dollar, date_col, start_date, end_date,
                                   format=False)
    ir_home = data_preprocessing(ir_home, date_col, start_date, end_date)
    ir_oil = data_preprocessing(ir_oil, date_col, start_date, end_date)
    ir_car = data_preprocessing(ir_car, date_col, start_date, end_date)
    ir_gold = data_preprocessing(ir_gold, date_col, start_date, end_date)

    # 6. plot each dataset
    if plotting:
        plotting(ir_dollar, titles[0])
        plotting(ir_home, titles[1])
        plotting(ir_oil, titles[2])
        plotting(ir_car, titles[3])
        plotting(ir_gold, titles[4])

    # 7. array datasets for horizontally stack in multivariate prediction
    datasets = [ir_dollar, ir_home, ir_oil, ir_car, ir_gold]

    dataset = stack_datasets(datasets, prediction_col)
    # print('sample dataset after stack ', dataset)
    train, test, last = train_test_split(dataset, n_steps)
    # print('train ', train)
    cnn = multivariate_parallel_series_CNN(train, test, n_steps, epochs_for_multivariate_series)
