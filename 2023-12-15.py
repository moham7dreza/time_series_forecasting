# 2023-12-15

# comparison refactored

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
    # print(dataset)

    if (format):
        # Assuming 'date' is the column containing date in integer format
        dataset[date_col] = pd.to_datetime(dataset[date_col], format='%Y%m%d')
    else:
        dataset[date_col] = pd.to_datetime(dataset[date_col])

    # Convert object columns to strings
    object_columns = dataset.select_dtypes(include='object').columns
    dataset[object_columns] = dataset[object_columns].astype(str)

    # print(dataset.dtypes)

    # Identify and exclude object columns
    non_object_columns = dataset.select_dtypes(exclude='object').columns
    # Create a new DataFrame without object columns
    dataset = dataset[non_object_columns]

    # print(dataset)
    dataset = dataset.set_index(date_col)
    # print(dataset.dtypes)

    dataset = dataset.resample('W-Sat').mean().ffill()
    # print(dataset)

    dataset = dataset.loc[start_date:end_date]
    # print(dataset)

    return dataset


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


def get_mutlti_output_CNN_model(n_steps, n_features):
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
    # define output 5
    output5 = Dense(1)(cnn)
    # tie together
    model = Model(inputs=visible, outputs=[output1, output2, output3, output4, output5])
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
    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    # separate output
    y1 = y[:, 0].reshape((y.shape[0], 1))
    y2 = y[:, 1].reshape((y.shape[0], 1))
    y3 = y[:, 2].reshape((y.shape[0], 1))
    y4 = y[:, 3].reshape((y.shape[0], 1))
    y5 = y[:, 4].reshape((y.shape[0], 1))
    # define model
    model = get_mutlti_output_CNN_model(n_steps, n_features)
    # fit model
    model.fit(X, [y1, y2, y3, y4, y5], epochs, verbose=0)
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


def run_for_univariate_series_ir(dataset, col, n_steps, epochs):
    index = -n_steps - 1
    dataset = dataset[col].values
    train = dataset[:index]
    test = dataset[index:-1]
    u_last = flatten_arr(dataset[-1])

    u_cnn = univariant_CNN(train, test, n_steps, epochs)
    u_lstm = univariant_LSTM(train, test, n_steps, epochs)
    u_b_lstm = univariant_BLSTM(train, test, n_steps, epochs)

    return u_cnn, u_lstm, u_b_lstm, u_last


def run_for_multivariate_series_ir(datasets, col, n_steps, epochs):
    index = -n_steps - 1

    dataset = stack_datasets(datasets, col)
    # print('dataset len : ', len(dataset))
    train = dataset[:index]
    test = dataset[index:-1]
    last = flatten_arr(dataset[-1])
    # print('train len : ', len(train))
    # print('test len', len(test))
    # print('test is : ', test)
    # print('last is : ', last)

    cnn = multivariate_parallel_series_CNN(train, test, n_steps, epochs)
    b_lstm = multivariate_parallel_series_BLSTM(train, test, n_steps, epochs)
    s_lstm = multivariate_parallel_series_Stacked_LSTM(train, test, n_steps, epochs)
    mo_cnn = multivariate_parallel_series_CNN_multiple_output(train, test, n_steps, epochs)

    return cnn, b_lstm, s_lstm, mo_cnn, last


if __name__ == '__main__':
    # 1. set csv dataset file names
    dollar_file_name = 'dollar_tjgu_from_2012.csv'
    car_file_name = 'Iran.Khodro_from_2001.csv'
    oil_file_name = 'S_Parsian.Oil&Gas_from_2012.csv'
    home_file_name = 'Maskan.Invest_from_2014.csv'
    gold_file_name = 'Lotus.Gold.Com.ETF_from_2017.csv'

    # 2. set env
    colab = True
    plotting = False

    # 3. get each dataset according to env
    if colab:
        dfs = read_csv_files_from_drive_in_colab()

        ir_dollar = dfs[dollar_file_name]
        ir_home = dfs[home_file_name]
        ir_oil = dfs[oil_file_name]
        ir_car = dfs[car_file_name]
        ir_gold = dfs[gold_file_name]
    else:
        folder_path = 'iran_stock/'

        ir_dollar = pd.read_csv(folder_path + dollar_file_name)
        ir_home = pd.read_csv(folder_path + home_file_name)
        ir_oil = pd.read_csv(folder_path + oil_file_name)
        ir_car = pd.read_csv(folder_path + car_file_name)
        ir_gold = pd.read_csv(folder_path + gold_file_name)

    # 4. set date col name for preprocess data
    date_col = "<DTYYYYMMDD>"
    # 4.1 set start and end dates for data splitting
    start_date = '2017-06-10'
    end_date = '2022-12-03'
    # end_date = datetime.now().date().isoformat()

    # 5. preprocess each dataset (sort, date col check, set index, resample)
    ir_dollar = data_preprocessing(ir_dollar, date_col, start_date, end_date, format=False)
    ir_home = data_preprocessing(ir_home, date_col, start_date, end_date)
    ir_oil = data_preprocessing(ir_oil, date_col, start_date, end_date)
    ir_car = data_preprocessing(ir_car, date_col, start_date, end_date)
    ir_gold = data_preprocessing(ir_gold, date_col, start_date, end_date)

    # 6. plot each dataset
    if plotting:
        plotting(ir_dollar, 'Dollar')
        plotting(ir_home, 'Home')
        plotting(ir_oil, 'Oil')
        plotting(ir_car, 'Car')
        plotting(ir_gold, 'Gold')

    # 7. array datasets for horizontally stack in multivariate prediction
    datasets = [ir_dollar, ir_car, ir_gold, ir_home, ir_oil]

    # 8. select col name for prediction
    col = '<CLOSE>'

    # 8.1 set number of epochs
    epochs_for_multivariate_series = 100
    epochs_for_univariate_series = 100

    # 8.2
    n_steps = 3

    # 9. run algorithms on multivariate series
    cnn, b_lstm, s_lstm, mo_cnn, last = run_for_multivariate_series_ir(datasets, col, n_steps,
                                                                       epochs_for_multivariate_series)

    # 10. run algorithms on univariate dollar
    u_cnn, u_lstm, u_b_lstm, u_last = run_for_univariate_series_ir(ir_dollar, col, n_steps,
                                                                   epochs_for_univariate_series)

    # 11. plot all results of algorithms on dollar
    plot_result(['CNN', 'U-CNN', 'BLSTM', 'U-BLSTM', 'LSTM', 'U-LSTM', 'MO CNN', 'REAL'],
                [cnn[0], u_cnn[0], b_lstm[0], u_b_lstm[0], s_lstm[0], u_lstm[0], mo_cnn[0], u_last], 'dollar')

    # 12. run algorithms on univariate car
    u_cnn, u_lstm, u_b_lstm, u_last = run_for_univariate_series_ir(ir_car, col, n_steps, epochs_for_univariate_series)

    # 13. plot all results of algorithms on car
    plot_result(['CNN', 'U-CNN', 'BLSTM', 'U-BLSTM', 'LSTM', 'U-LSTM', 'MO CNN', 'REAL'],
                [cnn[1], u_cnn[0], b_lstm[1], u_b_lstm[0], s_lstm[1], u_lstm[0], mo_cnn[1], u_last], 'car')

    # 14. run algorithms on univariate gold
    u_cnn, u_lstm, u_b_lstm, u_last = run_for_univariate_series_ir(ir_oil, col, n_steps, epochs_for_univariate_series)

    # 15. plot all results of algorithms on gold
    plot_result(['CNN', 'U-CNN', 'BLSTM', 'U-BLSTM', 'LSTM', 'U-LSTM', 'MO CNN', 'REAL'],
                [cnn[2], u_cnn[0], b_lstm[2], u_b_lstm[0], s_lstm[2], u_lstm[0], mo_cnn[2], u_last], 'gold')

    # 16. run algorithms on univariate home
    u_cnn, u_lstm, u_b_lstm, u_last = run_for_univariate_series_ir(ir_home, col, n_steps, epochs_for_univariate_series)

    # 17. plot all results of algorithms on home
    plot_result(['CNN', 'U-CNN', 'BLSTM', 'U-BLSTM', 'LSTM', 'U-LSTM', 'MO CNN', 'REAL'],
                [cnn[3], u_cnn[0], b_lstm[3], u_b_lstm[0], s_lstm[3], u_lstm[0], mo_cnn[3], u_last], 'home')

    # 18. run algorithms on univariate oil
    u_cnn, u_lstm, u_b_lstm, u_last = run_for_univariate_series_ir(ir_gold, col, n_steps, epochs_for_univariate_series)

    # 19. plot all results of algorithms on oil
    plot_result(['CNN', 'U-CNN', 'BLSTM', 'U-BLSTM', 'LSTM', 'U-LSTM', 'MO CNN', 'REAL'],
                [cnn[4], u_cnn[0], b_lstm[4], u_b_lstm[0], s_lstm[4], u_lstm[0], mo_cnn[4], u_last], 'oil')
