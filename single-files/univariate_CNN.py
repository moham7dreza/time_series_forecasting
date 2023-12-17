# univariate cnn example
import io
import os

import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files, drive
from keras.layers import Dense, Flatten, Input, Conv1D, MaxPooling1D, LSTM, Bidirectional
from keras.models import Sequential, Model
from numpy import array, hstack

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


def train_test_split(dataset, test_size):
    index = -test_size - 1
    train = dataset[:index]
    test = dataset[index:-1]
    u_last = flatten_arr(dataset[-1])

    return train, test, u_last


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


def run(train, test):
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(train, n_steps)
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
    x_input = test.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)


def plot_result(x, h, title):
    plt.bar(x, h)
    plt.ylabel("Price")
    plt.title("Average Stock Price Predictions for " + title)
    plt.show()


def flatten_arr(result):
    items = array(result).flatten()
    return [round(item, 2) for item in items]


if __name__ == '__main__':
    colab = True
    plotting = False
    date_col = "<DTYYYYMMDD>"
    start_date = '2017-06-10'
    end_date = '2022-12-03'
    epochs_for_multivariate_series = 2000
    epochs_for_univariate_series = 100
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

    dfs = read_csv_files_from_drive_in_colab()

    ir_dollar = dfs[dollar_file_name]
    ir_home = dfs[home_file_name]
    ir_oil = dfs[oil_file_name]
    ir_car = dfs[car_file_name]
    ir_gold = dfs[gold_file_name]

    ir_dollar = data_preprocessing(ir_dollar, date_col, start_date, end_date,
                                   format=False)
    ir_home = data_preprocessing(ir_home, date_col, start_date, end_date)
    ir_oil = data_preprocessing(ir_oil, date_col, start_date, end_date)
    ir_car = data_preprocessing(ir_car, date_col, start_date, end_date)
    ir_gold = data_preprocessing(ir_gold, date_col, start_date, end_date)

    labels = ['CNN', 'U-CNN', 'BLSTM', 'U-BLSTM', 'LSTM', 'U-LSTM', 'MO CNN', 'REAL']

    ir_dollar = ir_dollar[prediction_col].values
    ir_home = ir_home[prediction_col].values
    ir_oil = ir_oil[prediction_col].values
    ir_car = ir_car[prediction_col].values
    ir_gold = ir_gold[prediction_col].values

    train, test, u_last = train_test_split(ir_dollar, 3)
    print(u_last)
    run(train, test)

    train, test, u_last = train_test_split(ir_home, 3)
    print(u_last)
    run(train, test)

    train, test, u_last = train_test_split(ir_oil, 3)
    print(u_last)
    run(train, test)

    train, test, u_last = train_test_split(ir_car, 3)
    print(u_last)
    run(train, test)

    train, test, u_last = train_test_split(ir_gold, 3)
    print(u_last)
    run(train, test)