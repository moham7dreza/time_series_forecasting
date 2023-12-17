# last work 2023-11-23

# multivariant parallel series -> multi-output CNN model

from numpy import array, hstack, ravel, float32, concatenate
from keras.layers import (
    Dense,
    Flatten,
    Conv1D,
    MaxPooling1D,
    concatenate,
    Input,
    LSTM,
    TimeDistributed,
    Bidirectional,
)
from keras.models import Model, Sequential
import yfinance as yf
import matplotlib.pyplot as plt


class Multivariant_Multiple_Input_CNN:
    def __init__(self, n_steps, tickers, start, end, epoches, input):
        self.n_steps = n_steps
        self.model = None
        self.n_features = None
        self.n_output = None
        self.tickers = tickers
        self.start_date = start
        self.end_date = end
        self.epoches = epoches
        self.X, self.y = self.extract_input_output()
        self.input = input
        self.models = {}

    def define_models(self):
        self.models = {
            "LSTM": Sequential(
                [
                    LSTM(
                        50,
                        activation="relu",
                        input_shape=(self.n_steps, self.n_features),
                    ),
                    Dense(1),
                ]
            ),
            "BiLSTM": Sequential(
                [
                    Bidirectional(
                        LSTM(50, activation="relu"),
                        input_shape=(self.n_steps, self.n_features),
                    ),
                    Dense(1),
                ]
            ),
            "Stacked LSTM": Sequential(
                [
                    LSTM(
                        50,
                        activation="relu",
                        return_sequences=True,
                        input_shape=(self.n_steps, self.n_features),
                    ),
                    LSTM(50, activation="relu"),
                    Dense(1),
                ]
            ),
            "CNN": Sequential(
                [
                    Conv1D(
                        filters=64,
                        kernel_size=2,
                        activation="relu",
                        input_shape=(self.n_steps, self.n_features),
                    ),
                    MaxPooling1D(pool_size=2),
                    Flatten(),
                    Dense(50, activation="relu"),
                    Dense(1),
                ]
            ),
        }

    # split a multivariate sequence into samples
    def split_sequences(self, sequences):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + self.n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
            X.append(seq_x)
            y.append(seq_y)

        return array(X), array(y)

    def simple_LSTM_model(self):
        model = Sequential()
        model.add(
            LSTM(50, activation="relu", input_shape=(self.n_steps, self.n_features))
        )
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def simple_BiLSTM_model(self):
        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(50, activation="relu"), input_shape=(self.n_steps, self.n_features)
            )
        )
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def stacked_LTSM_model(self):
        model = Sequential()
        model.add(
            LSTM(
                50,
                activation="relu",
                return_sequences=True,
                input_shape=(self.n_steps, self.n_features),
            )
        )
        model.add(LSTM(50, activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def cnn_LSTM_model(self):
        model = Sequential()
        model.add(
            TimeDistributed(
                Conv1D(filters=64, kernel_size=1, activation="relu"),
                input_shape=(None, self.n_steps, self.n_features),
            )
        )
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(50, activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def simple_cnn_model(self):
        model = Sequential()
        model.add(
            Conv1D(
                filters=64,
                kernel_size=2,
                activation="relu",
                input_shape=(self.n_steps, self.n_features),
            )
        )
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def multi_headed_cnn_model(self):
        # first input model
        visible1 = Input(shape=(self.n_steps, self.n_features))
        cnn1 = Conv1D(filters=64, kernel_size=2, activation="relu")(visible1)
        cnn1 = MaxPooling1D(pool_size=2)(cnn1)
        cnn1 = Flatten()(cnn1)
        # second input model
        visible2 = Input(shape=(self.n_steps, self.n_features))
        cnn2 = Conv1D(filters=64, kernel_size=2, activation="relu")(visible2)
        cnn2 = MaxPooling1D(pool_size=2)(cnn2)
        cnn2 = Flatten()(cnn2)
        # merge input models
        merge = concatenate([cnn1, cnn2])
        dense = Dense(50, activation="relu")(merge)
        output = Dense(1)(dense)
        model = Model(inputs=[visible1, visible2], outputs=output)
        model.compile(optimizer="adam", loss="mse")
        return model

    def train_model(self, X):
        # fit model
        self.model.fit(X, self.y, epochs=self.epoches, verbose=0)

    def prepare_data(self):
        # Define input sequence
        in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
        out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

        # Convert to [rows, columns] structure
        in_seq1 = in_seq1.reshape((len(in_seq1), 1))
        in_seq2 = in_seq2.reshape((len(in_seq2), 1))
        out_seq = out_seq.reshape((len(out_seq), 1))

        # Horizontally stack columns
        dataset = hstack((in_seq1, in_seq2, out_seq))

        return dataset

    def download_data(self):
        sequences = []
        for ticker in self.tickers:
            # Download data from Yahoo Finance
            data = yf.download(ticker, start=self.start_date, end=self.end_date)

            # Extract the 'Close' prices
            seq = data["Close"].values

            # Normalize the sequence
            seq /= max(seq)

            sequences.append(seq)

        # Combine the sequences
        combined_seq = hstack([seq.reshape(-1, 1) for seq in sequences])

        return combined_seq

    def extract_input_output(self):
        dataset = self.prepare_data()

        # dataset = self.download_data()

        # Convert into input/output
        X, y = self.split_sequences(dataset)

        return X, y

    def run_multi_headed_cnn_model(self):
        # one time series per head
        self.n_features = 1

        # separate input data
        X1 = self.X[:, :, 0].reshape(self.X.shape[0], self.X.shape[1], self.n_features)
        X2 = self.X[:, :, 1].reshape(self.X.shape[0], self.X.shape[1], self.n_features)

        # Define and compile the model
        self.model = self.multi_headed_cnn_model()

        # Fit the model
        self.train_model(X=[X1, X2])

        self.input = array([[80, 85], [90, 95], [100, 105]])

        x1 = self.input[:, 0].reshape((1, self.n_steps, self.n_features))
        x2 = self.input[:, 1].reshape((1, self.n_steps, self.n_features))

        yhat = self.model.predict([x1, x2], verbose=0)

        return yhat

    def run_simple_cnn_model(self):
        # the dataset knows the number of features, e.g. 2
        self.n_features = self.X.shape[2]

        # Define and compile the model
        self.model = self.simple_cnn_model()

        # Fit the model
        self.train_model(X=self.X)

        self.input = array([[80, 85], [90, 95], [100, 105]])

        self.input = self.input.reshape((1, self.n_steps, self.n_features))

        yhat = self.model.predict(self.input, verbose=0)

        return yhat

    def run_simple_LSTM_model(self):
        # the dataset knows the number of features, e.g. 2
        self.n_features = self.X.shape[2]

        # Define and compile the model
        self.model = self.simple_LSTM_model()

        # Fit the model
        self.train_model(X=self.X)

        self.input = array([[80, 85], [90, 95], [100, 105]])

        self.input = self.input.reshape((1, self.n_steps, self.n_features))

        yhat = self.model.predict(self.input, verbose=0)

        return yhat

    def run_simple_BiLSTM_model(self):
        # the dataset knows the number of features, e.g. 2
        self.n_features = self.X.shape[2]

        # Define and compile the model
        self.model = self.simple_BiLSTM_model()

        # Fit the model
        self.train_model(X=self.X)

        self.input = array([[80, 85], [90, 95], [100, 105]])

        self.input = self.input.reshape((1, self.n_steps, self.n_features))

        yhat = self.model.predict(self.input, verbose=0)

        return yhat

    def run_stacked_LTSM_model(self):
        # the dataset knows the number of features, e.g. 2
        self.n_features = self.X.shape[2]

        # Define and compile the model
        self.model = self.stacked_LTSM_model()

        # Fit the model
        self.train_model(X=self.X)

        self.input = array([[80, 85], [90, 95], [100, 105]])

        self.input = self.input.reshape((1, self.n_steps, self.n_features))

        yhat = self.model.predict(self.input, verbose=0)

        return yhat

    def run_cnn_LSTM_model(self):
        # the dataset knows the number of features, e.g. 2
        self.n_features = self.X.shape[2]

        # Define and compile the model
        self.model = self.cnn_LSTM_model()

        # Fit the model
        self.train_model(X=self.X)

        self.input = array([[80, 85], [90, 95], [100, 105]])

        self.input = self.input.reshape((1, self.n_steps, self.n_features))

        yhat = self.model.predict(self.input, verbose=0)

        return yhat

    def flatten_arr(self, result):
        # Convert the array to a readable format
        flatten_arr = array(result).flatten()

        # flatten_arr = ravel(result)

        return flatten_arr

    def plot_result(self, x, h):
        plt.bar(x, h)
        plt.ylabel("Price")
        plt.title("Average Stock Price Predictions")
        plt.show()

    def run_app(self):
        keys = ["CNN", "MH CNN", "Stacked LTSM", "BiLSTM", "LSTM"]

        results = [
            self.run_simple_cnn_model(),
            self.run_multi_headed_cnn_model(),
            self.run_stacked_LTSM_model(),
            self.run_simple_BiLSTM_model(),
            self.run_simple_LSTM_model(),
        ]
        results = [model for model in self.flatten_arr(results)]

        self.plot_result(keys, results)

        print(
            "\n\n [ + ] result of run -->( CNN )<-- model : ",
            self.flatten_arr(self.run_simple_cnn_model()),
        )

        print(
            "\n\n [ + ] result of run -->( Multi Headed CNN )<-- model : ",
            self.flatten_arr(self.run_multi_headed_cnn_model()),
        )

        print(
            "\n\n [ + ] result of run -->( Stacked LTSM )<-- model : ",
            self.flatten_arr(self.run_stacked_LTSM_model()),
        )

        print(
            "\n\n [ + ] result of run -->( BiLSTM )<-- model : ",
            self.flatten_arr(self.run_simple_BiLSTM_model()),
        )

        print(
            "\n\n [ + ] result of run -->( LSTM )<-- model : ",
            self.flatten_arr(self.run_simple_LSTM_model()),
        )


tickers = ["GOOGL", "AAPL", "MSFT", "TSLA"]
start = "2020-1-1"
end = "2021-1-1"
epoches = 100
n_steps = 3
x_input = array([[80, 85], [90, 95], [100, 105]])

# Create an instance of the MultivariateCNN class
model = Multivariant_Multiple_Input_CNN(
    n_steps=n_steps,
    tickers=tickers,
    start=start,
    end=end,
    epoches=epoches,
    input=x_input,
)

# Run the models
model.run_app()
