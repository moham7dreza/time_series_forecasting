# last work 2023-11-23

# multivariant parallel series -> multi-output CNN model

from numpy import array, hstack
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, concatenate, Input
from keras.models import Model, Sequential
import yfinance as yf


class Multivariant_Multiple_Input_CNN:
    def __init__(self, n_steps, tickers, start, end, epoches, mode):
        self.n_steps = n_steps
        self.model = None
        self.n_features = None
        self.n_output = None
        self.tickers = tickers
        self.start_date = start
        self.end_date = end
        self.epoches = epoches
        self.mode = mode

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

    def define_model(self):
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

    def train_model(self, X, y):
        # fit model
        self.model.fit(X, y, epochs=self.epoches, verbose=0)

    def predict_sequence(self, x_input):
        if self.mode == "multi-headed":
            x1 = x_input[:, 0].reshape((1, self.n_steps, self.n_features))
            x2 = x_input[:, 1].reshape((1, self.n_steps, self.n_features))
            yhat = self.model.predict([x1, x2], verbose=0)

        elif self.mode == "normal":
            x_input = x_input.reshape((1, self.n_steps, self.n_features))
            yhat = self.model.predict(x_input, verbose=0)

        return yhat

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

    def separate_input(self, X):
        # separate input data
        X1 = X[:, :, 0].reshape(X.shape[0], X.shape[1], self.n_features)
        X2 = X[:, :, 1].reshape(X.shape[0], X.shape[1], self.n_features)

        return X1, X2

    def run(self):
        dataset = self.prepare_data()

        # dataset = self.download_data()

        # Convert into input/output
        X, y = self.split_sequences(dataset)

        if self.mode == "multi-headed":
            # one time series per head
            self.n_features = 1

            X1, X2 = self.separate_input(X)

            # Define and compile the model
            self.model = self.multi_headed_cnn_model()

            # Fit the model
            self.train_model([X1, X2], y)

        elif self.mode == "normal":
            # the dataset knows the number of features, e.g. 2
            n_features = X.shape[2]

            # Define and compile the model
            self.model = self.define_model()

            # Fit the model
            self.train_model(X, y)

        x_input = array([[80, 85], [90, 95], [100, 105]])

        result = self.predict_sequence(x_input)

        return result


tickers = ["GOOGL", "AAPL", "MSFT", "TSLA"]
start = "2020-1-1"
end = "2021-1-1"
epoches = 100
n_steps = 3
mode = "normal"

# Create an instance of the MultivariateCNN class
model = Multivariant_Multiple_Input_CNN(
    n_steps=n_steps, tickers=tickers, start=start, end=end, epoches=epoches, mode=mode
)

# Run the model
print(model.run())
