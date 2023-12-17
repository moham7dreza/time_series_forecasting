from pandora.Config import Config
from pandora.DataSampler import DataSampler
from pandora.Helper import Helper
from pandora.ModelBuilder import ModelBuilder


class Multivariate:
    epochs = Config.epochs_for_multivariate_series
    @staticmethod
    def multivariate_parallel_series_CNN(dataset, x_input, n_steps, epochs):
        # convert into input/output
        X, y = DataSampler.split_sequences_for_multivariate_parallel_series(dataset, n_steps)
        # the dataset knows the number of features, e.g. 2
        n_features = X.shape[2]
        # define model
        model = ModelBuilder.get_CNN_model(n_steps, n_features)
        # fit model
        model.fit(X, y, epochs, verbose=0)
        # demonstrate prediction
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        yhat = Helper.flatten_arr(yhat)
        print(yhat)
        return yhat

    @staticmethod
    def multivariate_parallel_series_Stacked_LSTM(dataset, x_input, n_steps, epochs):
        # convert into input/output
        X, y = DataSampler.split_sequences_for_multivariate_parallel_series(dataset, n_steps)
        # the dataset knows the number of features, e.g. 2
        n_features = X.shape[2]
        # define model
        model = ModelBuilder.get_stacked_LSTM_model(n_steps, n_features)
        # fit model
        model.fit(X, y, epochs, verbose=0)
        # demonstrate prediction
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        yhat = Helper.flatten_arr(yhat)
        print(yhat)
        return yhat

    @staticmethod
    def multivariate_parallel_series_BLSTM(dataset, x_input, n_steps, epochs):
        # convert into input/output
        X, y = DataSampler.split_sequences_for_multivariate_parallel_series(dataset, n_steps)
        # the dataset knows the number of features, e.g. 2
        n_features = X.shape[2]
        # define model
        model = ModelBuilder.get_bidirectional_LSTM_model(n_steps, n_features)
        # fit model
        model.fit(X, y, epochs, verbose=0)
        # demonstrate prediction
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        yhat = Helper.flatten_arr(yhat)
        print(yhat)
        return yhat

    @staticmethod
    def multivariate_parallel_series_CNN_multiple_output(dataset, x_input, n_steps, epochs):
        # convert into input/output
        X, y = DataSampler.split_sequences_for_multivariate_parallel_series(dataset, n_steps)
        # the dataset knows the number of features, e.g., 2
        n_features = X.shape[2]

        # separate output dynamically based on n_features
        y_outputs = [y[:, i].reshape((y.shape[0], 1)) for i in range(n_features)]

        # define model
        model = ModelBuilder.get_multi_output_CNN_model(n_steps, n_features)

        # fit model
        model.fit(X, y_outputs, epochs=epochs, verbose=0)

        # demonstrate prediction
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        yhat = Helper.flatten_arr(yhat)
        print(yhat)
        return yhat
